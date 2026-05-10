import os
import sys

sys.path.append('./')

from datetime import datetime

current_date = datetime.now()
formatted_date = current_date.strftime('%m-%d')

import argparse
import yaml
import torch
import numpy as np
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.DSCNet import PRO_DSC
from model.sink_distance import SinkhornDistance
from data.data_utils import FeatureDataset
from loss.loss_fn import TotalCodingRate
from utils import *
from metrics.clustering import spectral_clustering_metrics_with_ari_and_subspace_discovery_error
import scipy.io as sio
import pandas as pd
import wandb
import time

parser = argparse.ArgumentParser(description='PRO-DSC Training')
parser.add_argument('--desc', type=str, default='exp',
                    help='description of the experiment')

parser.add_argument('--data', type=str, default='cifar100',
                    help='dataset to use')
parser.add_argument('--gamma', type=int, default=100,
                    help='coeff for expr loss')
parser.add_argument('--beta', type=int, default=100,
                    help='coeff for block prior loss')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

parser.add_argument('--hidden_dim', type=int, default=4096,
                    help='hidden dimension of the pre_feature layer')
parser.add_argument('--z_dim', type=int, default=128,
                    help='dimension of the learned representation')
parser.add_argument('--n_clusters', type=int, default=10,
                    help='number of subspaces to cluster')
parser.add_argument('--epo', type=int, default=15,
                    help='number of epochs for training')
parser.add_argument('--bs', type=int, default=1024,
                    help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momo', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd1', type=float, default=1e-4,
                    help='weight decay for all other parameters except clustering head (default: 1e-4)')
parser.add_argument('--wd2', type=float, default=5e-3,
                    help='weight decay for clustering head (default: 5e-3)')
parser.add_argument('--eps', type=float, default=0.1,
                    help='eps squared for total coding rate (default: 0.1)')
parser.add_argument('--warmup', type=int, default=-1,
                    help='Steps of warmup-up training')

parser.add_argument('--save_every', type=int, default=50,
                    help='model save every')
parser.add_argument('--validate_every', type=int, default=25,
                    help='validate to check the clustering performance')
parser.add_argument('--experiment_name', type=str, default="subspace_coil100")
parser.add_argument('--out_dir', type=str, default="results")

args = parser.parse_args()

datasets_list = ['eyaleb', 'coil100', 'orl']
assert args.data.lower() in datasets_list, "Only {} are supported".format(','.join(datasets_list))

# parse configurations from yaml
with open(os.path.join('configs', '{}.yaml'.format(args.data.lower())), 'r', encoding='utf-8') as file:
    yaml_data = yaml.safe_load(file)
    for key, value in yaml_data.items():
        if key == "experiment_name" or key == "seed":
            continue
        setattr(args, key, value)
args.desc = '_'.join(
    [formatted_date, args.data, 'gamma{}'.format(args.gamma), 'beta{}'.format(args.beta), args.desc])
print(args)
global_config = vars(args)

def init_pipeline_with_config(model_dir, config):
    """Initialize folders and Seed for experiments"""


    """Initialize folders and Seed for experiments"""
    # project folder
    if os.path.exists(model_dir):
        print('EXP PATH EXISTS, PLEASE BE CAUTIOUS')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'codes'), exist_ok=True)

    # save exp settings
    save_params(model_dir, config)

    # GPU and seed setup
    os.environ['PYTHONHASHSEED'] = str(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True

    # tensorboard settings
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(model_dir, 'tensorboard'))

    # copy codes
    for filepath in os.listdir('./'):
        if filepath.endswith('.py'):
            shutil.copyfile(os.path.join('./', filepath), os.path.join(model_dir, 'codes', filepath))
    return writer

#################################################################################################################
def train(config):

    desc = config['desc']
    dir_name = os.path.join(f'exps/{desc}')
    writer = init_pipeline_with_config(dir_name, config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PRO_DSC(hidden_dim=config['hidden_dim'], z_dim=config['z_dim'], channels=config['channels'], kernels=config['kernels']).to(device)
    sink_layer = SinkhornDistance(config['pieta'], max_iter=1)

    if config['data'].lower() == 'orl':
        # Loading features and labels
        data = sio.loadmat(config['data_dir'])
        x, y = data['X'].reshape((-1, 1, 32, 32)), data['Y']  # data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        # network and optimization parameters
        num_sample = x.shape[0]
        temp = model.pre_feature.load_state_dict(torch.load('DSCNet_AE_pretrain/orl.pkl'), strict=False)

    elif config['data'].lower() == 'eyaleb':
        data = sio.loadmat(config['data_dir'])
        img = data['Y']
        I = []
        Label = []
        for i in range(img.shape[2]):
            for j in range(img.shape[1]):
                temp = np.reshape(img[:, j, i], [42, 48])
                Label.append(i)
                I.append(temp)
        I = np.array(I)
        y = np.array(Label[:])
        Img = np.transpose(I, [0, 2, 1])
        x = np.expand_dims(Img[:], 1).astype(float)
        y = y - y.min()

        num_class = 38
        num_sample = num_class * 64
        temp = model.pre_feature.load_state_dict(torch.load('DSCNet_AE_pretrain/yaleb.pkl'), strict=False)

    elif  config['data'].lower() == 'coil100':
        data = sio.loadmat(config['data_dir'])
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        num_sample = x.shape[0]
        temp = model.pre_feature.load_state_dict(torch.load('DSCNet_AE_pretrain/coil100.pkl'), strict=False)

    #### construct dataloader for batch training
    config['bs'] = num_sample
    feature_set = FeatureDataset(x, y)
    train_loader = DataLoader(feature_set, batch_size=config['bs'], shuffle=True, drop_last=True, num_workers=0)
    feature_set_test = FeatureDataset(x, y)
    test_loader = DataLoader(feature_set_test,  batch_size=config['bs'], shuffle=True, drop_last=False, num_workers=0)

    #### loss of TCR
    warmup_criterion = TotalCodingRate(eps=config['eps'])

    ### learning opt strategy
    param_list = [p for p in model.pre_feature.parameters() if p.requires_grad] + [p for p in model.subspace.parameters() if
                                                                                   p.requires_grad]
    param_list_c = [p for p in model.cluster.parameters() if p.requires_grad]
    optimizer = optim.SGD(param_list, lr=config['lr'], momentum=config['momo'], weight_decay=config['wd1'], nesterov=False)
    optimizerc = optim.SGD(param_list_c, lr=config['lr'], momentum=config['momo'], weight_decay=config['wd1'], nesterov=False)
    scaler = GradScaler()

    ### warmup iteration setting
    total_wamup_steps = config['warmup']
    warmup_step = 0
    print("before training configs:", config)

    with tqdm(total=config['epo']) as progress_bar:
        t_begin = time.time()
        for epoch in range(config['epo']):
            progress_bar.set_description('Epoch: ' + str(epoch) + '/' + str(config['epo']))
            model.train()
            ### learning loss storage
            loss_dict = {'loss_TCR': [], 'loss_Exp': [], 'loss_Block': []}

            for step, (x, y) in enumerate(train_loader):
                x, y = x.float().to(device), y.to(device)
                y_np = y.detach().cpu().numpy()
                with torch.amp.autocast('cuda', enabled=True):
                    z, logits = model(x)
                    self_coeff = (logits @ logits.T)
                    Sign_self_coeff = torch.sign(self_coeff)

                    ### Sinkhorn projection
                    self_coeff = self_coeff.abs().unsqueeze(0)
                    Pi = sink_layer(self_coeff)[0]
                    Pi = Pi * Pi.shape[-1]
                    self_coeff = Pi[0]
                    # eliminate the diagonal value of self_coeff, which fits the constraint of C
                    self_coeff = self_coeff - torch.diag(torch.diag(self_coeff))

                    ### compute the affinity matrix
                    A = 0.5 * (self_coeff.abs() + self_coeff.abs().T)
                    A_np = A.detach().cpu().numpy()
                    ### compute W for BDR
                    L = torch.diag(A.sum(1)) - A
                    with torch.no_grad():
                        try:
                            _, U = torch.linalg.eigh(L) # to do what happen when pytorch fail to converge?
                        except Exception as e:
                            print(e)
                            assert torch.isfinite(L).all(), "A contains NaN or Inf"
                            A = L.to(torch.float64)

                            # Force symmetry / Hermitian
                            A = 0.5 * (A + A.mH)

                            # Normalize scale to avoid huge/small values
                            scale = A.norm(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(A.dtype).tiny)
                            A = A / scale

                            # Ridge regularization: good for covariance / PSD matrices
                            I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
                            eps = torch.finfo(A.dtype).tiny
                            A = A + eps * I

                            _, U = torch.linalg.eigh(A)
                        U_hat = U[:, :config['n_clusters']]
                        W = U_hat @ U_hat.T

                    if warmup_step <= total_wamup_steps:
                        loss = warmup_criterion(z)
                        loss_dict['loss_TCR'].append(loss.item())
                    else:
                        loss_tcr = warmup_criterion(z)  # logdet() loss
                        loss_exp = 0.5 * (torch.linalg.norm(
                            z.T - z.T @ Sign_self_coeff.mul(self_coeff))) ** 2 / config['bs']  # ||Z-ZC||_F loss
                        loss_bl = torch.trace(L.T @ W) / config['bs']  # r() loss
                        loss = loss_tcr + config['gamma'] * loss_exp + config['beta'] * loss_bl

                        loss_dict['loss_TCR'].append(loss_tcr.item())
                        loss_dict['loss_Exp'].append(loss_exp.item())
                        loss_dict['loss_Block'].append(loss_bl.item())

                if warmup_step <= total_wamup_steps:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.zero_grad()
                    optimizerc.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.step(optimizerc)
                    scaler.update()

                if warmup_step == total_wamup_steps:
                    print("Warmup Ends...Start training...")
                    model = update_pi_from_z(model)

                if warmup_step <= total_wamup_steps:
                    progress_bar.set_postfix(tcr_loss="{:5.4f}".format(loss.item()))
                else:
                    progress_bar.set_postfix(
                        tcr_loss="{:5.4f}".format(loss_tcr.item()),
                        exp_loss="{:5.4f}".format(loss_exp.item()),
                        block_loss="{:5.4f}".format(loss_bl.item()),
                    )
                warmup_step += 1
            progress_bar.update(1)

            for k in loss_dict.keys():
                if len(loss_dict[k]) != 0:
                    writer.add_scalar(k, np.mean(loss_dict[k]), global_step=epoch)
                else:
                    writer.add_scalar(k, 0, global_step=epoch)

            if (epoch + 1) % config['save_every'] == 0 or (epoch + 1) == config['epo']:
                torch.save(model.state_dict(), '{}/checkpoints/model{}.pt'.format(dir_name, epoch))

            ### evaluate on test set
            if (epoch + 1) % config['validate_every'] == 0 or (epoch + 1) == config['epo']:
                print('EVAL on VALIDATE DATASETS')
                model.eval()
                t_end = time.time()
                with torch.no_grad():
                    logits_list = []
                    z_list = []
                    y_list = []

                    for step, (x, y) in enumerate(test_loader):
                        x, y = x.float().to(device), y.to(device)
                        y_list.append(y.detach().cpu().numpy())
                        z, logits = model(x)
                        logits_list.append(logits)
                        z_list.append(z)

                    logits = torch.cat(logits_list, dim=0)
                    z = torch.cat(z_list, dim=0)

                    self_coeff = (logits @ logits.T).abs().unsqueeze(0)
                    Pi = sink_layer(self_coeff)[0]
                    Pi = Pi * Pi.shape[-1]
                    self_coeff = Pi[0]

                    y_np = np.concatenate(y_list, axis=0)
                    acc_lst, nmi_lst, pred_lst, ari_lst, sde_lst = spectral_clustering_metrics_with_ari_and_subspace_discovery_error(
                        self_coeff.detach().cpu().numpy(), args.n_clusters, y_np,
                        seed=config['seed'])
                    writer.add_scalar('ACC', np.max(acc_lst), global_step=epoch)

                    with open('{}/acc.txt'.format(dir_name), 'a') as f:
                        f.write(
                            'Logits head mean acc: {} max acc: {} mean nmi: {} max nmi: {}, mean ari: {} max ari: {}, mean sdi: {}, max sdi: {}  epoch {}\n'.format(
                                np.mean(acc_lst), np.max(acc_lst),
                                np.mean(nmi_lst), np.max(nmi_lst), np.mean(ari_lst), np.max(ari_lst), np.mean(sde_lst),
                                np.max(sde_lst), epoch))
                    print(
                        'Logits head mean acc: {} max acc: {} mean nmi: {} max nmi: {}, mean ari: {} max ari: {}, mean sdi: {}, max sdi: {}  epoch {}\n'.format(
                            np.mean(acc_lst), np.max(acc_lst),
                            np.mean(nmi_lst), np.max(nmi_lst), np.mean(ari_lst), np.max(ari_lst), np.mean(sde_lst),
                            np.max(sde_lst), epoch))

                    result_df = pd.concat([result_df, pd.DataFrame.from_records(
                        [{'seq_name': args.data.lower(), 'seed': config['seed'], 'epoch': epoch, 'gamma': config['gamma'],
                          'acc': np.mean(acc_lst),
                          'nmi': np.mean(nmi_lst),
                          'ari': np.mean(ari_lst),
                          'subspace_discovery_err:': np.mean(sde_lst)
                          }])])

                    result_df.to_csv(
                        '{}/{}_{}.csv'.format(
                            args.out_dir, args.data.lower(), args.experiment_name), index=False, mode='a')



                    if wandb.run:
                        wandb.log({
                            "epoch": epoch,
                            "acc": np.mean(acc_lst),
                            "nmi": np.mean(nmi_lst),
                            "ari": np.mean(ari_lst),
                            "gamma": config['gamma'],
                            "seed": config['seed'],
                            'subspace_discovery_err:': np.mean(sde_lst)
                        })

def load_sweep_config():
    sweep_config = {"method": "grid"}
    parameters_dict = {}
    gamma_list = [200]+np.arange(30, 1000, 10).tolist()+[10,20]
    parameters_dict.update({'gamma':{"values":gamma_list}})
    eval_metric = {"name": "ari", "goal": "maximize"}
    sweep_config["parameters"] = parameters_dict
    sweep_config["metric"] = eval_metric

    return sweep_config


def interface_to_train():
    config = global_config
    with wandb.init(project="pro_dsc"+'_'+config["experiment_name"], config=config):

        for key in wandb.config.as_dict():
            config[key] = wandb.config.as_dict().get(key)

        train(config)

if __name__ == '__main__':
    sweep_config = load_sweep_config()
    sweep_id = ""
    count = 200

    if sweep_id == "":
        sweep_id = wandb.sweep(sweep_config, project="pro_dsc"+'_'+global_config["experiment_name"])

    wandb.agent(
        sweep_id,
        function=interface_to_train,
        count=count,
    )
    api= wandb.Api()
    best_run = api.sweep(api.default_entity + "/pro_dsc_"+global_config["experiment_name"]+"/"+sweep_id ).best_run()
    result = {
        "parameters": dict(best_run.config),
        "metrics": dict(best_run.summary._json_dict)
    }
    print("result to serialize: ", result)
    with open(global_config['out_dir']+"/"+global_config["data"]+'_'+global_config["experiment_name"] + "_"+"best_gamma_sweep_result.json", "w") as f:
        json.dump(result, f, indent=4)