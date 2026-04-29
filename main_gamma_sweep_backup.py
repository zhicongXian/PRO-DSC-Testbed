import os
import sys
import wandb

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
from model.model import PRO_DSC
from model.sink_distance import SinkhornDistance
from data.data_utils import FeatureDataset
from loss.loss_fn import TotalCodingRate
from utils import *
from metrics.clustering import spectral_clustering_metrics, spectral_clustering_metrics_with_ari
import pandas as pd

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
parser.add_argument('--epo', type=int, default=5000,
                    help='number of epochs for training')
parser.add_argument('--bs', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--lr_c', type=float, default=1e-4,
                    help='learning rate for clustering head (default: 0.0001)')
parser.add_argument('--momo', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd1', type=float, default=1e-4,
                    help='weight decay for all other parameters except clustering head (default: 1e-4)')
parser.add_argument('--wd2', type=float, default=5e-3,
                    help='weight decay for clustering head (default: 5e-3)')
parser.add_argument('--pieta', type=float, default=0.1,
                    help='hyper-parameter for Sinkhorn projection')
parser.add_argument('--piiter', type=int, default=1,
                    help='hyper-parameter for Sinkhorn projection')
parser.add_argument('--eps', type=float, default=0.1,
                    help='eps squared for total coding rate (default: 0.1)')
parser.add_argument('--warmup', type=int, default=-1,
                    help='Steps of warmup-up training')

parser.add_argument('--save_every', type=int, default=50,
                    help='model save every')
parser.add_argument('--validate_every', type=int, default=25,
                    help='validate to check the clustering performance')

############## addtitional to this script:

parser.add_argument('--gamma_list', type=int, default=[100,120],
                    help='list of coeff for expr loss')
parser.add_argument('--experiment_name', type=str, default="subspace_coil100")
parser.add_argument('--out_dir', type=str, default="results")

parser.add_argument('--input_dim', type=int, default=128,
                    help='list of coeff for expr loss')

args = parser.parse_args()

datasets_list = ['cifar10','cifar100','cifar10-mcr']#,'cifar20','tinyimagenet','imagenet','imagenetdogs']
assert args.data.lower() in datasets_list, "Only {} are supported".format(','.join(datasets_list))

# parse configurations from yaml
with open(os.path.join('configs','{}.yaml'.format(args.data.lower())), 'r', encoding='utf-8') as file:
    yaml_data = yaml.safe_load(file)
    for key, value in yaml_data.items():
        if hasattr(args, key):
            continue
        setattr(args, key, value) # he does it another way around.
args.desc = '_'.join(
    [formatted_date, args.data, 'gamma{}'.format(args.gamma), 'beta{}'.format(args.beta), args.desc])
print(args)
#################################################################################################################


###################load args to config ##############################################################
global_config = vars(args)
#########################################################################################


def load_dataset(config):
    # Loading features and labels
    if config['data'].lower() in ['cifar10','cifar100','cifar20']:
        feature_dict = torch.load(config['data_dir'])
        clip_features = feature_dict['features'][:50000]
        clip_labels = feature_dict['ys'][:50000]

        feature_dict = torch.load(config['data_dir_val'])
        clip_features_test = feature_dict['features'][-10000:]
        clip_labels_test = feature_dict['ys'][-10000:]

        if config['data'].lower() == 'cifar20':
            from data.dataset import sparse2coarse
            clip_labels = torch.from_numpy(sparse2coarse(clip_labels))
            clip_labels_test = torch.from_numpy(sparse2coarse(clip_labels_test))
    elif config['data'].lower() == "cifar10-mcr":

            with open('data/datasets/CIFAR10-MCR2/cifar10-features.npy', 'rb') as f:
                full_samples = np.load(f)
                clip_features = full_samples[:50000]
                clip_features_test = full_samples[-10000:]
            with open('data/datasets/CIFAR10-MCR2/cifar10-labels.npy', 'rb') as f:
                full_labels = np.load(f)
                clip_labels = full_labels[:50000]
                clip_labels_test = full_labels[-10000:]
            train_ids = np.arange(len(clip_labels))

    else:
        feature_dict = torch.load(config['data_dir'])
        clip_features = feature_dict['features']
        clip_labels = feature_dict['ys']

        feature_dict = torch.load(config['data_dir_val'])
        clip_features_test = feature_dict['features']
        clip_labels_test = feature_dict['ys']

    #### construct dataloader for batch training
    clip_feature_set = FeatureDataset(clip_features, clip_labels)
    train_loader = DataLoader(clip_feature_set, batch_size=config['bs'], shuffle=True, drop_last=True)
    clip_feature_set_test = FeatureDataset(clip_features_test, clip_labels_test)
    test_loader = DataLoader(clip_feature_set_test, batch_size=config['bs'], shuffle=True, drop_last=False)

    return train_loader, test_loader


################################custom model log files #####################
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

######## load the model #####################################


def train(config):
    previous_nmi = None
    same_seeds(config['seed'])
    ######load dataset ############
    train_loader, test_loader = load_dataset(config)
    ############### set up writer ##############################
    desc = config['desc']
    dir_name = os.path.join(f'exps/{desc}')
    writer = init_pipeline_with_config(dir_name, config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PRO_DSC(input_dim=config['input_dim'], hidden_dim=config['hidden_dim'], z_dim=config['z_dim']).to(device) # input_dim=768
    sink_layer = SinkhornDistance(config['pieta'], max_iter=config['piiter'])

    #### loss of logdet()
    warmup_criterion = TotalCodingRate(eps=config['eps'])

    ### optimizer
    param_list = [p for p in model.pre_feature.parameters() if p.requires_grad] + [p for p in model.subspace.parameters() if p.requires_grad]
    param_list_c = [p for p in model.cluster.parameters() if p.requires_grad]
    optimizer = optim.SGD(param_list, lr=config['lr'], momentum=config['momo'], weight_decay=config['wd1'], nesterov=False)
    optimizerc = optim.SGD(param_list_c, lr=config['lr_c'], momentum=config['momo'], weight_decay=config['wd2'], nesterov=False)
    scaler = GradScaler()

    ### warmup iteration setting
    total_wamup_steps = config['warmup']
    warmup_step = 0
    result_df = pd.DataFrame()
    with tqdm(total=config['epo']) as progress_bar:
        for epoch in range(config['epo']):
            progress_bar.set_description('Epoch: '+str(epoch)+'/'+str(config['epo']))
            model.train()
            ### learning loss storage
            loss_dict = {'loss_TCR': [], 'loss_Exp': [], 'loss_Block': []}

            for step, (x, y) in enumerate(train_loader):
                x, y = x.float().to(device), y.to(device)
                y_np = y.detach().cpu().numpy()
                with autocast(enabled=True):
                    z, logits = model(x)
                    self_coeff = (logits @ logits.T)
                    Sign_self_coeff = torch.sign(self_coeff)

                    ### Sinkhorn projection
                    self_coeff = self_coeff.abs().unsqueeze(0)
                    Pi = sink_layer(self_coeff)[0]
                    Pi = Pi * Pi.shape[-1]
                    self_coeff = Pi[0]
                    # eliminate the diagonal value of self_coeff, which fits the constraint of C
                    self_coeff = self_coeff - torch.diag(torch.diag(self_coeff)) # here he also does this!

                    ### compute the affinity matrix
                    A = 0.5 * (self_coeff.abs() + self_coeff.abs().T)
                    A_np = A.detach().cpu().numpy()
                    ### compute W for BDR
                    L = torch.diag(A.sum(1)) - A
                    with torch.no_grad():
                        _, U = torch.linalg.eigh(L)
                        U_hat = U[:, :config['n_clusters']]
                        W = U_hat @ U_hat.T

                    if warmup_step <= total_wamup_steps:
                        loss = warmup_criterion(z)
                        loss_dict['loss_TCR'].append(loss.item())
                    else:
                        loss_tcr = warmup_criterion(z) # logdet() loss
                        loss_exp = 0.5 * (torch.linalg.norm(z.T - z.T @ Sign_self_coeff.mul(A) )) ** 2 / config['bs'] # ||Z-ZC||_F loss
                        loss_bl = torch.trace(L.T @ W) / config['bs'] # r() loss
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
                with torch.no_grad():
                    logits_list = []
                    y_list = []

                    for step, (x, y) in enumerate(test_loader):
                        x, y = x.float().to(device), y.to(device)
                        y_list.append(y.detach().cpu().numpy())
                        _, logits = model(x)
                        logits_list.append(logits)

                    logits = torch.cat(logits_list, dim=0)

                    self_coeff = (logits @ logits.T).abs()

                    y_np = np.concatenate(y_list, axis=0)
                    acc_lst, nmi_lst, pred_lst, ari_lst = spectral_clustering_metrics_with_ari(
                        self_coeff.detach().cpu().numpy(),config['n_clusters'], y_np)
                    writer.add_scalar('ACC', np.max(acc_lst), global_step=epoch)
                    with open('{}/acc.txt'.format(dir_name), 'a') as f:
                        f.write('Logits head mean acc: {} max acc: {} mean nmi: {} max nmi: {}, mean ari: {} max ari: {}, '
                                'epoch {}\n'.format(np.mean(acc_lst), np.max(acc_lst),
                                                    np.mean(nmi_lst), np.max(nmi_lst),
                                                    np.mean(ari_lst), np.max(ari_lst), epoch))
                    print('Logits mean acc: {} max acc: {} mean nmi: {} max nmi: {} ,'
                          ' mean ari: {} max ari: {}, epoch {}\n'.format(np.mean(acc_lst), np.max(acc_lst),
                                                                    np.mean(nmi_lst), np.max(nmi_lst),
                                                                        np.mean(ari_lst), np.max(ari_lst), epoch))
                    ######## add early stop ######################
                    early_stopper = EarlyStopper(patience=5)
                    if early_stopper.early_stop(-np.mean(ari_lst)):
                        break

                    ##############################################

                    if previous_nmi is None:
                        previous_nmi = np.mean(nmi_lst)
                        torch.save(model.state_dict(), '{}/checkpoints/best_model{}.pt'.format(dir_name, epoch))

                        result_df = pd.concat([result_df, pd.DataFrame.from_records(
                            [{'dataset': config['data'].lower(), 'seed': config['seed'],
                              'gamma': config['gamma'],
                              'epoch': epoch, 'acc': np.mean(acc_lst),
                              'nmi': np.mean(nmi_lst),
                              'ari': np.mean(ari_lst),
                              }])])

                        if not os.path.exists(config['out_dir']):
                            os.makedirs(config['out_dir'])
                        result_df.to_csv(
                            '{}/{}_{}.csv'.format(
                                config['out_dir'], config['data'].lower(), config['experiment_name']), index=False)

                    elif np.mean(nmi_lst) > previous_nmi:
                        previous_nmi = np.mean(nmi_lst)

                        torch.save(model.state_dict(), '{}/checkpoints/best_model{}.pt'.format(dir_name, epoch))

                        result_df = pd.concat([result_df, pd.DataFrame.from_records(
                            [{'seq_name': config['data'].lower(), 'seed': config['seed'], 'epoch': epoch, 'acc': np.mean(acc_lst),
                              'nmi': np.mean(nmi_lst), 'ari': np.mean(ari_lst)
                              }])])

                        result_df.to_csv(
                            '{}/{}_{}.csv'.format(
                                config['out_dir'], config['data'].lower(),config['experiment_name']), index=False)


                    if wandb.run:
                        wandb.log({
                            "epoch": epoch,
                            "acc": np.mean(acc_lst),
                            "nmi": np.mean(nmi_lst),
                            "ari": np.mean(ari_lst),
                            "gamma":config['gamma']
                        })



def load_sweep_config():
    sweep_config = {"method": "grid"}
    parameters_dict = {}
    gamma_list = list(np.linspace(10, 1000, 20))
    parameters_dict.update({'gamma':{"values":gamma_list}})
    eval_metric = {"name": "ari", "goal": "maximize"}
    sweep_config["parameters"] = parameters_dict
    sweep_config["metric"] = eval_metric

    return sweep_config

def interface_to_train():
    config = global_config
    with wandb.init(project="pro_dsc_cifar", config=config):

        for key in wandb.config.as_dict():
            config[key] = wandb.config.as_dict().get(key)



        train(config)

if __name__ == '__main__':
    sweep_config = load_sweep_config()
    sweep_id = ""
    count = 200

    if sweep_id == "":
        sweep_id = wandb.sweep(sweep_config, project="pro_dsc_cifar")

    wandb.agent(
        sweep_id,
        function=interface_to_train,
        count=count,
    )
    api= wandb.Api()
    best_run = api.sweep(api.default_entity + "/pro_dsc_cifar/"+sweep_id ).best_run()
    result = {
        "parameters": dict(best_run.config),
        "metrics": dict(best_run.summary._json_dict)
    }
    print("result to serialize: ", result)
    with open(global_config['out_dir']+"/"+global_config["data"]+'_'+global_config["experiment_name"] + "_"+"best_gamma_sweep_result.json", "w") as f:
        json.dump(result, f, indent=4)
    # best_run = sweep.best_run()
    #
    # best_params = best_run.config
    # best_metric = best_run.summary
    #
    # print("Best parameters:", best_params)
    # print("Best metric:", best_metric)