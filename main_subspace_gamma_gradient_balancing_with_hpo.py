import os
import sys

sys.path.append('./')

from datetime import datetime

current_date = datetime.now()
formatted_date = current_date.strftime('%m-%d')
import math
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
from metrics.clustering import spectral_clustering_metrics_with_ari_and_subspace_discovery_error_with_seeds
import scipy.io as sio
import pandas as pd
import wandb
import time
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from copy import deepcopy
import json
from sklearn.metrics import silhouette_score
import logging
import  jax.scipy as jsc
import  jax.numpy as jnp


logging.basicConfig(level=logging.DEBUG,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
optuna.logging.set_verbosity(optuna.logging.INFO)
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()
logger = logging.getLogger("subspace_auto_gamma_gradient_balancing")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger.addHandler(handler)


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
parser.add_argument('--constant_factor', type=float, default=1.0,
                    help='pro dsc input dim')


def update_balanced_weights_from_tensor(
    L_se,
    L_bd,
    intermediate_tensor,
    lambda_se,
    lambda_bd,
    momentum=0.9,
    eps=1e-8
):
    g_se = grad_norm_wrt_tensor(L_se, intermediate_tensor)
    g_bd = grad_norm_wrt_tensor(L_bd, intermediate_tensor)

    g_avg = 0.5 * (g_se + g_bd)

    new_lambda_se = g_avg / (g_se + eps)
    new_lambda_bd = g_avg / (g_bd + eps)

    lambda_se = momentum * lambda_se + (1 - momentum) * new_lambda_se
    lambda_bd = momentum * lambda_bd + (1 - momentum) * new_lambda_bd

    return lambda_se.detach(), lambda_bd.detach(), g_se.item(), g_bd.item()


def parse_list(value):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON list: {e}")

    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("Argument must be a JSON list")

    return parsed


parser.add_argument('-s', '--seeds', type=parse_list, help='here you can set a list of seeds', default=[1, 2, 3])
# Use like:

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
logger.info(args)
global_config = vars(args)
wandb_kwargs = {"project": "pro_dsc" + '_' + global_config['data'] + '_' + global_config["experiment_name"]}
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)
def init_pipeline_with_config(model_dir, config):
    """Initialize folders and Seed for experiments"""


    """Initialize folders and Seed for experiments"""
    # project folder
    if os.path.exists(model_dir):
        logger.info('EXP PATH EXISTS, PLEASE BE CAUTIOUS')
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
def imqrginv_fixed(a: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    # q, r, p = sla.qr(a, mode="economic", pivoting=True)
    q, r = jnp.linalg.qr(a, mode="reduced")

    r_take = np.any(np.abs(r) > tol, axis=1)
    r = r[r_take, ::]
    q = q[::, r_take]

    return (
        q@ np.asarray(jsc.linalg.solve(
            a=r @ r.T,
            b=r,
            assume_a="pos",
            check_finite=False,
            overwrite_a=True,
            overwrite_b=True,
        ))
    ).T  # [np.argsort(p), ::]

def grad_norm_wrt_tensor(loss, tensor, eps=1e-12):
    g = torch.autograd.grad(
        loss,
        tensor,
        retain_graph=True,
        create_graph=False,
        allow_unused=True
    )[0]

    if g is None:
        return torch.tensor(0.0, device=tensor.device)

    return torch.sqrt(torch.sum(g.detach() ** 2) + eps)

def init_trial(config, device, train_loader, test_loader):
    model = PRO_DSC(hidden_dim=config['hidden_dim'], z_dim=config['z_dim'], channels=config['channels'], kernels=config['kernels']).to(device)
    sink_layer = SinkhornDistance(config['pieta'], max_iter=1)

    #### loss of logdet()
    warmup_criterion = TotalCodingRate(eps=config['eps'])

    # load dataset:
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


    ### optimizer
    param_list = [p for p in model.pre_feature.parameters() if p.requires_grad] + [p for p in model.subspace.parameters() if
                                                                                   p.requires_grad]
    param_list_c = [p for p in model.cluster.parameters() if p.requires_grad]
    optimizer = optim.SGD(param_list, lr=config['lr'], momentum=config['momo'], weight_decay=config['wd1'], nesterov=False)
    optimizerc = optim.SGD(param_list_c, lr=config['lr'], momentum=config['momo'], weight_decay=config['wd1'], nesterov=False)
    scaler = GradScaler()

    ### warmup iteration setting
    warmup_epochs = config['warmup']
    warmup_step = 0
    result_df = pd.DataFrame()
    final_ari = 0
    gamma_estimated_list = []
    gradient_ratio = 1
    gamma = None
    gamma_previous = None
    init_epoch = warmup_epochs + 3
    lambda_se = torch.tensor(1.0)
    lambda_bd = torch.tensor(1.0)
    with tqdm(total=init_epoch) as progress_bar:
        t_begin = time.time()
        for epoch in range(init_epoch):
            progress_bar.set_description('Epoch: ' + str(epoch) + '/' + str(init_epoch))
            model.train()
            ### learning loss storage
            loss_dict = {'loss_TCR': [], 'loss_Exp': [], 'loss_Block': []}
            if len(gamma_estimated_list) > 0:
                gamma_estimated_list = [np.nan if x is None else x for x in gamma_estimated_list],
                gamma = np.nanmean(np.array(gamma_estimated_list))

                # # remove the scheduling
                if gamma_previous is None:
                    gamma_previous = gamma
                elif gamma_previous < gamma:
                    gamma = gamma_previous
                else:
                    gamma_previous = gamma

                gamma_estimated_list = []
                logger.info(f"estimated gamma {gamma}, default gamma is {args.gamma}")

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
                    self_coeff = self_coeff - torch.diag(torch.diag(self_coeff))  # here he also does this!

                    ### compute the affinity matrix
                    A = 0.5 * (self_coeff.abs() + self_coeff.abs().T)
                    A_np = A.detach().cpu().numpy()
                    ### compute W for BDR
                    L = torch.diag(A.sum(1)) - A
                    with torch.no_grad():
                        _, U = torch.linalg.eigh(L)
                        U_hat = U[:, :config['n_clusters']]
                        W = U_hat @ U_hat.T

                    if epoch <= warmup_epochs:
                        loss = warmup_criterion(z)
                        loss_dict['loss_TCR'].append(loss.item())
                    else:
                        loss_tcr = warmup_criterion(z)  # logdet() loss
                        loss_exp = 0.5 * (torch.linalg.norm(z.T - z.T @ Sign_self_coeff.mul(A))) ** 2 / config[
                            'bs']  # ||Z-ZC||_F loss
                        loss_bl = torch.trace(L.T @ W) / config['bs']  # r() loss

                        # here add the gamma estimations:
                        if epoch > warmup_epochs:  # run on every steps and warmup_step <= total_wamup_steps + nb_steps_per_epoch   no initial pretraining is used:
                            with torch.no_grad():
                                block = z.detach().clone().double()
                                ########## Old way to calculate pseudo inverse and somehow does not lead to identity matrix ##
                                approx_pseudo = imqrginv_fixed(block.detach().cpu().numpy())
                                c_matrix = np.dot(block.detach().cpu().numpy(),
                                                  approx_pseudo)
                                #######################################

                                diagIndices = np.diag_indices(c_matrix.shape[0])
                                c_matrix[diagIndices] = 0

                                # this is especially psueo inverse leads to identity matrices
                                logger.debug(f"constant factor is: {config['constant_factor']}")
                                gamma_estimated = config['constant_factor'] * (np.linalg.norm(c_matrix, 1,
                                                                                              axis=0).sum() / args.bs) * args.beta
                                logger.debug(f"before gardient ration: {gamma_estimated}")
                                logger.debug(f"after gardient ration: , {gamma_estimated / gradient_ratio}")
                                block_reconstructed = torch.from_numpy(c_matrix).to(device) @ block
                                approx_err = torch.sum((block - block_reconstructed) ** 2).item() / args.bs

                                logger.debug(f"current approx err: , {approx_err}")
                                 #/ gradient_ratio
                                if math.sqrt(approx_err) < 0.3:
                                    gamma_estimated = gamma_estimated * gradient_ratio
                                    gamma_estimated_list.append(gamma_estimated)
                                else:
                                    gamma_estimated_list.append(gamma_estimated)
                                logger.debug(f"current estimated gamma: {gamma_estimated}")
                        if gamma is None:
                            loss = loss_tcr + config['gamma'] * loss_exp + config['beta'] * loss_bl
                        else:
                            loss = loss_tcr + gamma * loss_exp + config['beta'] * loss_bl
                            logger.debug(f"estimated gamma {gamma}, default gamma is {args.gamma}")

                        loss_dict['loss_TCR'].append(loss_tcr.item())
                        loss_dict['loss_Exp'].append(loss_exp.item())
                        loss_dict['loss_Block'].append(loss_bl.item())
                        lambda_se, lambda_bd, g_se, g_bd = update_balanced_weights_from_tensor(
                            L_se=loss_exp,
                            L_bd=loss_bl,
                            intermediate_tensor=self_coeff,
                            lambda_se=lambda_se,
                            lambda_bd=lambda_bd
                        )
                        logger.debug(f"new lambda_bd, {g_bd}")
                        logger.debug(f"new lambda_se {g_se}", )
                        logger.debug(f"ratio: {(g_bd / g_se)}")
                        gradient_ratio = g_bd # / g_se

                if epoch <= warmup_epochs:
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

                if epoch == warmup_epochs:
                    logger.info("Warmup Ends...Start training...")
                    model = update_pi_from_z(model)

                if epoch <= warmup_epochs:
                    progress_bar.set_postfix(tcr_loss="{:5.4f}".format(loss.item()))
                else:
                    progress_bar.set_postfix(
                        tcr_loss="{:5.4f}".format(loss_tcr.item()),
                        exp_loss="{:5.4f}".format(loss_exp.item()),
                        block_loss="{:5.4f}".format(loss_bl.item()),
                    )
                warmup_step += 1
            progress_bar.update(1)
    if gamma >1000:
        gamma = 1000
    return gamma

@wandbc.track_in_wandb()
def objective( trial : optuna.trial.Trial):
    config = global_config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['constant_factor'] = trial.suggest_float("constant_factor", 0.25, 4)

    model = PRO_DSC(hidden_dim=config['hidden_dim'], z_dim=config['z_dim'], channels=config['channels'], kernels=config['kernels']).to(device)
    sink_layer = SinkhornDistance(config['pieta'], max_iter=1)

    # load dataset:
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



    same_seeds(config['seed'])
    desc = config['desc']
    dir_name = os.path.join(f'exps/{desc}')
    writer = init_pipeline_with_config(dir_name, config)


    gamma = init_trial(config, device, train_loader, test_loader)




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
    logger.debug(f"before training configs: {config}")
    result_df = pd.DataFrame()
    final_ari = 0
    early_stopper = EarlyStopper(patience=20, min_delta=0.005)
    early_stop = False
    with tqdm(total=config['epo']) as progress_bar:
        t_begin = time.time()
        for epoch in range(config['epo']):
            progress_bar.set_description('Epoch: ' + str(epoch) + '/' + str(config['epo']))
            model.train()
            ### learning loss storage
            loss_dict = {'loss_TCR': [], 'loss_Exp': [], 'loss_Block': []}
            loss_per_epoch = []
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
                            logger.error(e)
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

                    if epoch <= total_wamup_steps:
                        loss = warmup_criterion(z)
                        loss_dict['loss_TCR'].append(loss.item())
                    else:
                        loss_tcr = warmup_criterion(z)  # logdet() loss
                        loss_exp = 0.5 * (torch.linalg.norm(
                            z.T - z.T @ Sign_self_coeff.mul(self_coeff))) ** 2 / config['bs']  # ||Z-ZC||_F loss
                        loss_bl = torch.trace(L.T @ W) / config['bs']  # r() loss
                        loss = loss_tcr + gamma * loss_exp + config['beta'] * loss_bl

                        loss_dict['loss_TCR'].append(loss_tcr.item())
                        loss_dict['loss_Exp'].append(loss_exp.item())
                        loss_dict['loss_Block'].append(loss_bl.item())
                    loss_per_epoch.append(loss.item())


                if epoch <= total_wamup_steps:
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

                if epoch == total_wamup_steps:
                    logger.info("Warmup Ends...Start training...")
                    model = update_pi_from_z(model)

                if epoch <= total_wamup_steps:
                    progress_bar.set_postfix(tcr_loss="{:5.4f}".format(loss.item()))
                else:
                    progress_bar.set_postfix(
                        tcr_loss="{:5.4f}".format(loss_tcr.item()),
                        exp_loss="{:5.4f}".format(loss_exp.item()),
                        block_loss="{:5.4f}".format(loss_bl.item()),
                    )
                warmup_step += 1
            progress_bar.update(1)

            if epoch > total_wamup_steps:
                if early_stopper.early_stop(np.mean(loss_per_epoch)):
                    early_stop = True

            for k in loss_dict.keys():
                if len(loss_dict[k]) != 0:
                    writer.add_scalar(k, np.mean(loss_dict[k]), global_step=epoch)
                else:
                    writer.add_scalar(k, 0, global_step=epoch)

            if (epoch + 1) % config['save_every'] == 0 or (epoch + 1) == config['epo']:
                torch.save(model.state_dict(), '{}/checkpoints/model{}.pt'.format(dir_name, epoch))

            ### evaluate on test set
            if (epoch + 1) % config['validate_every'] == 0 or (epoch + 1) == config['epo'] or early_stop:
                logger.info('EVAL on VALIDATE DATASETS')
                model.eval()
                t_end = time.time()
                with torch.no_grad():
                    logits_list = []
                    z_list = []
                    y_list = []
                    x_list = []

                    for step, (x, y) in enumerate(test_loader):
                        x, y = x.float().to(device), y.to(device)
                        y_list.append(y.detach().cpu().numpy())
                        x_list.append(x.detach().cpu().numpy())
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
                    x_np = np.concatenate(x_list, axis=0)

                    acc_lst, nmi_lst, pred_lst, ari_lst, sde_lst, si_list = spectral_clustering_metrics_with_ari_and_subspace_discovery_error_with_seeds(
                        np.reshape(x_np, (len(x_np), -1)), self_coeff.detach().cpu().numpy(), args.n_clusters, y_np,
                        seeds=[config['seed']])
                    writer.add_scalar('ACC', np.max(acc_lst), global_step=epoch)

                    si_score = np.mean(si_list) # silhouette_score(x_np, pred_lst[0])  # since we now set the same seed
                    with open('{}/acc.txt'.format(dir_name), 'a') as f:
                        f.write(
                            'Logits head mean acc: {} max acc: {} mean nmi: {} max nmi: {}, mean ari: {} max ari: {}, mean sdi: {}, max sdi: {}  epoch {}\n'.format(
                                np.mean(acc_lst), np.max(acc_lst),
                                np.mean(nmi_lst), np.max(nmi_lst), np.mean(ari_lst), np.max(ari_lst), np.mean(sde_lst),
                                np.max(sde_lst), epoch))
                    logger.info(
                        'Logits head mean acc: {} max acc: {} mean nmi: {} max nmi: {}, mean ari: {} max ari: {}, mean sdi: {}, max sdi: {}, mean si: {}, max si: {}  epoch {}\n'.format(
                            np.mean(acc_lst), np.max(acc_lst),
                            np.mean(nmi_lst), np.max(nmi_lst), np.mean(ari_lst), np.max(ari_lst), np.mean(sde_lst),
                            np.max(sde_lst), np.mean(si_list), np.max(si_list), epoch))

                    result_df = pd.concat([result_df, pd.DataFrame.from_records(
                        [{'seq_name': args.data.lower(), 'seed': config['seed'], 'epoch': epoch, 'gamma_default': config['gamma'],
                          'gamma_estimated': gamma,
                          'acc': np.mean(acc_lst),
                          'acc_std': np.std(acc_lst),
                          'nmi': np.mean(nmi_lst),
                          'nmi_std': np.std(nmi_lst),
                          'ari': np.mean(ari_lst),
                          'ari_std': np.std(ari_lst),
                          'subspace_discovery_err': np.mean(sde_lst),
                          'subspace_discovery_err_std': np.std(sde_lst),
                          'silhouette_score': si_score,
                          'silhouette_score_std': np.std(si_list),
                          'time': t_end - t_begin
                          }])]
                        )

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
                            'subspace_discovery_err:': np.mean(sde_lst),
                            'silhouette_score': si_score,
                        })
                    final_ari = np.mean(ari_lst)

                    if early_stop:
                        logger.info("Early Stopping Ends...")
                        break



    return -si_score

# def interface_to_train():
#     config = global_config
#     with wandb.init(project="pro_dsc"+'_'+config["experiment_name"], config=config):
#
#         for key in wandb.config.as_dict():
#             config[key] = wandb.config.as_dict().get(key)
#
#         train(config)
if __name__ == '__main__':
    # sampler = optuna.samplers.TPESampler(
    #     multivariate=True,
    #     group=True,
    #     seed=42,
    # )
    optuna.logging.set_verbosity(optuna.logging.INFO)
    initial_points = [
        {"constant_factor": 1},
        {"constant_factor": 0.25},
        {"constant_factor": 4},
    ]

    for seed in global_config['seeds']:
        global_config['seed'] = deepcopy(seed)
        sampler = optuna.samplers.GPSampler(
            seed=global_config['seed'],
            n_startup_trials=5,
        )

        pruner = optuna.pruners.HyperbandPruner()
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )
        for params in initial_points:
            study.enqueue_trial(params)
        # load results on weights and biases
        study.optimize(objective, n_trials=5,callbacks=[wandbc], timeout=3600*24) # time out after one day



    # for seed in global_config['seeds']:
    # load results on weights and biases



        best_run = {
            "trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
        }

        with open("{}/{}_{}_best_run_seed.json".format(args.out_dir, args.data.lower(), args.experiment_name), "w") as f:
            json.dump([best_run], f, indent=4)




