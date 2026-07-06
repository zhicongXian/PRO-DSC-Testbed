import os
import sys
import wandb
import  jax.scipy as jsc
import  jax.numpy as jnp

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

from model.model_with_transformer_encoder import PRO_DSC
from model.sink_distance import SinkhornDistance
from data.data_utils import FeatureDataset
from loss.loss_fn import TotalCodingRate
from utils import *
from metrics.clustering import spectral_clustering_metrics, spectral_clustering_metrics_with_ari_and_subspace_discovery_error_with_seeds
import pandas as pd
import pickle

import wandb
import time
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from copy import deepcopy
from sklearn.metrics import silhouette_score
import logging
import json
import time
import scipy
import re
import math

logging.basicConfig(level=logging.ERROR,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
optuna.logging.set_verbosity(optuna.logging.INFO)
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()
logger = logging.getLogger("auto_gamma_with_optuna_trajectory_embedding")

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

parser.add_argument('--input_dim', type=int, default=768,
                    help='pro dsc input dim')
def parse_list(value):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON list: {e}")

    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("Argument must be a JSON list")

    return parsed

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

parser.add_argument('-s', '--seeds', type=parse_list, help='here you can set a list of seeds', default=[1, 2, 3])
# Use like:

args = parser.parse_args()

datasets_list = ['cifar10','cifar100','cifar10-mcr','mnist','cifar20','tinyimagenet','imagenet','imagenetdogs','trajectory_embedding']
assert args.data.lower() in datasets_list, "Only {} are supported".format(','.join(datasets_list))

# parse configurations from yaml
with open(os.path.join('configs','{}.yaml'.format(args.data.lower())), 'r', encoding='utf-8') as file:
    yaml_data = yaml.safe_load(file)
    for key, value in yaml_data.items():
        if key == "experiment_name" or key =="seed":
            continue
        setattr(args, key, value) # he does it another way around.
args.desc = '_'.join(
    [formatted_date, args.data, 'gamma{}'.format(args.gamma), 'beta{}'.format(args.beta), args.desc])
print(args)
#################################################################################################################


###################load args to config ##############################################################
global_config = vars(args)
wandb_kwargs = {"project": "pro_dsc" + '_' + global_config['data'] + '_' + global_config["experiment_name"]}
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)
global_config = vars(args)
#########################################################################################


def load_dataset(config):
    # Loading features and labels
    if config['data'].lower() in ['cifar10','cifar100','cifar20']:
        feature_dict = torch.load(config['data_dir'])
        features_train = feature_dict['features'][:50000]
        train_labels = feature_dict['ys'][:50000]

        feature_dict = torch.load(config['data_dir_val'])
        features_test = feature_dict['features'][-10000:]
        test_labels = feature_dict['ys'][-10000:]

        if config['data'].lower() == 'cifar20':
            from data.dataset import sparse2coarse
            train_labels = torch.from_numpy(sparse2coarse(train_labels))
            test_labels = torch.from_numpy(sparse2coarse(test_labels))
    elif config['data'].lower() == "cifar10-mcr":

            with open('data/datasets/CIFAR10-MCR2/cifar10-features.npy', 'rb') as f:
                full_samples = np.load(f)
                features_train = full_samples[:50000]
                features_test = full_samples[-10000:]
            with open('data/datasets/CIFAR10-MCR2/cifar10-labels.npy', 'rb') as f:
                full_labels = np.load(f)
                train_labels = full_labels[:50000]
                test_labels = full_labels[-10000:]
            train_ids = np.arange(len(train_labels))
    elif config['data'].lower() == 'mnist':
        # downsample = 1280
        # previous_path = 'D:/Python_code/Self-Expressive-Network-main/Self-Expressive-Network-main/datasets/'
        previous_path = './data/datasets'  # 'D:/Python_code/Self-Expressive-Network-main/Self-Expressive-Network-main/datasets/'
        with open(previous_path + '/{}/{}_scattering_train_data.pkl'.format( config['data'].upper(),  config['data'].upper()),
                  'rb') as f:
            train_samples = pickle.load(f)
        with open(previous_path + '/{}/{}_scattering_train_label.pkl'.format( config['data'].upper(),  config['data'].upper()),
                  'rb') as f:
            train_labels = pickle.load(f)
        with open(previous_path + '/{}/{}_scattering_test_data.pkl'.format( config['data'].upper(),  config['data'].upper()),
                  'rb') as f:
            test_samples = pickle.load(f)
        with open(previous_path + '/{}/{}_scattering_test_label.pkl'.format( config['data'].upper(),  config['data'].upper()),
                  'rb') as f:
            test_labels = pickle.load(f)
        full_samples = np.concatenate([train_samples, test_samples], axis=0)  # [:downsample] #[nb_samples,1, 217, 4,4]
        full_labels = np.concatenate([train_labels, test_labels], axis=0)  # [:downsample]
        full_samples = np.reshape(full_samples, (len(full_samples), -1))
        args.input_dim = full_samples.shape[-1]
        features_train = train_samples
        features_test = test_samples
        train_labels = train_labels
        test_labels = test_labels
    elif config['data'].lower() in ['trajectory_embedding']:

        #  for seq_name in list(data_dict.keys()):
        seq_name = config["seq_name"]

        root_dir = config["root_dir"]
        # now we load the actual data:

        # for seq_name in sorted(os.listdir(root_dir)):
        seq_path = os.path.join(root_dir, seq_name)
        if os.path.isdir(seq_path):
            mat_file_name = f"{seq_name}_truth.mat"
            mat_file_path = os.path.join(seq_path, mat_file_name)

        try:
            mat_data = scipy.io.loadmat(mat_file_path)
            x_data_load = None
            if "x" in mat_data:
                x_data_load = mat_data["x"]

            if "y" in mat_data:
                y_data_load = mat_data["y"]

            coords_2PF = x_data_load[0:2, :, :]  # (2, P, F)
            num_points = coords_2PF.shape[1]
            num_frames = coords_2PF.shape[2]
            trajectories = np.transpose(coords_2PF, (1, 0, 2))  # (P, F, 2)
            base_time = torch.arange(num_frames, dtype=torch.float32) / (
                num_frames - 1
            )

            y_coords_2PF = y_data_load[0:2, :, :]  # (2, P, F)

            if "s" in mat_data:
                labels_load = mat_data["s"].reshape(-1)

            pattern = r"_g(.+)$"
            match = re.search(pattern, seq_name)

        except Exception as e:
            print(f"Error loading or processing {mat_file_path}: {e}")

        full_samples = trajectories.astype(np.float32)
        full_labels = labels_load.astype(np.int64)


        features_train = full_samples
        features_test = full_samples


        full_labels = full_labels - np.min(full_labels)
        train_labels = full_labels
        test_labels = full_labels


        config['n_clusters'] = len(np.unique(full_labels))
        config['n_neighbors'] = 20
        config['bs'] = len(full_samples)


    # y in [0, 1, ..., K-1]

    else:
        feature_dict = torch.load(config['data_dir'])
        features_train = feature_dict['features']
        train_labels = feature_dict['ys']

        feature_dict = torch.load(config['data_dir_val'])
        features_test = feature_dict['features']
        test_labels = feature_dict['ys']

    #### construct dataloader for batch training
    train_feature_set = FeatureDataset(features_train, train_labels)
    train_loader = DataLoader(train_feature_set, batch_size=config['bs'], shuffle=True, drop_last=True)
    test_feature_set = FeatureDataset(features_test, test_labels)
    test_loader = DataLoader(test_feature_set, batch_size=config['bs'], shuffle=True, drop_last=False)

    return train_loader, test_loader

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

def make_objective(seq_name):

    @wandbc.track_in_wandb()
    def objective( trial : optuna.trial.Trial):

        # wandb.run.save()
        config = global_config
        wandb.run.name = f"auto_gamma_with_optuna_{trial.number}_{config['data']}"
        config['constant_factor'] = trial.suggest_float("constant_factor", 0.05, 8,log=True)
        config['seq_name'] = seq_name
        previous_nmi = None
        same_seeds(config['seed'])
        print("current seed: {}".format(config['seed']))
        ######load dataset ############
        train_loader, test_loader = load_dataset(config)
        ############### set up writer ##############################
        desc = config['desc']
        dir_name = os.path.join(f'exps/{desc}')
        writer = init_pipeline_with_config(dir_name, config)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = PRO_DSC( hidden_dim=config['hidden_dim'], z_dim=config['z_dim'],dropout_rate=config['dropout_rate']).to(device) # input_dim=768
        sink_layer = SinkhornDistance(config['pieta'], max_iter=config['piiter'])
        # load pretrain models:
        temp = model.pre_feature.load_state_dict(torch.load('trajectory_subspace_model/model_unsupervised_transformer.pth'), strict=False)

        #### loss of logdet()
        warmup_criterion = TotalCodingRate(eps=config['eps'])

        ### optimizer
        param_list = [p for p in model.pre_feature.parameters() if p.requires_grad] + [p for p in model.subspace.parameters() if p.requires_grad]
        param_list_c = [p for p in model.cluster.parameters() if p.requires_grad]
        optimizer = optim.SGD(param_list, lr=config['lr'], momentum=config['momo'], weight_decay=config['wd1'], nesterov=False)
        optimizerc = optim.SGD(param_list_c, lr=config['lr_c'], momentum=config['momo'], weight_decay=config['wd2'], nesterov=False)
        scaler = GradScaler()

        ### warmup iteration setting
        warmup_epochs = config['warmup']
        warmup_step = 0
        result_df = pd.DataFrame()
        final_ari = 0
        lambda_se = torch.tensor(1.0)
        lambda_bd = torch.tensor(1.0)

        early_stopper = EarlyStopper(patience=100, min_delta=0.005)
        early_stop = False
        parameter_estimate_epos = 1
        gamma_estimated_list = []
        gamma = None

        with tqdm(total=config['epo']) as progress_bar:
            t_begin = time.time()
            for epoch in range(config['epo']):
                progress_bar.set_description('Epoch: '+str(epoch)+'/'+str(config['epo']))
                model.train()
                ### learning loss storage
                loss_dict = {'loss_TCR': [], 'loss_Exp': [], 'loss_Block': []}
                if len(gamma_estimated_list) > 0:
                    gamma_estimated_list = [np.nan if x is None else x for x in gamma_estimated_list],
                    gamma = np.nanmean(np.array(gamma_estimated_list))
                    gamma_estimated_list = []
                    logger.info(f"estimated gamma {gamma}, default gamma is {args.gamma}")
                loss_per_epoch = []
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

                        if epoch <= warmup_epochs:
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

                            # here we add the auto-gamma part:
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
                            gradient_ratio = g_bd  # / g_se

                            # add here the estimation
                            if warmup_epochs < epoch <= warmup_epochs + parameter_estimate_epos:  # run on every steps and warmup_step <= total_wamup_steps + nb_steps_per_epoch   no initial pretraining is used:
                                # with torch.no_grad():
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

                                ##################### here calculate the new gradient #########################
                                estimated_c =  torch.tensor(c_matrix, requires_grad=True).double()
                                estimated_c = estimated_c - torch.diag(torch.diag(estimated_c))  # here he also does this!

                                ### compute the affinity matrix
                                A_c = 0.5 * (estimated_c.abs() + estimated_c.abs().T)

                                ### compute W for BDR
                                L_c = torch.diag(A_c.sum(1)) - A_c
                                with torch.no_grad():
                                    _, U_c = torch.linalg.eigh(L_c)
                                    U_c_hat = U_c[:, :config['n_clusters']]
                                    W_c = U_c_hat @ U_c_hat.T

                                loss_bl_c = torch.trace(L_c.T @ W_c) / config['bs']
                                g_bd_c = grad_norm_wrt_tensor(loss_bl_c, estimated_c)

                                logger.debug(f"calculated g_bd from l2 norm solution: {g_bd_c}")


                                logger.debug(f"before gardient ration: {gamma_estimated}")
                                logger.debug(f"after gardient ration: , {gamma_estimated / gradient_ratio}")
                                block_reconstructed = torch.from_numpy(c_matrix).to(device) @ block
                                approx_err = torch.sum((block - block_reconstructed) ** 2).item() / args.bs

                                logger.info(f"current approx err: , {approx_err}")
                                logger.info(f"initial estimated gamma value: , {gamma_estimated}")

                                if math.sqrt(approx_err) < 0.6:
                                    gamma_estimated = gamma_estimated * g_bd_c # gradient_ratio
                                    gamma_estimated_list.append(gamma_estimated)
                                # not clear when this will satisfy.....
                                elif gamma_estimated < 10 or gamma_estimated > 1000:

                                    B = (np.eye(len(c_matrix)) - c_matrix) @ (np.eye(
                                        len(c_matrix)) - c_matrix).T  # this is from the minimizing l2 norm. !
                                    # soft_rank_global = #  soft_rank_global = frobi**2/(l2_norm_b**2 + 1e-16)effective_intrinsic_dimension_from_Z(B)

                                    frobi = np.linalg.norm(B, "fro")

                                    try:
                                        l2_norm_b = np.linalg.norm(B, 2)
                                        soft_rank_global = frobi ** 2 / (l2_norm_b ** 2 + 1e-16)
                                        print("soft_rank_global", soft_rank_global)
                                        gamma_estimated = config['beta'] * math.sqrt(soft_rank_global) / config['n_clusters']
                                    # to catch the SVD does not converge error:
                                    except Exception as e:
                                        print(e)
                                        try:  # retrial for SVD computation
                                            print("add to check numerical instability")
                                            l2_norm_b = np.linalg.norm(B + 1e-16 * np.eye(len(B)), 2)
                                            soft_rank_global = frobi ** 2 / (l2_norm_b ** 2 + 1e-16)
                                            print("soft_rank_global", soft_rank_global)
                                            gamma_estimated = config['beta'] * math.sqrt(
                                                soft_rank_global) / config['n_clusters']
                                        except Exception as e:
                                            print(e)

                                    logger.info(f"soft_rank_global {soft_rank_global}")

                                    gamma_estimated = gamma_estimated * \
                                                      config[
                                                          'constant_factor']

                                    gamma_estimated_list.append(gamma_estimated)

                                else:
                                    gamma_estimated_list.append(gamma_estimated)

                                logger.debug(f"current estimated gamma: {gamma_estimated}")

                            # update the loss:
                            if gamma is None:
                                loss = loss_tcr  + config['gamma'] * loss_exp + config['beta'] * loss_bl
                            else:
                                loss = loss_tcr + gamma * loss_exp + config['beta'] * loss_bl
                                logger.debug(f"estimated gamma {gamma}, default gamma is {config['gamma']}")

                        loss_per_epoch.append(loss.item())

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
                        print("Warmup Ends...Start training...")
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

                if epoch > warmup_epochs:
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
                if (epoch + 1) % config['validate_every'] == 0 or (epoch + 1) == config['epo'] or epoch ==warmup_epochs or early_stop:
                    t_end = time.time()
                    print('EVAL on VALIDATE DATASETS')
                    model.eval()
                    with torch.no_grad():
                        logits_list = []
                        y_list = []
                        x_list = []

                        for step, (x, y) in enumerate(test_loader):
                            x, y = x.float().to(device), y.to(device)
                            y_list.append(y.detach().cpu().numpy())
                            x_list.append(x.detach().cpu().numpy())
                            _, logits = model(x)
                            logits_list.append(logits)

                        logits = torch.cat(logits_list, dim=0)

                        self_coeff = (logits @ logits.T).abs()

                        y_np = np.concatenate(y_list, axis=0)
                        x_np = np.concatenate(x_list, axis=0)
                        if len(x_np.shape) > 2:
                            x_np = x_np.reshape(len(x_np), -1)
                        acc_lst, nmi_lst, pred_lst, ari_lst, sde_lst, si_lst = spectral_clustering_metrics_with_ari_and_subspace_discovery_error_with_seeds(x_np, self_coeff.detach().cpu().numpy(),args.n_clusters, y_np,
                                                                                                                                         seeds=[config['seed']])
                        # evaluate on the silhouette score:
                        si_score = np.mean(np.asarray(si_lst))  #silhouette_score(x_np, pred_lst[0]) # since we now set the same seed
                        writer.add_scalar('ACC', np.max(acc_lst), global_step=epoch)

                        with open('{}/acc.txt'.format(dir_name), 'a') as f:
                            f.write(
                                'Logits head mean acc: {} max acc: {} mean nmi: {} max nmi: {}, mean ari: {} max ari: {}, mean sdi: {}, max sdi: {}  epoch {}\n'.format(
                                    np.mean(acc_lst), np.max(acc_lst),
                                    np.mean(nmi_lst), np.max(nmi_lst), np.mean(ari_lst), np.max(ari_lst), np.mean(sde_lst), np.max(sde_lst), epoch))
                        print(
                            'Logits head mean acc: {} max acc: {} mean nmi: {} max nmi: {}, mean ari: {} max ari: {}, mean sdi: {}, max sdi: {}  epoch {}\n'.format(
                                    np.mean(acc_lst), np.max(acc_lst),
                                    np.mean(nmi_lst), np.max(nmi_lst), np.mean(ari_lst), np.max(ari_lst), np.mean(sde_lst), np.max(sde_lst), epoch))

                        result_df = pd.concat([result_df, pd.DataFrame.from_records(
                            [{'seq_name': seq_name, 'seed': config['seed'], 'epoch': epoch, 'gamma_estimated': gamma,
                              'acc': np.mean(acc_lst),
                              'acc_std': np.std(acc_lst),
                              'nmi': np.mean(nmi_lst),
                              'nmi_std': np.std(nmi_lst),
                              'ari': np.mean(ari_lst),
                              'ari_std': np.std(ari_lst),
                              'subspace_discovery_err:': np.mean(sde_lst),
                              'subspace_discovery_err_std': np.std(sde_lst),
                              'silhouette_score': si_score,
                              'silhouette_score_std': np.std(si_lst),
                              'time': t_end - t_begin
                              }])]
                            )

                        result_df.to_csv(
                            '{}/{}_{}.csv'.format(
                                args.out_dir, args.data.lower(), args.experiment_name), index=False)


                        ######## add early stop ######################
                        # early_stopper = EarlyStopper(patience=5)
                        # if early_stopper.early_stop(-np.mean(ari_lst)):
                        #     break

                        ##############################################

                        if previous_nmi is None:
                            previous_nmi = np.mean(nmi_lst)
                            torch.save(model.state_dict(), '{}/checkpoints/best_model{}.pt'.format(dir_name, epoch))


                            if not os.path.exists(config['out_dir']):
                                os.makedirs(config['out_dir'])

                        elif np.mean(nmi_lst) > previous_nmi:
                            previous_nmi = np.mean(nmi_lst)

                            torch.save(model.state_dict(), '{}/checkpoints/best_model{}.pt'.format(dir_name, epoch))

                        if wandb.run:
                            wandb.log({
                                "epoch": epoch,
                                "acc": np.mean(acc_lst),
                                "nmi": np.mean(nmi_lst),
                                "ari": np.mean(ari_lst),
                                'subspace discovery error': np.mean(sde_lst),
                                'silhouette_score': si_score,
                                "gamma":gamma
                            })
                        final_ari = np.mean(ari_lst)
                        if early_stop:
                            logger.info("Early Stopping Ends...")
                            break

        return -si_score
    return objective



def load_sweep_config():
    sweep_config = {"method": "grid"}
    parameters_dict = {}
    gamma_list = list(np.linspace(10, 1000, 200))
    parameters_dict.update({'gamma':{"values":gamma_list}})
    eval_metric = {"name": "ari", "goal": "maximize"}
    sweep_config["parameters"] = parameters_dict
    sweep_config["metric"] = eval_metric

    return sweep_config

# def interface_to_train():
#     config = global_config
#     with wandb.init(project="pro_dsc_cifar", config=config):
#
#         for key in wandb.config.as_dict():
#             config[key] = wandb.config.as_dict().get(key)
#
#
#
#         train(config)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)
    # sampler = optuna.samplers.TPESampler(
    #     multivariate=True,
    #     group=True,
    #     seed=42,
    # )

    for seed in global_config['seeds']:

        global_config['seed'] = deepcopy(seed)

        with open('data/datasets/trajectory_embedding_unsupervised', 'rb') as f:
            data_dict = pickle.load(f)

        with open('data/datasets/trajectory_embedding_labels_unsupervised', 'rb') as f:
            label_dict = pickle.load(f)
        for seq_name in data_dict.keys():
            global_config['seq_name'] = seq_name
            initial_points = [
                {"constant_factor": 1.0},
                {"constant_factor": 4.0},

            ]


            sampler = optuna.samplers.GPSampler(
                seed= seed,
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
            study.optimize(make_objective(seq_name) , n_trials=5, callbacks=[wandbc])
            logger.debug(" seed: ", seed, " best param", study.best_params)
            logger.debug(" seed: ", seed, " best values", study.best_value)

