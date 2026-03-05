import os
import sys
import math
import torch.nn.functional as F
from distributed.utils_test import double
import  jax.scipy as jsc
import  jax.numpy as jnp
import numpy as np
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


sys.path.append('./')

from datetime import datetime
current_date = datetime.now()
formatted_date = current_date.strftime('%m-%d')

import argparse
import yaml
import torch
import numpy as np
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.model_with_transformer_encoder import PRO_DSC_Transformer
from model.model import PRO_DSC
from model.sink_distance import SinkhornDistance
from data.data_utils import FeatureDatasetWithIDs, FeatureDataset
from loss.loss_fn import TotalCodingRate
from utils import *
from metrics.clustering import spectral_clustering_metrics
import pandas as pd

parser = argparse.ArgumentParser(description='PRO-DSC Training')
parser.add_argument('--desc', type=str, default='exp',
                    help='description of the experiment')

parser.add_argument('--data', type=str, default='cifar100',
                    help='dataset to use')
parser.add_argument('--gamma', type=float, default=100,
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
parser.add_argument('--bs', type=int, default=32,
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
parser.add_argument('--quantile_prob', type=float, default=0.99)

parser.add_argument('--seeds', type=int, default=[1,2],
                    help='seeds')

parser.add_argument('--experiment_name', type=str, default="subspace_coil100")
parser.add_argument('--out_dir', type=str, default="results")

args = parser.parse_args()

datasets_list = ['cifar10','cifar100','cifar10-mcr']#,'cifar20','tinyimagenet','imagenet','imagenetdogs']
assert args.data.lower() in datasets_list, "Only {} are supported".format(','.join(datasets_list))

import torch

def self_representation_ls(X: torch.Tensor) -> torch.Tensor:
    """
    Solve min_C ||X - X C||_F^2  s.t. diag(C)=0, by column-wise least squares.

    Args:
        X: (d, n) tensor (float32/float64), columns are data points.

    Returns:
        C: (n, n) tensor with diag(C)=0.
    """
    if X.dim() != 2:
        raise ValueError("X must be a 2D tensor of shape (d, n).")
    d, n = X.shape
    device, dtype = X.device, X.dtype

    C = torch.zeros((n, n), device=device, dtype=dtype)

    # Precompute a mask template to select all columns except i
    for i in range(n):
        mask = torch.ones(n, device=device, dtype=torch.bool)
        mask[i] = False

        X_ni = X[:, mask]          # (d, n-1)
        x_i = X[:, i]              # (d,)

        # Solve min ||X_ni c - x_i||_2  (PyTorch returns (n-1, 1))
        sol = torch.linalg.lstsq(X_ni, x_i.unsqueeze(1)).solution.squeeze(1)  # (n-1,)

        # Insert into column i, skipping row i
        C[mask, i] = sol

    # Ensure exact zero diagonal
    C.fill_diagonal_(0.0)
    return C


# parse configurations from yaml
with open(os.path.join('configs','{}.yaml'.format(args.data.lower())), 'r', encoding='utf-8') as file:
    yaml_data = yaml.safe_load(file)
    for key, value in yaml_data.items():
        setattr(args, key, value)
args.desc = '_'.join(
    [formatted_date, args.data, 'gamma{}'.format(args.gamma), 'beta{}'.format(args.beta), args.desc])
print(args)
#################################################################################################################
dir_name = os.path.join(f'exps/{args.desc}')
writer = init_pipeline(dir_name, args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading features and labels
if args.data.lower() in ['cifar10','cifar100','cifar20']:
    feature_dict = torch.load(args.data_dir)
    clip_features = feature_dict['features'][:50000]
    clip_labels = feature_dict['ys'][:50000]

    feature_dict = torch.load(args.data_dir_val)
    clip_features_test = feature_dict['features'][-10000:]
    clip_labels_test = feature_dict['ys'][-10000:]

    train_ids = np.arange(len(clip_labels))

    if args.data.lower() == 'cifar20':
        from data.dataset import sparse2coarse
        clip_labels = torch.from_numpy(sparse2coarse(clip_labels))
        clip_labels_test = torch.from_numpy(sparse2coarse(clip_labels_test))

elif args.data.lower() == "cifar10-mcr":

        with open('data/datasets/CIFAR10-MCR2/cifar10-features.npy', 'rb') as f:
            full_samples = np.load(f)
            clip_features = full_samples
            clip_features_test = full_samples
        with open('data/datasets/CIFAR10-MCR2/cifar10-labels.npy', 'rb') as f:
            full_labels = np.load(f)
            clip_labels = full_labels
            clip_labels_test = full_labels
        train_ids = np.arange(len(clip_labels))

else:
    feature_dict = torch.load(args.data_dir)
    clip_features = feature_dict['features']
    clip_labels = feature_dict['ys']
    train_ids = np.arange(len(clip_labels))
    feature_dict = torch.load(args.data_dir_val)
    clip_features_test = feature_dict['features']
    clip_labels_test = feature_dict['ys']

#### construct dataloader for batch training  
clip_feature_set = FeatureDatasetWithIDs(clip_features, clip_labels, train_ids)
train_loader = DataLoader(clip_feature_set, batch_size=args.bs, shuffle=True, drop_last=True)
clip_feature_set_test = FeatureDataset(clip_features_test, clip_labels_test)
test_loader = DataLoader(clip_feature_set_test, batch_size=args.bs, shuffle=True, drop_last=False)

#### construct the model:

model = PRO_DSC(input_dim=clip_features.shape[-1], hidden_dim=args.hidden_dim, z_dim=args.z_dim).to(device) # input_dim=768
sink_layer = SinkhornDistance(args.pieta, max_iter=args.piiter)
quantile_prob = args.quantile_prob


#### loss of logdet()
warmup_criterion = TotalCodingRate(eps=args.eps)

### optimizer
param_list = [p for p in model.pre_feature.parameters() if p.requires_grad] + [p for p in model.subspace.parameters() if p.requires_grad]
param_list_c = [p for p in model.cluster.parameters() if p.requires_grad]
optimizer = optim.SGD(param_list, lr=args.lr, momentum=args.momo, weight_decay=args.wd1, nesterov=False)
optimizerc = optim.SGD(param_list_c, lr=args.lr_c, momentum=args.momo, weight_decay=args.wd2, nesterov=False)
scaler = GradScaler()

### warmup iteration setting 
total_wamup_steps = args.warmup
warmup_step = 0
gamma_estimated_list = []
# calculate the number of steps per epoch:
candidate_quantile = torch.from_numpy(np.zeros_like(clip_labels)).float().to(device)
nb_steps_per_epoch = math.ceil(len(clip_features)/args.bs)

result_df = pd.DataFrame()
for seed in args.seeds:
    same_seeds(seed)
    previous_nmi = None
    with tqdm(total=args.epo) as progress_bar:
        for epoch in range(args.epo):
            progress_bar.set_description('Epoch: '+str(epoch)+'/'+str(args.epo))
            model.train()
            ### learning loss storage
            loss_dict = {'loss_TCR': [], 'loss_Exp': [], 'loss_Block': []}
            if len(gamma_estimated_list) > 0:
                gamma = np.nanmean(np.array(gamma_estimated_list))
                gamma_estimated_list = []
                print(f"estimated gamma {gamma}, default gamma is {args.gamma}")

            for step, (x, y, id_num) in enumerate(train_loader):

                x, y, id_num = x.float().to(device), y.to(device), id_num.long().to(device)
                y_np = y.detach().cpu().numpy()

                ############## calculate the pairwise cosine similarities between the data:
                pairwise_dist = torch.mm(x / torch.linalg.norm(x, axis=1, keepdims=True),
                                                (x / torch.linalg.norm(
                                                    x, axis=1, keepdims=True)).T).abs()
                pairwise_dist = pairwise_dist - torch.diag(torch.diag(pairwise_dist)) # remove the pairwise similarity
                # calculate the ranking
                weights = torch.zeros_like(pairwise_dist)
                if epoch == 0:
                    softmax = F.softmax(pairwise_dist.float(), dim=1)
                    prob_threshold = torch.quantile(softmax, quantile_prob, dim=1)
                    pairwise_dist_mask = softmax.ge(
                        torch.repeat_interleave(prob_threshold[:, None], pairwise_dist.shape[1],
                                                dim=1)).bool()  # Output bit mask, when for a given block sample, its pairwise distance larger than prob_threshold



                    non_zero_mask = pairwise_dist_mask
                    count_non_zero = non_zero_mask.sum(dim=1)

                    # 4. Ensure the denominator is at least 1 to avoid division by zero
                    #    This is important if a row/column is all zeros.
                    safe_count = count_non_zero.clamp(min=1)

                    # 5. Calculate the average of non-zero values
                    avg_non_zero = sum(count_non_zero) / len(
                        count_non_zero)  # .float() for float division
                    quantile_dist = torch.where(non_zero_mask, pairwise_dist, 1)

                    candidate_quantile[id_num] = torch.where(
                        candidate_quantile[id_num].le(quantile_dist.min(0)[0]),
                        quantile_dist.min(0)[0], candidate_quantile[id_num])
                else:


                    decrease_delta_1 = torch.where(pairwise_dist.ge(
                        torch.repeat_interleave(candidate_quantile[None, id_num], pairwise_dist.shape[0],
                                                dim=0)), pairwise_dist, torch.zeros_like(pairwise_dist))
                    # decrease_delta_2 = torch.where(abs_latent_dist.ge(
                    #     torch.repeat_interleave(last_layer_sim_candidate_quantile[None, block_id], abs_latent_dist.shape[0],
                    #                             dim=0)), abs_latent_dist,  torch.zeros_like(c))
                    weights =   decrease_delta_1  # decrease_delta_2*decrease_delta_1


                with autocast(enabled=True,device_type="cuda"):
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
                    # print("the shape of self-expressive coefficient is: ", self_coeff.shape)

                    ### compute the affinity matrix
                    A = 0.5 * (self_coeff.abs() + self_coeff.abs().T)

                    # d =A.sum(1)
                    A_np = A.detach().cpu().numpy()
                    ### compute W for BDR
                    L = torch.diag(A.sum(1)) - A



                    # D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-7))
                    # L = D_inv_sqrt @ L @ D_inv_sqrt
                    with torch.no_grad():

                        # add jitering:
                        # L = L.double()+ 1e-6 * torch.eye(L.shape[0]).to(device)
                        # A_sym = 0.5 * (L + L.T)
                        # cond_hint = torch.linalg.cond(A_sym)  # expensive but useful for debugging
                        # print("condition number is: ", cond_hint.item())
                        _, U = torch.linalg.eigh(L) # this is the laplacian matrix for spectral clustering, L is coming from the self-expressive coefficient C
                        U_hat = U[:, :args.n_clusters] # U is the eigenvectors
                        W = U_hat @ U_hat.T # L is a square matrix again
                        # W = W.double()

                    if epoch <= total_wamup_steps:
                        loss = warmup_criterion(z)
                        loss_dict['loss_TCR'].append(loss.item())
                    else:
                        loss_tcr = warmup_criterion(z) # logdet() loss
                        loss_exp = 0.5 * (torch.linalg.norm(z.T - z.T @ Sign_self_coeff.mul(A) )) ** 2 / args.bs # ||Z-ZC||_F loss
                        loss_bl = torch.trace(L.T @ W) / args.bs # r() loss
                        # # here you can add your initial estimates for Z and for C. and subsequently update!
                        if epoch> total_wamup_steps: # run on every steps and warmup_step <= total_wamup_steps + nb_steps_per_epoch   no initial pretraining is used:

                            block = z.detach().clone().double()

                            # approx_pseudo = imqrginv_fixed(block.detach().cpu().numpy())

                            G = block @ block.T
                            diagIndices = np.diag_indices(G.shape[0])

                            P = imqrginv_fixed(G.detach().cpu().numpy())
                            P = np.array(P)
                            B = P / (-np.diag(P) + 1e-7 * np.eye(G.shape[0]) )
                            B[diagIndices] = 0
                            # c_matrix = np.dot(block.detach().cpu().numpy(),
                            #                         B)
                            c_matrix = B
                            # print("size of the array c_matrix: ", c_matrix.shape )
                            # print("size of the array block: ", block.shape)
                            # print("size of the array B: ", B.shape)

                            # to print descriptive statistics of a numpy array
                            # tmp = pd.DataFrame(np.diag(c_matrix))
                            # print("summary of coefficient matrices: ", tmp.describe()) # this is especially psueo inverse leads to identity matrices
                            c_matrix = torch.from_numpy(c_matrix)

                            ########## Old way to calculate pseudo inverse and somehow does not lead to identity matrix ##
                            # c_matrix = np.dot(block.detach().cpu().numpy(),
                            #                         approx_pseudo)
                            # even older, no fast implementation
                            # c_matrix = self_representation_ls(block.T)
                            # c_matrix = block @ (
                            #              torch.linalg.pinv(block @ block.t()) @ block).t()  ##--This needs to be TODO!
                            #
                            #######################################

                            # conver to laplacian matrix:
                            A = c_matrix #0.5 * (c_matrix.abs() + c_matrix.abs().T)
                            L_c = torch.diag(A.sum(1)) - A
                            _, c_u = torch.linalg.eigh(
                                c_matrix)  # this is the laplacian matrix for spectral clustering, L is coming from the self-expressive coefficient C
                            c_u_hat = c_u[:, :args.n_clusters]  # U is the eigenvectors
                            c_W = c_u_hat @ c_u_hat.T  # L is a square matrix again

                            gamma_estimated =300* 1/ (torch.trace(L_c.T @ c_W)/args.bs + 1e-7) # 1/( 0.25 * 1 / torch.sum(torch.abs(c_matrix)))/len(x) # 1/500*torch.ones([1]).cuda() #
                            # print("current estimated gamma: ", gamma_estimated.item())
                            gamma_estimated_list.append(gamma_estimated.detach().cpu().numpy())

                        # if (warmup_step-  total_wamup_steps -1) % nb_steps_per_epoch == 0   and warmup_step!=-1:# epoch > 0:
                        #     gamma = np.mean(np.array(gamma_estimated_list))
                        #     gamma_estimated_list = []
                        if epoch >= total_wamup_steps+2:
                            M = L.T @ W
                            pairwise_eigenspace_dist = torch.cdist(M, M)
                            loss_enforce_same_block = torch.sum(weights * pairwise_eigenspace_dist)/args.bs
                            loss = loss_tcr + gamma * loss_exp + args.beta * loss_bl + 1e-3*loss_enforce_same_block
                            print(f"estimated gamma {gamma}, default gamma is {args.gamma}")
                            # else:
                            #     loss = loss_tcr + gamma * loss_exp + args.beta  * loss_bl
                        else:
                            loss = loss_tcr + args.gamma * loss_exp + args.beta * loss_bl
                        # if  epoch == total_wamup_steps + nb_steps_per_epoch +  1 and warmup_step!=-1:
                        #     print(f"estimated gamma {gamma }, default gamma is {args.gamma}")

                        loss_dict['loss_TCR'].append(loss_tcr.item())
                        loss_dict['loss_Exp'].append(loss_exp.item())
                        loss_dict['loss_Block'].append(loss_bl.item())

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
                    print("Warmup Ends...Start training...")
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

            for k in loss_dict.keys():
                if len(loss_dict[k]) != 0:
                    writer.add_scalar(k, np.mean(loss_dict[k]), global_step=epoch)
                else:
                    writer.add_scalar(k, 0, global_step=epoch)

            if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epo:
                torch.save(model.state_dict(), '{}/checkpoints/model{}.pt'.format(dir_name, epoch))

            ### evaluate on test set
            if (epoch + 1) % args.validate_every == 0 or (epoch + 1) == args.epo:
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
                    acc_lst, nmi_lst, pred_lst = spectral_clustering_metrics(self_coeff.detach().cpu().numpy(),args.n_clusters, y_np)
                    writer.add_scalar('ACC', np.max(acc_lst), global_step=epoch)
                    with open('{}/acc.txt'.format(dir_name), 'a') as f:
                        f.write('Logits head mean acc: {} max acc: {} mean nmi: {} max nmi: {}, epoch {}\n'.format(np.mean(acc_lst), np.max(acc_lst),
                                                                                            np.mean(nmi_lst), np.max(nmi_lst),epoch))
                    print('Logits mean acc: {} max acc: {} mean nmi: {} max nmi: {} epoch {}\n'.format(np.mean(acc_lst), np.max(acc_lst),
                                                                                            np.mean(nmi_lst), np.max(nmi_lst),epoch))

                    if previous_nmi is None:
                        previous_nmi = np.mean(nmi_lst)
                        torch.save(model.state_dict(), '{}/checkpoints/best_model{}.pt'.format(dir_name, epoch))

                        result_df = pd.concat([result_df, pd.DataFrame.from_records(
                            [{'seq_name': args.data.lower(), 'seed': seed, 'epoch': epoch, 'acc': np.mean(acc_lst),
                              'nmi': np.mean(nmi_lst),
                              }])])

                        if not os.path.exists(args.out_dir):
                            os.makedirs(args.out_dir)
                        result_df.to_csv(
                            '{}/{}_{}.csv'.format(
                                args.out_dir, args.data.lower(), args.experiment_name), index=False)

                    elif np.mean(nmi_lst) > previous_nmi:
                        previous_nmi = np.mean(nmi_lst)

                        torch.save(model.state_dict(), '{}/checkpoints/best_model{}.pt'.format(dir_name, epoch))

                        result_df = pd.concat([result_df, pd.DataFrame.from_records(
                            [{'seq_name': args.data.lower(), 'seed': seed, 'epoch': epoch, 'acc': np.mean(acc_lst),
                              'nmi': np.mean(nmi_lst),
                              }])])

                        result_df.to_csv(
                            '{}/{}_{}.csv'.format(
                                args.out_dir, args.data.lower(), args.experiment_name), index=False)


