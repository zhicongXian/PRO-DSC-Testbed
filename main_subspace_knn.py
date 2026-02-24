import os
import sys
import math
import torch.nn.functional as F
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
from data.data_utils import *
from loss.loss_fn import TotalCodingRate
from utils import *
from metrics.clustering import spectral_clustering_metrics
import scipy.io as sio
import  jax.scipy as jsc
import  jax.numpy as jnp
from PIL import Image
import cv2
from skimage.transform import resize

def convert_numpy_grb_to_greyscale(image_np):

    # Apply formula
    gray = np.dot(image_np[..., :3], [0.299, 0.587, 0.114])

    # Convert to uint8
    gray = gray.astype(np.uint8)

    return gray

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
parser.add_argument('--validate_every', type=int, default=100,
                    help='validate to check the clustering performance')
parser.add_argument('--quantile_prob', type=float, default=0.99)
args = parser.parse_args()

datasets_list = ['eyaleb','coil100','orl']
assert args.data.lower() in datasets_list, "Only {} are supported".format(','.join(datasets_list))

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
model = PRO_DSC(hidden_dim=args.hidden_dim, z_dim=args.z_dim,channels=args.channels,kernels=args.kernels).to(device)
sink_layer = SinkhornDistance(args.pieta, max_iter=1)

if args.data.lower() == 'orl':
    # Loading features and labels
    data = sio.loadmat(args.data_dir)
    x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
    y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
    # network and optimization parameters
    num_sample = x.shape[0]
    temp = model.pre_feature.load_state_dict(torch.load('DSCNet_AE_pretrain/orl.pkl'),strict=False)

elif args.data.lower() == 'eyaleb':
    data = sio.loadmat(args.data_dir)
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
    temp = model.pre_feature.load_state_dict(torch.load('DSCNet_AE_pretrain/yaleb.pkl'),strict=False)

elif args.data.lower() == 'coil100':
    data = sio.loadmat(args.data_dir)

    # with open("./data/datasets/scn_coil100_pretrained_weights.pkl", 'rb') as f:
    #     model_weights = np.load(f, allow_pickle=True)

    # with open(args.data_dir, 'rb') as f:
    #     data = np.load(f, allow_pickle=True)
    #     samples = data["images"]
    #     labels = data["labels"]
    #     rgb_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    #     gray = samples @ rgb_weights
    #     resized_imgs = []
    #
    #     for i in range(len(gray)):
    #         resized = resize(gray[i], (32, 32), anti_aliasing=True)
    #         resized_imgs.append(resized)
    #     resized = np.array(resized_imgs)
    #     from scipy.io import savemat
    #
    #     # Example dictionary
    #     data = {
    #         "fea": resized,
    #         "gnd": labels
    #     }
    #
    #     # Save to .mat file
    #     savemat("./data/datasets/coil100.mat", data)

    x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
    y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
    num_sample = x.shape[0]
    # checkpoint_file = "./data/datasets/pretrain-model-COIL100/model50.ckpt.data-00000-of-00001"
    # checkpoint = torch.load(checkpoint_file, weights_only=False)
    temp =model.pre_feature.load_state_dict(torch.load('DSCNet_AE_pretrain/coil100.pkl'),strict=False)
 #model.pre_feature.load_state_dict(checkpoint['model'], strict = False)

    model = temp #
nb_steps_per_epoch = math.ceil(len(x)/args.bs)
#### construct dataloader for batch training  
# args.bs = num_sample
clip_feature_set = FeatureDatasetWithIDs(x, y, np.arange(len(x)))
train_loader = DataLoader(clip_feature_set, batch_size=args.bs, shuffle=True, drop_last=True)
clip_feature_set_test = FeatureDataset(x, y)
test_loader = DataLoader(clip_feature_set_test, batch_size=args.bs, shuffle=True, drop_last=False)

gamma_estimated_list = []
# calculate the number of steps per epoch:
candidate_quantile = torch.from_numpy(np.zeros_like(x)).float().to(device)
nb_steps_per_epoch = math.ceil(len(x)/args.bs)
#### loss of TCR 
warmup_criterion = TotalCodingRate(eps=args.eps)
quantile_prob = args.quantile_prob

### learning opt strategy
param_list = [p for p in model.pre_feature.parameters() if p.requires_grad] + [p for p in model.subspace.parameters() if p.requires_grad]
param_list_c = [p for p in model.cluster.parameters() if p.requires_grad]
optimizer = optim.SGD(param_list, lr=args.lr, momentum=args.momo, weight_decay=args.wd1, nesterov=False)
optimizerc = optim.SGD(param_list_c, lr=args.lr, momentum=args.momo, weight_decay=args.wd2, nesterov=False)
scaler = GradScaler()

### warmup iteration setting 
total_wamup_steps = args.warmup
warmup_step = 0


with tqdm(total=args.epo) as progress_bar:
    for epoch in range(args.epo):
        progress_bar.set_description('Epoch: '+str(epoch)+'/'+str(args.epo))
        model.train()
        ### learning loss storage 
        loss_dict = {'loss_TCR': [], 'loss_Exp': [], 'loss_Block': []}

        for step, (x, y, id_num) in enumerate(train_loader):
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
                self_coeff = self_coeff - torch.diag(torch.diag(self_coeff))
            
                ### compute the affinity matrix 
                A = 0.5 * (self_coeff.abs() + self_coeff.abs().T)
                A_np = A.detach().cpu().numpy()
                ### compute W for BDR
                L = torch.diag(A.sum(1)) - A
                with torch.no_grad():
                    _, U = torch.linalg.eigh(L)
                    U_hat = U[:, :args.n_clusters]
                    W = U_hat @ U_hat.T
                
                if warmup_step <= total_wamup_steps:
                    loss = warmup_criterion(z)
                    loss_dict['loss_TCR'].append(loss.item())
                else:
                    loss_tcr = warmup_criterion(z) # logdet() loss
                    loss_exp = 0.5 * (torch.linalg.norm(z.T - z.T @ Sign_self_coeff.mul(self_coeff) )) ** 2 / args.bs # ||Z-ZC||_F loss
                    loss_bl = torch.trace(L.T @ W) / args.bs # r() loss
                    # # here you can add your initial estimates for Z and for C. and subsequently update!
                    if warmup_step> total_wamup_steps  and warmup_step!=-1: # run on every steps and warmup_step <= total_wamup_steps + nb_steps_per_epoch   no initial pretraining is used:

                        block = z.detach().clone().double()

                        approx_pseudo = imqrginv_fixed(block.detach().cpu().numpy())
                        c_matrix = np.dot(block.detach().cpu().numpy(),
                                                approx_pseudo)
                        c_matrix = torch.from_numpy(c_matrix)

                        # c_matrix = self_representation_ls(block.T)
                        # c_matrix = block @ (
                        #              torch.linalg.pinv(block @ block.t()) @ block).t()  ##--This needs to be TODO!
                        L_c = torch.diag(c_matrix.sum(1)) - c_matrix
                        _, c_u = torch.linalg.eigh(
                            c_matrix)  # this is the laplacian matrix for spectral clustering, L is coming from the self-expressive coefficient C
                        c_u_hat = c_u[:, :args.n_clusters]  # U is the eigenvectors
                        c_W = c_u_hat @ c_u_hat.T  # L is a square matrix again

                        gamma_estimated =3* 1/ (torch.trace(L_c.T @ c_W)/args.bs) # 1/( 0.25 * 1 / torch.sum(torch.abs(c_matrix)))/len(x) # 1/500*torch.ones([1]).cuda() #
                        gamma_estimated_list.append(gamma_estimated.detach().cpu().numpy())

                    if (warmup_step-  total_wamup_steps -1) % nb_steps_per_epoch == 0   and warmup_step!=-1:# epoch > 0:
                        gamma = np.mean(np.array(gamma_estimated_list))
                        gamma_estimated_list = []
                        if epoch > total_wamup_steps // nb_steps_per_epoch + 1:
                            M = L.T @ W
                            pairwise_eigenspace_dist = torch.cdist(M, M)
                            loss_enforce_same_block = torch.sum(1 * pairwise_eigenspace_dist)/args.bs
                            loss = loss_tcr + gamma * loss_exp + args.beta * loss_bl + 1e-3*loss_enforce_same_block
                            print(f"estimated gamma {gamma}, default gamma is {args.gamma}")
                        else:
                            loss = loss_tcr + gamma * loss_exp + args.beta  * loss_bl

                    loss = loss_tcr + args.gamma * loss_exp + args.beta * loss_bl

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

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epo:
            torch.save(model.state_dict(), '{}/checkpoints/model{}.pt'.format(dir_name, epoch))

        ### evaluate on test set
        if (epoch + 1) % args.validate_every == 0 or (epoch + 1) == args.epo:
            print('EVAL on VALIDATE DATASETS')
            model.eval()
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
                acc_lst, nmi_lst, pred_lst = spectral_clustering_metrics(self_coeff.detach().cpu().numpy(),args.n_clusters, y_np)
                writer.add_scalar('ACC', np.max(acc_lst), global_step=epoch)
                with open('{}/acc.txt'.format(dir_name), 'a') as f:
                    f.write('Logits head mean acc: {} max acc: {} mean nmi: {} max nmi: {}, epoch {}\n'.format(np.mean(acc_lst), np.max(acc_lst),
                                                                                        np.mean(nmi_lst), np.max(nmi_lst),epoch))
                print('Logits mean acc: {} max acc: {} mean nmi: {} max nmi: {} epoch {}\n'.format(np.mean(acc_lst), np.max(acc_lst),
                                                                                        np.mean(nmi_lst), np.max(nmi_lst),epoch))
                                                                                        