import os
import sys
sys.path.append('./')
import pickle
import os
import math
import pandas as pd
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
from model.model import  PRO_DSC_Image, DSCNet
from model.sink_distance import SinkhornDistance
from data.data_utils import FeatureDataset
from loss.loss_fn import TotalCodingRate
from utils import *
from metrics.clustering import spectral_clustering_metrics, spectral_clustering_metrics_with_ari_and_subspace_discovery_error
import torch.nn.functional as F
import scipy.io as sio
from model.DSCNet import PRO_DSC
import  jax.scipy as jsc
import  jax.numpy as jnp
import math as math


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
parser.add_argument('--load_pretrain', dest='load_pretrain', action='store_true')
parser.add_argument('--experiment_name', type=str, default="subspace_coil100")
parser.add_argument('--out_dir', type=str, default="results")
# parser.add_argument('--seeds', type=int, default=[42],
#                     help='random seed')

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

datasets_list = ['cifar10','cifar100','coil100', 'orl']#,'cifar20','tinyimagenet','imagenet','imagenetdogs']
assert args.data.lower() in datasets_list, "Only {} are supported".format(','.join(datasets_list))

# parse configurations from yaml
# if args.load_pretrain:
#     with open(os.path.join('configs', '{}_2_load_pretrain.yaml'.format(args.data.lower())), 'r', encoding='utf-8') as file:
#         yaml_data = yaml.safe_load(file)
#         for key, value in yaml_data.items():
#             setattr(args, key, value)
# else:
with open(os.path.join('configs','{}.yaml'.format(args.data.lower())), 'r', encoding='utf-8') as file:
    yaml_data = yaml.safe_load(file)
    for key, value in yaml_data.items():
        if key == "seeds":
            continue
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

# Loading features and labels
if args.data.lower() in ['cifar10','cifar100','cifar20']:
    feature_dict = torch.load(args.data_dir)
    x_train = feature_dict['features'][:50000]
    y_labels = feature_dict['ys'][:50000]

    feature_dict = torch.load(args.data_dir_val)
    x_test = feature_dict['features'][-10000:]
    y_test = feature_dict['ys'][-10000:]

    if args.data.lower() == 'cifar20':
        from data.dataset import sparse2coarse
        y_labels = torch.from_numpy(sparse2coarse(y_labels))
        y_test = torch.from_numpy(sparse2coarse(y_test))
    model = PRO_DSC(hidden_dim=args.hidden_dim, z_dim=args.z_dim).to(device)  # input_dim=768
    sink_layer = SinkhornDistance(args.pieta, max_iter=args.piiter)
elif args.data.lower() == "coil100":
    if args.load_pretrain:
        data = sio.loadmat(args.data_dir)
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd'] # he has another parameter setting than the original dataset 128 x 128 x3
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 50] # input channels 1, output channels 50
        kernels = [5]
        epochs = 120
        weight_coef = 1.0
        weight_selfExp = 15

        dscnet = DSCNet(num_sample=args.bs, channels=args.channels, kernels=args.kernels)
        dscnet.to(device)

        # load the pretrained weights which are provided by the original author in
        # https://github.com/panji1990/Deep-subspace-clustering-networks
        ae_state_dict = torch.load('data/datasets/pretrained_weights_original/%s.pkl' % args.data.lower())

        dscnet.ae.load_state_dict(ae_state_dict)
        print("Pretrained ae weights are loaded successfully.")

        model = PRO_DSC(hidden_dim=args.hidden_dim, z_dim=args.z_dim, channels=args.channels, kernels=args.kernels).to(
            device)

        # model = PRO_DSC(input_dim=args.hidden_dim, hidden_dim=args.hidden_dim, z_dim=args.z_dim,
        #                       ae_channels=args.channels, ae_kernels=args.kernels).to(device)  # input_dim=768 --TODO later for combi
        # weight transfer:
        # this works, but could be dangerous, if you are not careful
        model.load_state_dict(dscnet.state_dict(), strict=False)
        # > _IncompatibleKeys(missing_keys=['new_layer.weight', 'new_layer.bias'], unexpected_keys=[])
        # check
        print("Checking if the feature extraction parameters identical",(share_parameters(model.pre_feature, dscnet.ae)))

        sink_layer = SinkhornDistance(args.pieta, max_iter=args.piiter)

        x_train = x
        y_labels = y
        y_test = y
        x_test = x
    else:
        with open(args.data_dir, 'rb') as f:
            feature_dict = np.load(f,allow_pickle=True)

            x_train = feature_dict["images"] # [n_data, 128, 128, 3]
            x_train = np.transpose(x_train, (0, 3, 1, 2)) # by default pytorch conv2d is channel first, therefore, we need to transponse

            x_test = x_train

            y_labels = feature_dict["labels"]
            y_labels = y_labels - np.min(y_labels)

            y_test = y_labels
        model = PRO_DSC_Image(input_dim=32*32, hidden_dim=args.hidden_dim, z_dim=args.z_dim, ae_channels=args.channels, ae_kernels= args.kernels).to(device)  # input_dim=768
        sink_layer = SinkhornDistance(args.pieta, max_iter=args.piiter)

# here space for orl eylab
elif args.data.lower() == 'orl':
    # Loading features and labels
    data = sio.loadmat(args.data_dir)
    x, y = data['X'].reshape((-1, 1, 32, 32)), data['Y'] # data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
    y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
    # network and optimization parameters
    num_sample = x.shape[0]
    temp = model.pre_feature.load_state_dict(torch.load('DSCNet_AE_pretrain/orl.pkl'),strict=False)
    x_train = x_test = x
    y_labels = y_test = y

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

    x_train = x_test = x
    y_labels = y_test = y
else:
    feature_dict = torch.load(args.data_dir)
    x_train = feature_dict['features']
    y_labels = feature_dict['ys']

    feature_dict = torch.load(args.data_dir_val)
    x_test = feature_dict['features']
    y_test = feature_dict['ys']
    model = PRO_DSC(input_dim=768, hidden_dim=args.hidden_dim, z_dim=args.z_dim).to(device)  # input_dim=768
    sink_layer = SinkhornDistance(args.pieta, max_iter=args.piiter)
#### construct dataloader for batch training
args.bs = num_sample
train_feature_set = FeatureDataset(x_train, y_labels)
train_loader = DataLoader(train_feature_set, batch_size=args.bs, shuffle=True, drop_last=True)
feature_set_test = FeatureDataset(x_test, y_test)
test_loader = DataLoader(feature_set_test, batch_size=args.bs, shuffle=True, drop_last=False)

#### loss of logdet()
warmup_criterion = TotalCodingRate(eps=args.eps)

previous_nmi = None
##-- TODO later for different seeds:

result_df = pd.DataFrame()
gamma_estimated_list = []
gamma = None
gamma_previous = None
approx_err_list = []
approx_err_previous = 1

for seed in args.seeds:
    same_seeds(seed)

    ### optimizer
    param_list = [p for p in model.pre_feature.parameters() if p.requires_grad] + [p for p in
                                                                                   model.subspace.parameters() if
                                                                                   p.requires_grad]
    param_list_c = [p for p in model.cluster.parameters() if p.requires_grad]
    optimizer = optim.SGD(param_list, lr=args.lr, momentum=args.momo, weight_decay=args.wd1, nesterov=False)
    optimizerc = optim.SGD(param_list_c, lr=args.lr_c, momentum=args.momo, weight_decay=args.wd2, nesterov=False)
    scaler = GradScaler()

    ### warmup iteration setting
    total_wamup_epochs = args.warmup
    warmup_step = 0
    ae_z = []
    ys = []
    with tqdm(total=args.epo) as progress_bar:
        for epoch in range(args.epo):
            progress_bar.set_description('Epoch: ' + str(epoch) + '/' + str(args.epo))
            model.train()
            ### learning loss storage
            loss_dict = {'loss_TCR': [], 'loss_Exp': [], 'loss_Block': []}
            if len(approx_err_list) >0:
                approx_err_previous = np.median(np.array(approx_err_list))
                approx_err_list = []

            if len(gamma_estimated_list) >0:
                gamma = np.mean(np.array(gamma_estimated_list))
                gamma_estimated_list = []

                if gamma_previous is None:
                    gamma_previous = gamma
                elif gamma_previous > gamma:
                    gamma = gamma_previous
                else:
                    gamma_previous = gamma

                gamma_estimated_list = []

                print(f"estimated gamma {gamma}, default gamma is {args.gamma}")


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
                    self_coeff = self_coeff - torch.diag(torch.diag(self_coeff))

                    ### compute the affinity matrix
                    A = 0.5 * (self_coeff.abs() + self_coeff.abs().T)
                    A_np = A.detach().cpu().numpy()
                    ### compute W for BDR
                    L = torch.diag(A.sum(1)) - A
                    with torch.no_grad():

                        _, U = torch.linalg.eigh(L)

                        U_hat = U[:, :args.n_clusters]
                        W = U_hat @ U_hat.T # [bs, bs]


                    # here estimate the gamma:
                    if epoch > total_wamup_epochs:
                        with torch.no_grad():
                            block = z.detach().clone().double()

                            # approx_pseudo = imqrginv_fixed(block.detach().cpu().numpy())
                            # c_matrix = np.dot(block.detach().cpu().numpy(),
                            #                   approx_pseudo)
                            # this does not get from raw features, why the pseudo inverse not identity matrices?
                            # plot the distance measurement

                            # c_matrix = torch.from_numpy(c_matrix)
                            #
                            # # c_matrix = self_representation_ls(block.T)
                            # # c_matrix = block @ (
                            # #              torch.linalg.pinv(block @ block.t()) @ block).t()  ##--This needs to be TODO!
                            # L_c = torch.diag(c_matrix.sum(1)) - c_matrix
                            # _, c_u = torch.linalg.eigh(
                            #     c_matrix)  # this is the laplacian matrix for spectral clustering, L is coming from the self-expressive coefficient C
                            # c_u_hat = c_u[:, :args.n_clusters]  # U is the eigenvectors
                            # c_W = c_u_hat @ c_u_hat.T  # L is a square matrix again
                            # --TODO here:
                            G = block @ block.T # block.T @ block # B of shape: [bs, ft_sz]
                            diagIndices = np.diag_indices(G.shape[0])

                            P = jnp.linalg.inv(G.detach().cpu().numpy())  # imqrginv_fixed(G.detach().cpu().numpy())


                            P = np.array(P)
                            B = P / (-np.diag(P) + 1e-7 * np.eye(G.shape[0]))
                            B[diagIndices] = 0

                            c_matrix = B
                            B = B.T @ W.detach().cpu().numpy()
                            frobi= np.linalg.norm(B, "fro")
                            block_reconstructed = torch.from_numpy(B).to(device) @ block
                            approx_err = torch.mean((block - block_reconstructed) ** 2).item()
                            approx_err_list.append(approx_err)

                            try:
                                l2_norm_b = np.linalg.norm(B, 2)
                                soft_rank_global = frobi**2/(l2_norm_b**2 + 1e-16)
                                # print("soft_rank_global", soft_rank_global)
                                gamma_estimated = args.beta *  soft_rank_global / 4  # 1 / lambda_hat
                            except Exception as e:
                                print("trying to calculate soft rank ")
                                print(e)
                                try:
                                    print("add to check numerical instability")
                                    l2_norm_b = np.linalg.norm(B + 1e-16 * np.eye(len(B)), 2)
                                    soft_rank_global = frobi ** 2 / (l2_norm_b ** 2 + 1e-16)
                                    # print("soft_rank_global", soft_rank_global)
                                    gamma_estimated = args.beta * soft_rank_global / 4
                                except Exception as e:
                                    print(e)
                                gamma_estimated = gamma_previous


                            # gamma_estimated = 3 * 1 / (torch.trace(
                            #     L_c.T @ c_W) / args.bs)  # 1/( 0.25 * 1 / torch.sum(torch.abs(c_matrix)))/len(x) # 1/500*torch.ones([1]).cuda() #
                            gamma_estimated_list.append(gamma_estimated)




                    if epoch <= total_wamup_epochs:
                        loss = warmup_criterion(z)
                        loss_dict['loss_TCR'].append(loss.item())
                    else:
                        loss_tcr = warmup_criterion(z)  # logdet() loss
                        loss_exp = 0.5 * (torch.linalg.norm(
                            z.T - z.T @ Sign_self_coeff.mul(self_coeff))) ** 2 / args.bs  # ||Z-ZC||_F loss
                        loss_bl = torch.trace(L.T @ W) / args.bs  # r() loss
                        if epoch >= total_wamup_epochs+2:

                            loss = loss_tcr + gamma * loss_exp + args.beta * loss_bl

                        else:
                            loss = loss_tcr + args.gamma * loss_exp + args.beta * loss_bl



                        loss_dict['loss_TCR'].append(loss_tcr.item())
                        loss_dict['loss_Exp'].append(loss_exp.item())
                        loss_dict['loss_Block'].append(loss_bl.item())

                if epoch <= total_wamup_epochs:
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

                if epoch == total_wamup_epochs:
                    print("Warmup Ends...Start training...")
                    model = update_pi_from_z(model)

                if epoch <= total_wamup_epochs:
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
            if epoch >= total_wamup_epochs and ((epoch + 1) % args.validate_every == 0 or (epoch + 1) == args.epo):
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

                    acc_lst, nmi_lst, pred_lst, ari_lst, sde_lst = spectral_clustering_metrics_with_ari_and_subspace_discovery_error(
                        self_coeff.detach().cpu().numpy(), args.n_clusters, y_np,
                        seed=seed)
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
                        [{'seq_name': args.data.lower(), 'seed': seed, 'epoch': epoch, 'gamma_default': args.gamma,
                          'gamma_estimated': gamma,
                          'acc': np.mean(acc_lst),
                          'nmi': np.mean(nmi_lst),
                          'ari': np.mean(ari_lst),
                          'subspace_discovery_err:': np.mean(sde_lst)
                          }])])

                    result_df.to_csv(
                        '{}/{}_{}.csv'.format(
                            args.out_dir, args.data.lower(), args.experiment_name), index=False, mode='a')

                    # if previous_nmi is None:
                    #     previous_nmi = np.mean(nmi_lst)
                    #     torch.save(model.state_dict(), '{}/checkpoints/best_model{}.pt'.format(dir_name, epoch))
                    #
                    #     result_df = pd.concat([result_df, pd.DataFrame.from_records(
                    #         [{'seq_name': args.data.lower(), 'seed': seed, 'epoch': epoch, 'gamma_default': args.gamma,
                    #           'gamma_estimated': gamma,
                    #           'acc': np.mean(acc_lst),
                    #           'nmi': np.mean(nmi_lst),
                    #           'ari': np.mean(ari_lst)
                    #           }])])
                    #
                    #     if not os.path.exists(args.out_dir):
                    #         os.makedirs(args.out_dir)
                    #     result_df.to_csv(
                    #         '{}/{}_{}.csv'.format(
                    #             args.out_dir, args.data.lower(), args.experiment_name), index=False, mode="a")
                    #
                    # elif np.mean(nmi_lst) > previous_nmi:
                    #     previous_nmi = np.mean(nmi_lst)
                    #
                    #     torch.save(model.state_dict(), '{}/checkpoints/best_model{}.pt'.format(dir_name, epoch))
                    #
                    #     result_df = pd.concat([result_df, pd.DataFrame.from_records(
                    #         [{'seq_name': args.data.lower(), 'seed': seed, 'epoch': epoch, 'gamma_default': args.gamma,
                    #           'gamma_estimated': gamma,
                    #           'acc': np.mean(acc_lst),
                    #           'nmi': np.mean(nmi_lst),
                    #           'ari': np.mean(ari_lst)
                    #           }])])
                    #
                    #     result_df.to_csv(
                    #         '{}/{}_{}.csv'.format(
                    #             args.out_dir, args.data.lower(), args.experiment_name), index=False, mode="a")
                    #
                    # else:
                    #     result_df = pd.concat([result_df, pd.DataFrame.from_records(
                    #         [{'seq_name': args.data.lower(), 'seed': seed, 'epoch': epoch, 'gamma_default': args.gamma,
                    #           'gamma_estimated': gamma,
                    #           'acc': np.mean(acc_lst),
                    #           'nmi': np.mean(nmi_lst),
                    #           'ari': np.mean(ari_lst)
                    #           }])])
                    #
                    #     result_df.to_csv(
                    #         '{}/{}_{}.csv'.format(
                    #             args.out_dir, args.data.lower(), args.experiment_name), index=False, mode="a")