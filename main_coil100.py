import os
import sys
sys.path.append('./')
import pickle
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
from model.model import PRO_DSC, PRO_DSC_Image
from model.sink_distance import SinkhornDistance
from data.data_utils import FeatureDataset
from loss.loss_fn import TotalCodingRate
from utils import *
from metrics.clustering import spectral_clustering_metrics
import torch.nn.functional as F

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
args = parser.parse_args()

datasets_list = ['cifar10','cifar100','coil100']#,'cifar20','tinyimagenet','imagenet','imagenetdogs']
assert args.data.lower() in datasets_list, "Only {} are supported".format(','.join(datasets_list))

# parse configurations from yaml
with open(os.path.join('configs','{}_2.yaml'.format(args.data.lower())), 'r', encoding='utf-8') as file:
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

    if args.data.lower() == 'cifar20':
        from data.dataset import sparse2coarse
        clip_labels = torch.from_numpy(sparse2coarse(clip_labels))
        clip_labels_test = torch.from_numpy(sparse2coarse(clip_labels_test))
    model = PRO_DSC(input_dim=768, hidden_dim=args.hidden_dim, z_dim=args.z_dim).to(device)  # input_dim=768
    sink_layer = SinkhornDistance(args.pieta, max_iter=args.piiter)
elif args.data.lower() == "coil100":
    with open(args.data_dir, 'rb') as f:
        feature_dict = np.load(f,allow_pickle=True)

        clip_features = feature_dict["images"] # [n_data, 128, 128, 3]
        clip_features = np.transpose(clip_features, (0, 3, 1, 2)) # by default pytorch conv2d is channel first, therefore, we need to transponse

        clip_features_test = clip_features

        clip_labels = feature_dict["labels"]
        clip_labels = clip_labels-np.min(clip_labels)

        clip_labels_test = clip_labels
    model = PRO_DSC_Image(input_dim=32*32, hidden_dim=args.hidden_dim, z_dim=args.z_dim, ae_channels=args.channels, ae_kernels= args.kernels).to(device)  # input_dim=768
    sink_layer = SinkhornDistance(args.pieta, max_iter=args.piiter)

else:
    feature_dict = torch.load(args.data_dir)
    clip_features = feature_dict['features']
    clip_labels = feature_dict['ys']

    feature_dict = torch.load(args.data_dir_val)
    clip_features_test = feature_dict['features']
    clip_labels_test = feature_dict['ys']
    model = PRO_DSC(input_dim=768, hidden_dim=args.hidden_dim, z_dim=args.z_dim).to(device)  # input_dim=768
    sink_layer = SinkhornDistance(args.pieta, max_iter=args.piiter)
#### construct dataloader for batch training  
clip_feature_set = FeatureDataset(clip_features, clip_labels)
train_loader = DataLoader(clip_feature_set, batch_size=args.bs, shuffle=True, drop_last=True)
clip_feature_set_test = FeatureDataset(clip_features_test, clip_labels_test)
test_loader = DataLoader(clip_feature_set_test, batch_size=args.bs, shuffle=True, drop_last=False)

#### loss of logdet()
warmup_criterion = TotalCodingRate(eps=args.eps)

previous_nmi = None

if args.data.lower() == "coil100":
    ### optimizer
    param_list = [p for p in model.ae.parameters() if p.requires_grad] + [p for p in
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
            progress_bar.set_description('Epoch: '+str(epoch)+'/'+str(args.epo))
            model.train()
            ### learning loss storage
            loss_dict = {'loss_TCR': [], 'loss_Exp': [], 'loss_Block': []}

            for step, (x, y) in enumerate(train_loader):
                x, y = x.float().to(device), y.to(device)
                y_np = y.detach().cpu().numpy()
                with autocast(enabled=True):
                    z, logits, x_recon = model(x)
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

                    if epoch <= total_wamup_epochs:
                        loss = warmup_criterion(z)
                        loss_tcr = loss
                        loss_dict['loss_TCR'].append(loss_tcr.item())
                        loss_ae = F.mse_loss(x_recon, x, reduction='mean')
                        if epoch == 0:
                            loss = loss_tcr + loss_ae
                        else:
                            loss = 0.0*loss + loss_ae

                    else:
                        if epoch == total_wamup_epochs + 1:
                            ae_z.append(z.detach().cpu().numpy())
                            ys.append(y_np)
                        elif epoch == total_wamup_epochs+2:
                            # start serializing files:
                            images = np.concatenate(ae_z)
                            labels = np.concatenate(ys)
                            with open(args.output_file, "wb") as f:
                                pickle.dump({
                                    "images": images,
                                    "labels": labels
                                }, f)

                            # save the pretrained model:

                            torch.save(model.state_dict(), '{}/checkpoints/model{}.pt'.format(dir_name, epoch))
                            # I need to freeze the feature extraction layers


                        # collect the z values and save to a file
                        loss_tcr = warmup_criterion(z) # logdet() loss
                        loss_exp = 0.5 * (torch.linalg.norm(z.T - z.T @ Sign_self_coeff.mul(A) )) ** 2 / args.bs # ||Z-ZC||_F loss
                        loss_bl = torch.trace(L.T @ W) / args.bs # r() loss
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
                    progress_bar.set_postfix(tcr_loss="{:5.4f}".format(loss_tcr.item()), ae_loss ="{:5.4f}".format(loss_ae.item()) )
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
                        _, logits,_ = model(x)
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
                    elif np.mean(nmi_lst) > previous_nmi:
                        previous_nmi = np.mean(nmi_lst)
                        torch.save(model.state_dict(), '{}/checkpoints/best_model{}.pt'.format(dir_name, epoch))
