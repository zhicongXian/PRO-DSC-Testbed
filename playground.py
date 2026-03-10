import pickle

# this is a file used to inspect
# with open("./data/datasets/coil100_dsc_pretrain.pkl", 'rb') as f:
#     pretrain_features = pickle.load(f)
#
# print("inspecting pretrain features: ",pretrain_features["images"].shape)

# This is the code block to inspect the debug z dimension files:

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly


file_name = r"D:\Python_code\PRO-DSC-master\PRO-DSC-master\debug\latent_emb_similarity_seed_1_epoch_5"
seed = 1
epoch = 5
experiment_name = "cifar100_debug_init"
df = pd.read_csv(file_name, index_col=0)

col_size = df.values.shape[1] - 1
first_k = col_size
# visualize the dataframe
print("the first k values", df.iloc[:first_k,:-1].describe())
# fig = go.Figure(data=go.Heatmap(z = df.to_numpy()[:first_k,:-1],
#                                 x = df['id_num'].values[:first_k],
#                                 y =df['id_num'].values[:first_k]))
print("does it contain nan values? ", df.isnull().values.any())
fig =  go.Figure(data = go.Heatmap(z = df.to_numpy()[:col_size,:-1], x = df['id_num'].astype(str).values[:col_size],
                                  y =df['id_num'].astype(str).values[:col_size]))
plotly.offline.plot(fig, filename="latent_sim_plot_seed_{0}_epoch_{1}_{2}.html".format(seed, epoch,experiment_name),
                    auto_open=True)
file_name2 = r"D:\Python_code\PRO-DSC-master\PRO-DSC-master\debug\pseudo_inv_similarity_seed_1_epoch_5"
df2 = pd.read_csv(file_name2, index_col=0)
fig =  go.Figure(data = go.Heatmap(z = df.to_numpy()[:col_size,:-1], x = df['id_num'].astype(str).values[:col_size],
                                  y =df['id_num'].astype(str).values[:col_size]))
plotly.offline.plot(fig, filename="pseudo_inverse_plot_seed_{0}_epoch_{1}_{2}.html".format(seed, epoch,experiment_name),
                    auto_open=True)
