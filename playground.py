import pickle

# this is a file used to inspect
with open("./data/datasets/coil100_dsc_pretrain.pkl", 'rb') as f:
    pretrain_features = pickle.load(f)

print("inspecting pretrain features: ",pretrain_features["images"].shape)