import pandas as pd
import pickle
filepath = "./results/trajectory_embedding_hopkins155_optuna_automl_new.csv"
knn_filepath = "./results/trajectory_embedding_hopkins155_auto_gamma_new.csv"
def read_df_and_remove_string_columns_for_automl(filepath, ignore_columns = ["seq_name"], gamma_name = "gamma_default"):
    df = pd.read_csv(filepath, index_col=False)#
    df.rename(columns={"NMI": "nmi"}, inplace=True)
    df.rename(columns={"ARI": "ari"}, inplace=True)
    # at first remove all string colums

    df_numeric_column_names = [i for i in df.columns.values if i !="seq_name"]
    # df.drop(columns=ignore_columns, inplace=True)

    df_original = df[df_numeric_column_names].apply(lambda x: pd.to_numeric(x, errors="coerce"), axis = 0).dropna(axis=0)


    # now assign sequence names to the data frames, remember I also have the number of epochs
    with open('data/datasets/trajectory_embedding_unsupervised', 'rb') as f:
        data_dict = pickle.load(f)

    with open('data/datasets/trajectory_embedding_labels_unsupervised', 'rb') as f:
        label_dict = pickle.load(f)
    list_of_seq_names = list(data_dict.keys())


    indices = df_original.index[df_original["epoch"].shift(-1) < df_original["epoch"] ]
    # current_seq_name = None
    # mask = df["epoch"].shift(-1) < df["epoch"]
    #
    # # Index label of the last occurrence
    # last_idx = df_original.index[mask]
    df_original = df_original.loc[indices]

    starts = df_original[gamma_name].eq(380.7947176588889)

    df_original["cycle"] = starts.cumsum()
    df_original['seq_name'] = [list_of_seq_names[i-1] for i in df_original.cycle.values]

    return df_original

def read_df_and_remove_string_columns_for_autobeta(filepath, ignore_columns = ["seq_name"], gamma_name = "gamma_default"):
    df = pd.read_csv(filepath, index_col=False)#
    df.rename(columns={"NMI": "nmi"}, inplace=True)
    df.rename(columns={"ARI": "ari"}, inplace=True)
    # at first remove all string colums

    df_numeric_column_names = [i for i in df.columns.values if i !="seq_name"]
    # df.drop(columns=ignore_columns, inplace=True)

    df_original = df[df_numeric_column_names].apply(lambda x: pd.to_numeric(x, errors="coerce"), axis = 0).dropna(axis=0)
    df_original = df_original.groupby(gamma_name, sort = False).last()

    # now assign sequence names to the data frames, remember I also have the number of epochs
    with open('data/datasets/trajectory_embedding_unsupervised', 'rb') as f:
        data_dict = pickle.load(f)

    with open('data/datasets/trajectory_embedding_labels_unsupervised', 'rb') as f:
        label_dict = pickle.load(f)
    list_of_seq_names = list(data_dict.keys())


    # indices = df_original.index[df_original["epoch"].shift(-1) < df_original["epoch"] ]
    # # current_seq_name = None
    # # mask = df["epoch"].shift(-1) < df["epoch"]
    # #
    # # # Index label of the last occurrence
    # # last_idx = df_original.index[mask]
    # df_original = df_original.loc[indices]
    seq_name_list = []
    current_seq_name = None
    df_original = df_original.reset_index()
    for row_id, row in df_original.iterrows():
        if row_id % 5 == 0:
            current_seq_name = list_of_seq_names.pop(0)
        seq_name_list.append(current_seq_name)


    # starts = df_original[gamma_name].eq(380.7947176588889)

    #df_original["cycle"] = starts.cumsum()
    df_original['seq_name'] = seq_name_list #[list_of_seq_names[i-1] for i in df_original.cycle.values]

    return df_original


df = read_df_and_remove_string_columns_for_autobeta(knn_filepath, gamma_name = "gamma_estimated")
# df.drop(columns=["cycle"], inplace=True)
df.to_csv("./results/trajectory_embedding_hopkins155_auto_gamma_new_modified.csv", index = False)