import logging
import logging
import os
import pathlib
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataloader import load_graph_adj_mtx, load_graph_node_features
# from model import GCN, NodeAttnMap, UserEmbeddings, Time2Vec, CategoryEmbeddings, FuseEmbeddings, TransformerModel
from model_batch import CombinedModel
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss
import warnings
warnings.filterwarnings('ignore')


def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    # %% ====================== Load data ======================
    # Read check-in train data
    

    # Build POI graph (built from train_df)
    print('Loading POI graph...')
    raw_A = load_graph_adj_mtx("./models/graph_A.csv")
    raw_X = load_graph_node_features("./models/graph_X.csv",
                                     args.feature1,
                                     args.feature2,
                                     args.feature3,
                                     args.feature4)
    logging.info(
        f"raw_X.shape: {raw_X.shape}; "
        f"Four features: {args.feature1}, {args.feature2}, {args.feature3}, {args.feature4}.")
    logging.info(f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency).")
    num_pois = raw_X.shape[0]

    # One-hot encoding poi categories
    logging.info('One-hot encoding poi categories id')
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 1:num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1:] = raw_X[:, 2:]
    logging.info(f"After one hot encoding poi cat, X.shape: {X.shape}")
    logging.info(f'POI categories: {list(one_hot_encoder.categories_[0])}')
    # Save ont-hot encoder
    with open(os.path.join(args.save_dir, 'one-hot-encoder.pkl'), "wb") as f:
        pickle.dump(one_hot_encoder, f)

    # Normalization
    print('Laplician matrix...')
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')


    PATH = "./models/best.pt"


    print("Load dict from:", PATH)
    poi_id2idx_dict = torch.load(PATH)["poi_id2idx_dict"]
    cat_id2idx_dict = torch.load(PATH)["cat_id2idx_dict"]
    poi_idx2cat_idx_dict = torch.load(PATH)["poi_idx2cat_idx_dict"]
    user_id2idx_dict = torch.load(PATH)["user_id2idx_dict"]

    # %% ====================== Build Models ======================
    # Model1: POI embedding model
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)

    # Model
    args.gcn_nfeat = X.shape[1]
    num_users = len(user_id2idx_dict)
    args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim
    in_features = X.shape[1]
    model = CombinedModel(
        args,
        in_features,
        num_users,
        num_cats,
        num_pois,
        poi_idx2cat_idx_dict,
        user_id2idx_dict,
        A
    )



    softxmax = torch.nn.Softmax(dim=-1).to(device=args.device)


    ### load model
    from collections import OrderedDict
    print("Load model from:", PATH)
    stage2_state_dict = OrderedDict()
    for key, weight in torch.load(PATH)["model_state_dict"].items():
        # key = key[8:]
        stage2_state_dict[key] = weight
    model.load_state_dict(stage2_state_dict)
    model = model.to(device=args.device)
    eval_mode = True
    if eval_mode:
        args.epochs = 1


    # %% ====================== Train ======================
    model = model.to(device=args.device)
    

    def Common_Point_Extraction():
        def predict(batch):
            y, batch_results = model(X, batch)
            y_poi = y["poi"]
            y_time = y["time"]
            y_cat = y["cat"]
            y_pred_poi = y["pred_poi"]
            y_pred_poi_adjusted = y["pred_poi_adjusted"]
            y_pred_time = y["pred_time"]
            y_pred_cat = y["pred_cat"]
            batch_input_seqs = batch_results["input_seqs"]
            batch_seq_lens = batch_results["seq_lens"]
            batch_seq_embeds = batch_results["seq_embeds"]
            batch_seq_labels_poi = batch_results["seq_labels_poi"]
            batch_seq_labels_time = batch_results["seq_labels_time"]
            batch_seq_labels_cat = batch_results["seq_labels_cat"]
            top1 =  torch.argmax(y_pred_poi,dim=2)

            y_pred_poi = softxmax(y_pred_poi)
            return 0, top1, y_pred_poi

        def common_point_extraction(user):
            fake_label = [[0,0]]
            time = 0.50
            traj_tag = str(user) + "_1"
            predition = []

            batch = []
            locs = np.random.randint(0, 4556, size=num_samples).tolist()
            #print(locs)
            ensemble_conf = np.zeros((1,num_pois))
            for loc in locs:
                input_loc = [traj_tag,[[loc,time]],fake_label]
                batch.append(input_loc)
                #print(len(batch))
                if len(batch)%498==0 and len(batch)!=0:
                    
                    _,y_pred_poi, y_pred_conf = predict(batch)
                    y_pred_poi = y_pred_poi.detach().cpu().numpy()
                    y_pred_conf = y_pred_conf.detach().cpu().numpy()
                    y_pred_conf = np.sum(y_pred_conf, 0)
                    ### update ensemble_conf
                    ensemble_conf = ensemble_conf + y_pred_conf

                    for poi in y_pred_poi:
                        y = int(poi)
                        predition.append(y)
                    batch = []
            _,y_pred_poi,y_pred_conf = predict(batch)
            y_pred_poi = y_pred_poi.detach().cpu().numpy()
            y_pred_conf = y_pred_conf.detach().cpu().numpy()
            if num_samples==2:
                y_pred_conf = y_pred_conf[0]
                l = 1
            else:
                y_pred_conf = np.sum(y_pred_conf, 0)
                l = len(locs)
            ### update ensemble_conf and normalize
            ensemble_conf += y_pred_conf
            ensemble_conf = ensemble_conf/l
            assert np.max(ensemble_conf)<=1
            assert np.min(ensemble_conf)>=0

            for poi in y_pred_poi:
                y = int(poi)
                predition.append(y)
            batch = []
            return predition, ensemble_conf
    
        # initialize
        df_nyc = pd.read_csv("./Training_data/getnext_nyc_train.csv")
        ### Revise-2
        query_result_dir = "./attack1_result/"
        query_result_path = "./attack1/getnext_results_es/{}.csv"

        if not os.path.exists(query_result_dir):
            os.makedirs(query_result_dir)
        user_id = df_nyc["user_id"].unique().tolist()
        loc_id = df_nyc["POI_id"].unique().tolist()
        query_num = len(loc_id)
        print(len(user_id), query_num)

        user_id_npy = []
        result_npy = np.empty((0,num_pois))
        for i in tqdm(user_id):
            predition, conf = common_point_extraction(i)
            user_id_npy.append(i)
            result_npy = np.vstack([result_npy,conf])
                
        np.save(query_result_dir+"/query/user_id.npy", np.array(user_id_npy))
        np.save(query_result_dir+"/query/result.npy", result_npy)
    
    model.eval()
    Common_Point_Extraction()


if __name__ == '__main__':
    args = parameter_parser()
    # The name of node features in NYC/graph_X.csv
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    for k in [500]:
        num_samples = k
        train(args)

# %%
