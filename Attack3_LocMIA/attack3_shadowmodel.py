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







def score(X,labels):
    ##stablize
    predictions = X - np.max(X, axis=1, keepdims=True)
    predictions = np.array(np.exp(predictions), dtype=np.float64)
    predictions = predictions/np.sum(predictions,axis=1,keepdims=True)

    COUNT = predictions.shape[0]
    y_true = predictions[np.arange(COUNT),labels[:COUNT]]

    predictions[np.arange(COUNT),labels[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=1)

    logit = (np.log(y_true.mean((0))+1e-45) - np.log(y_wrong.mean((0))+1e-45))
    return logit

def train(args,shadownum):
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
    raw_A = load_graph_adj_mtx("./dataset_4sq/lira/attack_target/graph_A.csv")
    raw_X = load_graph_node_features("./dataset_4sq/lira/attack_target/graph_X.csv",
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
    PATH = "./dataset_4sq/lira/models/" + dataset + "/" + str(shadownum)+".pt"
    print("Load dict from:", PATH)
    cat_id2idx_dict = torch.load(PATH)["cat_id2idx_dict"]
    poi_idx2cat_idx_dict = torch.load(PATH)["poi_idx2cat_idx_dict"]
    user_id2idx_dict = torch.load(PATH)["user_id2idx_dict"]
    print(len(poi_id2idx_dict))






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
    print(args.transformer_nlayers, args.transformer_nhid)
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
    params = list(model.parameters())

    # default: 0.001
    optimizer = optim.Adam(params=params, lr=0.0001, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)



    ### load model
    from collections import OrderedDict
    print("Load model from:", PATH)
    stage2_state_dict = OrderedDict()
    for key, weight in torch.load(PATH)["model_state_dict"].items():
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
            # print(batch)
            y, batch_results = model(X, batch)
            y_pred_poi = y["pred_poi"]
            top1 =  torch.argmax(y_pred_poi,dim=2)
            top1_conf = y_pred_poi
            return top1_conf, top1

        def common_point_extraction(user):
            fake_label = [[1,0]]
            time = 0.60
            traj_tag = str(int(user)) + "_1"
            predition = []
            preditionconf=[]

            batch = []
            locs = range(query_num)

            for loc in locs:
                input_loc = [traj_tag,[[loc,time]],fake_label]
                batch.append(input_loc)
                
                if len(batch)%500==0 and len(batch)!=0:

                    y_conf,y_pred_poi = predict(batch)
                    y_pred_poi = y_pred_poi.detach().cpu().numpy()
                    y_conf = y_conf.detach().cpu().numpy()
                    for i in range(len(y_pred_poi)):
                        y = int(y_pred_poi[i])
                        predition.append(y)
                        preditionconf.append(y_conf[i][0])
                    batch = []
                
            y_conf,y_pred_poi = predict(batch)
            y_pred_poi = y_pred_poi.detach().cpu().numpy()
            y_conf = y_conf.detach().cpu().numpy()

            for i in range(len(y_pred_poi)):
                y = int(y_pred_poi[i])
                predition.append(y)
                preditionconf.append(y_conf[i][0])
            batch = []
            return preditionconf

   
        # initialize
        df_target = pd.read_csv("./dataset_4sq/lira/attack_target/"+"graph_train.csv")
        alllocs = df_target["POI_id"].unique()
        query_num = len(alllocs)
        df_nyc = pd.read_csv("./dataset_4sq/lira/target/"+"loc_target.csv")

        unique_user = list(df_nyc["users"].unique())
  
        for u in tqdm(unique_user):
            result = common_point_extraction(u)
            print(len(result))
            df_u = df_nyc[df_nyc["users"]==u]
            target_in = df_u["in_loc"].tolist()
            for loc in target_in:
                emb_loc = poi_id2idx_dict.get(loc)
                labels  = np.array([emb_loc] * len(result))
                sc = score(result,labels)
                key = str(u) + "_" + str(emb_loc)
                if key in in_dict:
                    in_dict[key].append(sc)
                else:
                    in_dict[key] = [sc]

            target_out = df_u["out_loc"].tolist()
            for loc in target_out:
                emb_loc = poi_id2idx_dict.get(loc)
                labels  = np.array([emb_loc] * len(result))
                sc = score(result,labels)
                key = str(u) + "_" + str(emb_loc)
                if key in out_dict:
                    out_dict[key].append(sc)
                else:
                    out_dict[key] = [sc]


        np.savez(save_p + str(shadownum) + '_in_dict.npz', in_dict)

         # Save dict2 to an NPZ file
        np.savez(save_p + str(shadownum) + '_out_dict.npz', out_dict)

    
    model.eval()
    Common_Point_Extraction()


if __name__ == '__main__':
    args = parameter_parser()
    # The name of node features in NYC/graph_X.csv
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    dataset = "4sq"
    dataset="4sq"
    setting = "Best"

    PATH = "./dataset_4sq/lira/models/4sq/Target.pt"
    print(PATH)
    
    poi_id2idx_dict = torch.load(PATH)["poi_id2idx_dict"]


    df_target=pd.read_csv("./dataset_4sq/lira/target/loc_target.csv")
    target_users = list(df_target["users"].unique())


    in_dict = {}
    out_dict = {}
    
    save_p = './attack_result/'

    for shadownum in range(8):
        #shadownum="Target"
        train(args,shadownum)

