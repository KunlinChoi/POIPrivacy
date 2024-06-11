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


    # %% ====================== Define Dataset ======================
    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df, public=True):
            self.df = train_df
            self.traj_seqs = []  # traj id: user id + traj no.
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(train_df['trajectory_id'].tolist())):
                traj_df = train_df[train_df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
                time_feature = traj_df[args.time_feature].to_list()

                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

                if len(input_seq) < args.short_traj_thres:
                    continue

                self.traj_seqs.append(traj_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(df['trajectory_id'].tolist())):
                user_id = traj_id.split('_')[0]

                # Ignore user if not in training set
                if user_id not in user_id2idx_dict.keys():
                    continue

                # Ger POIs idx in this trajectory
                traj_df = df[df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = []
                time_feature = traj_df[args.time_feature].to_list()

                for each in poi_ids:
                    if each in poi_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[each])
                    else:
                        # Ignore poi if not in training set
                        continue

                # Construct input seq and label seq
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

                # Ignore seq if too short
                if len(input_seq) < args.short_traj_thres:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(traj_id)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    #train_dataset = TrajectoryDatasetTrain(train_df)
    #val_dataset = TrajectoryDatasetVal(val_df)
    #train_loader = DataLoader(train_dataset,
    #                          batch_size=args.batch,
    #                          shuffle=True, drop_last=False,
    #                          pin_memory=True, num_workers=args.workers,
    #                          collate_fn=lambda x: x)
    #val_loader = DataLoader(val_dataset,
    #                        batch_size=args.batch,
    #                        shuffle=False, drop_last=False,
    #                        pin_memory=True, num_workers=args.workers,
    #                        collate_fn=lambda x: x)
    #print("Dataset size:", len(train_dataset), len(val_dataset))

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
    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_time = maksed_mse_loss

    # default: 0.001
    optimizer = optim.Adam(params=params, lr=0.0001, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)



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

    """
    # optimizer DP
    print("Use DPSGD optimizer")
    from opacus import PrivacyEngine
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=0.1,
        max_grad_norm=10,
    )
    """


    # %% Tool functions for training
    def input_traj_to_embeddings(sample, poi_embeddings):
        # Parse sample
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        input_seq_time = [each[1] for each in sample[1]]
        input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]

        # User to embedding
        user_id = traj_id.split('_')[0]
        user_idx = user_id2idx_dict[user_id]
        input = torch.LongTensor([user_idx]).to(device=args.device)
        user_embedding = user_embed_model(input)
        user_embedding = torch.squeeze(user_embedding)

        # POI to embedding and fuse embeddings
        input_seq_embed = []
        for idx in range(len(input_seq)):
            poi_embedding = poi_embeddings[input_seq[idx]]
            poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)

            # Time to vector
            time_embedding = time_embed_model(
                torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
            time_embedding = torch.squeeze(time_embedding).to(device=args.device)

            # Categroy to embedding
            cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=args.device)
            cat_embedding = cat_embed_model(cat_idx)
            cat_embedding = torch.squeeze(cat_embedding)

            # Fuse user+poi embeds
            fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)
            fused_embedding2 = embed_fuse_model2(time_embedding, cat_embedding)

            # Concat time, cat after user+poi
            concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)

            # Save final embed
            input_seq_embed.append(concat_embedding)

        return input_seq_embed

    def adjust_pred_prob_by_graph(y_pred_poi):
        y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
        attn_map = node_attn_model(X, A)

        for i in range(len(batch_seq_lens)):
            traj_i_input = batch_input_seqs[i]  # list of input check-in pois
            for j in range(len(traj_i_input)):
                y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]

        return y_pred_poi_adjusted

    # %% ====================== Train ======================
    model = model.to(device=args.device)
    
    def Common_Point_Extraction():
        def predict(batch):
            # print(batch)
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
            top1_conf = y_pred_poi
            return top1_conf, top1

        def common_point_extraction(user,time):
            fake_label = [[1,0]]
            #time = 0.60
            traj_tag = str(int(user)) + "_1"
            predition = []
            preditionconf=[]

            batch = []
            locs = np.random.choice(list(range(query_num)), 5, replace=False) # same number of queries

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
            #print(len(y_conf),len(y_pred_poi))
            for i in range(len(y_pred_poi)):
                y = int(y_pred_poi[i])
                predition.append(y)
                preditionconf.append(y_conf[i][0])
            batch = []
            return preditionconf

        def gen_rank(user):
            df_user = df_nyc[df_nyc["user_id"] == user ]
            poi_ids = df_user["POI_id"].values
            new_poi_ids = []
            for i in poi_ids:
                new_poi_ids.append(poi_id2idx_dict[i])

            x = Counter(new_poi_ids)
            sorted_x = sorted(x, key=x.get, reverse=True)
            return sorted_x
    
        # initialize
        df_target = pd.read_csv("./dataset_4sq/lira/attack_target/"+"graph_train.csv")
        alllocs = df_target["POI_id"].unique()
        query_num = len(alllocs)
        df_nyc = pd.read_csv("./dataset_4sq/lira/target/"+"loc_target.csv")

        unique_user = list(df_nyc["users"].unique())
        
        for u in tqdm(unique_user):
            all_result = []
            for t in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                result = common_point_extraction(u,t)
                all_result.append(result)
            #result = common_point_extraction(u)
            #print(len(result))
            df_u = df_nyc[df_nyc["users"]==u]
            target_in = df_u["in_loc"].tolist()
            for loc in target_in:
                emb_loc = poi_id2idx_dict.get(loc)
                temp = []
                for con in all_result:
                    labels  = np.array([emb_loc] * len(con))
                    sc = score(con,labels)
                    temp.append(sc)
                sc = max(temp)
                key = str(u) + "_" + str(emb_loc)
                if key in in_dict:
                    in_dict[key].append(sc)
                else:
                    in_dict[key] = [sc]

            target_out = df_u["out_loc"].tolist()
            for loc in target_out:
                emb_loc = poi_id2idx_dict.get(loc)
                temp = []
                for con in all_result:
                    labels  = np.array([emb_loc] * len(con))
                    sc = score(con,labels)
                    temp.append(sc)
                sc = max(temp)
                key = str(u) + "_" + str(emb_loc)
                if key in out_dict:
                    out_dict[key].append(sc)
                else:
                    out_dict[key] = [sc]


        np.savez(save_p + 'time_in_dict.npz', in_dict)

         # Save dict2 to an NPZ file
        np.savez(save_p + 'time_out_dict.npz', out_dict)

    
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
    #   PATH = "./Pretrained_Models/"+setting+".pt"
    print(PATH)

    poi_id2idx_dict = torch.load(PATH)["poi_id2idx_dict"]


    df_target=pd.read_csv("./dataset_4sq/lira/target/loc_target.csv")
    target_users = list(df_target["users"].unique())


    in_dict = {}
    out_dict = {}
    
    save_p = './Target_'

    for shadownum in range(1):
        shadownum="Target"
        train(args,shadownum)

