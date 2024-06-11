from Model_Source.GETNext.model import GCN, NodeAttnMap, UserEmbeddings, Time2Vec, CategoryEmbeddings, FuseEmbeddings, TransformerModel
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd 
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from collections import Counter   
import sys
import math
import torch.nn.functional as F
import random



from scipy.stats import skewnorm, skellam,skewnorm,kstest,t, ks_2samp,anderson,cramervonmises
from scipy.integrate import quad
from scipy.optimize import minimize,differential_evolution
from scipy.optimize import curve_fit



target_model="GETNext" #attacktarget
setting="Best"
dataset = "4sq"
attack="blackbox2"
print(setting+dataset)
device = torch.device('cuda')
query_num = 3

if dataset=="Gowalla":
    loc_count = 2559
else:
    loc_count = 4556

"""Prepare Models for GETNext"""

def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_catid_code',
                             feature3='latitude', feature4='longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X

def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A

def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')
#=============== Load dictionaries first =================

PATH = "./models/GETNext/"+ dataset +"/" + setting + ".pt"
print(PATH)
#print(torch.load(PATH))
poi_id2idx_dict = torch.load(PATH)["poi_id2idx_dict"]
cat_id2idx_dict = torch.load(PATH)["cat_id2idx_dict"]
poi_idx2cat_idx_dict = torch.load(PATH)["poi_idx2cat_idx_dict"]
user_id2idx_dict = torch.load(PATH)["user_id2idx_dict"]

#===============initialize parameters ================================
feature1 = 'checkin_cnt'
feature2 = 'poi_catid'
feature3 = 'latitude'
feature4 = 'longitude'

raw_A = load_graph_adj_mtx("./models/GETNext/"+ dataset +"/graph_A.csv")
raw_X = load_graph_node_features("./models/GETNext/"+ dataset +"/graph_X.csv",
                                 feature1,
                                 feature2,
                                 feature3,
                                 feature4)
A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')
num_pois = raw_X.shape[0]

num_cats = 1
one_hot_encoder = OneHotEncoder()
cat_list = list(raw_X[:, 1])
one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
X[:, 0] = raw_X[:, 0]
X[:, 1:num_cats + 1] = one_hot_rlt
X[:, num_cats + 1:] = raw_X[:, 2:]

num_users = len(user_id2idx_dict)

#===============initialize model================================
if isinstance(X, np.ndarray):
    X = torch.from_numpy(X)
    A = torch.from_numpy(A)
X = X.to(device= torch.device('cuda'), dtype=torch.float)
A = A.to(device=torch.device('cuda'), dtype=torch.float)
poi_embed_model = GCN(ninput=X.shape[1],
                      nhid=[32, 64],
                      noutput=128,
                      dropout=0.3)

# Node Attn Model
node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=128, use_mask=False)

# %% Model2: User embedding model, nn.embedding
num_users = len(user_id2idx_dict)
user_embed_model = UserEmbeddings(num_users, 128)

# %% Model3: Time Model
time_embed_model = Time2Vec('sin', out_dim=32)

# %% Model4: Category embedding model
cat_embed_model = CategoryEmbeddings(num_cats, 32)

# %% Model5: Embedding fusion models
embed_fuse_model1 = FuseEmbeddings(128, 128)
embed_fuse_model2 = FuseEmbeddings(32, 32)

# %% Model6: Sequence model
seq_input_embed = 128 + 128 + 32 + 32
seq_model = TransformerModel(num_pois,
                             num_cats,
                             seq_input_embed,
                             2,
                             1024,
                             2,
                             dropout=0.3)

optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                list(node_attn_model.parameters()) +
                                list(user_embed_model.parameters()) +
                                list(time_embed_model.parameters()) +
                                list(cat_embed_model.parameters()) +
                                list(embed_fuse_model1.parameters()) +
                                list(embed_fuse_model2.parameters()) +
                                list(seq_model.parameters()),
                        lr=0.001,
                        weight_decay=5e-4)


#===============load model================================

poi_embed_model.load_state_dict(torch.load(PATH)["poi_embed_state_dict"])
node_attn_model.load_state_dict(torch.load(PATH)["node_attn_state_dict"])
user_embed_model.load_state_dict(torch.load(PATH)["user_embed_state_dict"])
time_embed_model.load_state_dict(torch.load(PATH)["time_embed_state_dict"])
cat_embed_model.load_state_dict(torch.load(PATH)["cat_embed_state_dict"])
embed_fuse_model1.load_state_dict(torch.load(PATH)["embed_fuse1_state_dict"])
embed_fuse_model2.load_state_dict(torch.load(PATH)["embed_fuse2_state_dict"])
seq_model.load_state_dict(torch.load(PATH)["seq_model_state_dict"])
optimizer.load_state_dict(torch.load(PATH)["optimizer_state_dict"])

poi_id2idx_dict = torch.load(PATH)["poi_id2idx_dict"]
cat_id2idx_dict = torch.load(PATH)["cat_id2idx_dict"]
poi_idx2cat_idx_dict = torch.load(PATH)["poi_idx2cat_idx_dict"]
user_id2idx_dict = torch.load(PATH)["user_id2idx_dict"]

attn_map = torch.load(PATH)["node_attn_map"]

poi_embed_model = poi_embed_model.to(device=device)
node_attn_model = node_attn_model.to(device=device)
user_embed_model = user_embed_model.to(device=device)
time_embed_model = time_embed_model.to(device=device)
cat_embed_model = cat_embed_model.to(device=device)
embed_fuse_model1 = embed_fuse_model1.to(device=device)
embed_fuse_model2 = embed_fuse_model2.to(device=device)
seq_model = seq_model.to(device=device)

poi_embed_model.eval()
node_attn_model.eval()
user_embed_model.eval()
time_embed_model.eval()
cat_embed_model.eval()
embed_fuse_model1.eval()
embed_fuse_model2.eval()
seq_model.eval()

#===============Helper Functions================================
def input_traj_to_embeddings(sample, poi_embeddings):
    # Parse sample
    
    poi_embeddings = poi_embed_model(X, A)
    
    traj_id = sample[0]
    input_seq = [each[0] for each in sample[1]]
    input_seq_time = [each[1] for each in sample[1]]
    input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]

    # User to embedding
    user_id = traj_id.split('_')[0]
    user_idx = user_id2idx_dict[user_id]
    input = torch.LongTensor([user_idx]).to(device=device)
    user_embedding = user_embed_model(input)
    user_embedding = torch.squeeze(user_embedding)

    # POI to embedding and fuse embeddings
    input_seq_embed = []
    for idx in range(len(input_seq)):
        poi_embedding = poi_embeddings[input_seq[idx]]
        poi_embedding = torch.squeeze(poi_embedding).to(device=device)

        # Time to vector
        time_embedding = time_embed_model(
            torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=device))
        time_embedding = torch.squeeze(time_embedding).to(device=device)

        # Categroy to embedding
        cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=device)
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

    for i in range(len(batch_seq_lens)):
        traj_i_input = batch_input_seqs[i]  # list of input check-in pois
        for j in range(len(traj_i_input)):
            y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]

    return y_pred_poi_adjusted

def find_all_loc_by_traj(traj_id):
    ret =[]
    df_train =  pd.read_csv("./Training_data/GETNext/"+dataset+"/getnext_nyc_train.csv")
    df_target = df_train[df_train["trajectory_id"]==traj_id]
    print(len(df_target),1)
    locs = df_target["POI_id"].unique()
    for loc in locs:
        new_loc = poi_id2idx_dict[loc]
        ret.append(new_loc)
    print(ret,"ret")
    return ret


def predict(batch):
    
    src_mask = seq_model.generate_square_subsequent_mask(seq_len).to(device)

    # For padding
    batch_input_seqs = []
    batch_seq_lens = []
    batch_seq_embeds = []

    poi_embeddings = poi_embed_model(X, A)

    for sample in batch:
            input_seq = [each[0] for each in sample[1]]
            input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings))
            batch_seq_embeds.append(input_seq_embed)
            batch_seq_lens.append(len(input_seq))
            batch_input_seqs.append(input_seq)


    batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)


    x = batch_padded.to(device=device, dtype=torch.float)
    y_pred_poi, y_pred_time, _ = seq_model(x, src_mask)
    
    return float(y_pred_time[-1].detach().cpu().numpy()[-1]),y_pred_poi

def perplexity(traj_id,traj, time):
    label = traj[1:]
    prob=[]
    input_lst = [traj_id,[]]
    for i in range(len(traj)-1):
        input_lst[1].append([traj[i],time[i]])
        _, output = predict([input_lst])
        prob.append(output[0][0][label[i]])
    
    ppl= 0
    for i in range(len(prob)):
        ppl=ppl+-math.log2(prob[i])
    return ppl


def log_perplexity(batch_traj,batch_output):
    ppl_lst = []
    for i in range(len(batch_output)):
        prob=[]
        traj = batch_traj[i]
        label = traj[1:]
        out = F.softmax(batch_output[i])
        for j in range(len(traj)-1):
            prob.append(out[j][label[j][0]])
        ppl= 0
        for j in range(len(prob)):
            ppl=ppl+(-math.log2(prob[j]))
        ppl_lst.append(ppl)
    return ppl_lst


def attack_batch(user,current_batch_id,all_samples,batch_size):
    traj_id = user +"_0"
    batch = []
    batch_traj = []
    for i in range(batch_size): 
        input_lst = [traj_id,all_samples[current_batch_id*batch_size+i]]
        batch.append(input_lst)
        batch_traj.append(input_lst[1])
    _, outputs = predict(batch)
    ppl_lst = log_perplexity(batch_traj,outputs)
    return ppl_lst



#============== initial target user =========================
"""
if dataset=="Gowalla":
    target_user = np.load('gowalla_user.npz')["user"].tolist()
else:
    users = pd.read_csv("./attack2_distributions/4sq/best/4.csv")
    target_user= users["user"].values.tolist()
"""
df_train =  pd.read_csv("./Training_data/GETNext/"+dataset+"/getnext_nyc_train.csv")

target_user=list(df_train["user_id"].unique())
print(len(target_user),"target_userlen")
#============== read training set =============


for seq_len in [4]:
    for user in target_user:
        df_user =df_train[df_train["user_id"]==user]
        training_seqs = []
        batch_traj = []
        traj_id_lst=[]
        seq_lst= []
        all_traj_id = df_user["trajectory_id"].unique()
        for traj in all_traj_id:
            df_traj = df_user[df_user["trajectory_id"]==traj]
            if len(df_traj)>=seq_len:
                locs = df_traj["POI_id"].values.tolist()
                for j in range(len(locs)-seq_len+1):
                    traj_lst = [str(user) +"_0",[]]
                    for i in range(seq_len):
                        traj_lst[1].append([poi_id2idx_dict.get(locs[j+i]),0.5])
                    training_seqs.append(traj_lst)
                    batch_traj.append(traj_lst[1])
                    traj_id_lst.append(traj)
                    seq_lst.append(str(locs))
        if len(training_seqs)>=1:
            ppl_lst = [] 
            for i in range(len(batch_traj)//500 + 1):
                _, outputs = predict(training_seqs[i*500:(i+1)*500])  
                ppl_lst += log_perplexity(batch_traj[i*500:(i+1)*500],outputs)
            
            start_id_lst = [a[0][0] for a in batch_traj]
            df_ret = pd.DataFrame({"traj_id": traj_id_lst,"ppl": ppl_lst,"locations": seq_lst,"input":batch_traj,"start_loc":start_id_lst})
            df_ret.to_csv("./attack2_target_all/"+target_model+"/"+ dataset +"/" + setting +"/"+str(seq_len)+"/"+ str(user)+".csv")

