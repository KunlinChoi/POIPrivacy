import pandas as pd
import pickle
import torch
import numpy as np
import os
from collections import Counter

df_nyc = pd.read_csv("./Training_data/getnext_nyc_train.csv")
user_id = df_nyc["user_id"].unique().tolist()


query = 50

path = "./models/best.pt"
poi_id2idx_dict = torch.load(path, map_location=torch.device('cpu'))["poi_id2idx_dict"]
result_npy = np.load("./attack1_result/query/result.npy")
user_id_npy = np.load("./attack1_result/query/user_id.npy")


def gen_rank(user):
    df_user = df_nyc[df_nyc["user_id"] == user]
    poi_ids = df_user["POI_id"].values
    new_poi_ids = []
    dic = {}
    for i in poi_ids:
        item = poi_id2idx_dict[i]
        dic[item] = dic.get(item, 0) + 1
    sorted_dict = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict

def gen_attack_result_dic(user_vote):
    sorted_indices = np.argsort(user_vote)[::-1]
    return sorted_indices


acc_count=0
acc3 = 0
acc5 = 0
acc10 =0
extractable_count = 0
total_user_num = 0
user_lst = []
conf_lst =[]
for i in range(len(user_id_npy)):
    this_user_id = user_id_npy[i]
    if this_user_id not in user_id:
        continue
    total_user_num += 1
    attack_dict = gen_attack_result_dic(result_npy[i])
    
    
    target_dict = gen_rank(this_user_id)

    keys = list(attack_dict)
    conf = result_npy[i][keys[0]]
    user_lst.append(i)
    conf_lst.append(conf)
    target_keys = list(target_dict.keys())
    if keys[0] == target_keys[0]:
        acc_count+=1
    if target_keys[0] in keys:
        extractable_count +=1
    if target_keys[0] in keys[:3]:
        acc3 +=1
    if target_keys[0] in keys[:5]:
        acc5 +=1
    if target_keys[0] in keys[:10]:
        acc10 +=1
print(total_user_num)
print(acc_count,acc3,acc5,total_user_num)
#print(acc_count,total_user_num,acc3,acc5,acc10,extractable_count)

df_r = pd.DataFrame()
df_r["user"] = user_lst
df_r["conf"] = conf_lst
df_r.to_csv("att1_conf_ana.csv")