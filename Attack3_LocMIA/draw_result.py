import pandas as pd
import numpy as np
import os
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

shadownumber = 128 #128
#load result dictionary
in_dict = np.load("./attack_result/" +str(shadownumber-1) + "_in_dict.npz",allow_pickle=True)["arr_0"].item()

out_dict =  np.load("./attack_result/" +str(shadownumber-1) + "_out_dict.npz",allow_pickle=True)["arr_0"].item()

in_dict.update(out_dict)
target_in = np.load("./Target_time_in_dict.npz",allow_pickle=True)["arr_0"]
target_out = np.load("./Target_time_out_dict.npz",allow_pickle=True)["arr_0"]

# load model dict

PATH = "./dataset_4sq/lira/models/4sq/Target.pt"

print(PATH)

poi_id2idx_dict = torch.load(PATH)["poi_id2idx_dict"]


df0 = pd.read_csv("./dataset_4sq/lira/data_loc/0.csv")
df1 = pd.read_csv("./dataset_4sq/lira/data_loc/1.csv")
df_all_traj = pd.concat([df0, df1], ignore_index=True)

df_target = pd.read_csv("./dataset_4sq/lira/target/loc_target.csv")


####################################create convertion dict####################################


df_in = df_all_traj[df_all_traj["POI_id"].isin(df_target["in_loc"])]
print(len(df_in))
df_out = df_all_traj[df_all_traj["POI_id"].isin(df_target["out_loc"])]
print(len(df_out))
#df_all_traj = df_in
df_all_traj = pd.concat((df_in,df_out))

traj_ids = df_all_traj["trajectory_id"].values


in_locs = df_target["in_loc"].values
out_locs = df_target["out_loc"].values
users_lst =  df_target["users"].values
id_conversion_d = {}
for i in range(len(in_locs)):
    key = str(users_lst[i]) + "_" + str(i)
    new_id = str(users_lst[i]) + "_" + str(poi_id2idx_dict.get(in_locs[i]))
    id_conversion_d[key] = new_id
for i in range(len(out_locs)):
    key = str(users_lst[i]) + "_" + str(i+301)
    new_id = str(users_lst[i]) + "_" + str(poi_id2idx_dict.get(out_locs[i]))
    id_conversion_d[key] = new_id

print(id_conversion_d)
##################################################
traj_log = pd.read_csv("./dataset_4sq/lira/traj_log.csv")
print(traj_log.columns)
traj_log = traj_log.rename(columns=id_conversion_d)

print(traj_log.columns)


#################################### start from here same as attack3 ##################################
################################### you don't need this if you have all score.csv ###########################

conf_dict = {}
cols = traj_log.columns.tolist()
for key in in_dict.keys():
    if key not in cols:
        print(key)
        continue
    if 0 in traj_log[str(key)].values:
        conf_dict[key+"_in"]=[in_dict[key][0]]
        conf_dict[key+"_out"] =[]
    else:
        conf_dict[key+"_out"]=[in_dict[key][0]]
        conf_dict[key+"_in"]=[]
    
print(conf_dict)


for key in in_dict.keys():
    for j in range(1,shadownumber):
        if key not in cols:
            print(key)
            continue
        if (j) in traj_log[str(key)].values:
            conf_dict[key+"_in"].append(in_dict[key][j])
        else:
            conf_dict[key+"_out"].append(in_dict[key][j])

# Save all score
import pandas as pd
all_traj_id = list(conf_dict.keys())
print(len(all_traj_id))
df_ret = pd.DataFrame()
for traj in all_traj_id:
    if len(conf_dict[traj]) !=shadownumber//2:
        continue
    df_ret[traj] = conf_dict[traj]

df_ret.to_csv("./all_scores_seq.csv")


####################################start from here load target model output ###############################
conf_dict = {}
cols = traj_log.columns.tolist()
for key in in_dict.keys():
    if key not in cols:
        print(key)
        continue
    if 0 in traj_log[str(key)].values:
        conf_dict[key+"_in"]=[in_dict[key][0]]
        conf_dict[key+"_out"] =[]
    else:
        conf_dict[key+"_out"]=[in_dict[key][0]]
        conf_dict[key+"_in"]=[]
    
print(conf_dict)


conf_dict = {}
in_out_dict = {}

in_array = np.load("./Target_time_in_dict.npz",allow_pickle=True)["arr_0"].item()
out_array = np.load("./Target_time_out_dict.npz",allow_pickle=True)["arr_0"].item()


in_sc = []
out_sc = []
print(in_array)

for i in range(len(in_array)):
    in_traj =list(in_array.keys())[i]
    sc = in_array[in_traj]

    conf_dict[in_traj] = sc
    in_sc.append(sc[0])

    in_out_dict[in_traj] = 1
    
for i in range(len(out_array)):
    out_traj =list(out_array.keys())[i]
    sc = out_array[out_traj]

    conf_dict[out_traj] = sc
    
    out_sc.append(sc[0])
    in_out_dict[out_traj] = 0

np.savez("./1_output.npz",in1=in_sc,out1=out_sc)


###############################################read all score ################################
import pandas as pd
import os
import scipy.stats

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
df_scores = pd.read_csv("./all_scores_seq.csv")
print("loaded all scores")




all_trajs =list(conf_dict.keys())
#print(all_trajs)

prediction = []
answers = []

for traj in all_trajs:
    key = traj+"_in"
    if key not in df_scores:
        print(2222,traj)
        continue
    dat_in = df_scores[traj+"_in"].values
    dat_out = df_scores[traj+"_out"].values
    
    mean_in = np.median(dat_in)
    mean_out = np.median(dat_out)
    std_in = np.std(dat_in)
    std_out = np.std(dat_out)
    
    sc = ([conf_dict[traj]])
    

    pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in+1e-30)
    pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out+1e-30)
    score = (pr_in-pr_out)[0]
    #print(score)
    prediction= prediction + [score[0]]
    answers= answers + [in_out_dict[traj]]


metric = "acc"
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def do_plot(prediction,answers,c,legend='', metric='auc', sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))
    print(np.where(fpr<.9)[0][-1],1231231243124124)
    low = tpr[np.where(fpr<.1)[0][-1]]
    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@10%%FPR of %.4f'%(legend, auc,acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    plt.plot(fpr, tpr, label=legend+metric_text,color=c, **plot_kwargs)
    return (acc,auc)
print(answers)
print(len(answers))
np.savez('result.npz', prediction=prediction, answers=answers)
do_plot(prediction,answers,"blue","Ours (online)\n", metric='auc')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot([0, 1], [0, 1], ls='--', color='gray')
plt.legend(fontsize=8)

plt.show()