import pandas as pd
import sys

shadownumber =128 # put your shadow model number here

df_log = pd.read_csv("./dataset_4sq/lira/traj_log.csv")


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

import numpy as np
conf_dict = {}
in_out_dict = {}

my_array = np.load("./dataset_4sq/lira/prediction/0.npz",allow_pickle=True)

my_array = my_array["dict1"].item()

conf_dict ={}

for traj_id in my_array.keys():
    sc = my_array[traj_id]
    if 0 in df_log[traj_id].values:
        conf_dict[traj_id+"_in"]=[sc]
        conf_dict[traj_id+"_out"] =[]
    else:
        conf_dict[traj_id+"_out"]=[sc]
        conf_dict[traj_id+"_in"]=[]



for j in range(1,shadownumber):
    my_array = np.load("./dataset_4sq/lira/prediction/"+str(j)+".npz",allow_pickle=True)

    my_array = my_array["dict1"].item()


    for traj_id in my_array.keys():
        if j in df_log[traj_id].values:
            conf_dict[traj_id+"_in"]+=[my_array[traj_id]]
        else:
            conf_dict[traj_id+"_out"]+=[my_array[traj_id]]

df_ret = pd.DataFrame()
all_traj_id = list(conf_dict.keys())
print(all_traj_id)
for traj in all_traj_id:
    df_ret[traj] = conf_dict[traj]

df_ret.to_csv("./all_scores_seq.csv")


import numpy as np

df_scores = pd.read_csv("./all_scores_seq.csv")

conf_dict = {}
in_out_dict = {}

my_array = np.load("./dataset_4sq/lira/prediction/target"+str(shadownumber)+".npz",allow_pickle=True)
conf_dict = my_array["dict1"].item()
in_out_dict = my_array["dict2"].item()


import scipy
all_trajs =list(conf_dict.keys())

prediction = []
answers = []

for traj in all_trajs:
    dat_in = df_scores[traj+"_in"].values
    dat_out = df_scores[traj+"_out"].values

    mean_in = np.median(dat_in)
    mean_out = np.median(dat_out)

    std_in = np.std(dat_in)
    std_out = np.std(dat_out)
    sc = ([conf_dict[traj]])
    

    pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in+1e-30)
    pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out+1e-30)
    score = pr_in-pr_out
    prediction= prediction + [score]
    answers= answers + [in_out_dict[traj]]


df_result = pd.DataFrame({"traj":all_trajs,"pred":prediction,"ans":answers})

df_result.to_csv("./result.csv")

import pandas as pd
import os
import scipy.stats

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve


metric = "acc"
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def do_plot(prediction,answers,legend='', metric='auc', sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr<.1)[0][-1]]

    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f'%(legend, auc,acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    plt.plot(fpr, tpr, label=legend+metric_text, **plot_kwargs)
    return (acc,auc)

do_plot(prediction,answers,"Ours (online)\n", metric='auc')

plt.semilogx()
plt.semilogy()
plt.xlim(1e-5,1)
plt.ylim(1e-5,1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot([0, 1], [0, 1], ls='--', color='gray')
plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
plt.legend(fontsize=8)

plt.show()