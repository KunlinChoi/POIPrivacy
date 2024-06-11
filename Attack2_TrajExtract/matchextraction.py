import pandas as pd
import numpy as np
import os
import ast
import math
from scipy.stats import skewnorm,kstest
from scipy.integrate import quad

target_model="GETNext" #attacktarget
setting="Best"
dataset = "4sq"
print(setting+dataset)
query_num = 500
seq_len = 4
beam = 10

folder_path = './attack2_extraction_result/'+target_model+'/'+dataset+'/Best/4/' + str(beam)
ppl_path = './attack2_target_all/'+target_model+'/'+dataset+'/Best/4'
count = 0
total_count = 0
user_count =0
total_user_count= 0
success_output=[]
all_output = []

def approximate_exposure(a,loc,scale,target_ppl):
    def integrand(x):
        return skewnorm.pdf(x, a, loc, scale)
    integral, error = quad(integrand, 0, target_ppl)
    return -math.log2(integral)

total_expo = 0

for filename in os.listdir(folder_path):
    # Print the name of each file
    if filename[-3:]!= "csv":
        continue
    user = filename.split(".")[0]
    df = pd.read_csv(folder_path+"/"+filename)
    pred1 = df["pred1"].values
    ppl = pd.read_csv(ppl_path+"/"+filename)
    ppl_value = ppl["input"].values
    ppl_value = [ast.literal_eval(a) for a in ppl_value]
    
    ppl_value = [str([b[0] for b in a]) for a in ppl_value]
   
    total_user_count +=1
    flag = 0
    

    
    for i in range(len(ppl_value)):
        total_count+=1
        ppl_input = ppl_value[i]
        if str(ppl_input) in pred1:
            count+=1
            flag =1
            success_output.append([user,ppl_input])
            all_output.append([user,ppl_input,1])
        else:
            all_output.append([user,ppl_input,0])
    if flag ==1:
        user_count+=1
print(count,total_count,user_count,total_user_count)
#np.savez('./Extraction_match_result/'+target_model +"/" + dataset +"/"+setting+"/"+str(seq_len) +"_extracted.npz", np.array(success_output))
#np.savez('./Extraction_match_result/'+target_model +"/" + dataset +"/"+setting+"/"+str(seq_len) +"_all.npz", np.array(all_output))
#print(success_output)
