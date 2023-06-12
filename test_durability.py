from run_single_attack import *
import pickle
import os
from tqdm import tqdm
from cal_acc import calculate_acc_weighted



def test_durability(test_times,kws_uni_size,datasets = ["enron","lucene"]):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./results/test_durability"):
        os.makedirs("./results/test_durability")

    for dataset in datasets:
        if dataset == "wiki":
            Eta = [2,10,20,30]
            observed_time = 30
            query_number_per_week = 5000
        else:
            Eta = [10,50,100,150]
            observed_time = 50
            query_number_per_week = 2000
        Acc_Eta = []
        for _ in tqdm(range(test_times)):
            acc_Eta = []
            for eta in Eta:
                attack_params = {"alg":"Ours","alpha":0.3,"step":3,\
                    "baseRec":100,"confRec":50,\
                    "beta":0.8,"no_F":None,"refinespeed":15}
                result = run_single_attack(kws_uni_size,kws_uni_size,"sorted",query_number_per_week,observed_time,eta,dataset,\
                    {"alg":None},attack_params)
                data_for_acc_cal = result["data_for_acc_cal"]
                tdid_2_kwid = result["results"][1]
                correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid)
                print("time offset(weeks):",eta,"| Acc: ",acc)
                acc_Eta.append(acc)
            Acc_Eta.append(acc_Eta)
        with open("results/test_durability/"+dataset+".pkl", "wb") as tf:
            pickle.dump([Eta,Acc_Eta],tf)
        print(np.average(np.array(Acc_Eta),axis=0))

test_durability(30,1000,["enron","lucene"])
test_durability(30,3000,["wiki"])
    
