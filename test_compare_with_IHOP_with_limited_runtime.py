import pickle
import time
from tqdm import tqdm
from cal_acc import calculate_acc_weighted
from run_single_attack import *
import os

def comparison(test_times,dataset,\
        Query_number_per_week = [2500],\
        Kws_uni_size = [500],observed_time=50,eta=50,kws_extraction="sorted",refspeed=10,Sim_data_rate=[1,0.5,0.1]):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./results/test_comparison_with_IHOP_with_limited_time"):
        os.makedirs("./results/test_comparison_with_IHOP_with_limited_time")
   

    
    if not os.path.exists("./results/test_comparison_with_IHOP_with_limited_time/Ours"):
        os.makedirs("./results/test_comparison_with_IHOP_with_limited_time/Ours")
    if not os.path.exists("./results/test_comparison_with_IHOP_with_limited_time/IHOP"):
        os.makedirs("./results/test_comparison_with_IHOP_with_limited_time/IHOP")
    for i in tqdm(range(test_times)):
        Our_Result = []
        IHOP_Result = []
        runtime = []
        for query_number_per_week in Query_number_per_week:
            our_result = []
            ihop_result = []
            for kws_uni_size in Kws_uni_size:
                print("Test Our attack")
                if kws_uni_size >=5000:
                    refspeed = int(kws_uni_size/10)
                elif kws_uni_size > 2000:
                    refspeed = 50
                else:
                    refspeed = 10
                attack_params = {"alg":"Ours","alpha":0.3,"beta":0.9,"step":3,\
                    "baseRec":45,"confRec":35,\
                    "no_F":False,"refinespeed":refspeed}
                r = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,\
                    query_number_per_week,observed_time,eta,dataset,\
                    {"alg":None},attack_params)
                our_result.append(r)   
                          
                correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(r["data_for_acc_cal"],r["results"][2])
                print("Our accuracy:",acc)
                print("Our runtime:",r["Attack_time"])
                runtime = r["Attack_time"]

                print("Test IHOP attack")
                attack_params = {"alg":"IHOP","niters":500,"pfree":0.25,"no_F":False,"runtime_limit":runtime}
                r = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,\
                    query_number_per_week,observed_time,eta,dataset,\
                    {"alg":None},attack_params)
                ihop_result.append(r)
                correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(r["data_for_acc_cal"],r["results"])
                print("IHOP accuracy:",acc)
                print("IHOP runtime:",r["Attack_time"])


            Our_Result.append(our_result)  
            IHOP_Result.append(ihop_result)
             
        with open("results/test_comparison_with_IHOP_with_limited_time/Ours/"+dataset+"_"+str(i)+"_"+str(kws_uni_size)+".pkl", "wb") as f:
            pickle.dump(Our_Result,f)
        with open("results/test_comparison_with_IHOP_with_limited_time/IHOP/"+dataset+"_"+str(i)+"_"+str(kws_uni_size)+".pkl", "wb") as f:
            pickle.dump(IHOP_Result,f)

if __name__ == "__main__":

    comparison(10,"wiki",Query_number_per_week=[10000],Kws_uni_size=[5000,10000,15000],observed_time=30,eta=10)
