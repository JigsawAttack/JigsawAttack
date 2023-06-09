import pickle
import time
from tqdm import tqdm
from cal_acc import calculate_acc_weighted
from run_single_attack import *
import os

def comparison(test_times,dataset,\
        Query_number_per_week = [500],\
        Kws_uni_size = [1000],\
        Known_query_number_in_RSA = 10,\
        kws_extraction="random",refspeed = 10,observe_weeks=50,tau=10):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./results/test_random"):
        os.makedirs("./results/test_random")
    
    name = [i for i in range(5)]
    Attack_params = [               
        {"alg":"Ours","alpha":0.3,"beta":0.9,"step":3,\
                        "baseRec":45,"confRec":35,\
                        "no_F":False,"refinespeed":refspeed}
    ]
    for j in range(1):
        print("Test our attack")
        if not os.path.exists("./results/test_random/Ours"):
            os.makedirs("./results/test_random/Ours")
        for i in tqdm(range(test_times)):
            Result = []
            for query_number_per_week in Query_number_per_week:
                result = []
                for kws_uni_size in Kws_uni_size:
                    
                    r = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,\
                        query_number_per_week,observe_weeks,tau,dataset,\
                        {"alg":None},Attack_params[j])
                    result.append(r)
                Result.append(result)         
            with open("results/test_random/Ours/"+str(j)+"_"+dataset+"_"+str(i)+".pkl", "wb") as f:
                pickle.dump(Result,f)

    # print("Test RSA attack")
    # if not os.path.exists("./results/test_random/RSA"):
    #     os.makedirs("./results/test_random/RSA")
    # for i in tqdm(range(test_times)):
    #     Result = []
    #     for query_number_per_week in Query_number_per_week:
    #         result = []
    #         for kws_uni_size in Kws_uni_size:
    #             attack_params = {"alg":"RSA","known_query_number":Known_query_number_in_RSA,"refinespeed":10}
    #             r = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,\
    #                 query_number_per_week,50,50,dataset,\
    #                 {"alg":None},attack_params)
    #             result.append(r)
    #         Result.append(result)         
    #     with open("results/test_random/RSA/"+dataset+"_"+str(i)+".pkl", "wb") as f:
    #         pickle.dump(Result,f)

    # print("Test IHOP attack")
    # if not os.path.exists("./results/test_random/IHOP"):
    #     os.makedirs("./results/test_random/IHOP")
    # for i in tqdm(range(test_times)):
    #     Result = []
    #     for query_number_per_week in Query_number_per_week:
    #         result = []
    #         for kws_uni_size in Kws_uni_size:
    #             attack_params = {"alg":"IHOP","niters":500,"pfree":0.25,"no_F":False}
    #             r = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,\
    #                 query_number_per_week,50,50,dataset,\
    #                 {"alg":None},attack_params)
    #             result.append(r)
    #         Result.append(result)         
    #     with open("results/test_random/IHOP/"+dataset+"_"+str(i)+".pkl", "wb") as f:
    #         pickle.dump(Result,f)
    
def draw_comparison(dataset,test_times,qua, Query_number_per_week = [500],Kws_uni_size = [1000]):
    
    Query_number_per_week = [500]
    Kws_uni_size = [1000]
   

    ###########get our results and plot########
    Our_results_acc = [[[] for i in Query_number_per_week] for j in Kws_uni_size]
    Our_results_time = [[[] for i in Query_number_per_week] for j in Kws_uni_size]
    for test_time in range(test_times):
        with open("results/test_random/Ours/0_"+dataset+"_"+str(test_time)+".pkl", "rb") as f:
            Result = pickle.load(f)
        for j in range(len(Query_number_per_week)):
            
            results=Result.pop(0)
            for i in range(len(Kws_uni_size)):
                result = results.pop(0)
                data_for_acc_cal = result["data_for_acc_cal"]
                tdid_2_kwid = result["results"][2]
                correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid)
                if qua == 4:
                    Our_results_acc[i][j].append(acc)
                Our_results_time[i][j].append(result["Attack_time"])
    
    Our_results_time = np.array(Our_results_time)
    
    Our_results_time = np.average(Our_results_time, axis=2)
    
    print("Accurracy of ours:",np.average(np.array(Our_results_acc),axis=2))

 ###########get ihop results and plot########
    # IHOP_results_acc = [[[] for i in Query_number_per_week] for j in Kws_uni_size]
    # IHOP_results_time = [[[] for i in Query_number_per_week] for j in Kws_uni_size]
    # for test_time in range(test_times):
    #     with open("results/test_random/IHOP/"+dataset+"_"+str(test_time)+".pkl", "rb") as f:
    #         Result = pickle.load(f)
    #     for j in range(len(Query_number_per_week)):
    #         results=Result.pop(0)
        
    #         for i in range(len(Kws_uni_size)):
    #             result = results.pop(0)
    #             data_for_acc_cal = result["data_for_acc_cal"]
    #             tdid_2_kwid = result["results"]
                
    #             correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid)
    #             if qua == 4:
    #                 IHOP_results_acc[i][j].append(acc)
    #             IHOP_results_time[i][j].append(result["Attack_time"])
    # IHOP_results_time = np.array(IHOP_results_time)
    # IHOP_results_time = np.average(IHOP_results_time, axis=2)
    # print("Accurracy of IHOP:",np.average(np.array(IHOP_results_acc),axis=2)) 


if __name__ == "__main__":
    #comparison(30,"enron",Query_number_per_week=[500],Kws_uni_size=[1000],kws_extraction="random")
    #comparison(30,"lucene",Query_number_per_week=[500],Kws_uni_size=[1000],kws_extraction="random")
    # draw_comparison("enron",30,4)
    # draw_comparison("lucene",30,4)
    # comparison(10,"wiki",Query_number_per_week=[5000],Kws_uni_size=[3000],refspeed=50,kws_extraction="random",observe_weeks=30,tau=10)
    draw_comparison("wiki",10,4,Query_number_per_week = [5000],Kws_uni_size = [3000])
