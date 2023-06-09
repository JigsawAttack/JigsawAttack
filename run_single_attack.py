from attack.attack import Attacker
from attack.graphmattack import GraphMattacker
from attack.ihopattack import ihopattack
from attack.sapattack import Sapattacker
from extract_info import get_all_for_attacks
from extract_info import get_all_for_attacks_wiki
from countermeasure import *
import time


from cal_acc import calculate_acc_weighted
def run_single_attack(
    user_kws_universe_size,
    attack_kws_universe_size,
    kws_extraction, # random choosen from 3000 most popular kws or just choosen the most popular kws
    observed_query_number_per_week,
    observe_weeks,
    observe_offset,
    dataset,
    countermeasure_params,
    attack_params,
    similar_data_p=None
    ):
    print("In run single_attack 1:",time.time())
    if dataset == "enron" or dataset == "lucene":
        print(similar_data_p)
        data_for_attack,data_for_acc_cal = get_all_for_attacks(
            user_kws_universe_size,
            attack_kws_universe_size,
            kws_extraction,
            observed_query_number_per_week,
            observe_weeks,
            observe_offset,
            countermeasure_params,dataset,similar_data_p=similar_data_p)
    elif dataset == "wiki":
        data_for_attack,data_for_acc_cal = get_all_for_attacks_wiki(
            user_kws_universe_size,
            attack_kws_universe_size,
            kws_extraction,
            observed_query_number_per_week,
            observe_weeks,
            observe_offset,
            countermeasure_params,dataset,similar_data_p=similar_data_p)
        
    # if countermeasure_params["alg"] == "padding_linear_2":
    #     print("simialr data size before padding CGPR:", len(data_for_attack["sim_kw_d"][0]))
    #     if countermeasure_params["n"]!=0:
    #         data_for_attack["sim_kw_d"] = padding_linear_2(data_for_attack["sim_kw_d"],\
    #             int(countermeasure_params["n"]*len(data_for_attack["sim_kw_d"][0])/data_for_attack["real_doc_num"]))
    #     print("simialr data size after padding CGPR:", len(data_for_attack["sim_kw_d"][0]))
    # elif countermeasure_params["alg"] == "padding_cluster":
    #     print("simialr data size before padding cluster:", len(data_for_attack["sim_kw_d"][0]))
    #     if countermeasure_params["knum_in_cluster"]!=1:
    #         data_for_attack["sim_kw_d"] = padding_cluster(data_for_attack["sim_kw_d"],countermeasure_params["knum_in_cluster"])
    #     print("simialr data size after padding cluster:", len(data_for_attack["sim_kw_d"][0]))
    # elif countermeasure_params["alg"] == "obfuscation":
    #     pass
    # elif countermeasure_params["alg"] == "padding_seal":
    #     print("simialr data size before padding seal:", len(data_for_attack["sim_kw_d"][0]))
    #     if countermeasure_params["n"]!=1:
    #         # rg = np.random.default_rng()
    #         # sim_kw_d_=rg.choice(data_for_attack["sim_kw_d"],data_for_attack["real_doc_num"],replace=True)
    #         # sim_kw_d_ = padding_seal(sim_kw_d_,countermeasure_params["n"])
    #         # data_for_attack["sim_kw_d"] = sim_kw_d_

    #         sim_kw_d = data_for_attack["sim_kw_d"]
    #         if len(sim_kw_d[0])< data_for_attack["real_doc_num"]:
    #             sim_kw_d_ = sim_kw_d
    #             while len(sim_kw_d_[0])<data_for_attack["real_doc_num"]:
    #                 if data_for_attack["real_doc_num"]-len(sim_kw_d_[0]) > len(sim_kw_d[0]):
    #                     sim_kw_d_ = np.hstack((sim_kw_d_,sim_kw_d))
    #                 else:
    #                     sim_kw_d_ = np.hstack((sim_kw_d_,sim_kw_d[:,:data_for_attack["real_doc_num"]-len(sim_kw_d_[0])]))
    #         print(len(sim_kw_d_),data_for_attack["real_doc_num"])
    #         sim_kw_d_ = padding_seal(sim_kw_d_,countermeasure_params["n"])
    #         data_for_attack["sim_kw_d"]=sim_kw_d_
    #     print("simialr data size after padding seal:", len(data_for_attack["sim_kw_d"][0]))
    if attack_params["alg"] == "Ours":

        # if countermeasure_params["alg"] == "padding_linear_2":
        #     data_for_attack["sim_kw_d"] = padding_linear_2(data_for_attack["sim_kw_d"],countermeasure_params["n"])
        # elif countermeasure_params["alg"] == "padding_cluster":
        #     data_for_attack["sim_kw_d"] = padding_cluster(data_for_attack["sim_kw_d"],countermeasure_params["knum_in_cluster"])
        # elif countermeasure_params["alg"] == "obfuscation":
        #     data_for_attack["sim_kw_d"] = obfuscate(data_for_attack["sim_kw_d"],\
        #         countermeasure_params["p"],countermeasure_params["q"],countermeasure_params["m"])
        
        if attack_params["baseRec"]>len(data_for_attack["real_query_d"]):
            baseRec = len(data_for_attack["real_query_d"])
        else:
            baseRec = attack_params["baseRec"]

        
        attacker = Attacker(data_for_attack["sim_kw_d"],
            data_for_attack["real_query_d"],
            data_for_attack["sim_F"],
            data_for_attack["real_F"],
            alpha=attack_params["alpha"],beta=attack_params["beta"],
            no_F=attack_params["no_F"],
            baseRec=baseRec,confRec=attack_params["confRec"],
            refinespeed = attack_params["refinespeed"],countermeasure_params = countermeasure_params,real_doc_num=data_for_attack["real_doc_num"])
        time1 = time.time()
        print("3:",time.time())
        attacker.attack_step_1()
        print("4:",time.time())
        if attack_params["step"]==2:
            attacker.attack_step_2()
        elif attack_params["step"]==3:
            attacker.attack_step_2()
            print("run step3:",time.time())
            attacker.attack_step_3()
        print("after step3:",time.time())
        time_cost = time.time()-time1
        results = [attacker.tdid_2_kwsid_step1,attacker.tdid_2_kwsid_step2,attacker.tdid_2_kwsid]

    elif attack_params["alg"] == "RSA":
        attacker = Attacker(data_for_attack["sim_kw_d"],
            data_for_attack["real_query_d"],
            refinespeed = attack_params["refinespeed"])
        id_query = data_for_acc_cal["id_query"]
        id_kws = data_for_acc_cal["id_kws"]

        known_tdid_2_kwid = {}
        id_query_list = list(id_query.keys())
        random.shuffle(id_query_list)
        for k in range(attack_params["known_query_number"]):
            for kwid in id_kws:
                if id_query[id_query_list[k]]==id_kws[kwid]:
                    known_tdid_2_kwid[id_query_list[k]]=kwid
        
        attacker.tdid_2_kwsid.update(known_tdid_2_kwid)
        time1 = time.time()
        attacker.RSA()
        time_cost = time.time()-time1
        results = [attacker.tdid_2_kwsid,known_tdid_2_kwid]
    
    elif attack_params["alg"] == "Sap":
        attacker = Sapattacker(
            data_for_attack["sim_kw_trend"],
            data_for_attack["real_td_trend"],
            observed_query_number_per_week,
            data_for_attack["sim_V"],data_for_attack["real_V"],
            data_for_attack["real_doc_num"],
            alpha=attack_params["alpha"],countermeasure=countermeasure_params["alg"]
            )
        time1 = time.time()
        attacker.attack()
        time_cost = time.time() - time1
        results = attacker.tdid_2_kwsid
    
    elif attack_params["alg"] == "Graphm":
        attacker = GraphMattacker(data_for_attack["sim_kw_d"],data_for_attack["real_query_d"],
        attack_params["alpha"],attack_params["match_alg"])
        time1= time.time()
        attacker.attack()
        time_cost = time.time() - time1
        results = attacker.tdid_2_kwsid
    elif attack_params["alg"] == "IHOP":
        ndocs = len(data_for_attack["real_query_d"][0])
        nqr = observed_query_number_per_week*observe_weeks
        ntok = len(data_for_attack["real_query_d"])
        nkw = len(data_for_attack["sim_kw_d"])
        Vexp = np.dot(data_for_attack["sim_kw_d"],data_for_attack["sim_kw_d"].T)/len(data_for_attack["sim_kw_d"][0])
        Vobs = np.dot(data_for_attack["real_query_d"],data_for_attack["real_query_d"].T)/ndocs
        fexp = data_for_attack["sim_F"]
        fobs = data_for_attack["real_F"]
        time_before = time.time()
        # if countermeasure_params["alg"] == "obfuscation":
        #     tpr = countermeasure_params["p"]
        #     fpr = countermeasure_params["q"]
        #     common_elements = np.matmul(data_for_attack["sim_kw_d"],data_for_attack["sim_kw_d"].T)
        #     common_not_elements = np.matmul(1-data_for_attack["sim_kw_d"],(1-data_for_attack["sim_kw_d"]).T)
        #     Vaux = common_elements * tpr * (tpr - fpr) + common_not_elements * fpr * (fpr - tpr) + len(data_for_attack["sim_kw_d"][0]) * tpr * fpr
        #     np.fill_diagonal(Vaux, np.diag(common_elements) * tpr + np.diag(common_not_elements) * fpr)
        #     Vaux = Vaux/len(data_for_attack["sim_kw_d"][0])
            
        #     Vexp = Vaux
        #     print(np.min(Vexp))


        results = ihopattack(ndocs,nqr,ntok,nkw,Vexp,Vobs,fexp,fobs,attack_params)
        time_cost = time.time()-time_before
    else:
        print("No attack")

    attack_results = {
            "results":results,
            "Attack_time":time_cost,
            "data_for_acc_cal":data_for_acc_cal,
            "real_F":data_for_attack["real_F"],
            "real_V":data_for_attack["real_V"],
            "attack_params":attack_params,
            "countermeasure_params":countermeasure_params
        }
    # correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(data_for_acc_cal,results[0])
    # print(acc)
    return attack_results
