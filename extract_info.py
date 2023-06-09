import numpy as np
import pickle
#import random
from countermeasure import *
import time

def get_F(kws_list,kws_dict,time):
    kws_F = np.zeros(len(kws_list))
    for i in range(len(kws_list)):
        kws_F[i] = np.sum(kws_dict[kws_list[i]]["trend"][time[0]:time[1]])
    kws_F = kws_F/np.sum(kws_F)
    return kws_F

def get_all_for_attacks(
    user_kws_universe_size,
    attacker_kws_universe_size, 
    kws_extraction, # randomly choosen from 3000 most popular kws or just choosen the most popular kws
    observed_query_number_per_week,
    observe_weeks,
    observe_offset,
    countermeasure_params,
    dataset,
    known_database_size = True,
    similar_data_p = None
):
    ## read documents, we treat half the documents as real documents and another half as simliar data
    if dataset == "enron":
        db_dir = "dataset/enron_db.pkl"
    elif dataset == "lucene":
        db_dir = "dataset/lucene_db.pkl"
    elif dataset == "wiki":
        db_dir = "dataset/wiki_0.pkl"
    else:
        assert False

    with open(db_dir,"rb") as f:
        total_doc,kws_dict = pickle.load(f)
    kws_list = list(kws_dict.keys())
    np.random.shuffle(total_doc)
    #print(similar_data_p)
    if similar_data_p == None:
        sim_doc = total_doc[:(int) (len(total_doc)/2)]
    else:
        sim_doc = total_doc[:(int) (len(total_doc)/2*similar_data_p)]
    real_doc = total_doc[(int) (len(total_doc)/2):]
    del total_doc

    ## Get kws universe, we suppose that the kws extraction algorithm of the user and tha attacker are the same.
    kws_count = []
    for k in kws_list:
        kws_count.append([k, kws_dict[k]["count"]])
    if kws_extraction == "sorted":
        kws_count.sort(reverse=True,key = lambda x: x[1])
        user_kws_universe = kws_count[:user_kws_universe_size]
        user_kws_universe = [tmp[0] for tmp in user_kws_universe]
        known_kws = kws_count[:attacker_kws_universe_size]
        known_kws = [tmp[0] for tmp in known_kws]
    elif kws_extraction == "random":
        np.random.shuffle(kws_count)
        user_kws_universe = kws_count[:user_kws_universe_size]
        user_kws_universe = [tmp[0] for tmp in user_kws_universe]
        # the kws known to the attacker includes the user kws universe
        known_kws = kws_count[:attacker_kws_universe_size]
        known_kws = [tmp[0] for tmp in known_kws]

    ## sample queries from real frequency and generate observed frequency
    sim_kw_trend = []
    for i in range(observe_weeks):
        sim_kw_trend.append(get_F(user_kws_universe,kws_dict,(i,i+1)))
    sim_kw_trend = np.array(sim_kw_trend)
    sim_kw_trend = sim_kw_trend.T

    query_all = set()
    query_freq_all = []
    query_freq_all_dict = {}
    for i in range(observe_weeks):
        f = get_F(user_kws_universe,kws_dict,(observe_offset+i,observe_offset+i+1))
        query = np.random.choice(user_kws_universe,observed_query_number_per_week,replace=True,p=f)
        query_freq = {}
        for q in query:
            if q in query_freq:
                query_freq[q] = query_freq[q]+1
            else:
                query_freq[q] = 1
        for q in query:
            if q in query_freq_all_dict:
                query_freq_all_dict[q] = query_freq_all_dict[q]+1
            else:
                query_freq_all_dict[q] = 1
        query_all.update(set(query.tolist()))
        query_freq_all.append(query_freq)
    
    distinct_query_number = len(query_all)
    queried_kws = list(query_all)
    real_td_trend = np.zeros((distinct_query_number,len(query_freq_all)))
    real_F = np.zeros((distinct_query_number))
    for i in range(len(query_freq_all)):
        for j in range(distinct_query_number):
            if queried_kws[j] in query_freq_all[i]:
                real_F[j] += query_freq_all[i][queried_kws[j]]
                real_td_trend[j][i] = query_freq_all[i][queried_kws[j]]
    real_td_trend = real_td_trend/observed_query_number_per_week

    ## pair shuffled id and queries
    ## pair id and keywords
    query_id = {}
    id_query = {}
    for k in range(len(queried_kws)):
        query_id[queried_kws[k]] = k
        id_query[k] = queried_kws[k]
    kws_id = {}
    id_kws = {}
    for k in range(len(known_kws)):
        kws_id[known_kws[k]] = k
        id_kws[k] = known_kws[k]

    ## prepare information for attack
    real_query_d = np.zeros((len(queried_kws),len(real_doc)))
    sim_kw_d = np.zeros((attacker_kws_universe_size,len(sim_doc)))
    for i in range(len(real_doc)):
        for k in real_doc[i]:
            if(k in query_id):
                real_query_d[query_id[k]][i] = 1

    com_cost_before = np.sum(np.sum(real_query_d,axis=1)*real_F)
    sto_cost_before = len(real_query_d[0])

    if countermeasure_params["alg"] == "padding_linear":
        real_query_d = padding_linear(real_query_d,countermeasure_params["n"])
    elif countermeasure_params["alg"] == "padding_linear_2":
        real_query_d = padding_linear_2(real_query_d,countermeasure_params["n"])
    elif countermeasure_params["alg"] == "padding_cluster":
        real_query_d = padding_cluster(real_query_d,countermeasure_params["knum_in_cluster"])
    elif countermeasure_params["alg"] == "obfuscation":
        real_query_d = obfuscate(real_query_d,countermeasure_params["p"],countermeasure_params["q"],countermeasure_params["m"])
    elif countermeasure_params["alg"] == "padding_seal":
        real_query_d = padding_seal(real_query_d,countermeasure_params["n"])

    com_cost_after = np.sum(np.sum(real_query_d,axis=1)*real_F)
    sto_cost_after = len(real_query_d[0])

    for i in range(len(sim_doc)):
        for k in sim_doc[i]:
            if(k in kws_id):
                sim_kw_d[kws_id[k]][i] = 1
    ## delete the information of never returned docs 
    if known_database_size != True:
        index=np.where(np.sum(real_query_d,axis=0)==0)[0]
        real_query_d = np.delete(real_query_d,index,axis=1)

    # get volume and total frequency for all kws
    sim_F = get_F(known_kws,kws_dict,(0,observe_weeks))
    real_F = real_F/np.sum(real_F)
    sim_V = np.sum(sim_kw_d,axis=1)
    sim_V = sim_V/np.sum(sim_V)
    real_V = np.sum(real_query_d,axis=1)
    real_V = real_V/np.sum(real_V)

    data_for_attacks = {
        "sim_kw_trend":sim_kw_trend,
        "real_td_trend":real_td_trend,
        "sim_F":sim_F,
        "real_F":real_F,
        "sim_V":sim_V,
        "real_V":real_V,
        "real_doc_num":len(real_doc),
        "sim_kw_d":sim_kw_d,
        "real_query_d":real_query_d
    }
    data_for_acc_cal = {
        "id_kws":id_kws,
        "id_query":id_query,
        "query_frequency":query_freq_all_dict,
        "communication overhead":com_cost_after/com_cost_before,
        "storage overhead":sto_cost_after/sto_cost_before
    }
    return data_for_attacks,data_for_acc_cal


def get_all_for_attacks_wiki(
    user_kws_universe_size,
    attacker_kws_universe_size, 
    kws_extraction, # randomly choosen from 3000 most popular kws or just choosen the most popular kws
    observed_query_number_per_week,
    observe_weeks,
    observe_offset,
    countermeasure_params,
    dataset,
    known_database_size = True,
    similar_data_p = None
):
    #print("Read wiki doc begin:",time.time())
    ## read documents, we treat half the documents as real documents and another half as simliar data
    if dataset == "wiki":
        #db_dir = "dataset/kws_dict_3000_sorted.pkl"
        #with open(db_dir,"rb") as f:
        #    kws_dict = pickle.load(f)
        if user_kws_universe_size>=attacker_kws_universe_size:
            max_size = user_kws_universe_size
        else:
            max_size = attacker_kws_universe_size
        if max_size <= 3000:
            kw_dir = "dataset/kws_dict_3000_sorted.pkl"
            db_dir = "dataset/kws_list_and_doc_kws_all_new_0.pkl"
        elif max_size <= 5000:
            kw_dir = "dataset/kws_dict_5000_sorted.pkl"
            db_dir = "dataset/kws_list_and_doc_kws_new_5000_0.pkl"
        else:
            assert False
        with open(kw_dir,"rb") as f:
            kws_dict = pickle.load(f)
        with open(db_dir,"rb") as f:
            kws_list,doc_kwsid = pickle.load(f)
    else:
        assert False
    #print("Point 2:",time.time())
    np.random.shuffle(doc_kwsid)
    if similar_data_p == None:
        sim_doc_kwsid = doc_kwsid[:30000]
    else:
        sim_doc_kwsid = doc_kwsid[:(int) (len(doc_kwsid)/2*similar_data_p)]
    real_doc_kwsid = doc_kwsid[(int) (len(doc_kwsid)/2):]
    del doc_kwsid

    ## Get kws universe, we suppose that the kws extraction algorithm of the user and tha attacker are the same.
    kws_count = []
    for k in range(len(kws_list)):
        kws_count.append([kws_list[k], kws_dict[kws_list[k]]["count"],k])
    if kws_extraction == "sorted":
        kws_count.sort(reverse=True,key = lambda x: x[1])
        user_kws_universe_with_id = kws_count[:user_kws_universe_size]
        user_kws_universe = [tmp[0] for tmp in user_kws_universe_with_id]
        known_kws_with_id = kws_count[:attacker_kws_universe_size]
        known_kws = [tmp[0] for tmp in known_kws_with_id]
    elif kws_extraction == "random":
        np.random.shuffle(kws_count)
        user_kws_universe_with_id = kws_count[:user_kws_universe_size]
        user_kws_universe = [tmp[0] for tmp in user_kws_universe_with_id]
        # the kws known to the attacker includes the user kws universe
        known_kws_with_id = kws_count[:attacker_kws_universe_size]
        known_kws = [tmp[0] for tmp in known_kws_with_id]

    ## sample queries from real frequency and generate observed frequency
    sim_kw_trend = []
    for i in range(observe_weeks):
        sim_kw_trend.append(get_F(user_kws_universe,kws_dict,(i,i+1)))
    sim_kw_trend = np.array(sim_kw_trend)
    sim_kw_trend = sim_kw_trend.T

    query_all = set()
    query_freq_all = []
    query_freq_all_dict = {}
    for i in range(observe_weeks):
        f = get_F(user_kws_universe,kws_dict,(observe_offset+i,observe_offset+i+1))
        query = np.random.choice(user_kws_universe,observed_query_number_per_week,replace=True,p=f)
        query_freq = {}
        for q in query:
            if q in query_freq:
                query_freq[q] = query_freq[q]+1
            else:
                query_freq[q] = 1
        for q in query:
            if q in query_freq_all_dict:
                query_freq_all_dict[q] = query_freq_all_dict[q]+1
            else:
                query_freq_all_dict[q] = 1
        query_all.update(set(query.tolist()))
        query_freq_all.append(query_freq)
    #print("Point 3:",time.time())
    distinct_query_number = len(query_all)
    queried_kws = list(query_all)
    real_td_trend = np.zeros((distinct_query_number,len(query_freq_all)))
    real_F = np.zeros((distinct_query_number))
    for i in range(len(query_freq_all)):
        for j in range(distinct_query_number):
            if queried_kws[j] in query_freq_all[i]:
                real_F[j] += query_freq_all[i][queried_kws[j]]
                real_td_trend[j][i] = query_freq_all[i][queried_kws[j]]
    real_td_trend = real_td_trend/observed_query_number_per_week

    ## pair shuffled id and queries
    ## pair id and keywords
    query_id = {}
    id_query = {}
    for k in range(len(queried_kws)):
        query_id[queried_kws[k]] = k
        id_query[k] = queried_kws[k]
    kws_id = {}
    id_kws = {}
    for k in range(len(known_kws)):
        kws_id[known_kws[k]] = k
        id_kws[k] = known_kws[k]
    #print("Point 4:",time.time())
    ## prepare information for attack
    
    # real_query_d = np.zeros((len(queried_kws),len(real_doc)))
    # sim_kw_d = np.zeros((attacker_kws_universe_size,len(sim_doc)))
    # for i in range(len(real_doc)):
    #     for k in real_doc[i]:
    #         if(k in query_id):
    #             real_query_d[query_id[k]][i] = 1
    real_query_d = np.zeros((len(queried_kws),len(real_doc_kwsid)))
    sim_kw_d = np.zeros((len(known_kws_with_id),len(sim_doc_kwsid)))
    #print("Point 4.1:",time.time())
    real_kwsid_doc = real_doc_kwsid.T.astype(float)
    sim_kwsid_doc = sim_doc_kwsid.T.astype(float)
    #print("Point 4.2:",time.time())
    for k in range(len(known_kws_with_id)):
        if(kws_list[k] in query_id):
            real_query_d[query_id[kws_list[k]]] = real_kwsid_doc[k]
    #print("Point 4.3:",time.time())
    for k in range(len(known_kws_with_id)):
        sim_kw_d[k] = sim_kwsid_doc[known_kws_with_id[k][2]]
     
    #print("Point 5:",time.time())
    com_cost_before = np.sum(np.sum(real_query_d,axis=1)*real_F)
    sto_cost_before = len(real_query_d[0])

    if countermeasure_params["alg"] == "padding_linear":
        real_query_d = padding_linear(real_query_d,countermeasure_params["n"])
    elif countermeasure_params["alg"] == "padding_linear_2":
        real_query_d = padding_linear_2(real_query_d,countermeasure_params["n"])
    elif countermeasure_params["alg"] == "padding_cluster":
        real_query_d = padding_cluster(real_query_d,countermeasure_params["knum_in_cluster"])
    elif countermeasure_params["alg"] == "obfuscation":
        real_query_d = obfuscate(real_query_d,countermeasure_params["p"],countermeasure_params["q"],countermeasure_params["m"])
    elif countermeasure_params["alg"] == "padding_seal":
        real_query_d = padding_seal(real_query_d,countermeasure_params["n"])

    com_cost_after = np.sum(np.sum(real_query_d,axis=1)*real_F)
    sto_cost_after = len(real_query_d[0])

    # for i in range(len(sim_doc)):
    #     for k in sim_doc[i]:
    #         if(k in kws_id):
    #             sim_kw_d[kws_id[k]][i] = 1
    ## delete the information of never returned docs 
    if known_database_size != True:
        index=np.where(np.sum(real_query_d,axis=0)==0)[0]
        real_query_d = np.delete(real_query_d,index,axis=1)
    #print("Point 6:",time.time())
    # get volume and total frequency for all kws
    sim_F = get_F(known_kws,kws_dict,(0,observe_weeks))
    real_F = real_F/np.sum(real_F)
    sim_V = np.sum(sim_kw_d,axis=1)
    sim_V = sim_V/np.sum(sim_V)
    real_V = np.sum(real_query_d,axis=1)
    real_V = real_V/np.sum(real_V)

    data_for_attacks = {
        "sim_kw_trend":sim_kw_trend,
        "real_td_trend":real_td_trend,
        "sim_F":sim_F,
        "real_F":real_F,
        "sim_V":sim_V,
        "real_V":real_V,
        "real_doc_num":len(real_doc_kwsid),
        "sim_kw_d":sim_kw_d,
        "real_query_d":real_query_d
    }
    data_for_acc_cal = {
        "id_kws":id_kws,
        "id_query":id_query,
        "query_frequency":query_freq_all_dict,
        "communication overhead":com_cost_after/com_cost_before,
        "storage overhead":sto_cost_after/sto_cost_before
    }
    #print("Point 7:",time.time())
    return data_for_attacks,data_for_acc_cal
