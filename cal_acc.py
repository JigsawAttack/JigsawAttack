def calculate_acc(id_known_kws,id_queried_kws,tdid_2_kwsid,is_print = True):
    if len(tdid_2_kwsid) == 0:
        print("No real pair to calculate accuracy")
        assert 0
    
    correct_count = 0
    for i in list(tdid_2_kwsid.keys()):
        if id_queried_kws[i]==id_known_kws[tdid_2_kwsid[i]]:
            correct_count+=1
    if is_print==True:
        print("Recover Number:      ",len(tdid_2_kwsid.keys()))
        print("Correctly Recover Number:    ",correct_count)
        print("Accuracy:        ",correct_count/len(tdid_2_kwsid.keys()))
    return correct_count,correct_count/len(tdid_2_kwsid.keys())

def calculate_acc_weighted(data_for_acc_cal,tdid_2_kwsid,is_print = False):
    if len(tdid_2_kwsid) == 0:
        print("No real pair to calculate accuracy")
        assert 0
    id_known_kws = data_for_acc_cal["id_kws"]
    id_queried_kws = data_for_acc_cal["id_query"]
    query_freq = data_for_acc_cal["query_frequency"]
    correct_count = 0
    total_count = 0
    correct_id = []
    wrong_id = []
    for i in list(tdid_2_kwsid.keys()):
        if id_queried_kws[i]==id_known_kws[tdid_2_kwsid[i]]:
            correct_count+=query_freq[id_queried_kws[i]]
            correct_id.append(i)
        else:
            wrong_id.append(i)
        total_count+=query_freq[id_queried_kws[i]]
    
    if is_print==True:
        print("Recover Number:      ",total_count)
        print("Correctly Recover Number:    ",correct_count)
        print("Accuracy:        ",correct_count/total_count)
    return correct_count,correct_count/total_count,correct_id,wrong_id

def show_results(correct_id,wrong_id,real_F,real_V,high_frequency_ratio,high_volume_ratio,is_print= True):
    ## show the results in four areas
    freq_line = sorted(real_F,reverse=True)[int(len(real_F)* high_frequency_ratio)]
    vol_line = sorted(real_V,reverse=True)[int(len(real_V)* high_volume_ratio)]
    hv_hf = {"count":0,"correct_count":0,"wrong_count":0,"qcount":0}
    hv_lf = {"count":0,"correct_count":0,"wrong_count":0,"qcount":0}
    lv_hf = {"count":0,"correct_count":0,"wrong_count":0,"qcount":0}
    lv_lf = {"count":0,"correct_count":0,"wrong_count":0,"qcount":0}
    for i in range(len(real_V)):
        if real_V[i]>=vol_line:
            if real_F[i]>=freq_line:
                hv_hf["count"] = hv_hf["count"]+1
                hv_hf["qcount"] = hv_hf["qcount"]+real_F[i]
                if i in correct_id:
                    hv_hf["correct_count"] = hv_hf["correct_count"]+real_F[i]
                if i in wrong_id:
                    hv_hf["wrong_count"] = hv_hf["wrong_count"]+real_F[i]
            else:
                hv_lf["count"] = hv_lf["count"]+1
                hv_lf["qcount"] = hv_lf["qcount"]+real_F[i]
                if i in correct_id:
                    hv_lf["correct_count"] = hv_lf["correct_count"]+real_F[i]
                if i in wrong_id:
                    hv_lf["wrong_count"] = hv_lf["wrong_count"]+real_F[i]
        else:
            if real_F[i]>=freq_line:
                lv_hf["count"] = lv_hf["count"]+1
                lv_hf["qcount"] = lv_hf["qcount"]+real_F[i]
                if i in correct_id:
                    lv_hf["correct_count"] = lv_hf["correct_count"]+real_F[i]
                if i in wrong_id:
                    lv_hf["wrong_count"] = lv_hf["wrong_count"]+real_F[i]
            else:
                lv_lf["count"] = lv_lf["count"]+1
                lv_lf["qcount"] = lv_lf["qcount"]+real_F[i]
                if i in correct_id:
                    lv_lf["correct_count"] = lv_lf["correct_count"]+real_F[i]
                if i in wrong_id:
                    lv_lf["wrong_count"] = lv_lf["wrong_count"]+real_F[i]
    if is_print==True:
        print("------------------------------------------------")
        print("Low Vol High Freq                      |High Vol High Freq")
        print("The ratio of queries in this area:%.2f"%lv_hf["qcount"],"|The ratio of queries in this area:%.2f"%hv_hf["qcount"])
        print("Accuracy:%.2f                         "%(lv_hf["correct_count"]/lv_hf["qcount"]),"|Accuarcy:%.2f"%(hv_hf["correct_count"]/hv_hf["qcount"]))
        print("------------------------------------------------")
        print("Low Vol low Freq                       |High Vol Low Freq")
        print("The ratio of queries in this area:%.2f"%lv_lf["qcount"],"|The ratio of queries in this area:%.2f"%hv_lf["qcount"])
        print("Accuracy:%.2f                         "%(lv_lf["correct_count"]/lv_lf["qcount"]),"|Accuarcy:%.2f"%(hv_lf["correct_count"]/hv_lf["qcount"]))
    return lv_lf,lv_hf,hv_lf,hv_hf