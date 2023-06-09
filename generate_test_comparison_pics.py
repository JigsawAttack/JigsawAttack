import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from cal_acc import show_results,calculate_acc_weighted
def draw_comparison(dataset,test_times,qua,Query_number_per_week = [100,500,2500],Kws_uni_size = [500,1000,2000]):
    #mpl.rcParams.update(mpl.rcParamsDefault)
    params = {
            'figure.figsize': '12, 6.5',  # set figure size
            "text.usetex": True ,
            "font.family": "stix",
            "font.serif": ["Times"],
            "font.size":22,
        }
    plt.rcParams.update(params)
    plt.style.context(['science', 'no-latex'])
    rtc1 = [0.05,0.28,0.9,0.65]
    rtc2 = [0.05,0.08,0.9,0.2]
    ax = plt.axes(rtc1)
    ax2 = plt.axes(rtc2)
    #fig, (ax,ax2) = plt.subplots(2,1)
    
    #ax2 = ax.twinx()

    

    ###########get our results and plot########
    Our_results_acc = [[[] for i in Query_number_per_week] for j in Kws_uni_size]
    Our_results_time = [[[] for i in Query_number_per_week] for j in Kws_uni_size]
    for test_time in range(test_times):
        with open("results/test_comparison/Ours/"+dataset+"_"+str(test_time)+".pkl", "rb") as f:
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
                else:
                    real_F = result["real_F"]
                    real_V = result["real_V"]
                    results = show_results(correct_id,wrong_id,real_F,\
                        real_V,high_frequency_ratio=0.1,high_volume_ratio=0.1,is_print=False)
                    Our_results_acc[i][j].append(results[qua]["correct_count"]/results[qua]["qcount"])

                Our_results_time[i][j].append(result["Attack_time"])
    
    Our_results_time = np.array(Our_results_time)
    
    Our_results_time = np.average(Our_results_time, axis=2)
    
    Positions = np.array([[i*3.3+j for j in range(len(Query_number_per_week))] for i in range(len(Kws_uni_size))])
    our_lineprops = dict(linewidth=1.5,color='darkgreen')
    for i in range(len(Our_results_acc)):
        Bp_Ours=ax.boxplot(Our_results_acc[i],positions=Positions[i]-0.4,\
            widths=0.13,patch_artist=True,boxprops=dict(facecolor="mediumseagreen",edgecolor= "darkgreen"),whiskerprops=our_lineprops,
                medianprops=our_lineprops,capprops=our_lineprops,showfliers=False)#,capprops=lineprops,\
    print("Accurracy of ours:",np.average(np.array(Our_results_acc),axis=2))
    for i in range(len(Our_results_time)):
        ax2.scatter(y=Our_results_time[i],x=Positions[i]-0.4,marker="o",color="mediumseagreen",s=60)

###########get RSA results and plot########

    RSA_results_acc = [[[] for i in Query_number_per_week] for j in Kws_uni_size]
    RSA_results_time = [[[] for i in Query_number_per_week] for j in Kws_uni_size]
    for test_time in range(test_times):
        with open("results/test_comparison/RSA/"+dataset+"_"+str(test_time)+".pkl", "rb") as f:
            Result = pickle.load(f)
        for j in range(len(Query_number_per_week)):
            results=Result.pop(0)
            
            for i in range(len(Kws_uni_size)):
                result = results.pop(0)
                data_for_acc_cal = result["data_for_acc_cal"]
                for key in result["results"][1].keys():
                    del result["results"][0][key]
                tdid_2_kwid = result["results"][0]
                correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid)
                if qua == 4:
                    RSA_results_acc[i][j].append(acc)
                else:
                    real_F = result["real_F"]
                    real_V = result["real_V"]
                    results = show_results(correct_id,wrong_id,real_F,\
                        real_V,high_frequency_ratio=0.1,high_volume_ratio=0.1,is_print=False)
                    RSA_results_acc[i][j].append(results[qua]["correct_count"]/results[qua]["qcount"])
                RSA_results_time[i][j].append(result["Attack_time"])
    RSA_results_time = np.array(RSA_results_time)
    RSA_results_time = np.average(RSA_results_time, axis=2)
    #print("here",RSA_results_time)
    #Positions = np.array([[i*4+j for j in range(len(Query_number_per_week))] for i in range(len(Kws_uni_size))])
    rsa_lineprops = dict(linewidth=1.5,color='darkblue')
    for i in range(len(RSA_results_acc)):
        Bp_RSA=ax.boxplot(RSA_results_acc[i],positions=Positions[i]-0.2,\
            widths=0.13,patch_artist=True,boxprops=dict(facecolor="cornflowerblue",edgecolor= "darkblue"),
            whiskerprops=rsa_lineprops,medianprops=rsa_lineprops,capprops=rsa_lineprops,showfliers=False)
        #boxprops=lineprops,capprops=lineprops,whiskerprops=lineprops,medianprops=lineprops)
    print("Accurracy of RSA:",np.average(np.array(RSA_results_acc),axis=2)) 
    for i in range(len(RSA_results_time)):
        ax2.scatter(y=RSA_results_time[i],x=Positions[i]-0.2,marker="s",color="cornflowerblue",s=60)
        
 ###########get ihop results and plot########
    IHOP_results_acc = [[[] for i in Query_number_per_week] for j in Kws_uni_size]
    IHOP_results_time = [[[] for i in Query_number_per_week] for j in Kws_uni_size]
    for test_time in range(7):
        with open("results/test_comparison/IHOP/"+dataset+"_"+str(test_time)+".pkl", "rb") as f:
            Result = pickle.load(f)
        for j in range(len(Query_number_per_week)):
            results=Result.pop(0)
        
            for i in range(len(Kws_uni_size)):
                result = results.pop(0)
                data_for_acc_cal = result["data_for_acc_cal"]
                tdid_2_kwid = result["results"]
                
                correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid)
                if qua == 4:
                    IHOP_results_acc[i][j].append(acc)
                else:
                    real_F = result["real_F"]
                    real_V = result["real_V"]
                    results = show_results(correct_id,wrong_id,real_F,\
                        real_V,high_frequency_ratio=0.1,high_volume_ratio=0.1,is_print=False)
                    IHOP_results_acc[i][j].append(results[qua]["correct_count"]/results[qua]["qcount"])

                IHOP_results_time[i][j].append(result["Attack_time"])
    IHOP_results_time = np.array(IHOP_results_time)
    IHOP_results_time = np.average(IHOP_results_time, axis=2)
    
    #Positions = np.array([[i*4+j for j in range(len(Query_number_per_week))] for i in range(len(Kws_uni_size))])
    ihop_lineprops = dict(linewidth=1.5,color='darkorange')
    for i in range(len(IHOP_results_acc)):
        Bp_IHOP=ax.boxplot(IHOP_results_acc[i],positions=Positions[i],\
            widths=0.13,patch_artist=True,boxprops=dict(facecolor="wheat",edgecolor= "darkorange"),whiskerprops=ihop_lineprops,
            medianprops=ihop_lineprops,capprops=ihop_lineprops,showfliers=False)#,capprops=lineprops,\
    print("Accurracy of IHOP:",np.average(np.array(IHOP_results_acc),axis=2)) 
    for i in range(len(IHOP_results_time)):
        ax2.scatter(y=IHOP_results_time[i],x=Positions[i],marker="p",color="wheat",s=60)


###########get sap results and plot########
    
    Sap_results_acc = [[[] for i in Query_number_per_week] for j in Kws_uni_size]
    Sap_results_time = [[[] for i in Query_number_per_week] for j in Kws_uni_size]
    for test_time in range(test_times):
        with open("results/test_comparison/SAP/"+dataset+"_"+str(test_time)+".pkl", "rb") as f:
            Result = pickle.load(f)
            
        for j in range(len(Query_number_per_week)):
            results=Result.pop(0)
            
            for i in range(len(Kws_uni_size)):
                result = results.pop(0)
                data_for_acc_cal = result["data_for_acc_cal"]
                tdid_2_kwid = result["results"]
                correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid)
                if qua == 4:
                    Sap_results_acc[i][j].append(acc)
                else:
                    real_F = result["real_F"]
                    real_V = result["real_V"]
                    results = show_results(correct_id,wrong_id,real_F,\
                        real_V,high_frequency_ratio=0.1,high_volume_ratio=0.1,is_print=False)
                    Sap_results_acc[i][j].append(results[qua]["correct_count"]/results[qua]["qcount"])
                Sap_results_time[i][j].append(result["Attack_time"])
    Sap_results_time = np.array(Sap_results_time)
    Sap_results_time = np.average(Sap_results_time, axis=2)
    
    #Positions = np.array([[i*4+j for j in range(len(Query_number_per_week))] for i in range(len(Kws_uni_size))])
    sap_lineprops = dict(linewidth=1.5,color='purple')
    for i in range(len(Sap_results_acc)):
        Bp_Sap=ax.boxplot(Sap_results_acc[i],positions=Positions[i]+0.2,\
            widths=0.13,patch_artist=True,boxprops=dict(facecolor="orchid",edgecolor= "purple"),whiskerprops=sap_lineprops,medianprops=sap_lineprops,capprops=sap_lineprops,showfliers=False)#boxprops=lineprops,capprops=lineprops,whiskerprops=lineprops,medianprops=lineprops)
    print("Accurracy of SAP:",np.average(np.array(Sap_results_acc),axis=2)) 
    for i in range(len(Sap_results_time)):
        ax2.scatter(y=Sap_results_time[i],x=Positions[i]+0.2,marker="^",color="orchid",s=60)
    
 
    

 ###########get Graphm results and plot########
    if dataset != "wiki":
        Kws_uni_size = [500]
        Graphm_results_acc = [[[] for i in Query_number_per_week] for j in Kws_uni_size]
        Graphm_results_time = [[[] for i in Query_number_per_week] for j in Kws_uni_size]
        for test_time in range(test_times):
            with open("results/test_comparison/Graphm/"+dataset+"_"+str(test_time)+".pkl", "rb") as f:
                Result = pickle.load(f)
            for j in range(len(Query_number_per_week)):
                results=Result.pop(0)
                for i in range(len(Kws_uni_size)):
                    result = results.pop(0)
                    data_for_acc_cal = result["data_for_acc_cal"]
                    
                    tdid_2_kwid = result["results"]
                    correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid)
                    if qua == 4:
                        Graphm_results_acc[i][j].append(acc)
                    else:
                        real_F = result["real_F"]
                        real_V = result["real_V"]
                        results = show_results(correct_id,wrong_id,real_F,\
                            real_V,high_frequency_ratio=0.1,high_volume_ratio=0.1,is_print=False)
                        Graphm_results_acc[i][j].append(results[qua]["correct_count"]/results[qua]["qcount"])
                    Graphm_results_time[i][j].append(result["Attack_time"])
        Graphm_results_time = np.array(Graphm_results_time)
        Graphm_results_time = np.average(Graphm_results_time, axis=2)
        
        Positions_G = np.array([[i*3.3+j for j in range(len(Query_number_per_week))] for i in range(len(Kws_uni_size))])
        graphm_lineprops = dict(linewidth=1.5,color='darkred')
        for i in range(len(Graphm_results_acc)):
            Bp_Graphm=ax.boxplot(Graphm_results_acc[i],positions=Positions_G[i]+0.4,\
                widths=0.1,patch_artist=True,boxprops=dict(facecolor="red",edgecolor= "darkred"),whiskerprops=graphm_lineprops,medianprops=graphm_lineprops,capprops=graphm_lineprops,showfliers=False)
        
        for i in range(len(Graphm_results_time)):
            ax2.scatter(y=Graphm_results_time[i],x=Positions_G[i]+0.4,marker="x",color="red",s=60)
        
    
    
    for pos in [0.5,1.5,3.8,4.8,7.1,8.1]:
        ax.plot([pos, pos], [0, 1], 'k-', alpha=0.2)
    for pos in [2.7,6]:
        ax.plot([pos, pos], [0, 1], c="black")
    
    
    # plt.xticks([0,1,2,4,5,6,8,9,10],["$\\eta=1000$","$\\eta=5000$\n$|W|=1000$","$\\eta=10000$",
    # "$\\eta=1000$","$\\eta=5000$\n$|W|=3000$","$\\eta=10000$",
    # "$\\eta=1000$","$\\eta=5000$\n$|W|=5000$","$\\eta=10000$"])


    plt.xticks(Positions.reshape((9,)),["$100$","$500$\n$\\eta$\n$|W|=500$","$2500$",
    "$100$","$500$\n$\\eta$\n$|W|=1000$","$2500$",
    "$100$","$500$\n$\\eta$\n$|W|=2000$","$2500$"])
    ax.set_ylabel("Accuracy",size = 20) #
    if dataset != "wiki":
        ax.legend([Bp_Ours["boxes"][0], Bp_RSA["boxes"][0],Bp_IHOP["boxes"][0], Bp_Sap["boxes"][0],Bp_Graphm["boxes"][0]],\
            ["Jigsaw","RSA","IHOP","Sap","Graphm"],\
            loc='upper center', bbox_to_anchor=(0.5, 1.17),ncol=5,\
            fancybox=True, shadow=True)
    else:
        plt.xticks(Positions.reshape((9,)),["$1000$","$5000$\n$\\eta$\n$|W|=1000$","$10000$",
        "$1000$","$5000$\n$\\eta$\n$|W|=3000$","$10000$",
        "$1000$","$5000$\n$\\eta$\n$|W|=5000$","$10000$"])
        ax.legend([Bp_Ours["boxes"][0], Bp_RSA["boxes"][0],Bp_IHOP["boxes"][0], Bp_Sap["boxes"][0]],\
            ["Jigsaw","RSA","IHOP","Sap"],\
            loc='upper center', bbox_to_anchor=(0.5, 1.17),ncol=5,\
            fancybox=True, shadow=True)
    ax.get_xaxis().set_visible(False)
    ax.set_xlim(-0.65,9.2)
    ax2.set_xlim(-0.65,9.2)
    ax.tick_params(labelsize=20)
    ax2.set_ylabel("Time (seconds)",size = 22)
    
    plt.tick_params(labelsize=22) 

    title = ["LVLF","LVHF","HVLF","HVHF","All"]
    #plt.title(title[qua])
    #ax2.tight_layout()
    plt.savefig("results/test_comparison/comparison_"+dataset+"_"+title[qua]+".pdf",bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_comparison("enron",30,4)
    draw_comparison("lucene",30,4)
    draw_comparison("wiki",10,4,Query_number_per_week=[1000,5000,10000],Kws_uni_size=[1000,3000,5000])
    
