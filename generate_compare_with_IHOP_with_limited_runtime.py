import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from cal_acc import show_results,calculate_acc_weighted
def draw_comparison(dataset,test_times,qua,Kws_uni_size = [50000]):
    #mpl.rcParams.update(mpl.rcParamsDefault)
    params = {
            'figure.figsize': '6, 3',  # set figure size
            "text.usetex": True ,
            "font.family": "stix",
            "font.serif": ["Times"],
            "font.size":22,
        }
    plt.rcParams.update(params)
    plt.style.context(['science', 'no-latex'])
    ax = plt.axes()
    ax2 = ax.twinx()

    ###########get our results and plot########
    Our_results_acc = [[] for j in Kws_uni_size]
    Our_results_time = [[] for j in Kws_uni_size]
    
    for test_time in range(test_times):
        Result = []
        for kws_uni_size in Kws_uni_size:
            with open("results/test_comparison_with_IHOP_with_limited_time/Ours/"+dataset+"_"+str(test_time)+"_"+str(kws_uni_size)+".pkl", "rb") as f:
                result = pickle.load(f)
            Result.append(result)
      
        for i in range(len(Kws_uni_size)):
            result = Result.pop(0)
            data_for_acc_cal = result[0][0]["data_for_acc_cal"]
            tdid_2_kwid = result[0][0]["results"][2]
            correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid)
            if qua == 4:
                Our_results_acc[i].append(acc)
            else:
                real_F = result["real_F"]
                real_V = result["real_V"]
                results = show_results(correct_id,wrong_id,real_F,\
                    real_V,high_frequency_ratio=0.1,high_volume_ratio=0.1,is_print=False)
                Our_results_acc[i].append(results[qua]["correct_count"]/results[qua]["qcount"])

            Our_results_time[i].append(result[0][0]["Attack_time"])
    
    Our_results_time = np.array(Our_results_time)
    
    Our_results_time = np.average(Our_results_time, axis=1)
    
    Positions = np.array([0,1,2])
    our_lineprops = dict(linewidth=1.5,color='darkgreen')
    
    Bp_Ours=ax.boxplot(Our_results_acc,positions=Positions-0.2,\
        widths=0.13,patch_artist=True,boxprops=dict(facecolor="mediumseagreen",edgecolor= "darkgreen"),whiskerprops=our_lineprops,
            medianprops=our_lineprops,capprops=our_lineprops,showfliers=False)
    print("Accurracy of ours:",np.average(np.array(Our_results_acc),axis=1))
    for i in range(len(Our_results_time)):
        ours_s=ax2.scatter(y=Our_results_time,x=Positions-0.2,marker="o",color="mediumseagreen",s=60)
        
###########get ihop results and plot########
    IHOP_results_acc = [[] for j in Kws_uni_size]
    IHOP_results_time = [[] for j in Kws_uni_size]
    
    for test_time in range(test_times):
        Result = []
        for kws_uni_size in Kws_uni_size:
            with open("results/test_comparison_with_IHOP_with_limited_time/IHOP/"+dataset+"_"+str(test_time)+"_"+str(kws_uni_size)+".pkl", "rb") as f:
                result = pickle.load(f)
            Result.append(result)
      
        for i in range(len(Kws_uni_size)):
            result = Result.pop(0)
            data_for_acc_cal = result[0][0]["data_for_acc_cal"]
            tdid_2_kwid = result[0][0]["results"]
            correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid)
            if qua == 4:
                IHOP_results_acc[i].append(acc)
            else:
                real_F = result["real_F"]
                real_V = result["real_V"]
                results = show_results(correct_id,wrong_id,real_F,\
                    real_V,high_frequency_ratio=0.1,high_volume_ratio=0.1,is_print=False)
                IHOP_results_acc[i].append(results[qua]["correct_count"]/results[qua]["qcount"])

            IHOP_results_time[i].append(result[0][0]["Attack_time"])
    
    IHOP_results_time = np.array(IHOP_results_time)
    
    IHOP_results_time = np.average(IHOP_results_time, axis=1)
    
    Positions = np.array([0,1,2])
    ihop_lineprops = dict(linewidth=1.5,color='darkorange')
    
    Bp_IHOP=ax.boxplot(IHOP_results_acc,positions=Positions+0.2,\
        widths=0.13,patch_artist=True,boxprops=dict(facecolor="wheat",edgecolor= "darkorange"),whiskerprops=ihop_lineprops,
            medianprops=ihop_lineprops,capprops=ihop_lineprops,showfliers=False)
    print("Accurracy of ours:",np.average(np.array(IHOP_results_acc),axis=1))
    for i in range(len(IHOP_results_time)):
        ihop_s=ax2.scatter(y=IHOP_results_time,x=Positions+0.2,marker="p",color="wheat",s=60)

    
    plt.xticks([0,1,2],["$|W|=5000$","$|W|=10000$","$|W|=15000$"])
    ax.tick_params(labelsize=18)
    ax2.tick_params(labelsize=18)
    ax2.set_ylabel("Time (seconds)",size = 22)
    ax.set_ylabel("Accuracy",size = 22)
    
    ax.set_ylim(0.6,1.05)
    ax2.set_ylim(0,15000)
    title = ["LVLF","LVHF","HVLF","HVHF","All"]
    
    
    plt.legend([Bp_Ours["boxes"][0], Bp_IHOP["boxes"][0],ours_s,ihop_s],\
        ["Jigsaw","IHOP","Jigsaw Time","IHOP Time"],\
        loc="center left", bbox_to_anchor=(-0.01, 0.35),\
        fancybox=True, shadow=True,prop={"size":14})
    plt.savefig("results/test_comparison_with_IHOP_with_limited_time/comparison_"+dataset+"_"+title[qua]+".pdf",bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    
    draw_comparison("wiki",10,4,Kws_uni_size=[5000,10000,15000])
    
