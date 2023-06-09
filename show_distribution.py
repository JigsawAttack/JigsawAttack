from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from run_single_attack import *
from cal_acc import *
from tqdm import tqdm
def show_distribution(correct_id,real_F,real_V,high_frequency_ratio,high_volume_ratio,dataset,save=False):
    freq_line = sorted(real_F,reverse=True)[int(len(real_F)* high_frequency_ratio)]
    vol_line = sorted(real_V,reverse=True)[int(len(real_V)* high_volume_ratio)]
    plt.rcParams.update({
    
    "text.usetex": True,
    "font.family": "stix",
    "font.serif": ["Times"],
    "font.size":40,
    })
    plt.style.context(['science', 'no-latex'])
    fig, ax = plt.subplots()
    ax.axhline(y=freq_line,linestyle="--")
    ax.axvline(x=vol_line,linestyle="--",)

    # ax.set_xlabel("Normalized Volume")
    # ax.set_ylabel("Normalized Frequency")
    
    ax.scatter(real_V,real_F,marker=".",linewidths=2,label="Real Queries")
    ax.scatter(real_V[correct_id],real_F[correct_id],marker=".",color="red",label="Correctly Recovered")
    #ax.legend(loc=4)
    ax.set_xlim(-0.0001,np.max(real_V)+0.001)
    ax.set_ylim(-0.001,np.max(real_F)+0.01)
    axins = inset_axes(ax, width="70%", height="70%", loc=1)
    axins.set_xlim(0,0.005)
    axins.set_ylim(0,0.01)
    axins.scatter(real_V,real_F,marker=".",linewidths=2,label="Real Queries")
    axins.scatter(real_V[correct_id],real_F[correct_id],marker=".",color="red",label="Correctly Recovered")
    axins.axhline(y=freq_line,linestyle="--")
    axins.axvline(x=vol_line,linestyle="--",)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='k', lw=1)
    axins.tick_params(labelleft=False, labelbottom=False)
    plt.tick_params(labelsize=10)

    ax.set_xticks([])
    ax.set_yticks([])

    if save == False:
        plt.show()
    else:
        plt.savefig("results/Distributiion_"+dataset+"_2.pdf",bbox_inches='tight')
    plt.clf()

def show_distribution_3(correct_id,real_F,real_V,high_frequency_ratio,high_volume_ratio,dataset,save=False):
    freq_line = sorted(real_F,reverse=True)[int(len(real_F)* high_frequency_ratio)]
    vol_line = sorted(real_V,reverse=True)[int(len(real_V)* high_volume_ratio)]
    
    fig, ax = plt.subplots()
    ax.axhline(y=freq_line,linestyle="--")
    ax.axvline(x=vol_line,linestyle="--",)
    
    ax.set_xlabel("Normalized Volume")
    ax.set_ylabel("Normalized Frequency")
    
    ax.scatter(real_V,real_F,marker=".",linewidths=2,label="Real Queries")
    ax.scatter(real_V[correct_id],real_F[correct_id],marker=".",color="red",label="Correctly Recovered")
    #ax.legend(bbox_to_anchor=(0.99,0.20))
    ax.set_xlim(-0.0001,np.max(real_V)+0.001)
    ax.set_ylim(-0.001,np.max(real_F)+0.01)
    # axins = inset_axes(ax, width="60%", height="60%", loc=1)
    # axins.set_xlim(0,0.005)
    # axins.set_ylim(0,0.01)
    # axins.scatter(real_V,real_F,marker=".",linewidths=2,label="Real data")
    # axins.scatter(real_V[correct_id],real_F[correct_id],marker=".",color="red",label="Correctly Recovered")
    # axins.axhline(y=freq_line,linestyle="--")
    # axins.axvline(x=vol_line,linestyle="--",)
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='k', lw=1)
    # axins.tick_params(labelleft=False, labelbottom=False)
    if save == False:
        plt.show()
    else:
        plt.savefig("results/Distributiion_"+dataset+".pdf",bbox_inches='tight')
    plt.clf()

def sigmoid_log(x):
    return 1.0/(1+np.exp(-np.log10(x)))
def show_distribution_2(id_query,real_F,real_V,high_frequency_ratio,high_volume_ratio,show_type,save=False):
    sorted_F = sorted(real_F)
    F_dict = {}
    for i in real_F:
        for j in range(len(sorted_F)):
            if i == sorted_F[j]:
                if i in F_dict:
                    break
                else:
                    F_dict[i] = j
                    break
    F_list = []
    for i in real_F:
        F_list.append(F_dict[i])
    #print(sorted(F_list))
    sorted_V = sorted(real_V)
    V_dict = {}
    for i in real_V:
        for j in range(len(sorted_V)):
            if i == sorted_V[j]:
                if i in V_dict:
                    break
                else:
                    V_dict[i] = j
                    break
    V_list = []
    for i in real_V:
        V_list.append(V_dict[i])
    #print(sorted(V_list))
    real_F = np.array(F_list)
    real_V = np.array(V_list)#sigmoid_log(real_V/np.max(real_V))

    freq_line = sorted(real_F,reverse=True)[int(len(real_F)* high_frequency_ratio)]
    vol_line = sorted(real_V,reverse=True)[int(len(real_V)* high_volume_ratio)]
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "stix",
    "font.serif": ["Times"],
    "font.size":0,
    })
    plt.style.context(['science', 'no-latex'])
    fig, ax = plt.subplots()
    ax.axhline(y=freq_line,linestyle="--",c="black")
    ax.axvline(x=vol_line,linestyle="--",c="black")

    # ax.set_xlabel("Normalized Volume")
    # ax.set_ylabel("Normalized Frequency")
    
    ax.scatter(real_V,real_F,marker=".",linewidths=2,label="Real Queries")
    ax.scatter(real_V[correct_id],real_F[correct_id],marker=".",color="red",label="Correctly Recovered")
    #ax.legend(loc=4)#bbox_to_anchor=(0.99,0.20))
    ax.set_xlim(-0.0001,np.max(real_V)+0.001)
    ax.set_ylim(-0.001,np.max(real_F)+0.01)

    Labels = [0.1,0.3,0.5,0.7,0.9]
    #Labels_ = ["$100$","$300$","$500$","$700$","$900$"]
    Freq_label = [sorted(real_F)[int(len(real_F)* Labels[i])] for i in range(len(Labels))]
    Vol_label = [sorted(real_V)[int(len(real_F)* Labels[i])] for i in range(len(Labels))]

    # if show_type == "dot":
    #     plt.scatter(real_V,real_F,marker=".",linewidths=2,label="Real Queries")
    # elif show_type=="text":
    #     for i in range(len(real_F)):
    #         plt.text(real_V[i],real_F[i],id_query[i])
    # plt.xlabel("Volume Rank")
    # plt.ylabel("Frequency Rank")
    # # plt.xlim(0.1,0.55)
    # # plt.ylim(0,0.52)
    # plt.xticks(Vol_label,Labels_,)
    # plt.yticks(Freq_label,Labels_,)
    #plt.tick_params(labelsize=15) 
    plt.xticks([])
    plt.yticks([])

    if save == False:
        plt.show()
    else:
        plt.savefig("results/Distributiion_"+dataset+".pdf",bbox_inches='tight')
    plt.clf()
   
    

dataset = "wiki"
attack_params = {"alg":"Ours","alpha":0.5,"step":1,"baseRec":1000,\
                    "beta":None,"no_F":None,"confRec":None,"refinespeed":None}
Acc = [[],[],[],[]]
for i in tqdm(range(1)):
    result = run_single_attack(1000,1000,"sorted",3000,60,0,dataset,\
        {"alg":None},attack_params)

    real_F = result["real_F"]
    real_V = result["real_V"]

    high_frequency_ratio = 0.1
    high_volume_ratio = 0.1
    data_for_acc_cal = result["data_for_acc_cal"]
    tdid_2_kwid = result["results"][0]
    correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid,is_print=True)
    Results = show_results(correct_id,wrong_id,real_F,real_V,high_frequency_ratio,high_volume_ratio,is_print=True)
    for i in range(4):
        Acc[i].append(Results[i]["correct_count"]/Results[i]["qcount"])
print(np.average(np.array(Acc), axis=1))
# show_distribution_2(result["data_for_acc_cal"]["id_query"],\
#     real_F,real_V,high_frequency_ratio,high_volume_ratio,\
#     show_type="text",save=True)
# show_distribution_2(result["data_for_acc_cal"]["id_query"],\
#     real_F,real_V,high_frequency_ratio,high_volume_ratio,\
#     show_type="dot",save=True)

show_distribution(correct_id,real_F,real_V,high_frequency_ratio,high_volume_ratio,dataset,save=True)
show_distribution_2(correct_id,real_F,real_V,high_frequency_ratio,high_volume_ratio,dataset,save=True)
