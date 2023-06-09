import numpy as np
from tqdm import tqdm
import os
import stat
import subprocess
class GraphMattacker:
    def __init__(self,sim_kw_d,real_td_d,alpha = 0.5,alg="PATH",init="unif"):
        self.sim_kw_d = sim_kw_d
        self.real_td_d = real_td_d
        self.td_num = len(self.real_td_d)
        #self.real_td_d=np.vstack((self.real_td_d,np.zeros((len(self.sim_kw_d)-len(self.real_td_d),len(self.real_td_d[0])))))
        
        self.real_M = np.dot(self.real_td_d,self.real_td_d.T)/len(self.real_td_d[0])
        self.sim_M = np.dot(self.sim_kw_d,self.sim_kw_d.T)/len(self.sim_kw_d[0])

        self.tdid_2_kwsid = {}
        self.unrec_td_set = set([i for i in range(len(self.real_td_d))])
        self.id_known_kws=None
        self.id_queried_kws=None

        self.alpha = alpha
        self.alg = alg
        self.init = init

        

    def attack(self):
        G = self.sim_M
        np.fill_diagonal(G,0)
        H = self.real_M
        np.fill_diagonal(H,0)
        C = self.built_C()
        if not os.path.exists("./src_graphm/temp/graphm"):
            os.makedirs("./src_graphm/temp/graphm")
        with open("src_graphm/temp/graphm/G1.txt","wb") as f:
            for row in G:
                row_str = ' '.join("{:.6f}".format(val) for val in row) + '\n'
                f.write(row_str.encode('ascii'))
        with open("src_graphm/temp/graphm/G2.txt","wb") as f:
            for row in H:
                row_str = ' '.join("{:.6f}".format(val) for val in row) + '\n'
                f.write(row_str.encode('ascii'))
        with open("src_graphm/temp/graphm/C.txt","wb") as f:
            for row in C:
                row_str = ' '.join("{:.6f}".format(val) for val in row) + '\n'
                f.write(row_str.encode('ascii'))
        write_config([self.alg],self.init,self.alpha)
        with open("src_graphm/temp/graphm/run_script", 'w') as f:
            f.write("#!/bin/sh\n")
            f.write("src_graphm/graphm-0.51/bin/graphm src_graphm/temp/graphm/config.txt\n")
        st = os.stat("src_graphm/temp/graphm/run_script")
        os.chmod("src_graphm/temp/graphm/run_script", st.st_mode | stat.S_IEXEC)
        subprocess.run(["src_graphm/temp/graphm/run_script"], capture_output=True)
        
        with open("src_graphm/temp/graphm/X.txt") as f:
            
            while(f.readline()!=self.alg+" \n"):
                continue
            result = f.readlines()
            result = [int(i) for i in result]
            for i in range(len(result)):
                if result[i]-1<self.td_num:
                    self.tdid_2_kwsid[result[i]-1] = i

        os.remove("src_graphm/temp/graphm/X.txt")
        return self.tdid_2_kwsid
    def built_C(self):
        k_count = np.sum(self.sim_kw_d,axis=1)
        doc_num = len(self.sim_kw_d[0])
        k_freq = k_count/doc_num #g_i
        q_count = np.sum(self.real_td_d,axis=1) #k_i
        C = np.exp(compute_log_binomial_probability_matrix(len(self.real_td_d[0]),k_freq,q_count))
        return C

def _log_binomial(n, a):
    #  Computes an approximation of log(binom(n, n*a)) for a < 1
    if a == 0 or a == 1:
        return 0
    elif a < 0 or a > 1:
        raise ValueError("a cannot be negative or greater than 1 ({})".format(a))
    else:
        entropy = -a * np.log(a) - (1 - a) * np.log(1 - a)
        return n * entropy - 0.5 * np.log(2 * np.pi * n * a * (1 - a))
def compute_log_binomial_probability_matrix(ntrials, probabilities, observations):
    """
    This code is from https://github.com/simon-oya/USENIX21-sap-code/blob/master/attacks.py
    Computes the logarithm of binomial probabilities of each pair of probabilities and observations.
    :param ntrials: number of binomial trials
    :param probabilities: vector with probabilities
    :param observations: vector with integers (observations)
    :return log_matrix: |probabilities|x|observations| matrix with the log binomial probabilities
    """
    probabilities = np.array(probabilities)
    probabilities[probabilities == 0] = min(probabilities[probabilities > 0]) / 100  # To avoid numerical errors. An error would mean the adversary information is very off.
    log_binom_term = np.array([_log_binomial(ntrials, obs / ntrials) for obs in observations])  # ROW TERM
    column_term = np.array([np.log(probabilities) - np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    last_term = np.array([ntrials * np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    log_matrix = log_binom_term + np.array(observations) * column_term + last_term
    return log_matrix

def write_config(Alg,init,alpha):

    config_text = """//*********************GRAPHS**********************************
//graph_1,graph_2 are graph adjacency matrices,
//C_matrix is the matrix of local similarities  between vertices of graph_1 and graph_2.
//If graph_1 is NxN and graph_2 is MxM then C_matrix should be NxM
graph_1=src_graphm/temp/graphm/G1.txt s
graph_2=src_graphm/temp/graphm/G2.txt s
C_matrix=src_graphm/temp/graphm/C.txt s
//*******************ALGORITHMS********************************
//used algorithms and what should be used as initial solution in corresponding algorithms
algo={alg:s} s
algo_init_sol={init:s} s
solution_file=solution_im.txt s
//coeficient of linear combination between (1-alpha_ldh)*||graph_1-P*graph_2*P^T||^2_F +alpha_ldh*C_matrix
alpha_ldh={alpha:.6f} d
cdesc_matrix=A c
cscore_matrix=A c
//**************PARAMETERS SECTION*****************************
hungarian_max=10000 d
algo_fw_xeps=0.01 d
algo_fw_feps=0.01 d
//0 - just add a set of isolated nodes to the smallest graph, 1 - double size
dummy_nodes=0 i
// fill for dummy nodes (0.5 - these nodes will be connected with all other by edges of weight 0.5(min_weight+max_weight))
dummy_nodes_fill=0 d
// fill for linear matrix C, usually that's the minimum (dummy_nodes_c_coef=0),
// but may be the maximum (dummy_nodes_c_coef=1)
dummy_nodes_c_coef=0.01 d

qcvqcc_lambda_M=10 d
qcvqcc_lambda_min=1e-5 d


//0 - all matching are possible, 1-only matching with positive local similarity are possible
blast_match=0 i
blast_match_proj=0 i


//****************OUTPUT***************************************
//output file and its format
exp_out_file=src_graphm/temp/graphm/X.txt s
exp_out_format=Permutation s
//other
debugprint=0 i
debugprint_file=debug.txt s
verbose_mode=1 i
//verbose file may be a file or just a screen:cout
verbose_file=cout s
""".format(alg=" ".join(alg for alg in Alg), init=" ".join("unif" for _ in Alg),alpha=alpha)
    with open("src_graphm/temp/graphm/config.txt","w") as f:
        f.write(config_text)

        
