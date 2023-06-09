import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
class Sapattacker:
    def __init__(self,sim_kw_trend,real_td_trend,td_num_per_week,sim_v,real_v,real_doc_num,alpha = 0.5,countermeasure=None):
        self.sim_kw_trend = sim_kw_trend
        self.real_td_trend = real_td_trend
        
        self.td_num_per_week = td_num_per_week
        
        self.sim_v = sim_v
        self.real_v = real_v

        self.real_doc_num = real_doc_num
        self.alpha = alpha
        self.countermeasure = countermeasure
        self.tdid_2_kwsid = {}
    def attack(self):
        Cf = self.builtCf()
        Cv = self.builtCv()
        C_matrix = self.alpha*Cf+(1-self.alpha)*Cv
        
        row_ind, col_ind = hungarian(C_matrix)
        
        for td, kw in zip(col_ind, row_ind):
            
            self.tdid_2_kwsid[td] = kw

        return self.tdid_2_kwsid

    def builtCf(self):
        log_c_matrix = np.zeros((len(self.sim_kw_trend), len(self.real_td_trend)))
        for i in range(len(self.sim_kw_trend[0])):
            
            probabilities = self.sim_kw_trend[:, i].copy()
            probabilities[probabilities == 0] = min(probabilities[probabilities > 0]) / 100
            td_num_per_week = np.array([self.td_num_per_week]*len(self.real_td_trend))
            temp = (td_num_per_week * self.real_td_trend[:, i]) * np.log(np.array([probabilities]).T)

            log_c_matrix =log_c_matrix + temp
        return -log_c_matrix
    def builtCv(self):
        if self.countermeasure==None:
            log_prob_matrix = compute_log_binomial_probability_matrix(self.real_doc_num, self.sim_v/np.sum(self.sim_v), self.real_v)

        cost_vol = - log_prob_matrix
        return cost_vol

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
def _log_binomial(n, a):
    #  Computes an approximation of log(binom(n, n*a)) for a < 1
    if a == 0 or a == 1:
        return 0
    elif a < 0 or a > 1:
        raise ValueError("a cannot be negative or greater than 1 ({})".format(a))
    else:
        entropy = -a * np.log(a) - (1 - a) * np.log(1 - a)
        return n * entropy - 0.5 * np.log(2 * np.pi * n * a * (1 - a))