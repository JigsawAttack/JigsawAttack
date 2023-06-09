import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian

def compute_log_binomial_probability_matrix(ntrials, probabilities, observations):
    """
    Computes the logarithm of binomial probabilities of each pair of probabilities and observations.
    :param ntrials: number of binomial trials
    :param probabilities: vector with probabilities
    :param observations: vector with integers (observations)
    :return log_matrix: |probabilities| x |observations| matrix with the log binomial probabilities
    """
    probabilities = np.array(probabilities)
    if any(probabilities > 0):
        probabilities[probabilities == 0] = min(probabilities[probabilities > 0]) / 100  # To avoid numerical errors. An error would mean the adversary information is very off.
    else:
        probabilities += 1e-10
    column_term = np.array([np.log(probabilities) - np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    last_term = np.array([ntrials * np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    log_matrix = np.array(observations) * column_term + last_term
    return log_matrix


def ihopattack(ndocs,nqr,ntok,nkw,Vexp,Vobs,fexp,fobs,attack_params): 
    # Vobs is the co-occurrence matrix # fobs is the normalized frequency vector
    def _build_cost_Vol_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags):
        cost_vol = -compute_log_binomial_probability_matrix(ndocs, np.diagonal(Vexp)[free_keywords], np.diagonal(Vobs)[free_tags] * ndocs)
        for tag, kw in zip(fixed_tags, fixed_keywords):
            cost_vol -= compute_log_binomial_probability_matrix(ndocs, Vexp[kw, free_keywords], Vobs[tag, free_tags] * ndocs)
        return cost_vol

    def _build_cost_freq_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags):
        cost_freq = - (nqr * fobs[free_tags]) * np.log(np.array([fexp[free_keywords]]).T)
        return cost_freq
    
    def compute_coef_matrix(free_keywords, free_tags, fixed_keywords, fixed_tags):
        if attack_params["no_F"] == True:
            return _build_cost_Vol_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags)
        return _build_cost_Vol_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags)+\
            _build_cost_freq_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags)

    epsilon = 1e-20
    fexp = (fexp + epsilon / nkw) / (1 + epsilon * 2 / nkw)

    rep_to_kw = rep_to_kw = {rep: rep for rep in range(nkw)}
    pct_free = attack_params['pfree']
    n_iters = attack_params['niters']
    nrep = len(rep_to_kw)

    # 1) PROCESS GROUND-TRUTH INFORMATION
    
    ground_truth_tokens, ground_truth_reps = [], []

    unknown_toks = [i for i in range(ntok) if i not in ground_truth_tokens]
    unknown_reps = [i for i in range(nrep) if i not in ground_truth_reps]

    # First matching:
    c_matrix_original = compute_coef_matrix(unknown_reps, unknown_toks, ground_truth_reps, ground_truth_tokens)
    row_ind, col_ind = hungarian(c_matrix_original)
    replica_predictions_for_each_token = {token: rep for token, rep in zip(ground_truth_tokens, ground_truth_reps)}
    for j, i in zip(col_ind, row_ind):
        replica_predictions_for_each_token[unknown_toks[j]] = unknown_reps[i]

    # if 'niter_list' in attack_params:
    #     run_multiple_niters, niter_list, rep_pred_tok_list = True, att_params['niter_list'], []
    #     if 0 in niter_list:
    #         rep_pred_tok_list.append(replica_predictions_for_each_token.copy())
    # else:
    run_multiple_niters, niter_list, rep_pred_tok_list = False, [], []

    # Iterate using co-occurrence:
    n_free = int(pct_free * len(unknown_toks))
    assert n_free > 1
    for k in range(n_iters):
        random_unknown_tokens = list(np.random.permutation(unknown_toks))
        free_tokens = random_unknown_tokens[:n_free]
        fixed_tokens = random_unknown_tokens[n_free:] + ground_truth_tokens
        fixed_reps = [replica_predictions_for_each_token[token] for token in fixed_tokens]
        free_replicas = [rep for rep in unknown_reps if rep not in fixed_reps]

        c_matrix = compute_coef_matrix(free_replicas, free_tokens, fixed_reps, fixed_tokens)

        row_ind, col_ind = hungarian(c_matrix)
        for j, i in zip(col_ind, row_ind):
            replica_predictions_for_each_token[free_tokens[j]] = free_replicas[i]

        if run_multiple_niters and k + 1 in niter_list:
            rep_pred_tok_list.append(replica_predictions_for_each_token.copy())

        # if (k + 1) % (n_iters // 10) == 0:
        #     print("{:d}".format(((k + 1) // (n_iters // 10)) - 1), end='', flush=True)
    return replica_predictions_for_each_token
    
