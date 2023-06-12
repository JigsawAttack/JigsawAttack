# Implementation of Jigsaw

The attack folder contains the code for Jigsaw.

Run the python files with name starting with "test_" to simulate the attack under different situations.

Run the files starting with "generate_" to draw the results.

## About each file

The codes of Jigsaw are in the fold "attack". It also contains the code of all attacks with experiments in our paper. The file "attack.py" contains the implementation of Jigsaw and RSA.  

In "run_single_attack.py",  takes the parameters of an attack and simulate the attack under the given condition.

"test_alpha.py": tests the first module of Jigsaw with different $\alpha$. "generate_test_alpha.py": generates the pictures for the results of "test_alpha.py".

"test_base_conf_rec.py": tests Jigsaw with different $BaseRec$ and $ConfRec$.

"test_beta.py": tests Jigsaw with different $\beta$. "generate_test_beta.py": generates the pictures for the results of "test_beta.py".

"test_durability.py": tests Jigsaw with different $\tau$, where $\tau$ is the time offset between the observed frequency and the prior knowledge.

"show_distribution.py": runs the simple attack and draws the distribution of qieries.

"test_comparison.py": compares the Jigsaw, RSA, IHOP, SAP, and Graphm under different conditions. "generate_test_comparison_pics" generates the pictures of above comparisons.

"test_RSA_with_nkws.py": test RSA with different numbers of known queries.

"test_against_countermeasure.py": test Jigsaw, RSA, and IHOP against countermeasures. "generate_test_against_countermeasures.py": generates the pictures of comparisons when against countermeasures.


