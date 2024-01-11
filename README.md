# Implementations of Jigsaw

This is the python implementations of the attacks presented in :

"Query Recovery from Easy to Hard: Jigsaw Attack against SSE"

Run the python files with name starting with ``test_*.py`` to simulate the attack under different situations.

Run the files starting with ``generate_*.py`` to draw the results.

## About the dataset

The folder ``dataset`` contains the datasets. 

For Enron , the ``enron_db.pkl`` contains a document list and a keyword dict. The document list contains lists of keywords of each file. The keyword dict maps each keyword to its total counts in files and query trend. The same structure is used for Lucene.

For Wikipedia, ``wiki_kws_dict.pkl`` maps each keyword to its total counts in files and query trend. ``wiki_doc_0.pkl`` contains lists of keywords of each file. These two files can be download in "[https://drive.google.com/file/d/1ltB3oyiDV0Ef7v0dRBlXV_wwvBUwas05/view?usp=sharing](https://drive.google.com/file/d/1ltB3oyiDV0Ef7v0dRBlXV_wwvBUwas05/view?usp=drive_link)" and "[https://drive.google.com/file/d/18_RG5GiH65IAI64HE5Em7yC7hbIkfBSG/view?usp=drive_link](https://drive.google.com/file/d/18_RG5GiH65IAI64HE5Em7yC7hbIkfBSG/view?usp=drive_link)". Before using these files, we preprocess them to make the later attacks faster. Run the ``preprocess.py`` to change the keywords universe size in the Wikipedia dataset and preprocess the data. 

## About each file

The codes of Jigsaw are in the folder ``attack``. It also contains the code of all attacks with experiments in our paper. The file ``attack.py`` contains the implementation of Jigsaw and RSA.  

``run_single_attack.py``: takes the parameters of an attack and simulate the attack under the given condition.

``test_alpha.py``: tests the first module of Jigsaw with different $\alpha$. ``generate_test_alpha.py``: generates the pictures for the results of ``test_alpha.py``.

``test_base_conf_rec.py``: tests Jigsaw with different $BaseRec$ and $ConfRec$.

``test_beta.py``: tests Jigsaw with different $\beta$. ``generate_test_beta.py``: generates the pictures for the results of ``test_beta.py``.

``test_durability.py``: tests Jigsaw with different $\tau$, where $\tau$ is the time offset between the observed frequency and the prior knowledge.

``show_distribution.py``: runs the simple attack and draws the distribution of qieries.

``test_comparison.py``: compares the Jigsaw, RSA, IHOP, SAP, and Graphm under different conditions. ``generate_test_comparison_pics`` generates the pictures of above comparisons.

``test_RSA_with_nkws.py``: test RSA with different numbers of known queries.

``test_against_countermeasure.py``: test Jigsaw, RSA, and IHOP against countermeasures. ``generate_test_against_countermeasures.py``: generates the pictures of comparisons when against countermeasures.

``test_IHOP_with_different_alpha.py``: test IHOP with an alpha parameter to balance the frequency and volume.

``test_compare_with_IHOP_with_limited_runtime.py``: compares the Jigsaw and IHOP under the same runtime limits.


``extract_info.py``: prepares the necessary information for attacks.

``cal_acc.py``: calculate the accuracy from the attack results.

``countermeasure.py``: implementations of countermeasures.

