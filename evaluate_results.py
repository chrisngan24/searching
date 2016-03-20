import pandas as pd
from evaluate.metrics import compute_precision_at_k,\
                    compute_dcg

import numpy as np
import sys

if __name__ == '__main__':
    # use of pandas to load in csv files
    df_truth = pd.read_csv(
            'data/LA-only.trec8-401.450.minus416-423-437-444-447.txt', 
            sep=' ', 
            header=None,
            )
    dat_file = sys.argv[1]
    # the data files (note that its not the submission file)
    df_exp = pd.read_csv(dat_file)
    # since columns are not named
    df_truth.columns = ['topicID', 'q', 'docno', 'relevance']
    df_truth.drop(['q'], axis=1,inplace=True)
    df_j =pd.merge(df_exp, df_truth, on=['topicID', 'docno'], how='left').fillna(0)

    k = 10

    metrics = []

    for topicID, df_g in df_j.groupby('topicID'):
        if topicID not in [416, 423, 437, 444, 447]:
            
            relevances = df_g.sort('rank', ascending=True)['relevance']
            ranks = np.array(xrange(1,k+1))
            ### all results matching the query
            true_relevancy = df_truth[
                    df_truth['topicID'] == topicID
                    ].sort('relevance', ascending=False)['relevance']
            
            
            precision = compute_precision_at_k(relevances, k)
            
            ideal_dcg = compute_dcg(true_relevancy, ranks,k)
            m_dcg = compute_dcg(relevances, ranks,k)
            ndcg = m_dcg/ideal_dcg
            metrics.append(dict(
                    topicID = topicID,
                    k = k,
                    precision = precision,
                    ndcg = ndcg,
                    queryTime = np.mean(df_g['run_time']),
                    ))
    df = pd.DataFrame(metrics)
    # can read this for results
    df.to_csv("data/summary.csv")
    print df.describe()

