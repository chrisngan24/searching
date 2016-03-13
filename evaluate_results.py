import pandas as pd

if __name__ == '__main__':
    df_truth = pd.read_csv(
            'data/LA-only.trec8-401.450.minus416-423-437-444-447.txt', 
            sep=' ', 
            header=None,
            )
