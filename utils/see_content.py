import pandas as pd
import csv

def see_content(idx_pairs_path, data_path):
    idx_data = pd.read_csv(idx_pairs_path, quoting=csv.QUOTE_ALL)
    print(idx_data)

    data_text = pd.read_csv(data_path, quoting=csv.QUOTE_ALL)
    print(data_text)
    N = idx_data.shape[0]
    print(N)
    nb_error = 0
    num_error = []
    for i in range(N-1):
        # print(i)
        # print(idx_data.loc[i])
        if idx_data.loc[i,"1"]-idx_data.loc[i,"0"] < 2 :
            print(idx_data.loc[i])
            print(data_text.loc[idx_data.loc[i,"0"]-1:idx_data.loc[i,"1"]+1])
            print("\n")
            nb_error += 1
            num_error.append([idx_data.loc[i,"0"], idx_data.loc[i,"1"]])

    print(f"nb error : {nb_error}")
    print(num_error)
see_content("../data/idx_docs_pairs_dix_mille_short.csv", "../data/dix_mille_short.csv")
