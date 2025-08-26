import pandas as pd

def compare_extracted_ID(df_nrv, mj_df):
    set_nrv = set(df_nrv.iloc[:,0])
    set_mj = set(mj_df.iloc[:,0])
    commun = set_nrv & set_mj
    in_nrv_only = set_nrv - set_mj
    in_mj_only = set_mj - set_nrv
    print(f"en commun {len(commun)}, dans nrv seulement {len(in_nrv_only)}, dans mj seulement {len(in_mj_only)}")
    data = pd.read_csv("data/nahel_transcriptions_vocapia_26_06_2023_to_03_07_2023.csv")
    nrlis = list(in_nrv_only)
    nrlis.sort()
    print(data.iloc[nrlis[:10],:])

    mjlis = list(in_mj_only)
    mjlis.sort()

    print(data.iloc[mjlis[:10],:])
    a = data.iloc[mjlis,:]
    # for _, i in a.iterrows() :
    #     print(i["start"])
    #     print(i["channel"])
    #     print(i["text"])
    comlis= list(commun)
    comlis.sort()
    print(data.iloc[comlis[:10],:])

n = pd.read_csv("matrix/extracted_ids_050.csv")
m = pd.read_csv("extracted_docs/IDs_0.42_JT_nahel_transcriptions_vocapia_26_06_2023_to_03_07_2023_Lajavaness_sentence-camembert-large.csv")
compare_extracted_ID(n,m)