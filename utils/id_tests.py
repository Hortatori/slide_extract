import pandas as pd

def compare_extracted_ID(df_nrv, mj_df):
    set_nrv = set(df_nrv.iloc[:,0])
    set_mj = set(mj_df.iloc[:,0])
    commun = set_nrv & set_mj
    in_nrv_only = set_nrv - set_mj
    in_mj_only = set_mj - set_nrv
    print(f"en commun {len(commun)}, dans nrv seulement {len(in_nrv_only)}, dans mj seulement {len(in_mj_only)}")
    print(f"debut commun : {list(commun)[:20]}, début slmt nrv {list(in_nrv_only)[:20]}, début mj seulement {list(in_mj_only)[:20]}" )

n = pd.read_csv("matrix/extracted_ids_050.csv")
m = pd.read_csv("extracted_docs/IDs_0.42_JT_nahel_transcriptions_vocapia_26_06_2023_to_03_07_2023_Lajavaness_sentence-camembert-large.csv")
compare_extracted_ID(n,m)