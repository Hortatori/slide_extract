import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--channel', required=True, help="la chaîne concernée", choices=["TF1", "France2"])
args = parser.parse_args()

df = pd.read_csv("matrix/nrv_extracted_docs_035.csv")
df = df[df['channel'] == args.channel]
df ["start"] = pd.to_datetime(df['start'], dayfirst=True)
df['end'] = pd.to_datetime(df['end'], dayfirst=True)

if args.channel == 'France2':
    notice = pd.read_csv("data/fr2_notices_27_06_03_07.csv")
else :
    notice = pd.read_csv("data/tf1_notices_27_06_03_07.csv")

notice["start"] = notice["start"].dropna()
notice["datdifsec"] = notice["datdifsec"].dropna()
notice["start"] = pd.to_datetime(notice["start"], dayfirst=True)
notice["end"] = pd.to_datetime(notice["datdifsec"], dayfirst=True)

def compute_overlap(row_notice, df_corpus):


    # Calcul du recouvrement pour les chevauchements détectés
    total_overlap = pd.Timedelta(0)
    for _, row_c in df_corpus.iterrows():
        latest_start = max(row_notice['start'], row_c['start'])
        earliest_end = min(row_notice['end'], row_c['end'])
        delta = (earliest_end - latest_start)
        if delta.total_seconds() > 0:
            total_overlap += delta
    return total_overlap.total_seconds() #/ (row_notice['end'] - row_notice['start']).total_seconds()

def choose_df_to_overlaps(fixed_df, running_df) :
    overlap_ratio = list()
    i = 0
    for _, row in fixed_df.iterrows():
        overlap = compute_overlap(row, running_df)
        overlap_ratio.append(overlap)
        if overlap > 0.0 :
            i += 1
    fixed_df['overlap_ratio'] = overlap_ratio
    print(i)
    print(overlap_ratio)

choose_df_to_overlaps(fixed_df = notice, running_df = df)
"""
headers pertinents dans les notices :
    #datedif
    #heuremin
    #dureemin
    #datdifsec
    #start
    #date
    #heurefin
    #duree
    #Durée
"""