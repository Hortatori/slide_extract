import pandas as pd
import argparse
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tqdm as tqdm
import os
# --notice_path data/fr2_annotated_notices_27_06_03_07.csv --extract_path matrix/nrv_extracted_docs_030.csv 


def preprocess(args) :
    # for nrv minutes output
    if args.extract_path.split('_')[0] == 'matrix/nrv' :
        df = pd.read_csv(args.extract_path, dtype = {"channel": str, "start": str, "end": str , "start_id": int, "end_id": int, "text": str, "label": int})
        df ["start"] = pd.to_datetime(df["start"])
        df["end"] = pd.to_datetime(df["end"])
    # for mj lines output
    else:
        df = pd.read_csv(args.extract_path)
        df ["start"] = pd.to_datetime(df["start"], format="%d/%m/%Y %H:%M:%S")
        df["end"] = pd.to_datetime(df["end"], format="%d/%m/%Y %H:%M:%S")
    notice = pd.read_csv(args.notice_path)
    # filter docs with the notice channel
    channel = notice.at[0,"ch_code"]
    if channel == 'FR2' : channel = 'France2'
    df = df[df['channel'].apply(lambda x: bool(re.search(channel, x)))]

    notice = notice.dropna(subset=['Durée','Heure','Description'])
    # pas d heure de fin dans le fichier donc calcul
    # fin du segment = heure + duree
    notice['start'] = notice['date']+' '+notice['Heure']
    notice["start"] = pd.to_datetime(notice["start"], format='%d/%m/%Y %H:%M:%S')
    notice["Durée"] = pd.to_timedelta(notice["Durée"])
    notice["only_for_max_time_computation"] = notice["start"]+notice["Durée"]
    print(f"sur le total des jours, minimum heure { notice['start'].min()} :: maximum heure {notice['only_for_max_time_computation'].max()}")
    notice['datetime_str'] = notice['date'].astype(str) + ' ' + notice['Heure'].astype(str)
    notice["end"] = notice["start"]+notice["Durée"]
    notice = notice.drop(['only_for_max_time_computation', 'datetime_str'], axis=1)

    print(notice)
    return df, notice

def compute_overlap(row_pred, notice_corpus):

    # Calcul du recouvrement pour les chevauchements détectés
    total_overlap = pd.Timedelta(0) # pas vraiment utile, voir trompeur ? le temps de recouvrement sur l'ensemble des notices parcourues
    total_intersections = 0
    rld = {'row':[],'delta':[],'ratio_delta':[],'label':[]}
    for _, notice_c in notice_corpus.iterrows():
        latest_start = max(row_pred['start'], notice_c['start'])
        earliest_end = min(row_pred['end'], notice_c['end'])
        delta = (earliest_end - latest_start)

        earlier_start = min(row_pred['start'], notice_c['start'])
        lastest_end = max(row_pred['end'], notice_c['end'])
        largest = lastest_end - earlier_start

        #detection d'un chevauchement
        if delta.total_seconds() > 0:
            total_overlap += delta
            total_intersections += 1
            rld['row'].append(_)
            rld['delta'].append(delta)
            rld['ratio_delta'].append(delta.total_seconds()/ (row_pred['end'] - row_pred['start']).total_seconds())
            rld['label'].append(notice_c['label'])

    return total_overlap.total_seconds(), total_intersections, rld 

# doc extracted qui doit avoir une colonne label avec 1 si présent dans extraction et 0 sinon 
# crer cette colonne au moment de la generation des docs extracted : en conservant tout et en mettant 0 dans une colonne label ou 1 si >threshold
# modifs code pour avoir un label (fait sur nrv, pas sur mj)
    # maybe to do faire une loop sur chaque date (groupby) plutôt que tout le dataset où y aura des vides les mauvais jours.

def choose_df_to_overlaps(fixed_df, running_df) :
    overlap_duration = list()
    attributed_gold = list()
    i = 0
    for _, row in tqdm.tqdm(fixed_df.iterrows(), total=fixed_df.shape[0]):


        total_overlap, intersection_n, rld = compute_overlap(row, running_df)
        overlap_duration.append(total_overlap)
        # attribuer une notice à chaque ligne de pred : - si un seul chevauchement, attribution | si plus d'un chevauchement : label du plus gros chevauchement | si pas de chevauchement : pas de label
        if total_overlap > 0.0 :
            # rappel : ce sont des fenêtres glissantes, donc elles se chevauchent plusieurs fois sur les memes rangs de notices
            i += 1
        if len(rld['label']) > 1 :
            # probleme : si égalité il choisit le premier (pas grave, ratio est à 10 decimales près) 
            max_index = rld['ratio_delta'].index(max(rld['ratio_delta']))
            selected_label = int(rld['label'][max_index])
        elif len(rld['label']) == 0 :
            selected_label = None
        else :
            selected_label = int(rld['label'][0])

        attributed_gold.append(selected_label)
        # print(f"intersections = {intersection_n} :: {rld} :: in pred row {_}")
    fixed_df['overlap_duration'] = overlap_duration # not a ratio, duration of total notice overlap for each doc row of one minute windows
    fixed_df['attributed_gold'] = attributed_gold
    print("nb of rows with overlaps",i)
    print(fixed_df)
    # fixed_df.to_csv("test_eval.csv")
    return fixed_df

def evaluation(df) :
    print(f"nb of docs of with NaN: {df.shape} ")
    df = df.dropna(subset=['attributed_gold'])
    print(f"nb of docs of  without NaN: {df.shape} ")
    df = df.astype({'attributed_gold' : 'int'})
    print(df.dtypes)
    acc = accuracy_score(df['attributed_gold'], df['label'])
    f1 = f1_score(df['attributed_gold'], df['label'])
    p = precision_score(df['attributed_gold'], df['label'])
    r = recall_score(df['attributed_gold'], df['label'])
    print(df['attributed_gold'].value_counts())
    print(df['label'].value_counts())
    return {'acc': acc, 'f1': f1, 'p': p, 'r': r}


def main(args) :
    df, notice = preprocess(args)
    output_labeled = choose_df_to_overlaps(fixed_df = df, running_df = notice)
    scores = evaluation(output_labeled)
    scores["gold_file"] = args.notice_path
    scores["pred_file"] = args.extract_path
    print(scores)
    output = pd.DataFrame([scores])
    output.to_csv(os.path.join("eval_outputs","_".join([args.extract_path.split("/")[1].split(".")[0],"vs",args.notice_path.split("/")[1].split(".")[0],".csv"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--notice_path', required=True, help="path and file of notice")
    parser.add_argument('--extract_path', required=True, help="path and file of extracted docs")
    args = parser.parse_args()

    main(args)
