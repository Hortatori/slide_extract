import pandas as pd
import argparse
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tqdm as tqdm
import os
from pathlib import Path
# --gold_path gold_annotated_files --extract_path articles/labelled_minutes_from_run_encode_040.csv
# or
# --gold_path gold_annotated_files/ --extract_path data/label_keywords_nahel_transcriptions_vocapia_27_06_2023_to_03_07_2023.csv 

def preprocess(gold_path) :
    gold_df = pd.read_csv(gold_path)
    # if gold is from INA notices
    if "notice" in str(gold_path) :
        # filter docs with the notice channel
        if gold_df.at[0,"ch_code"] == "FR2" : 
            gold_df["channel"] = "France2"
        else :
            gold_df["channel"] = gold_df["ch_code"]
        gold_df = gold_df.dropna(subset=["Durée","Heure","Description"])
        # pas d heure de fin dans le fichier de notice donc calcul
        # fin du segment = heure + duree
        gold_df["start"] = gold_df["date"]+" "+gold_df["Heure"]
        gold_df["start"] = pd.to_datetime(gold_df["start"], format="%d/%m/%Y %H:%M:%S", dayfirst=True)
        gold_df["duration"] = pd.to_timedelta(gold_df["Durée"])
        gold_df["end"] = gold_df["start"]+gold_df["duration"]
        gold_df["text"] = gold_df["Description"]
        gold_df = gold_df.drop(["ch_code","datdifsec","date","Heure","Durée","ti","chap","de","Description"], axis=1)
    else :
        # regroup same labels in batches 
        #channel = gold_df.at[0,"channel"]
        # df = df[df['channel'].apply(lambda x: bool(re.search(channel, x)))]
        # labelled lines turned into larger segments for evaluation 
        gold_df = gold_df[["channel", "start", "end", "duration", "text", "label"]]
        gold_df['duration'] = pd.to_timedelta(gold_df['duration'], unit='s')
        gold_df["portion_id"] = (gold_df["label"].shift() != gold_df["label"]).cumsum()
        agg_gold_df = gold_df.groupby(["channel", "label", "portion_id"], as_index=False).agg(start=("start", "first"),end=("end", "last"),duration=("duration", "sum"),text=("text", " ".join))
        agg_gold_df = agg_gold_df.sort_values("portion_id")
        gold_df = agg_gold_df.drop(columns=["portion_id"])
        gold_df["end"] = pd.to_datetime(gold_df["end"], dayfirst=True)
        gold_df["start"] = pd.to_datetime(gold_df["start"], dayfirst=True)

    return gold_df


def compute_overlap(row_pred, notice_corpus):

    # Calcul du recouvrement pour les chevauchements détectés
    total_intersections = 0
    rld = {'row':[],'delta':[],'ratio_delta':[],'label':[]}
    for _, notice_c in notice_corpus.iterrows():
        latest_start = max(row_pred['start'], notice_c['start'])
        earliest_end = min(row_pred['end'], notice_c['end'])
        delta = (earliest_end - latest_start)

        earlier_start = min(row_pred['start'], notice_c['start'])
        lastest_end = max(row_pred['end'], notice_c['end'])
        # largest : initialement pour comparer le ratio obtenu avec la totalité des deux durée
        largest = lastest_end - earlier_start

        #detection d'un chevauchement
        if delta.total_seconds() > 0:
            total_intersections += 1
            rld['row'].append(_)
            rld['delta'].append(delta)
            rld['ratio_delta'].append(delta.total_seconds()/ (row_pred['end'] - row_pred['start']).total_seconds())
            rld['label'].append(notice_c['label'])

    return total_intersections, rld 


def choose_df_to_overlaps(fixed_df, running_df) :
    attributed_gold = list()
    # rappel : ce sont des fenêtres glissantes, donc elles se chevauchent plusieurs fois sur les memes rangs de notices
    history_n = 0
    for _, row in tqdm.tqdm(fixed_df.iterrows(), total=fixed_df.shape[0]):


        total_intersection, rld = compute_overlap(row, running_df)
        # attribuer une notice à chaque ligne de pred : - si un seul chevauchement, attribution | si plus d'un chevauchement : label du plus gros chevauchement | si pas de chevauchement : pas de label
        if len(rld['label']) > 1 :
            # plusieurs overlaps : récupère le label du pred row avec le plus long chevauchement. INFO : si égalité, max() prend le premier
            max_index = rld['ratio_delta'].index(max(rld['ratio_delta']))
            selected_label = int(rld['label'][max_index])
            history_n += 1
        elif len(rld['label']) == 0 :
            # no overlap in this row
            selected_label = None
        else :
            # un seul overlap : récupère le label du pred row
            selected_label = int(rld['label'][0])
            history_n += 1

        attributed_gold.append(selected_label)
        # print(f"intersections = {intersection_n} :: {rld} :: in pred row {_}")
    fixed_df["attributed_gold"] = attributed_gold
    fixed_df = fixed_df.dropna(subset=["attributed_gold"])
    print("nb of rows attributed", history_n)
    return fixed_df


def evaluation(df) :
    df = df.astype({'attributed_gold' : 'int'})
    df = df.astype({'label' : 'int'})
    acc = accuracy_score(df['attributed_gold'], df['label'])
    f1 = f1_score(df['attributed_gold'], df['label'])
    p = precision_score(df['attributed_gold'], df['label'])
    r = recall_score(df['attributed_gold'], df['label'])
    return {'acc': acc, 'f1': f1, 'p': p, 'r': r}


def main(args) :
    pd.options.mode.copy_on_write = True
    # for minutes output
    df = pd.read_csv(args.extract_path, dtype = {"channel": str, "start": str, "end": str , "start_id": int, "end_id": int, "text": str, "label": int})
    df ["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    all = [preprocess(Path(args.gold_path,f)) for f in os.listdir(args.gold_path)] 
    concat_gold=pd.concat(all, ignore_index=True)
    output_labelled = pd.DataFrame(columns=df.columns)
    for channel, df_gold_one_channel in concat_gold.groupby(["channel"]) :
        print(f"attrbuting gold label to minutes for channel {channel[0]}")
        df_channel = df[df["channel"] == channel[0]]
        this_channel_labeled = choose_df_to_overlaps(fixed_df = df_channel, running_df = df_gold_one_channel)
        output_labelled = pd.concat([output_labelled,this_channel_labeled], ignore_index=True)
    scores = evaluation(output_labelled)
    scores["gold_file"] = args.gold_path
    scores["pred_file"] = args.extract_path
    print(scores)
    output = pd.DataFrame([scores])
    output.to_csv(os.path.join("eval_outputs","_".join(["global_eval",args.extract_path.split("/")[1].split(".")[0],"VS","".join(args.gold_path.split("/")),".csv"])))

    # scores par chaines
    for channel, df_channel in output_labelled.groupby(["channel"]) :
        scores = evaluation(df_channel)
        print(f"for {channel[0]} :: {scores}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', required=True, help="path and file of notice")
    parser.add_argument('--extract_path', required=True, help="path and file of extracted docs")
    args = parser.parse_args()

    main(args)
