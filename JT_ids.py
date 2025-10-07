import pandas as pd
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import argparse
import os
import tqdm
import shutil
import re

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "--dataset", type=str, required=True, help="dataset to pass in JT_ids.py"
)


def count_time(name, dataset):
    """
    Processes the dataset to compute and categorize news sessions based on start and end times.

    This function iterates over the input dataset to identify batches of news sessions (JT).
    It calculates the duration and the number of lines for each session, categorized by channel.
    It also records the start and end indices of each session within the dataset.

    Args:
        dataset (pd.DataFrame): A DataFrame containing news data with columns "start", "end", and "channel".

    Returns:
        tuple: A tuple containing:
            - duration_lines_by_channel (defaultdict): A dictionary mapping each channel to a list of the number of lines per session.
            - durations_by_channel (defaultdict): A dictionary mapping each channel to a list of timedelta objects representing session durations.
            - simple_datetime (defaultdict): A dictionary mapping each channel to a list of timedelta objects representing session durations.
            - idx_docs_pairs (list): A list of lists, each containing the start and end indices of a session.
    """
    # applied on extracted documents
    if name.split('_')[0] == 'output' or 'formatted' :
        dataset["start"] = pd.to_datetime(dataset["start"])
        dataset["end"] = pd.to_datetime(dataset["end"])
        dataset["start"] = dataset["start"].dt.strftime("%d/%m/%Y %H:%M:%S")

        dataset["end"] = dataset["end"].dt.strftime("%d/%m/%Y %H:%M:%S")
    #applied on original data
    else :
        dataset = pd.read_csv(name, quoting=csv.QUOTE_ALL)

    n = 0
    # will contain docs indexes of start and end of one JT
    idx_list = 0
    idx_docs_pairs = list()
    durations_by_channel = defaultdict(list)
    duration_lines_by_channel = defaultdict(list)
    simple_datetime = defaultdict(list)
    end_time = 0
    while n < dataset.shape[0]:
        idx_docs_pairs.append([n])
        batch_start = datetime.strptime(dataset.loc[n, "start"], "%d/%m/%Y %H:%M:%S")
        end_time = datetime.strptime(dataset.loc[n, "end"], "%d/%m/%Y %H:%M:%S")
        following_start = datetime.strptime(
            dataset["start"][n + 1], "%d/%m/%Y %H:%M:%S"
        )
        # tant que le JT suivant a un écart de temps avec le JT actuel
        # update previous_n, n, following_start, duration, all record lists
        while end_time != following_start:
            previous_n = n
            n += 1
            end_time = datetime.strptime(
                dataset.loc[previous_n, "end"], "%d/%m/%Y %H:%M:%S"
            )
            following_start = datetime.strptime(
                dataset.loc[n, "start"], "%d/%m/%Y %H:%M:%S"
            )
            simple_datetime[dataset["channel"][previous_n]].append(
                end_time - batch_start
            )
            durations_by_channel[dataset["channel"][previous_n]].append(
                end_time - batch_start
            )

            idx_docs_pairs[idx_list].append(previous_n)
            duration_lines_by_channel[dataset["channel"][previous_n]].append(
                (idx_docs_pairs[idx_list][1] - idx_docs_pairs[idx_list][0]) + 1
            )
            idx_docs_pairs.append([n])
            idx_list += 1
            batch_start = datetime.strptime(
                dataset.loc[n, "start"], "%d/%m/%Y %H:%M:%S"
            )
            end_time = datetime.strptime(dataset.loc[n, "end"], "%d/%m/%Y %H:%M:%S")
            following_start = datetime.strptime(
                dataset.loc[n + 1, "start"], "%d/%m/%Y %H:%M:%S"
            )

        # tant que le JT suivant suit le JT actuel (et que ce n'est pas la dernière boucle)
        # update end_time, previous_n, n, following_start
        while n < dataset.shape[0] - 2 and end_time == following_start:
            end_time = datetime.strptime(dataset.loc[n, "end"], "%d/%m/%Y %H:%M:%S")
            previous_n = n
            n += 1
            following_start = datetime.strptime(
                dataset.loc[n, "start"], "%d/%m/%Y %H:%M:%S"
            )
        # update les temps issus de la boucle end_time == following_start si ce n'est pas la dernière ligne
        if n != dataset.shape[0] - 2:
            simple_datetime[dataset["channel"][previous_n]].append(
                end_time - batch_start
            )
            durations_by_channel[dataset["channel"][previous_n]].append(
                end_time - batch_start
            )

            idx_docs_pairs[idx_list].append(previous_n)

            duration_lines_by_channel[dataset["channel"][previous_n]].append(
                (idx_docs_pairs[idx_list][1] - idx_docs_pairs[idx_list][0]) + 1
            )
            idx_list += 1
        # update et enregistre les temps de la boucle précédente avec la dernière ligne du document
        else:
            n += 1
            end_time = datetime.strptime(dataset.loc[n, "end"], "%d/%m/%Y %H:%M:%S")
            durations_by_channel[dataset["channel"][n]].append(end_time - batch_start)
            simple_datetime[dataset["channel"][n]].append(end_time - batch_start)

            idx_docs_pairs[idx_list].append(n)
            duration_lines_by_channel[dataset["channel"][n]].append(
                (idx_docs_pairs[idx_list][1] - idx_docs_pairs[idx_list][0]) + 1
            )
            return (
                duration_lines_by_channel,
                durations_by_channel,
                simple_datetime,
                idx_docs_pairs,
            )

        # print(f"end of a  JT batch, n : {previous_n}, start : {n}, end_time : {end}, following start : {following_start}")


def stats_and_dataframing(
    duration_lines_by_channel, durations_by_channel, simple_datetime
):
    """
    Calculer des statistiques pour chaque chaîne :
    - Nombre de JT (nb_JTs)
    - Nombre de lignes (nb_lines)
    - Temps total (total time)
    - Temps moyen (avg time)
    - Temps minimum (min_time)
    - Temps maximum (max_time)

    Retourne trois DataFrames :
    - df_line : nombre de lignes par JT
    - df_time : durée de chaque JT
    - stats : les statistiques par chaîne
    """
    stats = {
        "channel": [],
        "nb_JTs": [],
        "nb_lines": [],
        "total time": [],
        "avg time": [],
        "min_time": [],
        "max_time": [],
    }
    # Calculer des statistiques
    for key in duration_lines_by_channel.keys():
        stats["channel"].append(key)
        stats["nb_JTs"].append(len(duration_lines_by_channel[key]))
        stats["nb_lines"].append(sum(duration_lines_by_channel[key]))
        dt_sum = sum(simple_datetime[key], timedelta())
        stats["total time"].append(dt_sum)
        dt_avg = sum(simple_datetime[key], timedelta()) / len(simple_datetime[key])
        stats["avg time"].append(dt_avg)
        dt_min = min(simple_datetime[key])
        stats["min_time"].append(dt_min)
        dt_max = max(simple_datetime[key])
        stats["max_time"].append(dt_max)

    # Compléter les listes avec NaN pour pv les transformer en csv
    max_len = max(len(lst) for lst in duration_lines_by_channel.values())
    for channel in duration_lines_by_channel.keys():
        duration_lines_by_channel[channel] += [np.nan] * (
            max_len - len(duration_lines_by_channel[channel])
        )
        durations_by_channel[channel] += [np.nan] * (
            max_len - len(durations_by_channel[channel])
        )

    stats = pd.DataFrame(stats)
    df_line = pd.DataFrame(duration_lines_by_channel)
    df_time = pd.DataFrame(durations_by_channel)
    return df_line, df_time, stats


def main(name):
    dataset = pd.read_csv(name, quoting=csv.QUOTE_ALL)

    print("indexing by JTs ...")
    duration_lines_by_channel, durations_by_channel, simple_datetime, idx_docs_pairs = (
        count_time(name, dataset)
    )
    print("count_time function ended, JTs indexing is ready")
    df_line, df_time, stats = stats_and_dataframing(
        duration_lines_by_channel, durations_by_channel, simple_datetime
    )
    print("DF LINES\n")
    print(df_line)
    print("DF TIME\n")
    print(df_time)
    print("STATS\n")
    print(stats)
    df_docs_pairs = pd.DataFrame(idx_docs_pairs)
    print(df_docs_pairs)

    # à utiliser dans slide.py : if not os.path.exists(name.split("/")[0] + "/" + name.split("/")[1] + "_reordered"): pour appeller JTtime si le reorder n'existe pas encore
    reorder_pairs = {"start_id" : [],"end_id" : [],"time":[]}
    # si utilisé pour un doc de texte extraits (code trop rigide pour tout changer) 
    if name.split('_')[0] == 'output' or 'formatted' :
        name = "data/"+name
    if not os.path.exists(os.path.join("data", name.split("/")[1] + "_reordered")):
        for row in tqdm.tqdm(df_docs_pairs.itertuples(), total=len(df_docs_pairs)) : 
            # print(row[1], row[2])
            reorder_pairs["start_id"].append(row[1])
            reorder_pairs["end_id"].append(row[2])
            reorder_pairs["time"].append(dataset.at[row[1],"start"])
    df_docs_pairs = pd.DataFrame(reorder_pairs)
    df_docs_pairs["time"] = pd.to_datetime(df_docs_pairs["time"], format = "%d/%m/%Y %H:%M:%S")
    df_docs_pairs = df_docs_pairs.sort_values(by='time')
    # reorder INA dataset by time
    reorder_dataset = pd.DataFrame(columns=dataset.columns)
    count = 0
    for row in tqdm.tqdm(df_docs_pairs.itertuples(), total=len(df_docs_pairs)) : 
        slice = dataset.loc[row[1]:row[2]]
        reorder_dataset = pd.concat([reorder_dataset, slice])
        if count % 10 == 0 or count == len(df_docs_pairs)-1:

            if os.path.exists("data/reordered/subsets/"):
                reorder_dataset.to_csv("data/reordered/subsets/" + re.split("\W+", name)[1] + "_" + str(count) + ".csv", index=False)
                reorder_dataset = pd.DataFrame(columns=dataset.columns)
            else :    
                os.makedirs("data/reordered/subsets/")
                reorder_dataset.to_csv("data/reordered/subsets/" + re.split("\W+", name)[1] + "_" + str(count) + ".csv", index=False)
                reorder_dataset = pd.DataFrame(columns=dataset.columns)
        count += 1

    # Concat all the csv files produced by the script in the subset directory
    fichiers_csv = [f for f in os.listdir(name.split("/")[0] + "/reordered/subsets/") if f.endswith(".csv")]
    fichiers_csv.sort(key=lambda x: int(re.split('\W+|_', x)[-2])) 

    df_list = [pd.read_csv(os.path.join(name.split("/")[0] + "/reordered/subsets/", fichier)) for fichier in fichiers_csv]
    df_final = pd.concat([df for df in df_list if len(df.index) > 0], ignore_index=True)
    df_final.to_csv(
        name.split("/")[0] + "/reordered/" 
        + name.split("/")[1],
        index=False,
    )
    shutil.rmtree(name.split("/")[0] + "/reordered/subsets/")

    df_docs_pairs.to_csv(name.split("/")[0] + "/idx_docs_pairs_" + name.split("/")[1], index=False)
    df_line.to_csv(name.split("/")[0] + "/line_" + name.split("/")[1], index=False)
    df_time.to_csv(name.split("/")[0] + "/time_" + name.split("/")[1], index=False)
    stats.to_csv(name.split("/")[0] + "/stats_" + name.split("/")[1], index=False)


args = parser.parse_args()
main(
    name = args.dataset
)
