import pandas as pd
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


def count_time(dataset):
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
            end_time = datetime.strptime(dataset.loc[previous_n, "end"], "%d/%m/%Y %H:%M:%S")
            following_start = datetime.strptime(
                dataset.loc[n, "start"], "%d/%m/%Y %H:%M:%S"
            )
            simple_datetime[dataset["channel"][previous_n]].append(end_time - batch_start)
            durations_by_channel[dataset["channel"][previous_n]].append(end_time - batch_start)

            idx_docs_pairs[idx_list].append(previous_n)
            duration_lines_by_channel[dataset["channel"][previous_n]].append(
                (idx_docs_pairs[idx_list][1] - idx_docs_pairs[idx_list][0]) + 1
            )
            idx_docs_pairs.append([n])
            idx_list += 1
            batch_start = datetime.strptime(dataset.loc[n, "start"], "%d/%m/%Y %H:%M:%S")
            end_time = datetime.strptime(dataset.loc[n, "end"], "%d/%m/%Y %H:%M:%S")
            following_start = datetime.strptime(
                dataset.loc[n+1, "start"], "%d/%m/%Y %H:%M:%S"
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
 
            simple_datetime[dataset["channel"][previous_n]].append(end_time - batch_start)
            durations_by_channel[dataset["channel"][previous_n]].append(end_time - batch_start)

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
    duration_lines_by_channel, durations_by_channel, simple_datetime, idx_docs_pairs = (
        count_time(dataset)
    )
    print("count_time ended")
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
    df_docs_pairs.to_csv(
        name.split("/")[0] + "/idx_docs_pairs_" + name.split("/")[1], index=False
    )
    df_line.to_csv(name.split("/")[0] + "/line_" + name.split("/")[1], index=False)
    df_time.to_csv(name.split("/")[0] + "/time_" + name.split("/")[1], index=False)
    stats.to_csv(name.split("/")[0] + "/stats_" + name.split("/")[1], index=False)


# main("data/formatted_medialex_transcriptions_vocapia_v1v2_20230301_20230731.csv")
# main("data/short_1000_04072023.csv")
# main("data/test_corpus.csv")
main("data/dix_mille_short.csv")
