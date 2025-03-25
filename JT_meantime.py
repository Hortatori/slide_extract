import pandas as pd
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


def count_time(dataset):
    n = 0
    durations_by_channel = defaultdict(list)
    duration_lines_by_channel = defaultdict(list)
    simple_datetime = defaultdict(list)
    end = 0
    while n < dataset.shape[0] - 2:
        size_jt_lines = 0
        batch_start = datetime.strptime(dataset["start"][n], "%d/%m/%Y %H:%M:%S")
        end = datetime.strptime(dataset["end"][n], "%d/%m/%Y %H:%M:%S")
        following_start = datetime.strptime(
            dataset["start"][n + 1], "%d/%m/%Y %H:%M:%S"
        )
        while end == following_start and n < dataset.shape[0] - 2:
            size_jt_lines += 1
            n += 1
            end = datetime.strptime(dataset["end"][n], "%d/%m/%Y %H:%M:%S")

            following_start = datetime.strptime(
                dataset["start"][n + 1], "%d/%m/%Y %H:%M:%S"
            )
            # print("\t into while, end : ", end , "following_start : ", following_start, " n + 1 : ", n+1)
        duration = end - batch_start
        hours, minutes = duration.seconds // 3600, duration.seconds // 60 % 60
        # print(f"duration in sc: {duration},\n{hours} hours, {minutes} minutes,\n time of batch start : {batch_start}, time of end : {end}")
        # print(f"size_jt_lines : {size_jt_lines}\n")
        durations_by_channel[dataset["channel"][n]].append(
            str(hours) + ":" + str(minutes)
        )
        duration_lines_by_channel[dataset["channel"][n]].append(size_jt_lines)
        simple_datetime[dataset["channel"][n]].append(duration)
        n += 1
    return duration_lines_by_channel, durations_by_channel, simple_datetime


def human_time(time):
    hours, minutes = time.seconds // 3600, time.seconds // 60 % 60
    return str(hours) + ":" + str(minutes)


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
    for key in duration_lines_by_channel.keys():
        stats["channel"].append(key)
        stats["nb_JTs"].append(len(duration_lines_by_channel[key]))
        stats["nb_lines"].append(sum(duration_lines_by_channel[key]))
        dt_sum = sum(simple_datetime[key], timedelta())
        stats["total time"].append(human_time(dt_sum))
        dt_avg = sum(simple_datetime[key], timedelta()) / len(simple_datetime[key])
        stats["avg time"].append(human_time(dt_avg))
        dt_min = min(simple_datetime[key])
        stats["min_time"].append(human_time(dt_min))
        dt_max = max(simple_datetime[key])
        stats["max_time"].append(human_time(dt_max))

    # Compléter les listes avec NaN pour pv les transformer en csv
    max_len = max(len(lst) for lst in duration_lines_by_channel.values())
    for key in duration_lines_by_channel.keys():
        duration_lines_by_channel[key] += [np.nan] * (
            max_len - len(duration_lines_by_channel[key])
        )
        durations_by_channel[key] += [np.nan] * (
            max_len - len(durations_by_channel[key])
        )

    stats = pd.DataFrame(stats)
    df_line = pd.DataFrame(duration_lines_by_channel)
    df_time = pd.DataFrame(durations_by_channel)
    return df_line, df_time, stats


def main(name):
    dataset = pd.read_csv(name, quoting=csv.QUOTE_ALL)
    duration_lines_by_channel, durations_by_channel, simple_datetime = count_time(
        dataset
    )
    df_line, df_time, stats = stats_and_dataframing(
        duration_lines_by_channel, durations_by_channel, simple_datetime
    )
    print(df_line)
    print(df_time)
    print(stats)
    df_line.to_csv(name.split("/")[0] + "/line_" + name.split("/")[1], index=False)
    df_time.to_csv(name.split("/")[0] + "/time_" + name.split("/")[1], index=False)
    stats.to_csv(name.split("/")[0] + "/stats_" + name.split("/")[1], index=False)


main("data/medialex_transcriptions_vocapia_v1v2_20230301_20230731.csv")
# main("data/short_1000_04072023.csv")
# main("data/test_corpus.csv")
