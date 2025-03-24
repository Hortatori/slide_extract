import pandas as pd
import csv
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
def count_time(dataset):
    n = 0
    durations_by_channel = defaultdict(list)
    duration_lines_by_channel = defaultdict(list)
    end = 0
    while n < dataset.shape[0]-2:
        size_jt_lines = 0
        batch_start = datetime.strptime(dataset["start"][n], "%d/%m/%Y %H:%M:%S")
        end = datetime.strptime(dataset["end"][n], "%d/%m/%Y %H:%M:%S")
        following_start = datetime.strptime(dataset["start"][n+1], "%d/%m/%Y %H:%M:%S")
        while end == following_start and n < dataset.shape[0]-2:
            size_jt_lines += 1
            n += 1
            end = datetime.strptime(dataset["end"][n], "%d/%m/%Y %H:%M:%S")
            
            following_start = datetime.strptime(dataset["start"][n+1], "%d/%m/%Y %H:%M:%S")
            # print("\t into while, end : ", end , "following_start : ", following_start, " n + 1 : ", n+1)
        duration = end - batch_start
        print("channel", dataset["channel"][n])
        hours, minutes = duration.seconds // 3600, duration.seconds // 60 % 60
        print(f"duration in sc: {duration},\n{hours} hours, {minutes} minutes,\n time of batch start : {batch_start}, time of end : {end}")
        print(f"size_jt_lines : {size_jt_lines}\n")
        durations_by_channel[dataset["channel"][n]].append(str(hours) +" : "+ str(minutes))
        duration_lines_by_channel[dataset["channel"][n]].append(size_jt_lines)
        n += 1
    [print(i) for i in durations_by_channel.items()]
    [print(i) for i in duration_lines_by_channel.items()]



def main(name):

    dataset = pd.read_csv(name, quoting=csv.QUOTE_ALL)
    count_time(dataset)

# main("data/medialex_transcriptions_vocapia_v1v2_20230301_20230731.csv")
main("data/short_1000_04072023.csv")
# main("data/test_corpus.csv")