import pandas as pd
import csv
data = pd.read_csv("medialex_transcriptions_vocapia_v1v2_20230301_20230731.csv",
                    quoting=csv.QUOTE_ALL
                    )
data.to_csv("medialex_transcriptions_vocapia_v1v2_20230301_20230731.tsv", sep="\t", index=False, quoting=csv.QUOTE_ALL)