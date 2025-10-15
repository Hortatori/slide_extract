import time
import os
from pathlib import Path
import polars as pl
from encode_articles import main as encode_article
from utils.to_deliver import main as to_deliver
import argparse
import glob

# fait tourner encode_article.py en utilisant les articles du jour pour comparer avec les JT du même jour. 

class Args :
    def __init__(self, jt, article,t):
        self.trs = "articles/"+jt
        self.output = "articles"
        self.meta_file = jt.split(".")[0]+"_meta.csv"
        self.similarity_file = article.split(".")[0]+"_sim.csv"
        self.npy_file = jt.split(".")[0]+".npy"
        self.otmedia = article
        self.threshold = t

class Args_deliver :
    def __init__(self, extracted, trs, dirname):
        self.extracted = extracted
        self.trs = trs
        self.dirname = dirname

def nasty_main(main_args) :
    start = time.time()
    if not os.path.exists("articles/"):
        os.mkdir("articles/")
    SIM_SCHEMA = {"channel": str, "start": pl.Datetime, "end": pl.Datetime, "max": float, "min": float, "avg": float}
    META_SCHEMA = {"":int, "channel": str, "start": pl.Datetime, "end": pl.Datetime, "duration": float, "text": str, "id": str, "created_at": str}
    LABEL_SCHEMA = {"":int, "channel": str, "start": pl.Datetime, "end": pl.Datetime, "duration": float, "text": str, "id": str, "created_at": str, "label":int}
    concat_days = pl.DataFrame(schema=META_SCHEMA)
    labelled_lines = pl.DataFrame(schema=LABEL_SCHEMA)
    EXTRACT_SCHEMA = {"channel": str, "start": pl.Datetime, "end": pl.Datetime, "start_id": int, "end_id": int, "text": str, "label": int}
    concat_label = pl.DataFrame(schema=EXTRACT_SCHEMA)
    for jt,article in [["06_27_JT.csv","06_27_otmedia.csv"],["06_28_JT.csv","06_28_otmedia.csv"],["06_29_JT.csv","06_29_otmedia.csv"],["06_30_JT.csv","06_30_otmedia.csv"],["07_01_JT.csv","07_01_otmedia.csv"],["07_02_JT.csv","07_02_otmedia.csv"],["07_03_JT.csv","07_03_otmedia.csv"]] :
        # ---- slicing previous meta and emb method cannot be used (order is first by channel, then by time => need to recalculate on DAY files (I manually) created for this ----
        if os.path.isfile(Path("articles",article)) and os.path.isfile(Path("articles", jt)):

            args = Args(jt,article, main_args.threshold)

            try:
                # run encode_article.py     => label a minute of text as 1 if it is close enough to the articles of the day
                print(f"running encode_article with {', '.join('%s: %s' % item for item in vars(args).items())}")
                encode_article(args)

                path_extracts = Path(args.output,"minutes_labelled_"+"".join(str(args.threshold).split("."))+"_"+args.otmedia.split(".")[0]+".csv")
                labelled_minutes = pl.read_csv(path_extracts, schema_overrides=EXTRACT_SCHEMA)
                concat_label.extend(labelled_minutes)
                # run to_deliver.py     => use indexes to retrieve lines of text instead of a sliding window of a minute
                arg_deliver = Args_deliver(path_extracts, Path(args.output,jt), args.output)
                to_deliver(arg_deliver)
                path_output = Path(args.output,'formatted_'+str(arg_deliver.extracted).split('/')[1])
                _this_day_deliver = pl.read_csv(path_output, schema_overrides=META_SCHEMA)
                print(f"shape of this day {jt} deliver : {_this_day_deliver.shape}")
                print(_this_day_deliver)
                concat_days = concat_days.extend(_this_day_deliver)
                print("concat all :")
                print(concat_days)

                path_output = Path(args.output,'trs_lignes_labelled_'+str(arg_deliver.extracted).split('/')[1])
                this_day_lines_labelled = pl.read_csv(path_output, schema_overrides=META_SCHEMA)
                labelled_lines.extend(this_day_lines_labelled)

            except Exception as e:
                print(e)
    end = time.time()
    # NOTE : concat_label, saved as "labelled_output_from_run_encode" is still in minutes
    concat_label = concat_label.sort("channel","start")
    concat_label.write_csv(Path(args.output,"labelled_minutes_from_run_encode_"+"".join(str(args.threshold).split("."))+".csv"))

    # NOTE : concat days contains only positives labels
    concat_days = concat_days.unique("id") #dedup
    concat_days = concat_days.sort("channel", "start")
    print(concat_days)
    concat_days.write_csv("formatted_output_from_run_encode_"+"".join(str(args.threshold).split("."))+".csv")

    # NOTE : labelled lines is the original transcription + one column label 0-1
    labelled_lines.write_csv(Path(args.output,"labelled_lines_output_from_run_encode_"+"".join(str(args.threshold).split("."))+".csv"))
    # similarity files
    sim_paths = glob.glob(args.output+"/*_sim.csv")
    sim_dfs = [pl.read_csv(f, schema_overrides=SIM_SCHEMA) for f in sim_paths]
    if sim_dfs:
        sim_concat = pl.concat(sim_dfs)
    else :
        "WARNING : no similarity files, there should be some"
    sim_concat = sim_concat.sort("start")
    sim_concat.write_csv(Path(args.output,"sim_from_run_encode.csv"))

    _ = [os.remove(f) for f in sim_paths]
    _ = [os.remove(f) for f in glob.glob(args.output+"/minutes_labelled_*.csv")]
    _ = [os.remove(f) for f in glob.glob(args.output+"/formatted_minutes_labelled*.csv")]
    _ = [os.remove(f) for f in glob.glob(args.output+"/trs_lignes_labelled_*.csv")]



    print(f"duration to encode and extract all articles : {end-start}")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', required=True, help="threshold to choose")
    main_args = parser.parse_args()
    nasty_main(main_args)
