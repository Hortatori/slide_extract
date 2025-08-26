import time
import os
from pathlib import Path
import polars as pl
from encode_articles import main as encode_article
from utils.to_deliver import main as to_deliver

# fait tourner encode_article.py en utilisant les articles du jour pour comparer avec les JT du même jour. 

class Args :
    def __init__(self, jt, article):
        self.trs = "articles/"+jt
        self.output = "articles"
        self.meta_file = jt.split(".")[0]+"_meta.csv"
        self.similarity_file = article.split(".")[0]+"_sim.csv"
        self.npy_file = jt.split(".")[0]+".npy"
        self.otmedia = article
        self.threshold = 0.40

class Args_deliver :
    def __init__(self, extracted, trs, dirname):
        self.extracted = extracted
        self.trs = trs
        self.dirname = dirname

def nasty_main() :
    start = time.time()
    META_SCHEMA = {"":int, "channel": str, "start": pl.Datetime, "end": pl.Datetime, "duration": float, "text": str, "id": str, "created_at": str, "tag": str}
    concat_days = pl.DataFrame(schema=META_SCHEMA)
    EXTRACT_SCHEMA = {"channel": str, "start": pl.Datetime, "end": pl.Datetime, "start_id": int, "end_id": int, "text": str, "label": int}
    concat_xtr = pl.DataFrame(schema=EXTRACT_SCHEMA)
    for jt,article in [["06_27_JT.csv","06_27_otmedia.csv"],["06_28_JT.csv","06_28_otmedia.csv"],["06_29_JT.csv","06_29_otmedia.csv"],["06_30_JT.csv","06_30_otmedia.csv"],["07_01_JT.csv","07_01_otmedia.csv"],["07_02_JT.csv","07_02_otmedia.csv"],["07_03_JT.csv","07_03_otmedia.csv"]] :
        # ---- slicing previous meta and emb method cannot be used (order is first by channel, then by time => need to recalculate on DAY files (I manually) created for this ----
        if os.path.isfile(Path("articles",article)) and os.path.isfile(Path("articles", jt)):

            args = Args(jt,article)

            try:
                # run encode_article.py     => label a minute of text as 1 if it is close enough to the articles of the day
                print(f"running encode_article with {', '.join('%s: %s' % item for item in vars(args).items())}")
                encode_article(args)

                path_extracts = Path(args.output,"nrv_extracted_docs_"+"".join(str(args.threshold).split("."))+"_"+args.otmedia.split(".")[0]+".csv")
                xtr = pl.read_csv(path_extracts, schema_overrides=EXTRACT_SCHEMA)
                concat_xtr.extend(xtr)
                # run to_deliver.py     => use indexes to retrieve lines of text instead of a sliding window of a minute
                arg_deliver = Args_deliver(path_extracts, Path(args.output,jt), args.output)
                to_deliver(arg_deliver)
                path_output = Path(args.output,'deliver_'+str(arg_deliver.extracted).split('/')[1])
                _this_day_deliver = pl.read_csv(path_output, schema_overrides=META_SCHEMA)
                print(f"shape of this day {jt} deliver : {_this_day_deliver.shape}")
                print(_this_day_deliver)
                concat_days = concat_days.extend(_this_day_deliver)
                print("concat all :")
                print(concat_days)
            except Exception as e:
                print(e)
    end = time.time()
    concat_xtr = concat_xtr.sort("channel","start")
    concat_xtr.write_csv(Path(args.output,"concat_all_nrv.csv"))

    concat_days = concat_days.unique("id") #dedup
    concat_days = concat_days.sort("channel", "start")
    print(concat_days)
    concat_days.write_csv("output_deliver_from_run_encode.csv")

    print(f"duration to encode and extract all articles : {end-start}")

if __name__ == "__main__" :

    nasty_main()
