# N. HERVE - INA Research - 05/05/2025
# Medialex Sprint 2
#TODO test if batch of minutes is different of batch of lines when extracting
from sentence_transformers import SentenceTransformer
import argparse
import logging
import json
import re
import string
import torch
import csv
from tqdm import tqdm
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("emb")

SPLIT_WORDS = re.compile("[" + re.escape(string.punctuation + " ") + "]+")
# BEGIN_DATE = datetime(2023, 6, 15, 0, 0, 0)
# END_DATE = datetime(2023, 7, 15, 23, 30, 0)
META_SCHEMA = {"channel": str, "start": pl.Datetime, "end": pl.Datetime, "start_id": int, "end_id": int}
SIMILARITY_SCHEMA = {"channel": str, "start": pl.Datetime, "end": pl.Datetime, "max": pl.Float64, "min": pl.Float64, "avg": pl.Float64}

META_FILE = None
SIMILARITY_FILE = None
NPY_FILE = None

def load_model():
    logger.info(f"Loading model")
    model = SentenceTransformer("Lajavaness/sentence-camembert-large")
    model.max_seq_length = 512
    model.similarity_fn_name = "dot"
    return model


def create_press_embeddings(model, otmedia_file):
    logger.info(f"Loading articles from {otmedia_file}")
    press_embeddings = []
    with open(otmedia_file, "rt") as f:
        for line in f:
            article = json.loads(line)
            text = article['document'].replace("\n", " ")
            words = SPLIT_WORDS.split(text);
            emb = model.encode(text, show_progress_bar=False, normalize_embeddings=True)
            logger.info(f" ~ {article['docTime']} [{article['media']}] [{len(text)} / {len(words)}] - {article['title']}")
            press_embeddings.append(emb)
    return np.array(press_embeddings)


def create_adhoc_embeddings(model):
    text = "La mort de Nahel Merzouk, un adolescent franco-algérien de 17 ans, est causée par le tir à bout portant d'un policier le 27 juin 2023 lors d'un contrôle routier à Nanterre dans les Hauts-de-Seine (Île-de-France). Deux autres adolescents, âgés de 14 et 17 ans, sont passagers à bord de la voiture. La version policière, celle d'une voiture refusant un contrôle avant de foncer sur un fonctionnaire de police qui a ouvert le feu dans son bon droit, est initialement reprise par les médias, mais contredite dans les heures qui suivent par les témoignages des deux passagers. La victime est également présentée à tort dans plusieurs médias comme ayant un casier judiciaire, allégations qui sont démenties par la suite, son nom ne figurant qu'au fichier des antécédents judiciaires. Le 29 juin, le policier Florian M. est mis en examen pour homicide volontaire et placé en détention provisoire avant d'être remis en liberté quelques mois plus tard. L'événement provoque des émeutes dans de nombreuses villes françaises ainsi qu'en Belgique et en Suisse, dont le bilan des dégâts et de la répression dépasse celui des émeutes de 2005. Cette affaire relance le débat sur les violences policières, la question du racisme au sein de la police française et son usage des armes à feu, ainsi que son traitement par les médias qui se sont d'abord appuyés sur des sources policières. Elle provoque de nombreuses réactions en France de personnalités politiques, sportives, artistiques et religieuses, ainsi que de gouvernements étrangers et de l'Organisation des Nations unies. Une marche blanche et des cagnottes sont par ailleurs organisées."
    # text = "L'accident du submersible Titan se produit lors d'une plongée dans les eaux internationales de l'océan Atlantique Nord au large de Terre-Neuve (Canada). Le Titan est un petit submersible à visée touristique, exploité par OceanGate et destiné particulièrement à assurer des visites payantes de l'épave du Titanic. Le 18 juin 2023, il amorce une descente en direction de l'épave au cours de laquelle il subit une implosion entraînant sa destruction et la mort de ses cinq occupants. Des débris du Titan sont retrouvés par 3 800 m de fond, non loin de l'épave du Titanic. Le 28 juin 2023, des restes humains sont retrouvés parmi des débris remontés à la surface. C'est l'accident sous-marin mortel le plus profond de l'Histoire. "
    emb = [model.encode(text, show_progress_bar=False, normalize_embeddings=True)]
    return np.array(emb)


def create_transcription_embeddings(model, transcription_file, output_dir):
    logger.info(f"Loading transcriptions from {transcription_file}")
    trs_table  = pl.read_csv(transcription_file, separator=",", quote_char='"')
    logger.info(f" - loaded   {trs_table.columns} : {trs_table.shape}")
    trs_table  = trs_table.with_columns(
        pl.col("start").str.to_datetime("%d/%m/%Y %H:%M:%S"),
        pl.col("end").str.to_datetime("%d/%m/%Y %H:%M:%S")
    )
    trs_table = trs_table.with_row_index(name="original_id")
    trs_table = trs_table.sort("channel", "start")
    # trs_table = trs_table.filter(pl.col("channel") == "TF1")
    # trs_table = trs_table.filter(pl.col("start") >= BEGIN_DATE, pl.col("start") <= END_DATE)
    logger.info(f" - filtered {trs_table.columns} : {trs_table.shape}")

    meta = pl.DataFrame(schema=META_SCHEMA)
    f = open(Path(output_dir, NPY_FILE), "wb")
    for start_idx in range(0, trs_table.shape[0]):
        start_row = trs_table.row(start_idx, named=True)
        start_start = start_row["start"]
        start_id = start_row["original_id"]
        start_channel = start_row["channel"]
        full_text = ""

        current_idx = start_idx
        while current_idx < trs_table.shape[0]:
            current_row = trs_table.row(current_idx, named=True)
            if current_row["channel"] != start_channel:
                break
            delta = current_row["start"] - start_start
            if delta.total_seconds() > 60:
                break
            current_end = current_row["end"]
            current_id = current_row["original_id"]
            full_text = full_text + " " + current_row["text"]
            current_idx += 1

        full_text = full_text.strip()
        if len(full_text) > 0:
            logger.info(f"    ~ {start_idx} ::: {start_channel} {start_start} {current_end} - {current_end - start_start} ::: start id {start_id}, end id {current_id}")
            ft_emb = model.encode(full_text, show_progress_bar=False, normalize_embeddings=True)
            np.save(f, ft_emb)
            this = pl.DataFrame({"channel": start_channel, "start": start_start, "end": current_end, "start_id": start_id, "end_id": current_id})
            meta = meta.extend(this)

    meta.write_csv(Path(output_dir, META_FILE), separator=",", quote_char='"', quote_style='non_numeric')
    f.close()


def process_embeddings(model, press_embeddings, output_dir, trs):
    meta = Path(output_dir, META_FILE)
    embeddings = Path(output_dir, NPY_FILE)

    trsc_full = Path(trs)
    logger.info(f"Loading transcription from {trsc_full}")
    full_text = pl.read_csv(trsc_full, separator=",", quote_char='"')

    logger.info(f"Loading metadata from {meta}")
    meta = pl.read_csv(meta, separator=",", quote_char='"', schema_overrides=META_SCHEMA)

    all_similarities = pl.DataFrame(schema=SIMILARITY_SCHEMA)
    all_extracted = pl.DataFrame(schema=[(col, dtype) for col, dtype in zip(full_text.columns, full_text.dtypes)])
    logger.info(f"Processing embeddings from {embeddings}")
    current_idx = 0
    with open(embeddings, "rb") as f:
        with tqdm(total=meta.shape[0]) as pbar:
            while current_idx < meta.shape[0]:
                current_row = meta.row(current_idx, named=True)
                ft_emb = np.load(f)
                pbar.update(1)

                current_start = current_row["start"]
                current_end = current_row["end"]
                current_channel = current_row["channel"]
                logger.info(f"current row {current_row}")
                old_idx = current_row["start_id"]
                similarities = model.similarity(ft_emb, press_embeddings)
                if torch.mean(similarities) > 0.42 :
                    df_text = pl.DataFrame(full_text[old_idx,:])
                    all_extracted = all_extracted.extend(df_text)
                # logger.info(f"[{current_idx}] {ft_emb.shape} - {current_channel} {current_start} ::: {similarities}")
                current_idx += 1

                this = pl.DataFrame({"channel": current_channel, "start": current_start, "end": current_end, "max": torch.max(similarities), "min": torch.min(similarities), "avg": torch.mean(similarities)})
                all_similarities = all_similarities.extend(this)
    all_similarities.write_csv(Path(output_dir, SIMILARITY_FILE), separator=",", quote_char='"', quote_style='non_numeric')
    all_extracted.write_csv(Path(output_dir, "nrv_extracted_docs_043.csv"), separator=",", quote_char='"', quote_style='non_numeric')


def main(args):
    model = load_model()
    # create_transcription_embeddings(model, args.trs, args.output)
    # press_embeddings = create_press_embeddings(model, args.otmedia)
    press_embeddings = create_adhoc_embeddings(model)
    process_embeddings(model, press_embeddings, args.output, args.trs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--otmedia",
                        type = str,
                        help = "Path to VSD json file")
    parser.add_argument("--trs",
                        type = str,
                        help = "Path to CSV transcription file")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="Path to store embeddings and results")
    parser.add_argument("--meta_file",
                        type=str,
                        help="File to store sliding window metadata",
                        )
    parser.add_argument("--similarity_file",
                        type=str,
                        help="File to store sliding window similarity scores")
    parser.add_argument("--npy_file",
                        type=str,
                        help="File to store sliding window ebeddings")

    parsed = parser.parse_args()

    # ugly, but idc
    META_FILE = parsed.meta_file
    SIMILARITY_FILE = parsed.similarity_file
    NPY_FILE = parsed.npy_file

    main(parsed)

