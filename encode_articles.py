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
META_SCHEMA = {"channel": str, "start": pl.Datetime, "end": pl.Datetime, "start_id": int, "end_id": int, "text": str}
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
    # text = "La mort de Nahel Merzouk, un adolescent franco-algérien de 17 ans, est causée par le tir à bout portant d'un policier le 27 juin 2023 lors d'un contrôle routier à Nanterre dans les Hauts-de-Seine (Île-de-France). Deux autres adolescents, âgés de 14 et 17 ans, sont passagers à bord de la voiture. La version policière, celle d'une voiture refusant un contrôle avant de foncer sur un fonctionnaire de police qui a ouvert le feu dans son bon droit, est initialement reprise par les médias, mais contredite dans les heures qui suivent par les témoignages des deux passagers. La victime est également présentée à tort dans plusieurs médias comme ayant un casier judiciaire, allégations qui sont démenties par la suite, son nom ne figurant qu'au fichier des antécédents judiciaires. Le 29 juin, le policier Florian M. est mis en examen pour homicide volontaire et placé en détention provisoire avant d'être remis en liberté quelques mois plus tard. L'événement provoque des émeutes dans de nombreuses villes françaises ainsi qu'en Belgique et en Suisse, dont le bilan des dégâts et de la répression dépasse celui des émeutes de 2005. Cette affaire relance le débat sur les violences policières, la question du racisme au sein de la police française et son usage des armes à feu, ainsi que son traitement par les médias qui se sont d'abord appuyés sur des sources policières. Elle provoque de nombreuses réactions en France de personnalités politiques, sportives, artistiques et religieuses, ainsi que de gouvernements étrangers et de l'Organisation des Nations unies. Une marche blanche et des cagnottes sont par ailleurs organisées."
    # text = "L'accident du submersible Titan se produit lors d'une plongée dans les eaux internationales de l'océan Atlantique Nord au large de Terre-Neuve (Canada). Le Titan est un petit submersible à visée touristique, exploité par OceanGate et destiné particulièrement à assurer des visites payantes de l'épave du Titanic. Le 18 juin 2023, il amorce une descente en direction de l'épave au cours de laquelle il subit une implosion entraînant sa destruction et la mort de ses cinq occupants. Des débris du Titan sont retrouvés par 3 800 m de fond, non loin de l'épave du Titanic. Le 28 juin 2023, des restes humains sont retrouvés parmi des débris remontés à la surface. C'est l'accident sous-marin mortel le plus profond de l'Histoire. "
    text = [
    "L'affaire Nahel concerne la mort de Nahel Merzouk, un adolescent de 17 ans, tué par un policier lors d'un contrôle routier à Nanterre le 27 juin 2023. Nahel, qui conduisait une voiture sans permis, a été arrêté par deux policiers à moto. Une vidéo, largement diffusée sur les réseaux sociaux, montre un policier pointant son arme sur lui à bout portant. Lorsque Nahel a tenté de redémarrer, le policier a tiré, provoquant sa mort.Cette affaire a suscité une immense indignation en France. La vidéo a contredit la version initiale des forces de l’ordre, entraînant une vague de colère et de violentes émeutes dans plusieurs villes. Les manifestations ont donné lieu à des affrontements avec la police et à de nombreux dégâts matériels. Sur le plan judiciaire, le policier responsable du tir a été mis en examen pour homicide volontaire et placé en détention provisoire.",
    "L’affaire a relancé les débats sur les violences policières, le racisme et les tensions entre la jeunesse des quartiers populaires et les forces de l’ordre. Le gouvernement a appelé au calme tandis que plusieurs personnalités politiques et culturelles ont exprimé leur indignation face à ce drame. L’affaire Nahel a ravivé un débat récurrent en France sur les violences policières, en particulier celles qui touchent les jeunes des quartiers populaires et les minorités. L’indignation a été massive et s’est exprimée à plusieurs niveaux : dans la rue, sur les réseaux sociaux, dans les médias et au sein de la classe politique.  Dès la diffusion de la vidéo du tir, la colère a explosé, car elle contredisait la version initiale des policiers, qui affirmaient que Nahel avait tenté de les percuter. ",
    "De nombreux habitants de quartiers populaires ont dénoncé un abus de pouvoir et une pratique récurrente des forces de l’ordre à leur égard. Les émeutes qui ont suivi traduisent une frustration plus large liée à des décennies de tensions entre la police et certaines populations. Plusieurs personnalités, notamment du monde du sport, du cinéma et de la musique, ont dénoncé la violence policière. Le footballeur Kylian Mbappé, l’acteur Omar Sy ou encore le rappeur Youssoupha ont exprimé leur tristesse et leur colère face à la mort de Nahel. Des associations comme SOS Racisme et la Ligue des droits de l’homme ont également condamné la brutalité des forces de l’ordre et appelé à des réformes profondes. Sur le plan politique, les réactions ont été vives et opposées. ",
    "À gauche, des figures comme Jean-Luc Mélenchon et Sandrine Rousseau ont dénoncé un racisme systémique dans la police et exigé des réformes structurelles, notamment la suppression de l’IGPN (Inspection Générale de la Police Nationale), jugée inefficace pour sanctionner les fautes policières. À droite et à l’extrême droite, des personnalités comme Gérald Darmanin et Marine Le Pen ont défendu les forces de l’ordre et condamné les émeutes, mettant en avant la nécessité de rétablir l’ordre et de punir sévèrement les violences urbaines. L’affaire Nahel s’inscrit dans un contexte où plusieurs jeunes hommes issus de l’immigration ont été tués ou blessés lors d’interventions policières. En 2022, un record avait été atteint avec 13 décès lors de contrôles routiers impliquant des refus d’obtempérer. ",
    "Des affaires comme celle d’Adama Traoré, mort en 2016 après une interpellation, avaient déjà mis en lumière ces violences. Face à l’ampleur des réactions, le président Emmanuel Macron a condamné la mort de Nahel tout en dénonçant les violences urbaines. Le policier mis en cause a été incarcéré pour homicide volontaire, un fait rare dans ce type d’affaire. Cependant, aucune réforme majeure de la police n’a été annoncée, et le débat reste très polarisé entre ceux qui réclament plus de fermeté et ceux qui demandent une refonte du maintien de l’ordre en France.",
    "Les violences urbaines qui ont suivi la mort de Nahel ont été d’une ampleur rare en France. Dès l’annonce de son décès, des émeutes ont éclaté à Nanterre et se sont rapidement propagées dans d’autres villes comme Paris, Marseille, Lyon et Lille. Pendant plusieurs nuits, des jeunes sont descendus dans la rue pour exprimer leur colère, attaquant des commissariats, incendiant des voitures, pillant des magasins et affrontant les forces de l’ordre.",
    "Ces violences n’étaient pas seulement une réaction à la mort de Nahel, mais le reflet d’un ras-le-bol plus large contre les discriminations, le contrôle au faciès et la manière dont les forces de l’ordre sont perçues dans certains quartiers populaires. Beaucoup de jeunes voient la police comme une institution qui les traite systématiquement avec méfiance ou agressivité, et la mort de Nahel a été la goutte d’eau qui a fait déborder le vase.",
    "Le gouvernement a réagi en déployant des dizaines de milliers de policiers et de gendarmes pour tenter de reprendre le contrôle. Le ministre de l’Intérieur, Gérald Darmanin, a parlé de réponse ferme et de tolérance zéro face aux violences. Des couvre-feux ont été instaurés dans certaines villes, et des centaines de personnes ont été arrêtées chaque nuit.",
    "Mais cette réponse musclée n’a pas calmé tout de suite la situation. Au contraire, beaucoup ont vu ces mesures comme une preuve supplémentaire que l’État répond toujours par la répression plutôt que d’essayer de comprendre les causes profondes du problème. Des élus et des sociologues ont rappelé que ce type d’émeutes éclate régulièrement en France après des bavures policières, comme en 2005 après la mort de Zyed et Bouna à Clichy-sous-Bois. À l’époque, trois semaines de violences avaient mis le pays sous tension.",
    "Sur les réseaux sociaux, les images des affrontements ont tourné en boucle, et les débats ont été très polarisés. D’un côté, certains dénonçaient l’attitude de la police et rappelaient que ces émeutes étaient un cri de désespoir de jeunes qui ne se sentent pas écoutés. De l’autre, certains insistaient sur les dégâts causés et demandaient un retour à l’ordre rapide, dénonçant des actes de casseurs sans lien direct avec la mort de Nahel.",
    "Au final, ces violences urbaines ont mis en lumière un problème plus profond : la fracture entre une partie de la population et les institutions, notamment la police. Tant que ces tensions ne seront pas prises en compte avec des réformes concrètes, il y a fort à parier que ce genre d’explosion sociale se reproduira.",
]
    # emb = [model.encode(text, show_progress_bar=False, normalize_embeddings=True)]
    # return np.array(emb)
    emb = model.encode(text, show_progress_bar=False, normalize_embeddings=True)

    return emb


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
        origin_start = start_row["original_id"]
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
            origin_end = current_row["original_id"]
            full_text = full_text + " " + current_row["text"]
            current_idx += 1

        full_text = full_text.strip()
        if len(full_text) > 0:
            logger.info(f"    ~ {start_idx} ::: {start_channel} {start_start} {current_end} - {current_end - start_start} ::: origin start id {origin_start}, origin end id {origin_end}")
            ft_emb = model.encode(full_text, show_progress_bar=False, normalize_embeddings=True)
            np.save(f, ft_emb)
            this = pl.DataFrame({"channel": start_channel, "start": start_start, "end": current_end, "start_id": origin_start, "end_id": origin_end, "text": full_text})
            meta = meta.extend(this)

    meta.write_csv(Path(output_dir, META_FILE), separator=",", quote_char='"', quote_style='non_numeric')
    f.close()


def process_embeddings(model, press_embeddings, output_dir, trs):
    meta = Path(output_dir, META_FILE)
    embeddings = Path(output_dir, NPY_FILE)

    logger.info(f"Loading metadata from {meta}")
    meta = pl.read_csv(meta, separator=",", quote_char='"', schema_overrides=META_SCHEMA)

    all_similarities = pl.DataFrame(schema=SIMILARITY_SCHEMA)
    all_extracted = pl.DataFrame(schema = META_SCHEMA)
    extracted_old_ids = set()
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
                similarities = model.similarity(ft_emb, press_embeddings)
                if torch.mean(similarities) > 0.50 :
                    id_list = range(current_row["start_id"], current_row["end_id"])
                    extracted_old_ids.update(id_list)
                    df_text = pl.DataFrame(current_row)
                    all_extracted = all_extracted.extend(df_text)
                # logger.info(f"[{current_idx}] {ft_emb.shape} - {current_channel} {current_start} ::: {similarities}")
                current_idx += 1

                this = pl.DataFrame({"channel": current_channel, "start": current_start, "end": current_end, "max": torch.max(similarities), "min": torch.min(similarities), "avg": torch.mean(similarities)})
                all_similarities = all_similarities.extend(this)
    all_similarities.write_csv(Path(output_dir, SIMILARITY_FILE), separator=",", quote_char='"', quote_style='non_numeric')
    all_extracted.write_csv(Path(output_dir, "nrv_extracted_docs_050.csv"), separator=",", quote_char='"', quote_style='non_numeric')
    ids_write = pl.DataFrame(list(extracted_old_ids))
    ids_write.write_csv(Path(output_dir,"extracted_ids_050.csv"))

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

