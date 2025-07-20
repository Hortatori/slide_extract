import pandas as pd
import csv
from sentence_transformers import SentenceTransformer
import os
import tqdm
import torch
import time
import argparse
import shutil
import subprocess
import sys

# ouvrir idx docs
# pour chaque paire d'index,selectionner le batch d'embedding correspondant
# puis faire tourner les fenêtres sur ce morceau
# au lieu de sauvegrader tous les 10000, sauvegarder à chaque fin de morceau


def encode_dataset(model, data):
    embbed_texts = model.encode(
        data.text.tolist(), show_progress_bar=True, convert_to_tensor=True
    )
    return embbed_texts


def encode_keywords(model, keywords):
    # keywords = [i[0:500] for i in keywords]
    vectors = model.encode(keywords, convert_to_tensor=True)
    return vectors

def load_X(path, model, data, saving_path):
    if not os.path.exists("matrix/"):
        os.mkdir("matrix/")

    if os.path.exists(path + ".pt"):
        print("embedding already computed, loading from ", path + ".pt")
        embedded_data = torch.load(path + ".pt", weights_only=True)
    else:
        print("computing embedding...")
        embedded_data = encode_dataset(model, data)
        print(
            "shape of embedded data ",
            embedded_data.shape,
            " saving to ",
            path,
            "type : ",
            type(embedded_data),
        )
        torch.save(embedded_data, path + ".pt")
    # add a directory to save extracted documents from the script
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    return embedded_data

class JT_sliding():
    def __init__(self, start_batch, end_batch, window_size, embeddings, step=1):
        self.window_size = window_size
        self.embeddings = embeddings
        self.step = step
        self.start_batch = start_batch
        self.end_batch = end_batch

    def iterate(self):
        start = self.start_batch
        N = self.end_batch - self.window_size + 1
        while start < N:
            embedded_texts = self.embeddings[start : start + self.window_size]
            doc_indices = [start, start + self.window_size]  # Index des docs
            yield embedded_texts, doc_indices, start
            start += self.step

def main(args):

    begin_tim = time.time()
    matrix_path = os.path.join("matrix", args.dataset.replace(".csv", "").split("/")[-1]+"_"+args.model_name.replace("/", "_"))

    # checking if reordering and JT batching has been done, if not, call JTtime
    if not os.path.exists(args.dataset.split("/")[0] + "/reordered/" + args.dataset.split("/")[1]):
        print("computing JTs indexes of ", args.dataset, "and reordering depending of time")
        try :
            full_command = f"python3 JT_ids.py --dataset {args.dataset}"
            subprocess.run(full_command, shell = True, check=True)
        except OSError as err:  
            print(err)
            sys.exit(1)

    data = pd.read_csv(args.dataset, quoting=csv.QUOTE_ALL)
    df_idx_paires = pd.read_csv(args.dataset.split("/")[0]+"/idx_docs_pairs_"+args.dataset.split("/")[-1], quoting=csv.QUOTE_ALL)
    saving_path = "extracted_docs"
    # AliBaba model need a SentenceTransformer(args.model_name, trust_remote_code=True) security issue if they change the config file (but huggingface should verify it)
    sim_history = {"similarity":[],"start_time":[], "channel":[]}
    model = SentenceTransformer(args.model_name)
    truncation = int(model.max_seq_length) - 2
    model.max_seq_length = truncation
    embedded_data = load_X(matrix_path, model, data, saving_path)
    cuda0 = torch.device('cuda:0')
    embedded_data = embedded_data.to(cuda0)
    keywords_embedded = encode_keywords(model, args.keywords)
    print(f"keywords shape : {keywords_embedded.shape}")
    # creating a spare index to keep when extracting documents
    data = data.reset_index()

    print(f"embedded data shape : {embedded_data.shape}")
    info_sim = []
    keep_oldID = set()
    # for each JTbatch :
    for row in df_idx_paires.itertuples():
        w_slide = JT_sliding(row[1], row[2], args.window_size, embedded_data)
        length = int(row[2]- (args.window_size-1)) - (int(row[1]))
        with tqdm.tqdm(total=length) as pbar:
            for batch, indexes, batch_index in w_slide.iterate():
                similarity = model.similarity(batch, keywords_embedded)
                avg_sim = torch.mean(similarity, dim=1)
                sim_history["similarity"].append(avg_sim)
                sim_history["start_time"].append(data.at[indexes[0],"start"])
                sim_history["channel"].append(data.at[indexes[0],"channel"])

                if (avg_sim > args.threshold).any():
                    current_old_ids = range(indexes[0], indexes[1])
                    keep_oldID.update(current_old_ids)

                    info_sim.append(avg_sim)
                pbar.update(1)


    if len(info_sim) == 0:
        print("aucune similarité superieure au threshold")
    else:
        data['label'] = data.index.isin(keep_oldID).astype(int)
    end_tim = time.time()
    
    sim_history["similarity"] = [i[0].detach().item() for i in sim_history["similarity"]]
    rg = range(0, len(sim_history["similarity"]))
    df_similarity = pd.DataFrame(sim_history, index=rg)
    df_similarity['start_time'] = pd.to_datetime(df_similarity['start_time'], dayfirst=True, format="mixed")
    df_similarity.to_csv("suite_similarite"
            + "_JT_"
            + args.dataset.replace(".csv", "").split("/")[-1]
            + "_"
            + args.model_name.replace("/", "_")
            + ".csv",)

    data.to_csv(os.path.join(saving_path, str(args.threshold).replace(".","")+"_labelledJT_"+args.dataset.replace(".csv", "").split("/")[-1]+"_"+args.model_name.replace("/", "_")+".csv"))
    delta = end_tim - begin_tim
    print(f"temps de traitement : {delta} secondes, soit {int(delta // 3600)} heures, {int((delta % 3600) // 60)} minutes, {int(delta % 60)} secondes")

# working combos for now : 
# -model_name Alibaba-NLP/gte-multilingual-base -threshold 0.63 
# (only for a shorter dataset, AliBaba is too heavy as an embedding)
# -model_name Lajavaness/sentence-camembert-large -threshold 0.4

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "--model_name",
    type=str,
    default="Lajavaness/sentence-camembert-large",
    help="Sentence Transformer model name, default to Lajavaness/sentence-camembert-large"   
)
parser.add_argument(
    "--keywords",
    type=str,
    required=True,
    default="gpt_nahel.csv",
    help="text that will be compared"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="data/nahel_transcriptions_vocapia_26_06_2023_to_03_07_2023.csv",
    required=True,
    help="path to dataset"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.42,
    help="threshold"
)
parser.add_argument(
    "--window_size",
    type=int,
    default=8,
    help="size of the sliding window"
)
parser.add_argument(
    "--sliding_type",
    type=str,
    choices=["JT", "time"],
    default="JT",
    help="JT = sliding windows take into account the JT divisions | time = sliding windows does not take into account the JT divisions"
)
args = parser.parse_args()
main(args)

