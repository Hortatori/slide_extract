import numpy as np
import pandas as pd
import csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import tqdm
import torch
import time
import argparse
"""
Parameters
* The SentenceBert model for representing text
* The threshold for distance cosinus
* Size of sliding window 
* Step of the slide 
* The dataset
each 10000 lines, write in a csv with the current results, then concat all to produce the complete results
"""

KEY_WORDS = [
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

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "-model_name",
    type=str,
    default="Lajavaness/sentence-camembert-large",
    help="Sentence Transformer model name, default to Lajavaness/sentence-camembert-large"   
)
parser.add_argument(
    "-dataset",
    type=str,
    default="data/formatted_medialex_transcriptions_vocapia_v1v2_20230301_20230731.csv",
    help="path to dataset",
    required=True
)
parser.add_argument(
    "-threshold",
    type=float,
    default=0.4,
    help="threshold"
)
parser.add_argument(
    "-window_size",
    type=int,
    default=8,
    help="size of the sliding window"
)

class sliding_windows:
    def __init__(self, window_size, embeddings, step=1):
        self.window_size = window_size
        self.embeddings = embeddings
        self.step = step

    def iterate(self):
        start = 0
        N = self.embeddings.shape[0] - self.window_size + 1
        while start < N:
            embedded_texts = self.embeddings[start : start + self.window_size]
            doc_indices = [start, start + self.window_size]  # Index des docs
            yield embedded_texts, doc_indices, start
            start += self.step


def encode_dataset(model, data):
    data["text"] = data.text.str.slice(0, 500)
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
    if not os.path.exists(saving_path.split("/")[0]+"/"):
        os.mkdir(saving_path.split("/")[0]+"/")
    # add a subdirectory to save each 10000 loop docs
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    return embedded_data

def main(model_name, dataset, threshold, window_size):
    begin_tim = time.time()
    data = pd.read_csv(dataset, quoting=csv.QUOTE_ALL)
    path = os.path.join("matrix", dataset.replace(".csv", "").split("/")[-1])
    saving_path = os.path.join("extracted_docs", dataset.replace(".csv", "").split("/")[-1]+"_subsets/")
    model = SentenceTransformer(model_name)
    embedded_data = load_X(path, model, data, saving_path)
    cuda0 = torch.device('cuda:0')
    embedded_data = embedded_data.to(cuda0)
    keywords_embedded = encode_keywords(model, KEY_WORDS)
    print(f"keywords shape : {keywords_embedded.shape}")
    sliding = sliding_windows(window_size, embedded_data)

    extracted_docs = pd.DataFrame(columns=data.columns)
    print(f"embedded data shape : {embedded_data.shape}")
    info_sim = []
    length = int(embedded_data.shape[0]) - (window_size - 1)
    with tqdm.tqdm(total=length) as pbar:
        for batch, indexes, batch_index in sliding.iterate():
            similarity = model.similarity(batch, keywords_embedded)
            avg_sim = torch.mean(similarity, dim=1)
            # print(f"shape of batch : {batch.shape}")
            # print(f"shape of keywords : {keywords_embedded.shape}")
            # print(f"shape of similarity : {similarity.shape}")
            # print(f"shape of similarity{avg_sim.shape}")
            if (avg_sim > threshold).any():
                extracted_docs = pd.concat(
                    [extracted_docs, data.iloc[indexes[0] : indexes[1]]],
                    ignore_index=True,
                )
                info_sim.append(avg_sim)
            extracted_docs.drop_duplicates(inplace=True)
            # save and delete the dataframe on every 10000 iterations to avoid memory issues
            if batch_index % 10000 == 0:
                extracted_docs.to_csv(
                    saving_path
                    + str(batch_index)
                    +"_"
                    + str(threshold)
                    + "_"
                    + dataset.replace(".csv", "").split("/")[-1]
                    + "_extracted_docs.csv",
                    index=False,
                )
                extracted_docs = pd.DataFrame(columns=data.columns)
            pbar.update(1)
    # save the last dataframe of the loop if it exists
    if extracted_docs.shape[0] > 0:
        extracted_docs.to_csv(
            saving_path
            + str(batch_index)
            +"_"
            + str(threshold)
            + "_"
            + dataset.replace(".csv", "").split("/")[-1]
            + "_extracted_docs.csv",
            index=False,
        )
    if len(info_sim) == 0:
        print("aucune similarité superieure au threshold")
    else:
        print(
            "similarité maximum : ",
            max(tensor.max() for tensor in info_sim),
            " similarité moyenne : ",
            sum(torch.mean(tensor, dim=0) for tensor in info_sim) / len(info_sim),
        )

    # Concat all the csv files produced by the script
    fichiers_csv = [f for f in os.listdir(saving_path) if f.endswith(".csv")]
    fichiers_csv.sort(key=lambda x: int(x.split('_')[0])) 

    df_list = [pd.read_csv(os.path.join(saving_path, fichier)) for fichier in fichiers_csv]
    df_final = pd.concat(df_list, ignore_index=True)
    df_final.to_csv(
        "extracted_docs/"
        + str(threshold)
        + "_"
        + dataset.replace(".csv", "").split("/")[-1]
        + "_extracted_docs.csv",
        index=False,
    )
    end_tim = time.time()
    delta = end_tim - begin_tim
    print(f"temps de traitement : {delta} secondes, soit {int(delta // 3600)} heures, {int((delta % 3600) // 60)} minutes, {int(delta % 60)} secondes")



# main(
#     model_name = "Lajavaness/sentence-camembert-large",
#     dataset = "data/formatted_medialex_transcriptions_vocapia_v1v2_20230301_20230731.csv",
#     threshold=0.36,
#     window_size = 8)
args = parser.parse_args()
main(
    model_name= args.model_name,
    dataset= args.dataset,
    threshold= args.threshold,
    window_size= args.window_size
)
