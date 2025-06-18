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
"""
Parameters
* The SentenceBert model for representing text
* The threshold for distance cosinus
* Size of sliding window 
* Step of the slide 
* The dataset
each JTs batch, write in a csv with the current results, then concat all to produce the complete results
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

# test with another event
# KEY_WORDS = [
#     "L'accident du submersible Titan se produit lors d'une plongée dans les eaux internationales de l'océan Atlantique Nord au large de Terre-Neuve (Canada)",
# "Le Titan est un petit submersible à visée touristique, exploité par OceanGate et destiné particulièrement à assurer des visites payantes de l'épave du Titanic. Le 18 juin 2023, il amorce une descente en direction de l'épave au cours de laquelle il subit une implosion entraînant sa destruction et la mort de ses cinq occupants. Des débris du Titan sont retrouvés par 3 800 m de fond, non loin de l'épave du Titanic. Le 28 juin 2023, des restes humains sont retrouvés parmi des débris remontés à la surface.",
# "C'est l'accident sous-marin mortel le plus profond de l'Histoire[1]."
# ]

# test with wiki hat
# KEY_WORDS = [
#     "La mort de Nahel Merzouk, un adolescent franco-algérien de 17 ans, est causée par le tir à bout portant d'un policier le 27 juin 2023 lors d'un contrôle routier à Nanterre dans les Hauts-de-Seine (Île-de-France). Deux autres adolescents, âgés de 14 et 17 ans, sont passagers à bord de la voiture. La version policière, celle d'une voiture refusant un contrôle avant de foncer sur un fonctionnaire de police qui a ouvert le feu dans son bon droit, est initialement reprise par les médias, mais contredite dans les heures qui suivent par les témoignages des deux passagers. La victime est également présentée à tort dans plusieurs médias comme ayant un casier judiciaire, allégations qui sont démenties par la suite, son nom ne figurant qu'au fichier des antécédents judiciaires. Le 29 juin, le policier Florian M. est mis en examen pour homicide volontaire et placé en détention provisoire avant d'être remis en liberté quelques mois plus tard. L'événement provoque des émeutes dans de nombreuses villes françaises ainsi qu'en Belgique et en Suisse, dont le bilan des dégâts et de la répression dépasse celui des émeutes de 2005. Cette affaire relance le débat sur les violences policières, la question du racisme au sein de la police française et son usage des armes à feu, ainsi que son traitement par les médias qui se sont d'abord appuyés sur des sources policières. Elle provoque de nombreuses réactions en France de personnalités politiques, sportives, artistiques et religieuses, ainsi que de gouvernements étrangers et de l'Organisation des Nations unies. Une marche blanche et des cagnottes sont par ailleurs organisées."
#     ]
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
    "--dataset",
    type=str,
    default="data/nahel_transcriptions_vocapia_26_06_2023_to_03_07_2023.csv",
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

def load_X(path, model, data, saving_path_subsets):
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
    if not os.path.exists(saving_path_subsets.split("/")[0]+"/"):
        os.mkdir(saving_path_subsets.split("/")[0]+"/")
    # add a subdirectory to save each loop docs
    if not os.path.exists(saving_path_subsets):
        os.mkdir(saving_path_subsets)
    return embedded_data

class time_sliding():
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

def main(model_name, dataset, threshold, window_size, sliding_type):
    begin_tim = time.time()
    matrix_path = os.path.join("matrix", dataset.replace(".csv", "").split("/")[-1]+"_"+model_name.replace("/", "_"))



    # checking if reordering has been done, if not, call JTtime
    if not os.path.exists(dataset.split("/")[0] + "/reordered/" + dataset.split("/")[1]):
        print("computing JTs indexes of ", dataset, "and reordering depending of time")
        try :
            full_command = f"python3 JT_ids.py --dataset {dataset}"
            subprocess.run(full_command, shell = True, check=True)
        except OSError as err:  
            print(err)
            sys.exit(1)
    if sliding_type == "time" :
        #if sliding_type == "time", we are using the reordered dataset, if not, we are using the reordered indexes  (sliding_time == "JTs")
        dataset = dataset.split("/")[0] + "/reordered/" + dataset.split("/")[1]

    data = pd.read_csv(dataset, quoting=csv.QUOTE_ALL)
    df_idx_paires = pd.read_csv(dataset.split("/")[0]+"/idx_docs_pairs_"+dataset.split("/")[-1], quoting=csv.QUOTE_ALL)
    saving_path_subsets = os.path.join("extracted_docs", dataset.replace(".csv", "").split("/")[-1]+"_subsets/")
    # AliBaba model need a SentenceTransformer(model_name, trust_remote_code=True) security issue if they change the config file (but huggingface should verify it)
    sim_history = {"similarity":[],"start_time":[], "channel":[]}
    model = SentenceTransformer(model_name)
    truncation = int(model.max_seq_length) - 2
    model.max_seq_length = truncation
    embedded_data = load_X(matrix_path, model, data, saving_path_subsets)
    cuda0 = torch.device('cuda:0')
    embedded_data = embedded_data.to(cuda0)
    keywords_embedded = encode_keywords(model, KEY_WORDS)
    print(f"keywords shape : {keywords_embedded.shape}")
    # creating a spare index to keep when extracting documents
    data = data.reset_index()

    extracted_docs = pd.DataFrame(columns=data.columns)
    print(f"embedded data shape : {embedded_data.shape}")
    info_sim = []
    print(f"sliding_type {sliding_type}")
    # sliding windows across the entire dataset without distinguishing between JTs
    if sliding_type == "time":
        sliding = time_sliding(window_size, embedded_data)
        length = int(embedded_data.shape[0]) - (window_size - 1)
        with tqdm.tqdm(total=length) as pbar:
            for batch, indexes, batch_index in sliding.iterate():
                similarity = model.similarity(batch, keywords_embedded)
                avg_sim = torch.mean(similarity)
                sim_history["similarity"].append(avg_sim)
                sim_history["start_time"].append(data.at[indexes[0],"start"])
                sim_history["channel"].append(data.at[indexes[0],"channel"])
                if (avg_sim > threshold).any():
                    if len(extracted_docs) != 0:
                        extracted_docs = pd.concat(
                            [extracted_docs, data.iloc[indexes[0] : indexes[1]]],
                            ignore_index=True,
                        )
                    else:
                        extracted_docs = data.iloc[indexes[0] : indexes[1]]
                    info_sim.append(avg_sim)
                extracted_docs.drop_duplicates(inplace=True)
                # save and delete the dataframe on every 10000 iterations to avoid memory issues
                if batch_index % 10000 == 0:
                    extracted_docs.to_csv(
                        saving_path_subsets
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
                    saving_path_subsets
                    + str(batch_index)
                    +"_"
                    + str(threshold)
                    + "_"
                    + dataset.replace(".csv", "").split("/")[-1]
                    + "_extracted_docs.csv",
                    index=False,
                )
    # sliding windows on each JT at a time
    else :
        keep_oldID = set()
        for row in df_idx_paires.itertuples():
            w_slide = JT_sliding(row[1], row[2], window_size, embedded_data)
            length = int(row[2]- (window_size-1)) - (int(row[1]))
            with tqdm.tqdm(total=length) as pbar:
                for batch, indexes, batch_index in w_slide.iterate():
                    similarity = model.similarity(batch, keywords_embedded)
                    avg_sim = torch.mean(similarity, dim=1)
                    sim_history["similarity"].append(avg_sim)
                    sim_history["start_time"].append(data.at[indexes[0],"start"])
                    sim_history["channel"].append(data.at[indexes[0],"channel"])

                    if (avg_sim > threshold).any():
                        current_old_ids = range(row[1], row[2])
                        keep_oldID.update(current_old_ids)
                        if len(extracted_docs) != 0:
                            extracted_docs = pd.concat(
                                [extracted_docs, data.iloc[indexes[0] : indexes[1]]],
                                ignore_index=True,
                            )
                        else:
                            extracted_docs = pd.DataFrame(data.iloc[indexes[0] : indexes[1]])
                        info_sim.append(avg_sim)
                    extracted_docs.drop_duplicates(inplace=True)
                    pbar.update(1)

                # save and delete the dataframe at end of batch to avoid memory issues
                if len(extracted_docs.index) > 0 :
                    extracted_docs.to_csv(
                        saving_path_subsets
                        + str(batch_index)
                        +"_"
                        + str(threshold)
                        + "_"
                        + dataset.replace(".csv", "").split("/")[-1]
                        + "_extracted_docs.csv",
                        index=False,
                    )
                    extracted_docs = pd.DataFrame(columns=data.columns)

    if len(info_sim) == 0:
        print("aucune similarité superieure au threshold")
    else:
        print(
            "similarité maximum : ",
            max(tensor.max() for tensor in info_sim),
            " similarité moyenne : ",
            sum(torch.mean(tensor, dim=0) for tensor in info_sim) / len(info_sim),
        )
        # Concat all the csv files produced by the script in the subset directory
        fichiers_csv = [f for f in os.listdir(saving_path_subsets) if f.endswith(".csv")]
        fichiers_csv.sort(key=lambda x: int(x.split('_')[0])) 
        df_old_ids = pd.DataFrame(list(keep_oldID))
        df_old_ids.to_csv("extracted_docs/"+ "IDs_"+str(threshold)+ "_"+ sliding_type+ "_"+ dataset.replace(".csv", "").split("/")[-1]+ "_"+ model_name.replace("/", "_")+ ".csv",index=False)
        df_list = [pd.read_csv(os.path.join(saving_path_subsets, fichier)) for fichier in fichiers_csv]
        df_final = pd.concat([df for df in df_list if len(df.index) > 0], ignore_index=True)
        df_final.to_csv(
            "extracted_docs/"
            + str(threshold)
            + "_"
            + sliding_type
            + "_"
            + dataset.replace(".csv", "").split("/")[-1]
            + "_"
            + model_name.replace("/", "_")
            + "_extracted_docs"
            + ".csv",
            index=False,
        )
    # delete subset directory after concat alls subset files
    shutil.rmtree(saving_path_subsets)
    end_tim = time.time()
    
    sim_history["similarity"] = [i[0].detach().item() for i in sim_history["similarity"]]
    rg = range(0, len(sim_history["similarity"]))
    df = pd.DataFrame(sim_history, index=rg)
    df['start_time'] = pd.to_datetime(df['start_time'], dayfirst=True, format="mixed")
    df.to_csv("suite_similarite"
            + "_"
            + sliding_type
            + "_"
            + dataset.replace(".csv", "").split("/")[-1]
            + "_"
            + model_name.replace("/", "_")
            + ".csv",)

    delta = end_tim - begin_tim
    print(f"temps de traitement : {delta} secondes, soit {int(delta // 3600)} heures, {int((delta % 3600) // 60)} minutes, {int(delta % 60)} secondes")


args = parser.parse_args()
#TODO :
# a slicing with time 

main(
    model_name= args.model_name,
    dataset= args.dataset,
    threshold= args.threshold,
    window_size= args.window_size,
    sliding_type = args.sliding_type
)
