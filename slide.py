import numpy as np
import pandas as pd
import csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

KEY_WORDS = KEYWORDS = [
    "jeune conducteur",
    "affrontements",
    "refus d'obtempérer",
    "émeutes",
    "émeutiers",
    "cagnotte",
    "l'haÿ-les-roses",
    "lhaylesroses",
    "violences",
    "nanterre",
    "clichy-sous-bois",
    "naël",
    "nahel",
]
# KEY_WORDS = KEYWORDS = [
#     "incendies",
#     "feux",
# ]

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
            yield embedded_texts, doc_indices
            start += self.step


def encode_dataset(model, data):
    data["text"] = data.text.str.slice(0, 500)
    # embbed_texts = np.array([model.encode(text) for text in tqdm(data.text.tolist(), desc="Encoding texts")])

    embbed_texts = np.array(model.encode(data.text.tolist(), show_progress_bar=True))
    return np.array(embbed_texts)

def encode_keywords(model, keywords):
    vectors = np.array(model.encode(keywords))
    return vectors

# reste calculer la similarité cos, trouver comment comparer vecteur embeddings phrase quand ils sont groupé par batch
# reste l'extraction de docs à faire
# tfiidf plus économique mais problématique pour des mots-clés
def sklearn_cosine(x, y):
    return cosine_similarity(x, y)

def main(model_name, dataset, threshold, window_size):
    data = pd.read_csv(dataset, quoting=csv.QUOTE_ALL)
    model = SentenceTransformer(model_name)
    path = os.path.join(dataset.replace(".csv", ""))
    if os.path.exists(os.path.join(path + ".npy")):
        print(os.path.join(path + ".npy"))
        embedded_data = np.load(path+".npy")
    else: 
        embedded_data = encode_dataset(model, data)
        print(embedded_data.shape, path)
        np.save(path, embedded_data)

    keywords_embedded = encode_keywords(model, KEY_WORDS)
    sliding = sliding_windows(window_size, embedded_data)

    extracted_docs = pd.DataFrame(columns=data.columns)
    print(embedded_data.shape)
    for batch, indexes in sliding.iterate():
        similarity = cosine_similarity(batch, keywords_embedded)
        avg_sim = np.mean(similarity)
        print(indexes)
        if avg_sim > threshold:
            print(avg_sim)
            # print(data[indexes[0]:indexes[1]])
            extracted_docs = pd.concat([extracted_docs,data.iloc[indexes[0]:indexes[1],:]], ignore_index=True)
    extracted_docs.drop_duplicates(inplace=True)
    print(extracted_docs)
    extracted_docs.to_csv("extracted_docs.csv", index=False)


main(model_name = "Lajavaness/sentence-camembert-large", dataset = "data/formatted_medialex_transcriptions_vocapia_v1v2_20230301_20230731.csv", threshold=0.30, window_size = 8)
# main(model_name = "Lajavaness/sentence-camembert-large", dataset = "data/test_corpus.csv", threshold=0.30, window_size = 8)
