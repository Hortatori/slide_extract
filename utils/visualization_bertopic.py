from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("stopwords")
stopwords = stopwords.words('french')
df = pd.read_csv("extracted_docs/0.42_JT_nahel_transcriptions_vocapia_26_06_2023_to_03_07_2023_Lajavaness_sentence-camembert-large_extracted_docs.csv")

sentence_model = SentenceTransformer("Lajavaness/sentence-camembert-large")
topic_model = BERTopic(embedding_model=sentence_model)
vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words=stopwords)

truncation = int(sentence_model.max_seq_length) - 2
sentence_model.max_seq_length = truncation
i = 0
for channel, group in df.groupby('channel'):

    embeddings = sentence_model.encode(group["text"].to_list(), show_progress_bar=True, convert_to_numpy=True)
    print(channel)
    # bertopic want embedding on cpu and in npy array
    # if embeddings.is_cuda:
    #     embeddings = embeddings.cpu()
    # embeddings = embeddings.to_numpy()
    try :
        topics, probs = topic_model.fit_transform(group["text"].to_list(), embeddings)
        topic_model.update_topics(group["text"].to_list(), vectorizer_model=vectorizer_model)
        fig = topic_model.visualize_topics()
        fig.write_html("utils/topics/"+channel+"_topics.html")
    except Exception as e:
        print(e)
        pass
