## Information extraction on TV news data

This repo aims to extract relevant interventions from TV news transcriptions by selecting those that are semantically similar to a text describing a specific event. (Currently, we focus on the interventions about the death of Nahel Merzouk and the following riots).

### Selection with the press as a reference
Use run_encode.py to select the news bulletins of each day depending of their similarity to the press of the same day, with the following command : 

```python3 run_encode.py --threshold <float>```

| Parameter            | Description                                                                                             |
|----------------------|---------------------------------------------------------------------------------------------------------|
| `--threshold`        | Minimum cosine similarity used to extract batches of lines. A higher threshold selects texts more semantically similar to the reference text. |                                                                  |

This script will :

* select the bulletins transcription of one day
* run encode_articles.py for each day
* all labelled transcriptions of each days are concatenated and saved
* an optionnal formatting is applied in case only the positive selection is needed

### Selection with a text as a reference

Use encode_article.py to select all news bulletins depending of their similarity to ONE text describing the event, withe following command :

```python3 encode_articles.py --trs <name_file.csv> --output <directory_name> --meta_file <name_file_meta.csv> --npy_file <name_file_emb.npy> --similarity_file <name_similarity_file.csv> --threshold <float value of a threshold>```

| Parameter            | Description                                                                                             |
|----------------------|---------------------------------------------------------------------------------------------------------|
| `--trs`              | Path to the INA JT transcription file.                                                                  |
| `--output`           | Directory or file path where the results will be saved.                                                 |
| `--meta_file`        | Output file containing metadata after the text is segmented into 1-minute sliding windows.              |
| `--npy_file`         | Output file for storing the text embeddings in `.npy` format.                                           |
| `--similarity_file`  | Output file containing similarity scores computed for each 1-minute window.                             |
| `--threshold`        | Minimum cosine similarity used to extract batches of lines. A higher threshold selects texts more semantically similar to the reference text. |

This script will :
* calculates a one-minute sliding window, advancing line by line, resetting to zero if the channel changes.
* compute an embedding for each minute
* compute one embedding for each article of the day
* calculates the similarities between each one minute embedding and each article embedding
* if the mean of these similarities is above a (chosen) threshold, the minute is labelled as 1, if not, 0
* the original transcription dataset is then labelled using the minutes' labels, and saved

## Evaluation

Attribute a gold value for each minute depending of the overlap with annotated intervals, then compute evaluation scores between prediction label and attributed gold.

```--gold_path <directory> --extract_path <path>```

| Parameter            | Description                                                                                             |
|----------------------|---------------------------------------------------------------------------------------------------------|
| `--gold_path`        | Directory where are all annotated files                                                                 |
| `--extract_path`     | Directory or file path where are prediction (labelled minutes by encode_articles).                      |

