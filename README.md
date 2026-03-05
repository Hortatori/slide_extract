## Information Extraction from TV News Data

This repository aims to extract relevant interventions from TV news transcriptions by selecting segments that are semantically similar to a text describing a specific event. Two pipelines allows to compare the performances of a summary text vs. daily press.

Currently, the project focuses on interventions related to the death of Nahel Merzouk and the riots that followed.

---

### Data

The pipeline expects the following input files:

- **TV news transcription file (`--trs`)**  a CSV file containing the transcription of TV news broadcasts.

    | channel | start | end | duration | text | id | created_at |
    |---------|-------|-----|----------|------|----|------------|
    | CNews | 27/06/... | 27/06/... | 24.32 | Et la Cour des comptes… | CNews27/06... | Tue Jun 27... |
    | CNews | 27/06/... | 27/06/... | 3.79 | On est en direct… | CNews27/06... | Tue Jun 27... |

- **Press articles file (`--articles`)**  a CSV file containing press articles used as reference texts.

    | docTime | media | title | text|
    |---------|-------|-----|----------|
    | 2023-06-27T.. | LE_REPUBLICAIN_LORRAIN | Hauts-de-Seine. Un homme... | Hauts-de-Seine Un homme... |
    | 2023-06-27T.. | 20_MINUTES_SUISSE | France: Il refuse d’obtempérer... | Il refuse d’obtempérer... |



### Structure

```
├── run_encode.py
├── encode_articles.py
└── articles/
    ├── transcription_of_tvnews.csv
    └── press.csv
└── data/
    ├── transcription_of_tvnews.csv
└── summaries/
    ├── summary_of_event.txt
└── utils/
    ├── eval.py
gold_annotated_files/
```


---

### Selection Using the Press as a Reference

Use `run_encode.py` to select news bulletins for each day based on their semantic similarity to press articles published on the same day.

```bash
python3 run_encode.py --threshold <float> --trs data/transcription_of_tvnews.csv --articles data/press.csv --start 27-06-2023 --end 03-07-2023
```

| Parameter | Description |
|-----------|-------------|
| `--threshold` | Minimum cosine similarity used to extract segments of text. |
| `--trs` | Path to the INA TV news transcription file. |
| `--articles` | Path to the press articles file. |
| `--start` | Start date (format: `day-month-year`). |
| `--end` | End date (format: `day-month-year`). |

This script will:

- split the news transcription and press files by day  
- run `encode_articles.py` for each day  
- compare press articles with TV and radio news from the same day  
- concatenate all labelled transcriptions into a single output file  

---

### Selection Using a Single Reference Text

Use `encode_articles.py` to select news bulletins based on their similarity to **one reference text describing the event**.

```bash
python3 encode_articles.py --trs <file.csv> --output <directory> --meta_file <meta.csv> --npy_file <embeddings.npy> --similarity_file <similarity.csv> --threshold <float>
```

| Parameter | Description |
|-----------|-------------|
| `--trs` | Path to the INA TV news transcription file. |
| `--output` | Directory where the results will be saved. |
| `--meta_file` | Metadata for 1-minute sliding windows. |
| `--npy_file` | File storing text embeddings. |
| `--similarity_file` | File storing similarity scores. |
| `--threshold` | Minimum cosine similarity used for selection. |

This script will:

- compute a **1-minute sliding window** over the transcription  
- compute embeddings for each window  
- compute embeddings for the reference articles  
- calculate similarity scores  
- label segments as **1** (relevant) or **0** (non-relevant)  
- save the labelled transcription dataset  

---

### Evaluation

Gold labels are assigned to each minute based on overlap with annotated intervals. Evaluation scores are then computed between predicted and gold labels.

Arguments:

```
--gold_path <directory> --extract_path <path>
```

| Parameter | Description |
|-----------|-------------|
| `--gold_path` | Directory containing annotated files. |
| `--extract_path` | Directory containing prediction outputs. |

---

### Example Output

Example of labelled transcription:

| channel | start | end | duration | text | id | created_at | label |
|---------|-------|-----|----------|------|----|------------|-------|
| CNews | 27/06/... | 27/06/... | 24.32 | Et la Cour des comptes… | CNews27/06... | Tue Jun 27... | 0 |
| CNews | 27/06/... | 27/06/... | 3.79 | On est en direct… | CNews27/06... | Tue Jun 27... | 0 |
