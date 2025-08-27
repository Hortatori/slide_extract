This repo aims to extract texts from TV news transcriptions, selecting those that are semantically similar to a text describing a specific event. (The event(s) here are the death of Nahel Mazrouk and the following riots).

# Select the texts similar to the press articles of the day
By running run_encode.py with the following command 
```python3 run_encode.py``` 
(will run encode_article.py for each day)

# Select the texts similar to only one text describing the event

### embeddings computation on a transcription file

```python encode_articles.py --trs <name_file.csv> --output <directory_name> --meta_file <name_file_meta.csv> --npy_file <name_file_emb.npy> --similarity_file <name_similarity_file.csv> --threshold <float value of a threshold>```

| Parameter            | Description                                                                                             |
|----------------------|---------------------------------------------------------------------------------------------------------|
| `--trs`              | Path to the INA JT transcription file.                                                                  |
| `--output`           | Directory or file path where the results will be saved.                                                 |
| `--meta_file`        | Output file containing metadata after the text is segmented into 1-minute sliding windows.              |
| `--npy_file`         | Output file for storing the text embeddings in `.npy` format.                                           |
| `--similarity_file`  | Output file containing similarity scores computed for each 1-minute window.                             |
| `--threshold`        | Minimum cosine similarity used to extract batches of lines. A higher threshold selects texts more semantically similar to the reference text. |



### JT_ids.py
For running JT_ids.py:    
```python3 JT_ids.py --dataset <data/dataset_filnename.csv>```  
Produces batch of news by gathering documents lines : if the end time of the record line is equal to the start time of the following record line, we consider they belong to the same batch.
* compute number of lines for each news "session" (JT)
* compute duration of each news session
* reorder JT session by time and save the reordered dataset.
* save index of the beginning and end of each session (to be used in future versions of slide.py)
* compute statistics for each channel of news

Will save the following files in data/ (same directory as the dataset file):  
- "idx_docs_pairs_<name>": indexes pairs of beginning and ending of each news bulletin, reordered by time
- "line_<name>": the number of lines by channel
- "time_<name>": the number of seconds by channel
- "stats_<name>": the statistics (nb_JTs, nb_lines, total time, avg time,
    min_time, max_time) by channel  
Will save the reordered dataset in data/reordered

### slide.py
For running slide.py with minimal customization :  
```python3 slide.py --dataset <data/dataset_filnename.csv>```

If you want to custom your run, you can use the following parameters :

```python3 slide.py --dataset <data/dataset_filename.csv> --model "Lajavaness/sentence-camembert-large" --windows 8 --threshold 0.4 --sliding_type JT```

* ```--dataset``` : the name of your INA JT dataset
* ```--model``` : the SentenceBert model you want to use. The embedding of reference texts and dataset texts has to be a Sentence Transformer architecture for now.
* ```--windows``` : windows parameter is the number of documents (equivalent of batches of lines here) which are taken into account when comparing with reference text
* ```--threshold``` : threshold parameter is the minimal cosine similarity for extracting a batch of lines. The higher the threshold, the closer the selected texts will be to the reference text.

This script will :
* Slice embeddings with a sliding windows and compute the cosine similarity between a windows & representatives documents (also embedded).
* Retrieve the documents of a windows if one of the documents mean similarity is above a threshold t.
Save the result in a csv file.

Will save the selected documents in a dedicated directory "extracted_docs", file will be named "<threshold>_<datasetfilename>_extracted_docs.csv"

