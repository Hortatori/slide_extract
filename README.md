These scripts extract texts from TV news bulletin transcriptions, selecting those that are sufficiently similar to a reference text describing a specific event.
(The event(s) here are the death of Nahel Mazrouk and the following riots).
### slide.py
* Slice embeddings with a sliding windows and compute the cosine similarity between a windows & representatives documents (also embedded).
* Retrieve the documents of a windows if one of the documents mean similarity is above a threshold t.
Save the result in a csv file.

The embedding of the key words and dataset texts has to be a Sentence Transformer architecture for now.
Will save the selected documents in a dedicated directory "extracted_docs", file will be named "<threshold>_<datasetfilename>_extracted_docs.csv"

### JTmeantime.py
Produces batch of news by gathering documents lines : if the end time of the record line is equal to the start time of the following record line, we consider they belong to the same batch.
* compute number of lines for each news "session" (JT)
* compute duration of each news session
* save index of the beginning and end of each session (to be used in future versions of slide.py)
* compute statistics for each channel of news
Save the results in four csv files.

Will save the following files in the same directory as the
dataset file:
- "idx_docs_pairs_<name>": indexes pairs of beginning and ending of each news bulletin
- "line_<name>": the number of lines by channel
- "time_<name>": the number of seconds by channel
- "stats_<name>": the statistics (nb_JTs, nb_lines, total time, avg time,
    min_time, max_time) by channel

