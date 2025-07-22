import pandas as pd
import tqdm as tqdm
import argparse

"""
utilise les indexes de début et fin de chaque fenêtre de minute (créés lors de la génération du fichier meta)
selectionne les lignes de ces mêmes indexes dans la transcription
ajoute les lignes contenant les mots clés refus d'obtempérer, pour le 27/06/2023
trie temporellement et enregistre
"""
def main(args) :
    extracted_docs= pd.read_csv(args.extracted, dtype = {"channel": str, "start": str, "end": str , "start_id": int, "end_id": int, "text": str, "label": int})
    extracted_docs ["start"] = pd.to_datetime(extracted_docs["start"])
    extracted_docs["end"] = pd.to_datetime(extracted_docs["end"])
    df = extracted_docs[extracted_docs['label'] == 1]
    # df = extracted_docs
    ids_set = set()
    for _, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        # ids_range = range(row['start_id'], row['end_id']+1)  # +1 pour end_id inclu
        # ids_set.update(ids_range)
        ids_set.add(row['start_id'])
    print(len(ids_set))
    transcription = pd.read_csv("data/nahel_transcriptions_vocapia_27_06_2023_to_03_07_2023.csv")
    trs_select = transcription[transcription.index.isin(ids_set)]
    trs_select["start"] = pd.to_datetime(trs_select["start"], format="%d/%m/%Y %H:%M:%S")
    trs_select["end"] = pd.to_datetime(trs_select["end"], format="%d/%m/%Y %H:%M:%S")

    # selection des lignes contenant 'refus d'obtemperer' le jour du 27/06/2023
    # command : xan search -s start '27/06/2023' data/nahel_transcriptions_vocapia_27_06_2023_to_03_07_2023.csv | xan search -s text -r "\brefus\s+(de\sobtempérer|d'obtempérer|d\sobtempérer)\b" > selected_refus.csv 
    refus_labelled = pd.read_csv("selected_refus.csv")
    refus_labelled["start"] = pd.to_datetime(refus_labelled["start"], format="%d/%m/%Y %H:%M:%S")
    refus_labelled["end"] = pd.to_datetime(refus_labelled["end"], format="%d/%m/%Y %H:%M:%S")
    print(f"trs shape {trs_select.shape}, keywords shape {refus_labelled.shape}")
    output = pd.concat([trs_select,refus_labelled])
    output = output.sort_values(by=["channel", "start"])

    ch_interest = {'Europe1' : 'Xdroite',
                   'CNews' : 'Xdroite',
                   'FranceInfo_TV' : 'public',
                   'FranceInfo_RD' : 'public',
                   'France2' : 'public',
                   'BFM_TV' : 'apo_prive',
                   'TF1' : 'apo_prive',
                   'ARTE' : 'gauche',
                   'FranceInter' : 'gauche'}
    # y a pas SudRadio dans les données
    channels_to_keep = ch_interest.keys()
    output = output[output['channel'].isin(channels_to_keep)]
    output['tag'] = output['channel'].map(ch_interest)

    output.to_csv('output_'+args.extracted.split('/')[1])

    print(output.shape)

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "--extracted",
    type=str,
    default="matrix/nrv_extracted_docs_040.csv",
    help="path and file to the chosen extracted doc"   
)
args = parser.parse_args()
main(args)