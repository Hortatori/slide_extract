import pandas as pd
import argparse
import re
# pas d heure de fin dans le fichier donc calcul
# fin du segment = heure + duree
# --notice_path data/fr2_annotated_notices_27_06_03_07.csv --extract_path matrix/nrv_extracted_docs_030.csv 
parser = argparse.ArgumentParser()
parser.add_argument('--notice_path', required=True, help="path and file of notice")
parser.add_argument('--extract_path', required=True, help="path and file of extracted docs")

args = parser.parse_args()
if args.extract_path.split('_')[0] == 'matrix/nrv' :
    df = pd.read_csv(args.extract_path, dtype = {"channel": str, "start": str, "end": str , "start_id": int, "end_id": int, "text": str, "label": int})
    df ["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
else:
    # my output has a different time encoding
    df = pd.read_csv(args.extract_path)
    df ["start"] = pd.to_datetime(df["start"], format="%d/%m/%Y %H:%M:%S")
    df["end"] = pd.to_datetime(df["end"], format="%d/%m/%Y %H:%M:%S")
notice = pd.read_csv(args.notice_path)
channel = notice.at[0,"ch_code"]
if channel == 'FR2' : channel = 'France2'
print(type(channel))

df = df[df['channel'].apply(lambda x: bool(re.search(channel, x)))]
notice = notice.dropna(subset=['Durée','Heure','Description'])
notice['start'] = notice['date']+' '+notice['Heure']
notice["start"] = pd.to_datetime(notice["start"], format='%d/%m/%Y %H:%M:%S')
notice["Durée"] = pd.to_timedelta(notice["Durée"])
notice["only_for_max_time_computation"] = notice["start"]+notice["Durée"]
print(f"sur le total des jours, minimum heure { notice['start'].min()} :: maximum heure {notice['only_for_max_time_computation'].max()}")
notice['datetime_str'] = notice['date'].astype(str) + ' ' + notice['Heure'].astype(str)
# overwrite start and end to get correct dates

notice["end"] = notice["start"]+notice["Durée"]
# notice['start'] = pd.to_datetime(notice['datetime_str'], format='%d/%m/%Y %H:%M:%S')
# notice['end'] = notice["start"]+notice["Durée"]
# notice = notice.drop(['only_for_max_time_computation', 'datetime_str'], axis=1)


print(notice)

# maybe to do faire une loop sur chaque date (groupby) plutôt que tout le dataset où y aura des vides les mauvais jours.
print()
# attribuer une notice à chaque ligne de pred : celle avec le plus gros chevauchement?

def compute_overlap(row_pred, notice_corpus):


    # Calcul du recouvrement pour les chevauchements détectés
    total_overlap = pd.Timedelta(0) # pas vraiment utile, voir trompeur ? le temps de recouvrement sur l'ensemble des notices parcourues
    total_intersections = 0
    rld = {'row':[],'delta':[],'ratio_delta':[],'label':[]}
    for _, notice_c in notice_corpus.iterrows():
        latest_start = max(row_pred['start'], notice_c['start'])
        earliest_end = min(row_pred['end'], notice_c['end'])
        delta = (earliest_end - latest_start)

        earlier_start = min(row_pred['start'], notice_c['start'])
        lastest_end = max(row_pred['end'], notice_c['end'])
        largest = lastest_end - earlier_start
        # if row_pred['start'] >= notice_c['start'] and row_pred['end'] <= notice_c['end'] :
        #     # print(earliest_end - latest_start)
        #     print("row_pred est complètement inclus dans notice, index", _)
        #     # print(delta.total_seconds()/ (notice_c['end'] - notice_c['start']).total_seconds())
        # if delta.total_seconds() > 0 and delta.total_seconds() <= 10 :
        #     print("::WEAK OVERLAP::")
        #detection d'un chevauchement
        if delta.total_seconds() > 0:
            total_overlap += delta
            total_intersections += 1
            # print("NEW ROW")
            # print("gold start ", row_pred["start"],":: end ",row_pred["end"])
            # print("pred start ", notice_c["start"],":: end ",notice_c["end"])
            # print("OVERLAP","index", _)
            # print("PRED index", row_pred.name)
            # print("ratio du chevauchement sur la durée pred",delta.total_seconds()/ (row_pred['end'] - row_pred['start']).total_seconds(), ":: durée de pred :", row_pred['end'] - row_pred['start'],":: START TIME PRED", row_pred['start'], ":: END TIME PRED", row_pred['end'] )
            # print("ratio du chevauchement sur la durée notice",delta.total_seconds()/ (notice_c['end'] - notice_c['start']).total_seconds(), ":: durée de notice:", notice_c['end'] - notice_c['start'], ":: NOTICE START", notice_c['start'], "NOTICE END", notice_c['end'])
            # print("ratio du chevauchement sur la durée des deux",delta.total_seconds()/ largest.total_seconds())

            # if notice_c["label"] != row_pred["label"] :
            #     print("--------------error label----------------")
            #     print("DURATION ", row_pred['end']-row_pred['start'], "START TIME PRED", row_pred['start'], "END TIME PRED", row_pred['end'] )
            #     print("TEXT ",row_pred["text"])
            #     print()
            #     print("DURATION ", notice_c['end']-notice_c['start'], "NOTCE START", notice_c['start'], "NOTICE END", notice_c['end'])
            #     print("GOLD", notice_c["chap"])
            #     print("--------------end of error label----------------")

            # print(f"intersections = {total_intersections} :: row delta = {delta} :: label pred {row_pred['label']} in pred row {row_pred.name}:: label gold {notice_c['label']} for notice row {_}")

            print()
            # print(f"intersections = {total_intersections} :: row delta = {delta} :: label gold {row_pred['label']} :: label pred {notice_c['label']}")
            # print(f"start noti {row_pred['start']}\nstart pred {notice_c['start']}\nend noti {row_pred['end']}\nend pred {notice_c['end']}")
            rld['row'].append(_)
            rld['delta'].append(delta)
            rld['ratio_delta'].append(delta.total_seconds()/ (row_pred['end'] - row_pred['start']).total_seconds())
            rld['label'].append(notice_c['label'])

    return total_overlap.total_seconds(), total_intersections, rld 

# def day() :
#     for day in notice(day):
#         min and max for this day in notice
#         for row in df :
#             if min >= min_notice and max < max_notice
#             compute_overlap()


# for day in list of day of df and notice :
#     groupby by day for df and notice
#     day(df, notice)
    # dans un df qui rassemble à la fois nahel_transcripion et le docs extrait
    # overlaps permet identifier le (les) meilleurs row candidats
    # plusieur candidats possibles
    # le meilleur candidat du row df (plus gros overlap) est associé au row de notice dans nouveau df  


# qui doit avoir une colonne label avec 1 si présent dans extraction et 0 sinon 
# crer cette colonne au moment de la generation des docs extracted : en conservant tout et en mettant 0 dans une colonne label ou 1 si >threshold
# OK pour création de fichier avec slmt des labels

def choose_df_to_overlaps(fixed_df, running_df) :
    overlap_duration = list()
    attributed_gold = list()
    i = 0
    for _, row in fixed_df.iterrows():


        total_overlap, intersection_n, rld = compute_overlap(row, running_df)
        overlap_duration.append(total_overlap)
        # print(f"for pred row {_}, number of notice rows with intersection : {intersection_n}")
        if total_overlap > 0.0 :
            # rappel : ce sont des fenêtres glissantes, donc elles se chevauchent plusieurs fois sur les memes rangs de notices
            i += 1
        if len(rld['label']) > 1 :

            # print(intersection_n, " intersection pour row", _, "of pred")
            # print(rld)
            # probleme max : si égalité il choisit le premier (ptet pas important, ratio est à 10 decimales pres) 
            max_index = rld['ratio_delta'].index(max(rld['ratio_delta']))
            selected_label = int(rld['label'][max_index])
            # print('label pred ', row['label'], ':: label gold attributed', selected_label)
            # print()
        elif len(rld['label']) == 0 :
            # print("no overlap")
            selected_label = None
        else :
            selected_label = int(rld['label'][0])
            # print('label pred ', row['label'], ':: label gold attributed', selected_label)
            # print()
        attributed_gold.append(selected_label)
        print(f"intersections = {intersection_n} :: {rld} :: in pred row {_}")
    fixed_df['overlap_duration'] = overlap_duration # not a ratio, time od overlap for each minute row
    fixed_df['attributed_gold'] = attributed_gold
    print("nb of minute rows with overlaps",i)
    # print(overlap_duration)
    print(fixed_df)
    fixed_df.to_csv("test_eval.csv")

choose_df_to_overlaps(fixed_df = df, running_df = notice)
