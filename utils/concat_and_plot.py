import pandas as pd
import os
import matplotlib.pyplot as plt
'''
concat all evaluations from evals_output directory
plot a graph of p,r,f1,acc for each threshold
filenames are hardcoded for now
'''

dir = "eval_outputs/"
all = [pd.read_csv(dir+f) for f in os.listdir(dir)] 
par_article = [i for i in all if i.at[0,"pred_file"].split("/")[1].split(".")[0].startswith("labelled_minutes_from_run_encode") is True] #select only pred by otmedia
par_txt_unique = [i for i in all if i.at[0,"pred_file"].split("/")[1].split(".")[0].startswith("minutes_labelled_")] # select only pred with one texte
keys_ex = [i for i in all if i.at[0,"pred_file"].split("/")[1].split(".")[0].startswith("label_keywords")]
keys_ex = pd.concat(keys_ex)
keys_ex["channel"] = [i.split("/")[1].split("_")[0]if "data/" in i else i for i in keys_ex["gold_file"]]
keys_ex["key_words"] = ["_".join(i.split("/")[1].split("_")[:3])  if "data/" in i else i for i in keys_ex["pred_file"]]
keys_ex = keys_ex[keys_ex["key_words"] == "label_keywords_nahel"] # choosing the largest regex

# courbes tous types evaluations selon threshold pour chaque chaines et chaque textes
for pred,name in zip([par_article, par_txt_unique],["par_article","par_texte_unique"]):
    t = [int(i.at[0,"pred_file"].split("/")[1].split("_")[-1].split(".")[0]) for i in pred] # threshold value
    pred = pd.concat(pred)
    pred["t"] = t
    pred = pred.sort_values(by='t')
    # remove global evaluation, then compute individual evaluations for channels (which are not saved).
    pred = pred[pred["gold_file"] != "gold_annotated_files"]

    # add info channel 
    pred["channel"] = [i.split("/")[1].split("_")[0] for i in pred["gold_file"]]
    # add info keyswords and channel on baseline

    for channel, group in pred.groupby('channel'):
        baseline = keys_ex[keys_ex["channel"] == channel]
        plt.figure(figsize=(10, 8))
        for i in [['Accuracy', 'acc','coral'], ['F1 Score', 'f1', 'dodgerblue'], ['Precision', 'p', 'forestgreen'], ['Recall', 'r', 'orange']] :
            line, = plt.plot(group['t'], group[i[1]], label=i[0], color = i[2],marker='o')
            if len(baseline) > 0 :
                # écrase les couleurs et label pour éviter multiples labels/couleurs
                frst = plt.axhline(y = baseline[i[1]].item(), label=line.get_label(), color=line.get_color(), linestyle='dotted')
                frst.set_label('_' + line.get_label()) 
                # scd = plt.axhline(y = baseline[baseline["key_words"] == "refus"][i[1]].item(), label=line.get_label(), color=line.get_color(), linestyle='dashdot')
                # scd.set_label('_' + line.get_label())
        plt.ylim(0.2, 1)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(str(channel).upper()+" "+name+' - Accuracy, F1, Precision, Recall selon threshold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # Sauvegarde des graphs avec nom de chaines
        plt.savefig("graphs/"+str(channel)+name+"courbes_performance_t.png", dpi=300)
        plt.show()


# comparaison résumé vs presse selon threshold pour chaque type evaluation
# pour article : need  pour justifier le choix du seuil vitef
#baseline = keys_ex.groupby("pred_file")[["acc", "f1", "p", "r"]].mean().round(2).reset_index()
global_par_article = [i for i in par_article if (i["gold_file"] == "gold_annotated_files").all() or (i["gold_file"] == "gold_annotated_files/").all()]
global_par_txt_unique = [i for i in par_txt_unique if (i["gold_file"] == "gold_annotated_files").all() or (i["gold_file"] == "gold_annotated_files/").all()]
baseline = keys_ex[keys_ex["gold_file"] == "gold_annotated_files"]
for i in [['Accuracy', 'acc'], ['F1 Score', 'f1'], ['Precision', 'p'], ['Recall', 'r']] :
    plt.figure(figsize=(10,6))
    for pred,name,color in zip([global_par_article, global_par_txt_unique],["articles quotidiens","texte unique"],["orange","dodgerblue"]):
        t = [int(i.at[0,"pred_file"].split("/")[1].split("_")[-1].split(".")[0]) for i in pred] # threshold value
        pred = pd.concat(pred)
        pred["t"] = t
        grouped_mean = pred.groupby("pred_file")[["acc", "f1", "p", "r", "t"]].mean(numeric_only=True).round(2).reset_index()
        grouped_mean = grouped_mean.sort_values(by='t')
        line, = plt.plot(grouped_mean['t'], grouped_mean[i[1]], label=name, color = color,marker='o')
    if len(baseline) > 0 :
        frst = plt.axhline(y = baseline[i[1]].item(), label="mot-clés", color="forestgreen")
        # frst.set_label('_' + line.get_label()) 
    plt.ylim(0.2, 1)
    plt.xlabel('Threshold', fontsize=18)
    plt.ylabel('Score', fontsize=18)
    # plt.title('Articles vs. unique - '+i[0]+' selon threshold', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    # Sauvegarde des graphs avec nom de chaines
    plt.savefig("graphs/"+i[1]+"_articleVSunique.png", dpi=300)
    plt.show()

# comparaison résumé VS presse pour chaque eval (bars) avec le meilleur threshold
# à coder après décision meilleur threshold
