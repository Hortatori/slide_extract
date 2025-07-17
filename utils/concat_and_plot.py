import pandas as pd
import os
import matplotlib.pyplot as plt

dir = "eval_outputs/"
all = [pd.read_csv(dir+f) for f in os.listdir(dir)] 

sim_ex = [i for i in all if i.at[0,"pred_file"].split(".")[0].endswith("2023") is False]
t = [int(i.at[0,"pred_file"].split("/")[1].split("_")[-1].split(".")[0]) for i in sim_ex]
sim_ex = pd.concat(sim_ex)
sim_ex["t"] = t
keys_ex = [i for i in all if i.at[0,"pred_file"].split(".")[0].endswith("2023") is True]
keys_ex = pd.concat(keys_ex)

sim_ex = sim_ex.sort_values(by='t')

# add info channel 
channel = [i.split("/")[1].split("_")[0] for i in sim_ex["gold_file"]]
sim_ex["channel"] = channel
# add info keyswords and channel on baseline
channel = [i.split("/")[1].split("_")[0] for i in keys_ex["gold_file"]]
type_of_keyword = [i.split("/")[1].split("_")[1] for i in keys_ex["pred_file"]]
keys_ex["channel"] = channel
keys_ex["key_words"] = type_of_keyword


print(sim_ex)
print(keys_ex)
# plt.style.use('ggplot')
for channel, group in sim_ex.groupby('channel'):
    baseline = keys_ex[keys_ex["channel"] == channel]

    plt.figure(figsize=(10, 6))
    for i in [['Accuracy', 'acc','coral'], ['F1 Score', 'f1', 'dodgerblue'], ['Precision', 'p', 'forestgreen'], ['Recall', 'r', 'orange']] :
        line, = plt.plot(group['t'], group[i[1]], label=i[0], color = i[2])
        if len(baseline) > 0 :
            # écrase les couleurs et label pour éviter multiples labels/couleurs
            frst = plt.axhline(y = baseline[baseline["key_words"] == "keyword"][i[1]].item(), label=line.get_label(), color=line.get_color(), linestyle='dotted')
            frst.set_label('_' + line.get_label()) 
            scd = plt.axhline(y = baseline[baseline["key_words"] == "refus"][i[1]].item(), label=line.get_label(), color=line.get_color(), linestyle='dashdot')
            scd.set_label('_' + line.get_label())


    plt.ylim(0.2, 1)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(str(channel).upper()+' - Accuracy, F1, Precision, Recall selon threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # Sauvegarde des graphs avec nom de chaines
    plt.savefig(str(channel)+"courbes_performance_t.png", dpi=300)
    plt.show()

