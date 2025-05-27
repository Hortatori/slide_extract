import matplotlib.pyplot as plt
import pandas as pd
import os
#INFO : remember the time orders is not coherent : different JT are overlapping when plotting more than one channel at a time

df = pd.read_csv("suite_similarite_JT_nahel_transcriptions_vocapia_26_06_2023_to_03_07_2023_Lajavaness_sentence-camembert-large.csv")
# df = pd.read_csv("sim_27_06.csv")
if not os.path.exists("utils/plots/"):
    os.mkdir("utils/plots/")
df["start_time"] = pd.to_datetime(df["start_time"], format = "%Y-%m-%d %H:%M:%S")
print(df)

for channel, group in df.groupby('channel'):
    fig, ax = plt.subplots(figsize=(10, 6))
    print('start plotting ', channel)
    ax.plot(group['start_time'], group['similarity'], label=channel)
    ax.axhline(y=0.42, color='orange', linestyle='--', label='Seuil 0.42')
    ax.set_xlabel("Start Time")
    ax.set_ylabel("Similarity")
    ax.set_title(channel + " : Courbe des similarités dans le temps")
    ax.grid(True)
    #reduire le nb de ticks
    n = max(1, len(group) // 10)
    xticks = group['start_time'].iloc[::n]
    ax.set_xticks(xticks)
    ax.set_xticklabels([ts for ts in xticks], rotation=45, ha='right', rotation_mode='anchor')

    fig.tight_layout()
    fig.savefig("utils/plots/"+channel+"_plot_sim.png")
    plt.close(fig)
