import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path
import os

# python utils/plot_sim.py --filepath articles/sim_from_run_encode.csv --output 7days_test__plot_sim.png
# python utils/plot_sim.py --filepath summaries/gpt_sim.csv --output 7days_test_gpt_plot_sim.png

def main(args):
    filepath = args.filepath
    output_file = args.output
    #INFO : remember the time orders is not coherent : different JT are overlapping when plotting more than one channel at a time
    df = pd.read_csv(filepath)

    df["start"] = pd.to_datetime(df["start"])
    print(df)

    if not os.path.exists("utils/plots/"):
        os.mkdir("utils/plots/")
    for channel, group in df.groupby('channel'):
        fig, ax = plt.subplots(figsize=(10, 6))
        print('start plotting ', channel)
        ax.plot(group['start'], group['avg'], label=channel)
        ax.axhline(y=0.30, color='orange', linestyle='--', label='Seuil 0.30')
        ax.set_xlabel("Start Time")
        ax.set_ylabel("Similarity")
        ax.set_title(channel + " : Courbe des similarités dans le temps pour le fichier " + filepath)
        ax.grid(True)
        #reduire le nb de ticks
        n = max(1, len(group) // 10)
        xticks = group['start'].iloc[::n]
        ax.set_xticks(xticks)
        ax.set_xticklabels([ts for ts in xticks], rotation=45, ha='right', rotation_mode='anchor')

        fig.tight_layout()
        fig.savefig("utils/plots/"+channel+"_plot_sim.png")
        plt.close(fig)
    df_sorted = df.sort_values("start")



    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_sorted['start'], df_sorted['avg'])
    ax.axhline(y=0.35, color='orange', linestyle='--', label='Seuil 0.35')
    ax.set_xlabel("Start Time")
    ax.set_ylabel("Similarity")
    ax.set_title("Courbe des similarités dans le temps")
    ax.grid(True)
    #reduire le nb de ticks
    n = max(1, len(df_sorted) // 10)
    xticks = df_sorted['start'].iloc[::n]
    ax.set_xticks(xticks)
    ax.set_xticklabels([ts for ts in xticks], rotation=45, ha='right', rotation_mode='anchor')

    fig.tight_layout()
    fig.savefig(filepath.split(".")[0]+"_plot_sim.png")
    plt.close(fig)


    df_for_days = df_sorted
    df_for_days["day"] = df['start'].dt.date
    grouped = df_for_days.groupby('day')
    fig, axes = plt.subplots(nrows=len(grouped),figsize=(20,10*len(grouped)))

    for (day, df), ax in zip(grouped,axes) :
        ax.plot(df["start"], df["avg"])
        ax.set_ylim(-0.1,0.6)
        ax.axhline(y=0.35, color='orange', linestyle='--', label='Seuil 0.35')
        ax.set_xlabel("Start Time")
        ax.set_ylabel("Similarity")
        ax.set_title("Courbe des similarités dans le temps")
        ax.grid(True)
        #reduire le nb de ticks (d'une manière bof))
        n = max(1, len(df) // 10)
        xticks = df['start'].iloc[::n]
        ax.set_xticks(xticks)
        ax.set_xticklabels([ts for ts in xticks], rotation=45, ha='right', rotation_mode='anchor')

        fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--filepath",
                        type = str,
                        help = "Path similarity filee")
    parser.add_argument("--output",
                        type = str,
                        help = "Path to save plot file")
    parsed = parser.parse_args()
    main(parsed)