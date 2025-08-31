import polars as pl
import argparse
from matplotlib import pyplot as plt
import numpy as np
"""
--path_extract formatted_output_from_run_encode.csv --path_transcription data/nahel_transcriptions_vocapia_27_06_2023_to_03_07_2023.csv
compute duration of transcription stream for each day and each channels, then compute percentage of time about Nahel vs. total time
"""

def stats(args) :
    SCHEMA = {"":int, "channel": str, "start": pl.Datetime, "end": pl.Datetime, "duration": float, "text": str, "id":str, "created_at":str, "tag":str}
    TRS = {"channel":str, "start":pl.Datetime, "end":pl.Datetime, "duration":float, "text":str, "id":str, "created_at":str, "tag":str}
    extract = pl.read_csv(args.path_extract, schema_overrides=SCHEMA)
    pl.Config.set_tbl_rows(100)
    trs = pl.read_csv(args.path_transcription, schema_overrides=TRS)
    extract = extract.with_columns(
        pl.col("start").dt.date().alias("day")  # datetime -> jour pour fichiers pred
    )
    trs = trs.with_columns(
        pl.col("start").dt.date().alias("day")  # datetime -> jour pour transcription
    )
    trs_sum = trs.select(["channel", "day", "duration"]).group_by(["channel", "day"]).sum()
    trs_sum = trs_sum.rename({"duration":"duration_trs"})
    pred_sum = extract.select(["channel", "day", "duration"]).group_by(["channel", "day"]).sum()
    pred_sum = pred_sum.rename({"duration":"duration_nahel"})
    cat_trs_pred = trs_sum.join(pred_sum, on=['day', 'channel'], how='full', coalesce=True)
    cat_trs_pred = cat_trs_pred.sort(["day", "channel"])

    # proportion par jour pour toutes chaînes
    per_day = cat_trs_pred.group_by("day").sum() 
    per_day = per_day.with_columns([
            (pl.col("duration_nahel") / pl.col("duration_trs")*100).round(2).alias("percent")
        ]).fill_null(0)
    per_day = per_day.with_columns(((pl.col("duration_trs") * 1_000_000).cast(pl.Duration("us"))).dt.to_string(format="polars").alias("total_duration"))
    per_day = per_day.with_columns(((pl.col("duration_nahel") * 1_000_000).cast(pl.Duration("us"))).dt.to_string(format="polars").alias("extract_duration"))
    print(per_day)

    # propotion jours seulement chaines télés
    filter_tv_ch = cat_trs_pred.filter(pl.col("channel").is_in(["ARTE","BFM_TV","C8","CNews","France2","France3","France5","FranceInfo_TV","LCI","LCP","M6","TF1","TMC"]))
    per_day_tv = filter_tv_ch.group_by("day").sum() 
    per_day_tv = per_day_tv.with_columns([
            (pl.col("duration_nahel") / pl.col("duration_trs")*100).round(2).alias("percent")
        ]).fill_null(0)
    per_day_tv = per_day_tv.with_columns(((pl.col("duration_trs") * 1_000_000).cast(pl.Duration("us"))).dt.to_string(format="polars").alias("dt_trs"))
    per_day_tv = per_day_tv.with_columns(((pl.col("duration_nahel") * 1_000_000).cast(pl.Duration("us"))).dt.to_string(format="polars").alias("extract_duration"))

    print(per_day_tv)

    #proportion par chaînes
    per_channels = cat_trs_pred.with_columns([
            (pl.col("duration_nahel") / pl.col("duration_trs")*100).round(2).alias("percent")
        ]).fill_null(0)
    per_channels = per_channels.with_columns(((pl.col("duration_trs") * 1_000_000).cast(pl.Duration("us"))).dt.to_string(format="polars").alias("dt_trs"))
    per_channels = per_channels.with_columns(((pl.col("duration_nahel") * 1_000_000).cast(pl.Duration("us"))).dt.to_string(format="polars").alias("extract_duration"))

    print(per_channels)
    # filtre proportion pour chaîne majeures
    filter_main_ch = per_channels.filter(pl.col("channel").is_in(["ARTE","BFM_TV","CNews","Europe1","France2","FranceInfo_RD","FranceInfo_TV","FranceInter","TF1"]))
    # filtre proportion par chaînes télé seulement
    per_channels_filter_tv_ch = per_channels.filter(pl.col("channel").is_in(["ARTE","BFM_TV","C8","CNews","France2","France3","France5","FranceInfo_TV","LCI","LCP","M6","TF1","TMC"]))
    

    per_channels.write_csv("TV_RADIO_stats_temps_chaines_jours.csv")
    per_channels_filter_tv_ch.write_csv("only_TV_stats_temps_chaines_jours.csv")

    per_day.write_csv("TV_RADIO_stats_temps_jours.csv")
    per_day_tv.write_csv("only_TV_stats_temps_jours.csv")
    return filter_main_ch


def plot_lines(df, colors_dict, channels):
    # Conversion de la colonne "day" en datetime
    # df = df.with_columns(pl.col("day").str.strptime(pl.Date, "%Y-%m-%d"))

    # Pivot (jours en index, channels en colonnes, valeurs = percent)
    pivot_df = df.pivot(index="day", on="channel", values="percent")
    dates = pivot_df["day"].to_numpy()
    # Tracé direct avec polars
    plt.figure(figsize=(12, 6))
    for channel in channels:
        plt.plot(dates, pivot_df[channel].to_numpy(), marker="o", label=channel, color=colors_dict[channel])

    plt.title("Évolution de la proportion de temps dédiée à Nahel par chaîne")
    plt.xlabel("Jour")
    plt.ylabel("Pourcentage (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("utils/stats_time_lines.png")
    
def plot_bars(df, colors_dict, channels) :

    # df = df.with_columns(pl.col("day").str.strptime(pl.Date, "%Y-%m-%d"))

    # Pivot : jours en index, chaînes en colonnes, valeurs = percent
    pivot_df = df.pivot(index="day", on="channel", values="percent")
    # Données pour le tracé
    days = pivot_df["day"].to_list()
    #channels = pivot_df.drop("day").columns
    n_channels = len(channels)
    n_days = len(days)

    # Position des groupes sur l'axe X
    x = np.arange(n_days)

    # Largeur de chaque barre
    bar_width = 0.8 / n_channels

    # Création de la figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Tracé des barres pour chaque chaîne
    for i, channel in enumerate(channels):
        values = pivot_df[channel].to_numpy()
        ax.bar(x + i * bar_width, values, width=bar_width, label=channel, color=colors_dict[channel])

    # Mise en forme
    ax.set_title("Proportion de temps dédié à Nahel par chaîne et par jour")
    ax.set_xlabel("Jour")
    ax.set_ylabel("Pourcentage (%)")
    ax.set_xticks(x + 0.8 / 2 - bar_width / 2)  # centrer les ticks
    ax.set_xticklabels([str(d) for d in days], rotation=45)
    ax.legend(title="Chaîne")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
    plt.savefig("utils/stats_time_bar.png")

    

def main(args) :
    df_stats = stats(args)
    channels = ["ARTE","BFM_TV","CNews","Europe1","France2","FranceInfo_RD","FranceInfo_TV","FranceInter","TF1"]
    colors_dict = {
        "ARTE": "maroon",          
        "BFM_TV": "yellowgreen",       
        "CNews": "blue",          
        "Europe1": "darkturquoise",       
        "France2": "gold",        
        "FranceInfo_RD": "darkgoldenrod",
        "FranceInfo_TV": "orange",
        "FranceInter": "red",
        "TF1": "slategray"
    }
    plot_bars(df=df_stats, colors_dict=colors_dict, channels=channels)
    plot_lines(df=df_stats, colors_dict=colors_dict, channels=channels)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_extract', required=True, help="path of extracted, post format deliver")
    parser.add_argument('--path_transcription', required=True, help="path of the full transcripiton")
    args = parser.parse_args()

    main(args)