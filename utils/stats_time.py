import polars as pl
import argparse
from matplotlib import pyplot as plt
import numpy as np
"""
--path_extract output_deliver_from_run_encode.csv --path_transcription data/nahel_transcriptions_vocapia_27_06_2023_to_03_07_2023.csv

compute duration of transcription stream for each day and each channel, then compute percentage of time about Nahel vs. total time
"""

def stats(args) :
    SCHEMA = {"channel": str, "start": pl.Datetime, "end": pl.Datetime, "duration": float, "text": str, "id":str, "created_at":str, "tag":str}
    TRS = {"channel":str, "start":pl.Datetime, "end":pl.Datetime, "duration":float, "text":str, "id":str, "created_at":str, "tag":str}
    extract = pl.read_csv(args.path_extract, schema_overrides=SCHEMA)

    # channel en commun entre extracted et transcription
    trs = pl.read_csv(args.path_transcription, schema_overrides=TRS)
    filtered_trs = trs.filter(
        pl.col("channel").is_in(extract.select("channel").to_series())
    )
    print(filtered_trs)

    # Tri de filtered_trs et ext par "channel"
    trs = filtered_trs.sort("channel", "start")
    extract = extract.sort("channel", "start")

    extract = extract.with_columns(
        pl.col("start").dt.date().alias("jour")  # extrait uniquement la date
    )
    trs = trs.with_columns(
        pl.col("start").dt.date().alias("jour")  # extrait uniquement la date
    )
    df_to_save = pl.DataFrame(schema={"channel":str, "transcription_jour":pl.Duration, "extract_jour":pl.Duration,"percent":pl.Float64, "day":str})
    ls = []
    noch_df_time_by_day = pl.DataFrame(schema={"transcription_jour":pl.Duration, "extract_jour":pl.Duration,"percent":pl.Float64, "day":str})
    #proportion of time about nahel/all time for each day
    for jour, data in trs.group_by("jour"):
        data = data.with_columns((pl.col("end") - pl.col("start")).alias("delta"))
        nochannel_trs = data.select(pl.sum("delta").alias("transcription_jour"))
        
        nochannel_day_extract = extract.filter(pl.col("jour") == jour[0])
        nochannel_day_extract = nochannel_day_extract.with_columns((pl.col("end") - pl.col("start")).alias("delta"))
        nochannel_duration_extract = nochannel_day_extract.select(pl.sum("delta").alias("extract_jour"))
      
        nochannel_out = pl.concat([nochannel_trs, nochannel_duration_extract], how="horizontal")
        nochannel_out = nochannel_out.with_columns([
            (pl.col("extract_jour") / pl.col("transcription_jour")*100).round(2).alias("percent")
        ])
        nochannel_out = nochannel_out.with_columns([
            pl.lit(str(jour[0])).alias("day")
        ])
        noch_df_time_by_day.extend(nochannel_out)

    #proportion of time about nahel/all time for each channel, each day
    for jour, data in trs.group_by("jour"): 
        print(jour)
        data = data.with_columns((pl.col("end") - pl.col("start")).alias("delta"))
        trs_duration = (
            data.group_by("channel")
            .agg([
                pl.sum("delta").alias("transcription_jour")
            ])
        )
        trs_duration = trs_duration.sort("channel")
        filter_jour = extract.filter(pl.col("jour") == jour[0])
        filter_jour = filter_jour.with_columns((pl.col("end") - pl.col("start")).alias("delta"))

        extract_jour = (
            filter_jour.group_by("channel").agg([
                pl.sum("delta").alias("extract_jour")
            ])
        )   
        extract_jour = extract_jour.sort("channel")
        out = trs_duration.join(extract_jour, on='channel', how='left')
        out = out.with_columns([
            (pl.col("extract_jour") / pl.col("transcription_jour")*100).round(2).alias("percent")
        ])
        out = out.with_columns([
            pl.lit(str(jour[0])).alias("day")
        ])
        print(out)
        # la liste de df pour calculer les durées totales (ordre des jours pas conservé)
        # le dict pour noter l'ordre des jours
        ls.append(out)
        df_to_save.extend(out)

    # this df "total" is not the one saved, its only for total on alll period
    total = pl.concat(ls)
    total_time = (
        total
        .group_by("channel")
        .agg([
            pl.sum("transcription_jour").alias("total_transcription"),
            pl.sum("extract_jour").alias("total_extract")
        ])
    )
    # % de Nahel sur la durée totale de la semaine pour chaque chaîne
    final_df = (
        total_time
        .with_columns([
            (pl.col("total_extract") / pl.col("total_transcription")*100).round(2).alias("percent")
        ])
    )
    print(final_df)

    df_to_save = df_to_save.sort('day')
    df_to_save = df_to_save.with_columns([pl.col("transcription_jour").dt.total_seconds().alias("transcription_jour"), pl.col("extract_jour").dt.total_seconds().alias("extract_jour")])
    df_to_save = df_to_save.fill_null(0)

    df_to_save.write_csv("stats_temps_chaines_jours.csv")

    noch_df_time_by_day = noch_df_time_by_day.sort('day')
    noch_df_time_by_day = noch_df_time_by_day.with_columns([pl.col("transcription_jour").dt.to_string(format="polars").alias("transcription_jour"), pl.col("extract_jour").dt.to_string(format="polars").alias("extract_jour")])

    noch_df_time_by_day.write_csv("stats_temps_jours.csv")
    return df_to_save

def plot_lines(df, colors_dict, channels):
    # Conversion de la colonne "day" en datetime
    df = df.with_columns(pl.col("day").str.strptime(pl.Date, "%Y-%m-%d"))

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

    df = df.with_columns(pl.col("day").str.strptime(pl.Date, "%Y-%m-%d"))

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

    channels = ["ARTE", "FranceInter", "France2", "FranceInfo_RD", "FranceInfo_TV", "BFM_TV", "Europe1", "TF1", "CNews"]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_extract', required=True, help="path of extracted, post format deliver")
    parser.add_argument('--path_transcription', required=True, help="path of the full transcripiton")
    args = parser.parse_args()

    main(args)