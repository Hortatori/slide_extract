import pandas as pd
import re
"""
création d'un fichier annoté par mot clés pour faire une baseline d'évaluation
"""

def annote_corpus_with_keyword() :
    # deux documents ont été générés avec ce script :
    # un avec la liste globale de mots clés
    # un avec le pattern regex de refus d obtempere seulement

    pattern_regex = re.compile(r"\brefus\s+(de\sobtempérer|d'obtempérer|d\sobtempérer)\b", flags=re.IGNORECASE)

    mots_cles = [
        "jeune conducteur", "affrontements", "refus d'obtempérer", "émeutes", "émeutiers",
        "cagnotte", "l'haÿ-les-roses", "lhaylesroses", "violences",
        "nanterre", "clichy-sous-bois", "naël", "nahel"
    ]

    # with keywords pattern
    def contient_mot_cle(texte):
        if pd.isna(texte):
            return 0
        texte = texte.lower()
        return int(any(mot in texte for mot in mots_cles))

    # with regex 'refus'
    def contient_refus(texte):
        if pd.isna(texte):
            return 0
        texte = texte.lower()
        return int(bool(pattern_regex.search(texte)))

    df = pd.read_csv("data/nahel_transcriptions_vocapia_27_06_2023_to_03_07_2023.csv")  
    df_refus = df.copy(deep=True)

    df["label"] = df["text"].apply(contient_mot_cle)
    df_refus["label"] = df_refus["text"].apply(contient_refus)
    print(df)
    print(df_refus)
    df.to_csv("data/label_keywords_nahel_transcriptions_vocapia_27_06_2023_to_03_07_2023.csv", index=False)
    df_refus.to_csv("data/label_keywords_refus_d_obtemperer_nahel_transcriptions_vocapia_27_06_2023_to_03_07_2023.csv", index=False)

annote_corpus_with_keyword()