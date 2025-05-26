import pandas as pd

df = pd.read_csv("data/short_1000_04072023.csv")
df ["start"] = pd.to_datetime(df['start'], dayfirst=True)
df['end'] = pd.to_datetime(df['end'], dayfirst=True)

notice = pd.read_csv("../medialex/fr2_nahel_notices.csv")
notice[]
ff = pd.Interval(df["start"][10], df["end"][15])
ii = pd.Interval(notice["start"][0], df["end"][10])
ii.overlaps(ff)
print(ii)

# datedif
#heuremin
#dureemin
#datdifsec
#start
#date
#heurefin
#duree
#Durée