from datetime import datetime
from email.utils import formatdate
import pytz
import pandas as pd
import csv
import tqdm

def format_date(date_str):

    # Conversion en objet datetime
    date_obj = datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")

    # Conversion en UTC (si nécessaire, ici supposé UTC, sinon ajuster avec pytz)
    utc_date = date_obj.replace(tzinfo=pytz.utc)

    # Format RFC 822 sans virgule
    rfc_822_custom = utc_date.strftime("%a %b %d %H:%M:%S +0000 %Y")

    return rfc_822_custom

data = pd.read_csv("data/medialex_transcriptions_vocapia_v1v2_20230301_20230731.csv",
                    quoting=csv.QUOTE_ALL
                    )

tqdm.tqdm.pandas(desc="progression of formatting dates")

data["created_at"] = data.start.progress_apply(format_date)
print(data)
data.to_csv("data/formatted_medialex_transcriptions_vocapia_v1v2_20230301_20230731.csv", index=False)
data = pd.read_csv("data/formatted_medialex_transcriptions_vocapia_v1v2_20230301_20230731.csv",
                    quoting=csv.QUOTE_ALL,
                    dtype={"created_at": str, "text": str}
                    )
print(data)