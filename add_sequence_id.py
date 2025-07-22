# Add a sequence_id column to the input file. A sequence is a set of contiguous transcriptions
# (the end of the first transcription is close enough to the beginning of the next one).
#
# If two lines are separated by less than SECONDS, and if they are from the same channel,
# they are grouped together in the same sequence.
#
# The input file is supposed to contain the following columns: start, end, channel.
# It is supposed to be sorted by channel and by start.
#

import argparse
import casanova
from datetime import datetime, timedelta

SECONDS = 30
parser = argparse.ArgumentParser()

parser.add_argument(
    "input",
    help="File containing the transcriptions sorted by channel and by start time",
)
parser.add_argument(
    "output",
    help="Name of the file created by this script",
)


def read_time(position):
    return datetime.strptime(row[position], "%Y-%m-%d %H:%M:%S")


args = parser.parse_args()

with open(args.input, "r") as input, open(args.output, "w") as output:
    enricher = casanova.enricher(input, output, add=["sequence_id"])
    start_pos = enricher.headers.start
    end_pos = enricher.headers.end
    channel_pos = enricher.headers.channel

    sequence_id = 0
    end = None
    channel = None
    for row in enricher:
        if end is None:
            end = read_time(start_pos)
            channel = row[channel_pos]
        delta = read_time(start_pos) - end
        if delta > timedelta(seconds=SECONDS) or row[channel_pos] != channel:
            sequence_id += 1
        channel = row[channel_pos]
        end = read_time(end_pos)
        enricher.writerow(row, add=[sequence_id])
