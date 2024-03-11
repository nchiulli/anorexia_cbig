import json
import os
from glob import glob
from natsort import natsorted
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Reformat BIDS json into CBIG slice timing. Written by Stephan Palm, modified by Christopher Lin")
parser.add_argument(
    "-d",
    required=True, help="BIDS directory")
parser.add_argument(
    "-p",
    required=True, help="Subject")
parser.add_argument(
    "-s",
    required=True, help="Session")
parser.add_argument(
    "-o",
    required=True, help="Output file")
args = parser.parse_args()

search_path = os.path.join(args.d, args.p, args.s, "func", "*task-rest*_bold.json")
print(search_path)
input_jsons = natsorted(glob(search_path))
input_jsons = [p for p in input_jsons if "echo" not in p]

# print(input_jsons)

df = pd.DataFrame()
for j in input_jsons:
    fname = os.path.basename(j)
    data = json.load(open(j,))
    df = pd.concat([df, pd.DataFrame(data["SliceTiming"])], axis=1)

df = df.fillna(-999)
# print(df.to_string(header=False, index=False))
df.to_csv(args.o, sep=" ", header = False, index = False)
# print(f"Printed to {args.o}")
