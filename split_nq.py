#!/usr/bin/env python3
import pandas as pd
import json
from tqdm import tqdm


def batch_iterator(fpath):
    yield from pd.read_json(fpath, lines=True, chunksize=1)


for i, row in tqdm(enumerate(batch_iterator("data/simplified-nq-train.jsonl"))):
    row.to_json(f"data/simplified-nq-individual/{i}.json", indent=2)
