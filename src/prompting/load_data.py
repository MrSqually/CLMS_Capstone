
import pandas as pd 

def preprocess_data(doc_file, batch_size):
    yield from pd.read_json(doc_file, lines=True, chunksize=batch_size)
