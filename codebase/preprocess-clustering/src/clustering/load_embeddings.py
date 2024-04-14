#!/usr/bin/env python3
# CLMS Capstone | Dean Cahill 
# Clustering Preprocess Code --- Embeddings
import argparse 
from tqdm import tqdm
from typing import Union, Generator
import ijson
import logging
import os 
import re

import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from transformers import DistilBertTokenizer, DistilBertModel 
from transformers import AutoTokenizer 

# TEST read in documents
def read_in_documents(fname: Union[str, os.PathLike]):
    """
    Load in data from Natural Questions dataset
    produces an iterator, as each document is very large
    """
    with open(fname) as f:
        question_documents = ijson.items(f)
        for i, document in enumerate(question_documents):
            yield document 

# TODO preprocess document
def preprocess_document(document: dict[str, str]):
    pass 


# TODO encoder_loader 
def encoder_loader(data, 
                   model,
                   text_fn,
                   batch_size,
                   max_len,
                   shuffle=False,
                   ) -> DataLoader:
    """Build a DataLoader for the NQ embeddings"""    
    device = torch.get_device()
    
    def collate_fn(batch):
        embed_list, mask_list = [], []

        # FIXME tokenizer and model should work on the batch
        
        for _text in batch:
            toks = text_fn(_text,
                           return_tensors="pt",
                           padding="max_length",
                           max_length=max_len,
                           truncation=True
            )
            mask_list.append(toks["attention_mask"])
            embedding = model(toks["input_ids"]).last_hidden_state

            embed_list.append(embedding)

        embedding_tensor = torch.vstack(embed_list)
        mask_tensor = torch.vstack(mask_list)

        return embedding_tensor.to(device), mask_list.to(device)
    
    return DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )

            

# TODO cached_encoder_loader


if __name__ == "__main__":
    docs = read_in_documents("data/toy_sample.jsonl")
    for doc in docs:
        preprocess_document(doc)