#!/usr/ bin/env python3
# CLMS Capstone | Data Preprocessing & Topic Model
# Copyright 2024 Dean Cahill
# ============================================================================|
# This script runs a BERTopic model on the Natural Questions dataset. This
# dataset is quite large and contains large documents, so we must implement the
# topic model in Online algorithms.

# ============================================================================|
# Import Statements
# ============================================================================|
from bertopic import BERTopic
from bertopic.vectorizers import OnlineCountVectorizer, ClassTfidfTransformer
import pandas as pd
from river import cluster, stream
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Standard Imports
import argparse
import json
import logging
import os
import random
import re
import tomli

# Typing Imports
from typing import Generator

random.seed("We each go through so many bodies in each other")

logger = logging.getLogger(__name__)
# ============================================================================|
# Data Iterator
# TODO - parse individual document for relevant information
# TODO - batch iterator for clustering pipeline
# ============================================================================|


def parse_document(doc):
    """Parse an individual document

    ## params
    ## returns
    STATUS: #TODO
    """
    textstr = re.search(r"Categories : <Ul>(.*)[<\/Ul>] Hidden", doc)
    text = ""
    if textstr:
        # Additional formatting on match text
        textval = textstr.group(0)
        text = re.sub("<[^<]+?>", "", textval).strip()
        text = [word for word in text.split("   ")][1:-1]
        text_tokens = "".join(word for word in text)
        return text_tokens
    return text


def batch_iterator(
    data_file: str | os.PathLike,
    batch_size: int,
) -> Generator:
    """Create a minibatch iterator for online topic modeling

    ## params
    ## returns
    """
    yield from pd.read_json(data_file, lines=True, chunksize=batch_size)


# ============================================================================|
# Topic Model
# ============================================================================|
def pipeline_parameters(fname: str | os.PathLike) -> dict[dict]:
    """Function to parse configuration into pipeline parameters
    
    ## params 
    - fname : config.toml file path 

    ## returns 
    * nested dictionary containing all sets of hyperparameters
    """
    with open(fname, "rb") as f:
        configs = tomli.load(f)
    return configs


def pipeline_components(
    **kwargs,
) -> tuple:
    """Function to generate the pipeline components for topic model

    ## params
    for the full set of parameters, see config/topic_model/config.toml

    ## returns
    a tuple containing the mapping, clustering, and vectorizing models
    """
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = IncrementalPCA(**kwargs["umap_params"])
    cluster_model = NQRiverClustering(
        model=cluster.DBSTREAM(**kwargs["cluster_params"])
    )
    vec_model = OnlineCountVectorizer(**kwargs["vectorizer_params"])
    ctfidf_model = ClassTfidfTransformer(**kwargs["tfidf_params"])

    return embedding_model, umap_model, cluster_model, vec_model, ctfidf_model


class NQRiverClustering:
    """BERTopic Model

    ## pipeline
    - Extract embeddings
    - Reduce dimensionality
    - Cluster
    - Tokenize topics
    - Extract topic words
    - (Fine tune topic words)
    """

    def __init__(self, model):
        self.model = model

    def partial_fit(self, umap_embeddings):
        """Fit clusters on umap embeddings
        In order to model such a large dataset, we must 
        be able to cluster incrementally. Thus, a partial 
        fit algortihm makes more sense than an ordinary 
        clustring algorithm
        """
        # Learn new embeddings
        for i, (umap_embedding, _) in enumerate(stream.iter_array(umap_embeddings)):
            self.model.learn_one(umap_embedding)

        # Predict labels
        labels = []
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            label = self.model.predict_one(umap_embedding)
            labels.append(label)
        self.labels_ = labels

        return self


# ============================================================================|
# Training & Inference Loops
# ============================================================================|


def train_topic_model(model, data, **hyperparams):
    """Learn topics from documents
    
    ## params 
    - model : BERTopic model for clustering 
    - data  : filename
    - params: hyperparameters
    """
    doc_batches = batch_iterator(data, batch_size=hyperparams["batch_size"])
    batch_docs = []
    topics = []
    for batch in tqdm(doc_batches):
        batchtext = [
            parse_document(doc) for doc in batch["document_text"].values.tolist()
        ]
        model.partial_fit(batchtext)
        batch_docs.append(batchtext)
        topics.extend(model.topics_)
    model.topics = topics
    model.save("models/NQ_topic_model.pkl", serialization="pickle")
    return batch_docs


def group_documents_by_topic():
    # TODO
    pass


# ============================================================================|
# Main
# ============================================================================|
def parse_args() -> argparse.Namespace:
    """Topic Model Argument Parser"""
    parser = argparse.ArgumentParser(
        description="topic model for Natural Questions dataset"
    )
    parser.add_argument(
        "--is_train",
        help="boolean for running train loop",
        action="store_true",
    )
    parser.add_argument(
        "--is_eval",
        help="boolean for performing clustering of documents",
        action="store_true",
    )
    parser.add_argument(
        "--train",
        help="Input file for dataset",
        default="data/simplified-nq-train.jsonl",
    )
    parser.add_argument(
        "--config",
        help="Configuration filename",
        default="config/topic_model/config.toml",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    # ================================|
    # Initialize Models
    model_params: dict = pipeline_parameters(args.config)
    hyperparams: dict = model_params["hyperparameters"]
    embed, umap, cluster, vectorizer, ctfidf = pipeline_components(**model_params)
    topic_model = BERTopic(
        hdbscan_model=cluster,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
    )
    # ================================|
    # Process Loops
    if args.is_train:
        batch_docs = train_topic_model(topic_model, args.train, **hyperparams)
        with open("results/topic_model/batch_docs", 'w') as f:
            for batch in batch_docs:
                f.writelines(batch) 


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
# ============================================================================|
