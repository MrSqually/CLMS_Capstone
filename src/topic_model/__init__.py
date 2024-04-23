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
import gzip 
import json
import logging 
import os
import random
import re 
import tempfile
import tomli

# Typing Imports
from typing import Generator

# Logging Imports
import wandb

random.seed = "We each go through so many bodies in each other"
logger = logging.getLogger(__name__)
# ============================================================================|
# Data Iterator
# TODO - parse individual document for relevant information
# TODO - batch iterator for clustering pipeline
# ============================================================================|


def parse_document(doc: dict) -> pd.DataFrame:
    """Parse an individual document

    ## params
    ## returns
    STATUS: #TODO
    """
    text = re.match(r"Categories : <Ul> (.*)<\/Ul> Hidden",doc["document_text"]).string
    text = text.replace("<Li>", "").replace("</Li>", "").replace("<Ul>", "").replace("Hidden", "")
    doc["document_text"] = text 
    return doc


def batch_iterator(
    data_dir: str | os.PathLike,
    batch_size: int = 10,
) -> Generator:
    """Create a minibatch iterator for online topic modeling

    ## params
    ## returns
    STATUS: #TODO
    """
    for root, dirs, chunk_zip_fnames in os.walk(data_dir):
        for chunk_zip in chunk_zip_fnames:
            batch = []
            with gzip.open(os.path.join(root, chunk_zip), 'rt') as compressed_data:
               
               for line in compressed_data.readlines():
                    batch.append(parse_document(json.laods(line)))
               
            yield batch
            
                


# ============================================================================|
# Topic Model
# ============================================================================|
def pipeline_parameters(fname: str | os.PathLike) -> dict[dict]:
    """Function to parse configuration into pipeline parameters"""
    with open(fname, "rb") as f:
        configs = tomli.load(f)
    return configs


def pipeline_components(
    **kwargs,
) -> tuple[IncrementalPCA, MiniBatchKMeans, OnlineCountVectorizer]:
    """Function to generate the pipeline components for topic model

    ## params
    ## returns

    a tuple containing the mapping, clustering, and vectorizing models
    STATUS: #TODO
    """
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = IncrementalPCA(**kwargs["umap_params"])
    cluster_model = NQRiverClustering(cluster.DBSTREAM(**kwargs["cluster_params"]))
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

    def __init__(self, model, **kwargs):
        self.model = model

    def partial_fit(self, umap_embeddings):
        """Fit clusters on umap embeddings

        ## params
        ## returns
        STATUS: #TODO
        """
        # Learn new embeddings
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            self.model = self.model.learn_one(umap_embedding)

        # Predict labels
        labels = [
            self.model.predict_one(u_emb)
            for u_emb, _ in stream.iter_array(umap_embeddings)
        ]
        self.labels_ = labels
        return self


# ============================================================================|
# Main
# ============================================================================|
def parse_args() -> argparse.Namespace:
    """Topic Model Argument Parser"""
    parser = argparse.ArgumentParser(
        description="topic model for Natural Questions dataset"
    )
    parser.add_argument(
        "--train_dir", help="Input directory for dataset", default="data/v1.0/train"
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
    model_params: dict[dict] = pipeline_parameters(args.config)
    hyperparams = model_params["hyperparameters"]
    embed, umap, cluster, vectorizer, ctfidf = pipeline_components(**model_params)
    topic_model = BERTopic(
        embedding_model=embed, 
        umap_model=umap,
        hdbscan_model=cluster,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
    )
    # ================================|
    # Data Preprocessing
    doc_batches = batch_iterator(args.train_dir, batch_size=hyperparams["batch_size"])
    for epoch in range(hyperparams["epochs"]):
        batch_docs = []
        topics = []
        for batch_id, batch in doc_batches:
            topic_model.fit(batch)
            batch_docs.append(batch)
            topics.extend(topic_model.topics_)
        topic_model.topics = topics

        fig = topic_model.visualize_document_datamap(batch_docs)
        fig.savefig(f"results/topics/epoch{epoch}_topics.png", bbox_inches="tight")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
# ============================================================================|
