#!/usr/bin/env python3
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

# Standard Imports
import argparse
from itertools import islice
import json
import os
import tempfile
from zipfile import ZipFile 

# Typing Imports 
from typing import Generator

# Logging Imports 
import wandb 
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

    pass

def batch_iterator(
    data_dir: str | os.PathLike,
    batch_size: int = 10,
) -> Generator[str]:
    """Create a minibatch iterator for online topic modeling

    ## params
    ## returns
    STATUS: #TODO
    """
    pass


# ============================================================================|
# Topic Model
# ============================================================================|
def pipeline_parameters(fname: str | os.PathLike) -> dict[dict]:
    """Function to parse configuration into pipeline parameters"""
    with open(fname, 'r') as f:
        configs = json.load(f)
    return configs

def pipeline_components(**kwargs) -> tuple[IncrementalPCA, 
                                           MiniBatchKMeans, 
                                           OnlineCountVectorizer]:
    """Function to generate the pipeline components for topic model
    
    ## params 
    ## returns

    a tuple containing the mapping, clustering, and vectorizing models
    STATUS: #TODO 
    """
    umap_model = IncrementalPCA(**kwargs["umap_params"])
    cluster_model = NQRiverClustering(cluster.DBSTREAM(**kwargs["cluster_params"]))
    vec_model = OnlineCountVectorizer(**kwargs["vectorizer_params"])
    ctfidf_model = ClassTfidfTransformer(**kwargs["tfidf_params"])

    return umap_model, cluster_model, vec_model, ctfidf_model

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
            self.model.predict_one(u_emb) for u_emb, _ in stream.iter_array(umap_embeddings)
        ]
        self.labels_ = labels
        return self 

# ============================================================================|
# Main
# ============================================================================|
def parse_args() -> argparse.Namespace:
    """Topic Model Argument Parser
    """
    parser = argparse.ArgumentParser(
        description="topic model for Natural Questions dataset"
    )
    parser.add_argument("--data_dir",
                        help="Input directory for dataset",
                        default="data/")
    parser.add_argument("--config",
                        help="Configuration filename",
                        default="config/topic_model/config.toml")
    return parser.parse_args()

def load_zip_as_file(fname):
    with ZipFile(fname, 'r') as f:
        for doc in list(f):
            yield json.loads(doc)


def main(args: argparse.Namespace):
    # ================================|
    # Initialize Models
    
    model_params: dict[dict] = pipeline_parameters(args.config)
    hyperparams = model_params["hyperparameters"]
    umap, cluster, vectorizer, ctfidf = pipeline_components(model_params)
    topic_model = BERTopic(umap_model=umap,
                           hdbscan_model=cluster,
                           vectorizer_model=vectorizer,
                           ctfidf_model=ctfidf)

    # ================================|
    # Data Preprocessing 
    # NQ is provided a series of gzip archives, which we read in one at a time
    # via a temporary staging file 

    for i, chunk_zip in os.listdir(args.data_dir):
        with tempfile.TemporaryFile() as tmp:
            with open(tmp.name, 'w') as f:
                f.write(line for line in load_zip_as_file(chunk_zip))
            doc_batches = batch_iterator(tmp.name, batch_size=hyperparams["batch_size"])
            for batch in doc_batches:
                topic_model.fit(batch)

if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
# ============================================================================|