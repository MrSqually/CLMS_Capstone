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

# Logging Imports
import wandb

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
    STATUS: REVIEW
    """
    yield from pd.read_json(data_file, lines=True, chunksize=batch_size)


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
) -> tuple:
    """Function to generate the pipeline components for topic model

    ## params
    ## returns

    a tuple containing the mapping, clustering, and vectorizing models
    STATUS: #TODO
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

        ## params
        ## returns
        STATUS: #TODO
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
# Main
# ============================================================================|
def parse_args() -> argparse.Namespace:
    """Topic Model Argument Parser"""
    parser = argparse.ArgumentParser(
        description="topic model for Natural Questions dataset"
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
    # Data Preprocessing
    doc_batches = batch_iterator(args.train, batch_size=hyperparams["batch_size"])

    for epoch in range(hyperparams["epochs"]):
        logger.info(f"Starting Epoch {epoch}")
        batch_docs = []
        topics = []
        for i, batch in tqdm(enumerate(doc_batches)):
            batchtext = [
                parse_document(doc) for doc in batch["document_text"].values.tolist()
            ]

            logger.info(f"Processing Batch {i}")
            topic_model.partial_fit(batchtext)
            batch_docs.append(batchtext)
            topics.extend(topic_model.topics_)
        logger.info(f"Batch Processed! Updating topics...")
        topic_model.topics = topics

    logger.info("Generating Document Maps")
    fig = topic_model.visualize_document_datamap(batch_docs)
    fig.savefig(f"results/topics/topics.png", bbox_inches="tight")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
# ============================================================================|
