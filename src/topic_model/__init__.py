#!/usr/bin/env python3

# CLMS Capstone | Data Preprocessing & Topic Model
# Copyright 2024 Dean Cahill


# ============================================================================|
# Import Statements
# ============================================================================|
from bertopic import BERTopic


# Standard Imports
import argparse
import json
import os

# Typing Imports
from typing import Iterator


# ============================================================================|
# Data Iterator
# TODO - parse individual document for relevant information
# TODO - batch iterator for clustering pipeline
# ============================================================================|
def parse_document(doc: dict) -> list[str]:
    """Parse an individual document

    ## params
    ## returns
    STATUS: TODO
    """

    pass


def batch_iterator(
    data_dir: str | os.PathLike,
    batch_size: int = 10,
) -> Iterator[str]:
    """Create a minibatch iterator for online topic modeling

    ## params
    ## returns
    STATUS: TODO
    """
    pass


# ============================================================================|
# Topic Model
# ============================================================================|
def run_topic_model(data_iter, **params) -> BERTopic:
    """Primary topic model training/clustering loop

    As NQ is a very large dataset with very large documents,
    it makes sense to approach this as an Online clustering 
    problem

    ## params
    - data_iter => batched data generator

    ## returns
    STATUS: TODO
    """
    pass


# ============================================================================|
# Main
# ============================================================================|
def parse_args() -> argparse.Namespace:
    """Topic Model Argument Parser
    STATUS: TODO
    """
    parser = argparse.ArgumentParser(
        description="topic model for Natural Questions dataset"
    )

    parser.add_argument("--data_dir")
    parser.add_argument("--config")

    return parser.parse_args()


def main(args: argparse.Namespace):
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
