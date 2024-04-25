#!/usr/bin/env python3
# CLMS Capstone | LLM Prompting
# Dean Cahill

# ==================================================================|
# Imports
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# from transformers import BartForConditionalGeneration, BartTokenizer

import argparse

# from prompting import prompts

# ==================================================================|
# Flan-T5
flan_t5 = "google/flan-t5-small"
flant5_model = AutoModelForSeq2SeqLM.from_pretrained(flan_t5)
flant5_tokenizer = AutoTokenizer.from_pretrained(flan_t5)


# AutoModel-based Prompt Function
def prompt_t5_llm(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    data,
    **params,
) -> list[str]:
    """Prompt `model` to generate output based on
    the question in `data`.

    ## params
    - model     : a huggingface transformer model
    - tokenizer : a huggingface tokenizer
    - data      : the document / batch being tested
    - params    : runtime hyperparameters

    ## returns
    list[str] -> answers to the batch of questions in data
    STATUS: TODO
    """
    ids = tokenizer(data, **params).input_ids
    output = model.generate(ids)
    return tokenizer.decode(output[0])


# =================================================================|
# GPT
# =================================================================|
# CLAUDE / LLAMA3 (newer model)
# =================================================================|
# Prompt Inference Loop & Data Collection
def bare_repetition(model, data, reps, **kwargs):
    """Bare Repetition

    ## params

    ## returns
    STATUS: TODO
    """
    pass


def contradictive_repetition():
    """

    ## params

    ## returns
    STATUS: TODO
    """
    pass


def instructive_repetition():
    """Instructive Repetition

    ## params

    ## returns
    STATUS: TODO
    """
    pass


def prompt_inference_loop():
    """

    ## params

    ## returns
    STATUS: TODO
    """

    # for i in number_of_repetitions:
    #   prompt llm to answer question
    #   store the responses in json
    pass


# =================================================================|
# Main
def parse_args() -> argparse.Namespace:
    """Runtime Arguments

    - config: configuration TOML
    - data_dir: location of QA dataset
    """
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main(args: argparse.Namespace):
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
