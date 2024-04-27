#!/usr/bin/env python3
# CLMS Capstone | LLM Prompting
# Dean Cahill

# ==================================================================|
# Imports
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


import argparse

from prompting import prompts

# ==================================================================|
# Flan-T5
flan_t5 = "google/flan-t5-small"
flant5_model = AutoModelForSeq2SeqLM.from_pretrained(flan_t5)
flant5_tokenizer = AutoTokenizer.from_pretrained(flan_t5)

# ==================================================================|
# Mixtral
mixtral = "mistralai/Mixtral-8x7B-v0.1"
mixtral_model = AutoModelForCausalLM.from_pretrained(mixtral, device_map="auto")
mixtral_tokenizer = AutoTokenizer.from_pretrained(mixtral)

# ==================================================================|
# Prompt Inference Loop & Data Collection


# AutoModel-based Prompt Function
def prompt_llm(
    model,
    tokenizer,
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
    STATUS: REVIEW
    """
    ids = tokenizer(data).input_ids
    output = model.generate(ids)
    return tokenizer.decode(output[0])


def bare_repetition_prompt(model, data, reps, **kwargs) -> tuple[str,str,str]:
    """Bare Repetition

    ## params

    ## returns
    (nocon_output, qacon_output, halucon_output)
    STATUS: TODO
    """
    # for doc in data:
    #   - prompt base
    #   - prompt QA context
    #   - prompt Halu context
    pass


def contradictive_repetition_prompt() -> tuple[str, str, str]:
    """

    ## params

    ## returns
    STATUS: TODO
    """
    # for doc in data:
    #   - prompt base
    #   - prompt QA context
    #   - prompt Halu context
    pass


def instructive_repetition_prompt() -> tuple[str, str, str]:
    """Instructive Repetition
    ## params

    ## returns
    STATUS: TODO
    """
    # for doc in data:
    #   - prompt base
    #   - prompt QA context
    #   - prompt Halu context

    pass


def chain_of_thought_repetition_prompt() -> tuple[str, str, str]:
    """"""
    # for doc in data:
    #   - prompt base
    #   - prompt QA context
    #   - prompt Halu context
    pass


def prompt_inference_loop() -> dict[str, tuple[str, str, str]]:
    """Inference loop for a given set of documents

    ## params

    ## returns
    STATUS: TODO
    """
    # for i in number_of_repetitions:
    #   prompt llm to answer question => triple (base, QA, halu)
    #   serialize & store responses
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
