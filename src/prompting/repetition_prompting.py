#!/usr/bin/env python3
# CLMS Capstone | LLM Prompting
# Dean Cahill

# ==================================================================|
# Imports
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


import argparse
from collections import defaultdict
import logging 
import json 
import tomli 
from tqdm import tqdm

from prompt_construction import get_prompts
from load_data import preprocess_data


logger = logging.getLogger(__name__)
# ====================================================================|
# Model Initialization Functions
def get_flant5_model(**params):
    flan_t5 = "google/flan-t5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(flan_t5, **params)
    tokenizer = AutoTokenizer.from_pretrained(flan_t5)
    return model, tokenizer

def get_mixtral_model(**params):
    mixtral = "mistralai/Mixtral-8x7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(mixtral, **params)
    tokenizer = AutoTokenizer.from_pretrained(mixtral)
    return model, tokenizer

def get_bart_model(**params):
    pass

def get_gpt_model(**params):
    pass 

models = {"flant5" : get_flant5_model,
          "mixtral": get_mixtral_model,
          "bart"   : get_bart_model,
          "gpt"    : get_gpt_model}
# ==================================================================|
# Prompting Functions
class Prompter:
    def __init__(self, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer


    def generate_llm_response(self, prompt) -> list[str]:
        """Prompt `model` to generate output based on
        the question in `data`. 
        *** This is where the LLM stuff happens ***

        ## params
        - prompt: a single question

        ## returns
        list[str] -> answers to the batch of questions in data
        """
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output = self.model.generate(ids, max_length=50)
        decoder_out = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoder_out


    def query_prompt(self, question, prompt_type) -> tuple[str, str, str]:
        """Generate prompts from a QA document
        augments `question` with the prompt boilerplate 
        from `prompt-type` and generates the 3 necessary responses
        from this.
        """
        raw, qa, halu = prompt_type(question)
        raw_response = self.generate_llm_response(raw)
        qa_response = self.generate_llm_response(qa)
        halu_response = self.generate_llm_response(halu)
        return raw_response, qa_response, halu_response


# ===============================================================|
# Response Loop 
def response_loop(dataset, prompter:Prompter, **params):
    """"""
    # prompt order: basic => chain of thought => contradiction => instruction
    for prompt in get_prompts():
        logger.info(f"Beginning {prompt} response generation")
        promptdict = defaultdict(lambda: defaultdict(list))
        for question in tqdm(dataset):
            for _ in range(params["num_repetitions"]):
                raw_resp, qa_resp, halu_resp = prompter.query_prompt(question, prompt)
                promptdict[question]["raw"].append(raw_resp)
                promptdict[question]["qa"].append(qa_resp)
                promptdict[question]["halu"].append(halu_resp)
        with open(f"results/prompting/{prompt.__name__}_answers.jsonl", 'a+') as f:
            f.write(json.dumps(promptdict) + "\n")


# ===============================================================|
# Main
def parse_args() -> argparse.Namespace:
    """Runtime Arguments

    - config: configuration TOML
    - data_dir: location of QA dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="configuration file",
        default="config/prompting/config.toml",
    )
    parser.add_argument(
        "-d",
        "--data_loc",
        help="file location of data",
        default="data/simplified-nq-train.jsonl",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    with open(args.config, 'rb') as f:
        hyperparameters = tomli.load(f)
    model_params = hyperparameters["model_args"]
    model_name = hyperparameters["model_name"].lower().replace("-", "")
    runtime_parameters = hyperparameters["run_params"]


    prompter = Prompter(*models[model_name](**model_params))

    documents = preprocess_data(args.data_loc, runtime_parameters["batch_size"])

    for batch in documents:
        response_loop(prompter=prompter, dataset=batch["question_text"], **runtime_parameters)


if __name__ == "__main__":
    args = parse_args()
    main(args)
