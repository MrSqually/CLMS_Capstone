#!/usr/bin/env python3
# CLMS Capstone | LLM Prompting
# Dean Cahill

# ==================================================================|
# Model Imports

### Huggingface
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

### OpenAI / GPT
from openai import OpenAI

# StandardLib
from abc import ABC, abstractmethod
import argparse
from collections import defaultdict
import logging 
import json 
import tomli  

# Local
from prompt_construction import HFPrompt, GPTPrompt
from load_data import preprocess_data

# Other Plumbing
logger = logging.getLogger(__name__)
# ====================================================================|

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


hfmodels = {"flant5" : get_flant5_model,
          "mixtral": get_mixtral_model,
          "bart"   : get_bart_model}
# ==================================================================|
class Prompter(ABC):

    @abstractmethod
    def generate_responses(self, prompt, **kwargs) -> list[str]:
        pass 

    @abstractmethod
    def response_loop(self, data, **kwargs):
        pass

class HFPrompter(Prompter):
    def __init__(self, model_name:str, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, **params):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.__dict__.update(**params)


    def generate_responses(self, prompt) -> list[str]:
        """Prompt `model` to generate output based on
        the question in `data`. 
        *** This is where the LLM stuff happens ***

        ## params
        - prompt: a single question

        ## returns
        list[list[str]] -> a list of multiple responses
        """

        # TODO prompt repetition
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output = self.model.generate(ids, max_length=self.max_len)
        decoder_out = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoder_out

    def response_loop(self, batch, batch_id, write_to_file=False) -> dict[dict]:
        """"""
        pr = HFPrompt()
        prompts = ((i, (pr.base_prompt(question),
                    pr.contradiction_prompt(question),
                    pr.instructive_prompt(question),
                    pr.cot_prompt(question)))
                                            for i, question in enumerate(batch))
        
        main_out = {}
        for idx, prompt_set in prompts:
            titles = ["base", "contradiction", "instructive", "chain_of_thought"]
            responses = {title: self.generate_responses(resp) for title, resp in zip(titles, prompt_set)}
            main_out[idx] = responses

        if write_to_file:
            with open(f'{self.model_name}.{batch_id}.json', 'w+') as f:
                f.write(json.dumps(main_out) + "\n")

        return main_out



class OAIPrompter(Prompter):
    def __init__(self, **kwargs):
        self.model_client = OpenAI()
        self.__dict__.update(**kwargs)

    def generate_responses(self, prompt) -> list[str]:
        pass 
    
    def response_loop(self, batch):
        pr = GPTPrompt()
        for document in batch:
            self.generate_response()
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
    # init configuration
    with open(args.config, 'rb') as f:
        hyperparameters = tomli.load(f)
    model_params = hyperparameters["model_args"]
    model_name = hyperparameters["model_name"].lower().replace("-", "")
    runtime_parameters = hyperparameters["run_params"]
   
   # init dataset
    documents = preprocess_data(args.data_loc, runtime_parameters["batch_size"])

    # init prompter

    # =========================================|
    # GPT API
    if model_name == "gpt":
        gpt_pr = OAIPrompter()
        for batch in documents:
            gpt_pr.response_loop(dataset=batch["question_text"])

    # =========================================|
    # Huggingface Models
    else:
        prompter = HFPrompter(model_name, *hfmodels[model_name](**model_params), **runtime_parameters)

        # run response loop
        for n, batch in enumerate(documents):
            prompter.response_loop(batch["question_text"], 
                                   batch_id = n, 
                                   write_to_file=True)




if __name__ == "__main__":
    args = parse_args()
    main(args)
