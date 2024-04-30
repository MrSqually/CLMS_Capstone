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
from tqdm import tqdm

# Local
from prompt_construction import FLANT5Prompt, GPTPrompt, MixtralPrompt
from load_data import preprocess_data

# Other Plumbing
logger = logging.getLogger(__name__)

# ==================================================================|
class Prompter(ABC):

    @abstractmethod
    def generate_responses(self, prompt, **kwargs) -> list[str]:
        pass 

    @abstractmethod
    def response_loop(self, data, **kwargs):
        pass

class FlanT5Prompter(Prompter):
    def __init__(self, **params):
        self.model_name = "google/flan-t5-small"
        self.model =  AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **params)
        self.pr = FLANT5Prompt()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
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
        encoder_in = prompt
        prompt_responses = []
        for i in range(self.num_repetitions):
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            output = self.model.generate(ids, max_length=self.max_len)
            decoder_out = self.tokenizer.decode(output[0], skip_special_tokens=True)
            prompt_responses.append(decoder_out)
            encoder_in = encoder_in + decoder_out + prompt
        return prompt_responses

    def response_loop(self, batch, batch_id, write_to_file=False) -> dict[dict]:
        """"""
        
        main_out, question_keys = {}, {}
        for i, question in enumerate(batch):
            prompts = (self.pr.base_prompt(question),
                          self.pr.contradiction_prompt(question),
                          self.pr.instructive_prompt(question),
                          self.pr.cot_prompt(question))
            question_keys[i] = question      
            titles = ["base", "contradiction", "instructive", "chain_of_thought"]
            responses = {title: self.generate_responses(resp) for title, resp in zip(titles, prompts)}
            main_out[i] = responses
            
        if write_to_file:
            with open(f"results/prompting/{self.model_name}.{batch_id}.keys.json", 'w+') as f:
                f.write(json.dumps(question_keys, indent=2))
            with open(f'results/prompting/{self.model_name}.{batch_id}.json', 'w+') as f:
                f.write(json.dumps(main_out, indent=2) + "\n")
        return main_out

class MixtralPrompter(Prompter):
    def __init__(self,  model_params, **kwargs):
        self.model_name = "mistralai/Mixtral-8x7B-v0.1"
        self.pr = MixtralPrompt()
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_params)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.__dict__.update(**kwargs)

    def generate_responses(self, prompt) -> list[str]:        
        model_in = prompt 
        for i in range(self.num_repetitions):
            # TODO 
            pass 

    def response_loop(self, batch, batch_id, write_to_file=False):
        main_out, question_keys = {}, {}

        for i, question in enumerate(batch):
            question_keys[i] = question
            prompts =  (self.pr.base_prompt(question),
                        self.pr.contradiction_prompt(question),
                        self.pr.instructive_prompt(question),
                        self.pr.cot_prompt(question))
            titles = ["base", "contradiction", "instructive", "chain_of_thought"]
            responses = {title: self.generate_responses(resp) for title, resp in zip(titles, prompts)}
            main_out[i] = responses

        if write_to_file:
            with open(f"results/prompting/{self.model_name}.{batch_id}.keys.json", 'w+') as f:
                f.write(json.dumps(question_keys, indent=2))
            with open(f'results/prompting/{self.model_name}.{batch_id}.json', 'w+') as f:
                f.write(json.dumps(main_out, indent=2) + "\n")

class OAIPrompter(Prompter):
    def __init__(self, **kwargs):
        self.model_client = OpenAI()
        self.pr = GPTPrompt()
        self.__dict__.update(**kwargs)

    def generate_responses(self, prompt) -> list[str]:
        
        model_in = prompt 
        for i in range(self.num_repetitions):
            completion = self.model_client.chat.completions.create(model="",
                                                      messages=model_in)
            model_out = completion.choices[0].message
            model_in.append(model_out)
            model_in.append(prompt)

    def response_loop(self, batch, batch_id, write_to_file=False):
        main_out, question_keys = {}, {}

        for i, question in enumerate(batch):
            question_keys[i] = question
            prompts =  (self.pr.base_prompt(question),
                        self.pr.contradiction_prompt(question),
                        self.pr.instructive_prompt(question),
                        self.pr.cot_prompt(question))
            titles = ["base", "contradiction", "instructive", "chain_of_thought"]
            main_out[i] = {title: self.generate_responses(resp) for title, resp in zip(titles, prompts)}

        if write_to_file:
            with open(f"results/prompting/{self.model_name}.{batch_id}.keys.json", 'w+') as f:
                f.write(json.dumps(question_keys, indent=2))
            with open(f'results/prompting/{self.model_name}.{batch_id}.json', 'w+') as f:
                f.write(json.dumps(main_out, indent=2) + "\n")

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
        gpt_pr = OAIPrompter({"model_name": model_name})
        for n, batch in tqdm(enumerate(documents)):
            gpt_pr.response_loop(dataset=batch["question_text"])

    # =========================================|
    # Huggingface Models
    if model_name == "flant5":
        prompter = FlanT5Prompter()
        # run response loop
        for n, batch in tqdm(enumerate(documents)):
            prompter.response_loop(batch["question_text"], 
                                   batch_id = n, 
                                   write_to_file=True)
    if model_name == "mixtral":
        prompter = MixtralPrompter()
        # run response loop
        for n, batch in tqdm(enumerate(documents)):
            prompter.response_loop(batch["question_text"], 
                                   batch_id = n, 
                                   write_to_file=True)        

if __name__ == "__main__":
    args = parse_args()
    main(args)
