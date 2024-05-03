#!/usr/bin/env python3
# CLMS Capstone | LLM Prompting
# Dean Cahill

# ==================================================================|
# Model Imports

### Huggingface

### (L)LLM APIs
from openai import OpenAI
import replicate 

# StandardLib & Misc
from abc import ABC, abstractmethod
import argparse
from collections import defaultdict
import logging 
import json 
import pandas as pd 
import random
import tomli  
from tqdm import tqdm

# Other Plumbing
logger = logging.getLogger(__name__)
# ============================================================================|
def generate_data(doc_file, batch_size):
    yield from pd.read_json(doc_file, lines=True, chunksize=batch_size)

# ============================================================================|
class Prompter(ABC):

    # ======================|
    # Abstract Methods
    @abstractmethod
    def modulate_context(prompt, context_type):
        pass
    @abstractmethod
    def generate_responses(self, prompt, **kwargs) -> list[str]:
        pass 
    @abstractmethod
    def response_loop(self, data, **kwargs):
        pass

    # ======================|
    # Static Methods
    @staticmethod
    def get_contra_ctx():
        return random.choice(["That's incorrect",
                          "You are wrong",
                          "That answer is wrong",
                          "You misunderstood my question",
                          "You've made a mistake",
                          "I don't believe that that's correct"])

    @staticmethod
    def get_instruct_ctx():
        return random.choice(["Could you repeat that?",
                          "Could you say that again?",
                          "Give me another answer",
                          "Say that again, but differently",
                          "Say that again, please",
                          "Repeat your response, but use different verbiage",
                          "Word that differently",
                          "Say that again, with different verbs",
                          "Say that again, but with more action",
                          "Say that again, with different adjectives",
                          "Say that again, but with different nouns"])
    
    @staticmethod
    def get_qa_initial_ctx():
          # Context Strings 
        return """CONTEXT: You are a Question-Answering service. Answer the following question as correctly as possible with as little extraneous information as possible."""
        
    @staticmethod
    def get_halu_initial_ctx():
        return """CONTEXT: You are a Question-Answering service being evaluated for the capacity to generate hallucinations through repetition. You will be asked the following question many times. Do not hallucinate extraneous information."""
    # ======================|
    # Concrete Methods
    def get_initial_ctx(self, init_ctx):
        if init_ctx == "qa":
            return self.get_qa_initial_ctx()      # 'naive' QA context
        if init_ctx == "halu":
            return self.get_halu_initial_ctx()    # 'aware' QA context
        else:
            return "You are a helpful assistant." # base gpt prompt

    def response_loop(self, batch, idx):
        """"""
        batch_out = {}
        batch_keys = {}
        for i, question in enumerate(batch):
            question_out = defaultdict(lambda: defaultdict(dict))
            for init_context in ("none", "qa", "halu"):
                prompt = self.get_initial_ctx(init_context) + question
                for context in ('base', 'contradict', 'instruct', 'chain-thought'):
                    question_out[init_context].update({context: self.generate_responses(prompt, context)})

            batch_out[i] = question_out
            batch_keys[i] = question
            with open(f"{self.model_name}.{idx}.json", 'a+') as f, open(f"{self.model_name}.{idx}.keys.json", "a+") as k:
                f.write(json.dumps(batch_out, indent=2))
                k.write(json.dumps(batch_keys, indent=2))
    
# ============================================================================|
class LlamaPrompter(Prompter):
    prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{context}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    def modulate_context(self,context_type):
        match context_type:
            case "contradict":
                return self.get_contra_ctx()
            case "instruct":
                return self.get_instruct_ctx()
            case _:
                return ""

    def generate_responses(self, prompt, context) -> list[str]:
        out = []
        model_in = {"prompt": prompt,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "temperature": self.temperature,
                    "system_prompt": self.get_initial_ctx(context),
                    "length_penalty": self.length_penalty,
                    "max_new_tokens": self.max_tokens,
                    "prompt_template": self.prompt_template,
                    "presence_penalty": self.presence_penalty}
        
        for i in range(self.num_repetitions):
            response = replicate.run(self.model,
                                      input=model_in)
            model_out = "".join(response)
            #TODO repetition
            
            
        return out 
# ============================================================================|
class MixtralPrompter(Prompter):

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
    
    def modulate_context(self, context_type):
        match context_type:
            case "contradict":
                return self.get_contra_ctx()
            case "instruct":
                return self.get_instruct_ctx()
            case _:
                return ""

    def generate_responses(self, prompt, context="base") -> list[str]:        
        out = []
        model_in = {"prompt": prompt,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "temperature": self.temperature,
                    "system_prompt": self.get_initial_ctx(context),
                    "length_penalty": self.length_penalty,
                    "max_new_tokens": self.max_tokens,
                    "prompt_template": "<s>[INST] {prompt} [/INST] ",
                    "presence_penalty": self.presence_penalty}
        
        for i in range(self.num_repetitions):
            response = replicate.run(self.model,
                                      input=input,
                                      )
            model_out = "".join(response)
            out.append(model_out)
            model_in["prompt"] = model_in["prompt"] + f"[INST] {model_out} [/INST]"
            model_in["prompt_template"] = f"<s>[INST]{self.modulate_context(context)}" + "{prompt}"
        return out 
# ============================================================================|
class GPTPrompter(Prompter):

    def __init__(self, **kwargs):
        self.client = OpenAI()
        self.__dict__.update(**kwargs)

    def modulate_context(self, prompt, context_type):
        match context_type:
            case "contradict":
                mod = self.get_contra_ctx()
                return {"role": "user", "content": f"{mod}\n{prompt}"}
            case "instruct":
                mod = self.get_instruct_ctx()
                return {"role":"user", "content": f"{mod}\n{prompt}"}
            case _:
                return {"role":"user", "content": f"{prompt}"}

    def generate_responses(self, prompt, context) -> list[str]:
        out = []
        formatted_init_prompt = [{"role":"system", "content": self.get_initial_ctx(context)},
                                 {"role":"user", "content": prompt}]
        model_in = formatted_init_prompt
        for _ in tqdm(range(self.num_reps)):
            response = self.client.chat.completions.create(
                model = self.model,
                frequency_penalty=self.frequency_penalty,
                max_tokens=self.max_tokens,
                presence_penalty=self.presence_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                messages = model_in)
            model_out = response.choices[0].message.content
            out.append(model_out)
            new_ctx = self.modulate_context(prompt, context) # modulate context by adding prompt repetition
            model_in.append(new_ctx)
        return out 

# ============================================================================|
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
    runtime_parameters = hyperparameters["experimental_params"]
   
   # init dataset
    documents = generate_data(args.data_loc, runtime_parameters["batch_size"])

    # init prompter
    match runtime_parameters["model_name"]:
        # GPT API
        case "gpt":
            runtime_parameters.update(hyperparameters["gpt_prompt_params"])
            pr = GPTPrompter(**runtime_parameters)
        # Mixtral API 
        case "mixtral":
            runtime_parameters.update(hyperparameters["mixtral_prompt_params"])
            pr = MixtralPrompter(**runtime_parameters)
        # Llama (3?)
        case "llama":
            runtime_parameters["prompt_params"] = hyperparameters["mixtral_prompt_params"]
            pr = LlamaPrompter(**runtime_parameters)


    # response-generation loop
    for n, batch in tqdm(enumerate(documents)):
        pr.response_loop(batch["question_text"], n)
        if n == hyperparameters["num_docs"]:
            break 

    print("Complete!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
