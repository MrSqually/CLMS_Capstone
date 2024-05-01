#!/usr/bin/env python3
## CLMS Capstone | LLM Evaluation
# Dean Cahill

from bert_score import BERTScorer
from collections import defaultdict
import json
import matplotlib 
import os 
import pandas as pd 
import re

from typing import Generator
# ==================================================================|
class HaluMetrics:

    def __init__(self):
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    def get_haluvar(self, cands: list[list[str]], refs: list[str], num_reps: int) -> int:
        pass 

    def get_bertscore(self, cands: list[str], refs: list[str]) -> tuple[tuple[float, float, float],str]:
        """Calculate the mean BERT score for a single 
        instance

        ## params 
        - cands => candidate sentences 
        - refs  => reference sentences

        ## returns
        (precision, recall, f1), hash_value
        """
        (p, r, f1), hash_val = self.bert_scorer.score(cands, refs, return_hash=True)
        return (p.mean(), r.mean(), f1.mean()), hash_val

    def iterator(self, fname: os.PathLike | str) -> Generator:
        yield from pd.read_json(fname, chunksize=1)


    def get_gold_answers(self, doc_id: str) -> dict[list]:
        """"""
        with open(f"data/simplified-nq-individual/{doc_id}.json", 'r') as f:
            json_dict = json.load(f)
        answers = defaultdict(list)
        
        doc_tokens = re.sub('<[^<]+?>', '', json_dict["document_text"][doc_id]).split(" ")

        for annotation in json_dict["annotations"][doc_id]:
            lstart, lend = annotation["long_answer"]["start_token"], annotation["long_answer"]["end_token"]
            if lstart == -1:
                continue
            answers["long_answer"] = doc_tokens[lstart: lend]
            answers["long_answer"] = answers["long_answer"][1:-1]
            for answer in annotation["short_answers"]:
                answer_start, answer_end = answer["start_token"], answer["end_token"]
                answers["short_answers"].append(doc_tokens[answer_start:answer_end])
        return answers


    def evaluation_loop(self, response_fname: os.PathLike | str):
        response_iter = self.iterator(response_fname)
        
        for response in response_iter:
            golds = self.get_gold_answers(response["id"])
            hv = self.get_haluvar(golds, response["text"])

         

    def visualize_hallucinatory_variance(self, hv_results: list[float]):
        pass 

if __name__ == "__main__":
    halu = HaluMetrics()
    for i in range(10):
        print(halu.get_gold_answers(str(i)))
