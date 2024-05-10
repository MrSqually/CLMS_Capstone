#!/usr/bin/env python3
## CLMS Capstone | LLM Evaluation
# Dean Cahill

from bert_score import BERTScorer
from collections import defaultdict
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from typing import Generator


# ==================================================================|
class HaluMetrics:
    def __init__(self):
        self.bert_scorer = BERTScorer(lang="en")

    def get_haluvar(self, fname) -> int:
        """ """

        model_name = fname.split("/")[2].split(".")[0]
        for init_ctx in ("none", "qa", "halu"):
            for ctx in ("base", "contradict", "instruct", "chain-thought"):
                question_bertscores = self.get_instance_haluvar(fname, ctx, init_ctx)
                for metric, values in question_bertscores.items():
                    val_array = np.array([list(x) for x in values if len(x) == 10])
                    q_scores = np.average(val_array, axis=1)
                    full_score = np.average(val_array)
                    expected_val = np.average(q_scores) / full_score
                    halu_var = (
                        np.average(val_array, axis=0) / full_score
                    ) - expected_val

                    df = pd.DataFrame(halu_var, index=[i for i in range(10)])
                    var_df = pd.DataFrame(val_array)

                    df.plot(title=f"{init_ctx}-{ctx}-{metric}", legend=False)
                    plt.savefig(
                        f"results/figures/{model_name}/{init_ctx}-{ctx}-{metric}-hv.png"
                    )
                    plt.clf()
                    plt.plot(val_array.T)
                    plt.savefig(
                        f"results/figures/{model_name}/{init_ctx}-{ctx}-{metric}-bertscore.png"
                    )
                    plt.clf()
                    var_df.var(axis=0).plot(title=f"{init_ctx}-{ctx}-{metric}")
                    plt.savefig(
                        f"results/figures/{model_name}/{init_ctx}-{ctx}-{metric}-regular-variance.png"
                    )
                    plt.clf()
                    plt.close()

    def get_instance_haluvar(self, fname, ctx, initctx):
        answer_lists = defaultdict(list)
        for qid, answer in tqdm(self.get_pred_answers(fname)):
            refs = self.get_gold_answers(qid)
            if not refs:
                continue
            answers = [x for x in answer[initctx][ctx]]
            scores = [self.get_bertscore(ans, refs) for ans in answers]
            p, r, f1 = zip(*scores)
            answer_lists["precision"].append(p)
            answer_lists["recall"].append(r)
            answer_lists["f1"].append(f1)
        return answer_lists

    def get_bertscore(
        self, cands: list[str], refs: list[str]
    ) -> tuple[tuple[float, float, float], str]:
        """Calculate the mean BERT score for a single
        instance

        ## params
        - cands => candidate sentences
        - refs  => reference sentences

        ## returns
        (precision, recall, f1), hash_value
        """
        answer = "".join(refs["long_answer"])
        cands = "".join(cands)
        answer = re.sub(r"\s+", " ", answer)
        cands = re.sub(r"\s+", " ", cands)

        (p, r, f1) = self.bert_scorer.score([cands], [answer])
        return (p.mean(), r.mean(), f1.mean())

    def iterator(self, fname: os.PathLike | str) -> Generator:
        yield from pd.read_json(fname)

    def get_gold_answers(self, doc_id: str) -> dict[list]:
        """"""
        with open(f"data/simplified-nq-individual/{doc_id}.json", "r") as f:
            json_dict = json.load(f)
        answers = defaultdict(list)

        doc_tokens = re.sub("<[^<]+?>", "", json_dict["document_text"][doc_id]).split(
            " "
        )

        for annotation in json_dict["annotations"][doc_id]:
            lstart, lend = (
                annotation["long_answer"]["start_token"],
                annotation["long_answer"]["end_token"],
            )
            if lstart == -1:
                continue
            answers["long_answer"] = doc_tokens[lstart:lend]
            answers["long_answer"] = answers["long_answer"][1:-1]
            for answer in annotation["short_answers"]:
                answer_start, answer_end = answer["start_token"], answer["end_token"]
                answers["short_answers"].append(doc_tokens[answer_start:answer_end])
        return answers

    def get_pred_answers(self, dirname) -> Generator:
        for fname in os.listdir(dirname):
            with open(f"{dirname}/{fname}") as f:
                json_obj = json.load(f)
            for q_id, answers in json_obj.items():
                yield q_id, answers


if __name__ == "__main__":
    halu = HaluMetrics()
    halu.get_haluvar("results/prompting/gpt")
