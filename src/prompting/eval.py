#!/usr/bin/env python3
## CLMS Capstone | LLM Evaluation
# Dean Cahill
"""
Four primary forms of evaluation:
1.) Simplex Correctness -  Does the response contain `short_answer` in the text?
    a.) "Strict" Recall (Adlakha et al 2024)
2.) Complex Correctness -  similarity full answer and response tokens (normalized)
    a.) ROUGE
    b.) BLEU
    c.) Precision/Recall/F1
3.) Faithfulness - Token Overlap Precision
    a.) K-Precision
        i.) "faithfulness wrt relevant knowledge"  (Adlakha et al 2024)
4.) Model Adjudication with DeBERTa

NOTE
traditional QA metrics penalize lexical matching too much -
verbosity is not inherently bad.
"""

import torch
# import DeBERTa


# ==================================================================|
class HaluMetrics:
    @staticmethod
    def simple_correctness(model_response, short_answer):
        pass

    @staticmethod
    def complex_correctness(model_response, full_answer):
        pass

    @staticmethod
    def faithfulness(model_response, full_answer):
        pass

    @staticmethod
    def model_adjudication(model_response):
        """Uses DeBERTa to adjudicate output 
        on the three criteria:

        - contradiction 
        - entailment 
        - neutral 

        """
        with torch.no_grad():
            label_mapping = ["contradiction", "entailment", "neutral"]
            labels = [
                label_mapping[score_max] for score_max in logits.argmax(dim=1)
            ]
        return labels
