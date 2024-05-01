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

From these evaluation metrics, we build HV. 

NOTE 
HaluVar should:

1.) reflect the deviation in hallucination rate from expected as context grows
2.) Given a hallucination rate, normalize the hallucination rate of the repetitions by their size, compare the hallucination rate at time point (t) to the overall average hallucination rate
3.) increase non-linearly with the raw number of hallucinations
4.) be bounded between [0,1)
5.) more...

NOTE
traditional QA metrics penalize lexical matching too much:
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
        pass 