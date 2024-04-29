# Can You Say That Again? Hallucinatory Variance in Large Language Models

This repository contains the source code, documentation, and writeup information
for my Computational Linguistics MS Capstone.

## Overview / Pipeline

### TODO

1. Incorporating topic model into prompting pipeline
    - currently, the topic model is more "proof of concept" than pipeline component. It functions, but
    doesn't actually cluster the documents.
2. Improving prompt pipeline
    - FLAN tuning uses a specific set of prompt templates - leverage these.
    - randomization & multiple-turn prompts
3. Evaluation
    - Basically just data-wrangling the "answer span" offsets into a string representation and comparing
    that for eval metrics listed in `evaluation.py`
4. Information Extraction (retrieval tuning)
    - A pre-process script for QAGNN was included in the source code. I still need to modify this to work on
    the NQ dataset.
    - piping QAGNN into this schema
5. Discussion & Deliverables
    - Paper length?
    - Presentation length?
