#!/usr/bin/env python3
# CLMS Capstone | Prompt Generation
# Dean Cahill
import random

""" A collection of functions which return prompts as a tuple of strings.
Each prompt returns 3 strings:
1. the prompt itself
2. A "naive context" prompt, modified by the QA task context. 
3. An "aware context" prompt, modified by the hallucination task context (a "metatask")
"""


qa_context: str = """CONTEXT: You are a Question-Answering service. Answer the following question as correctly as possible with as little extraneous information as possible."""

halu_context: str = """CONTEXT: You are a Question-Answering service being evaluated for
the capacity to generate hallucinations through repetition. You will be asked the following question many times. Do not hallucinate extraneous information."""

# ============================================================================|
def get_prompts():
    return basic_prompt, chain_of_thought_prompt, contradiction_prompt, instructive_prompt,


def basic_prompt(question) -> tuple[str, str, str]:
    return f"{question}", f"{qa_context}\n\n{question}", f"{halu_context}\n\n{question}"


def contradiction_prompt(question) -> tuple[str, str, str]:
    mod_string = random.choice(
        [
            "that is incorrect",
            "you're wrong",
        ]
    )
    tgt_string = f"{mod_string}. {question}"

    return tgt_string, f"{qa_context}\n\n{question}", f"{halu_context}\n\n{question}"


def instructive_prompt(question) -> tuple[str, str, str]:
    mod_string = random.choice(
        [
            "",
            ", but replace all the nouns",
            ", but replace all the verbs",
            ", and use synonyms",
            ", but word that differently",
            ", but more verbose",
            ", but less verbose",
        ]
    )
    tgt_string = f"Could you repeat that{mod_string}: {question}"
    return (
        tgt_string,
        f"{qa_context}\n\n{tgt_string}",
        f"{halu_context}\n\n{tgt_string}",
    )


def chain_of_thought_prompt(question) -> tuple[str, str, str]:
    tgt_string = f"Answer the following question, explaining each step of your reasoning: {question}"
    return (
        tgt_string,
        f"{qa_context}\n\n{tgt_string}",
        f"{halu_context}\n\n{tgt_string}",
    )
