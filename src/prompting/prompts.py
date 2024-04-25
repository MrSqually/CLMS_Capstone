#!/usr/bin/env python3
## CLMS Capstone | Prompt Generation
# Dean Cahill

# A collection of functions which return prompts as strings.
# ============================================================================|


def bare_prompt(question) -> str:
    """Prompt for basic QA
    STATUS TODO
    """
    return f"{question}"


def context_prompt(question, context) -> str:
    """Prompt for extended contextualization
    STATUS TODO
    """
    return f"{context} | {question}"


def contradiction_prompt() -> str:
    """Prompt for direct contradiction
    STATUS TODO
    """
    pass


def chain_of_thought_prompt(question) -> str:
    """Prompt for step-by-step (CoT) reasoning
    STATUS TODO
    """
    return f"Answer the following question, explaining each step of your reasoning: {question}"


qa_context: str = """CONTEXT: You are a Question-Answering service. Your job is to
ensure that the following question is answered as correctly as possible with as little
extraneous information as possible."""

halu_context: str = """CONTEXT: You are a Question-Answering service being evaluated for
the capacity to generate hallucinations. Answer the following question. Do not hallucinate
extraneous information."""
