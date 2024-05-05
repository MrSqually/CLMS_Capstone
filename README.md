# Can You Say That Again? Hallucinatory Variance in Large Language Models

This repository contains the source code, documentation, and writeup information
for my Computational Linguistics MS Capstone.

## Prompt Code

Prompts are generated via `src/prompting/repetition_prompting.py`. This script contains code to query various APIs for LLM reponse, as well as the
loop to concatenate context & generate responses. If the user has Replicate and OpenAI API keys, the code should work from jump.

This script is designed to choose a model to run on by reading the "model_name" parameter within the configuration file. This allowed us to run multiple
LLM prompt cycles concurrently by simply altering the configuration between script runs in separate terminals.

## Evaluation Code

The evaluation script is designed to run on the prompting output. Further optimization can be made, but the nature of the prompting task made data-wrangling for efficient
comparison somewhat difficult (the outermost structure of the data is the question, not the loci of comparison, such as initial context).

## Topic Model

While the topic model was not used in the final writeup, the code to generate clusters over Natural Questions is made available in `src/topic_modeling.py`. The
metric's inability to capture the phenomenon made testing on the clusters seem like a waste of time and resources.
