from openai import OpenAI
import anthropic
from utils import call_api
import argparse
import json
import os
import random 
import retry
import numpy as np

def tokens_to_confidence(logprobs_lst):
    '''
    Given a list of logprobs, calculate the confidence score.
    '''
    logprobs = [token.logprob for token in logprobs_lst]
    print(logprobs)
    confidence = 1
    for logprob in logprobs:
        confidence *= np.exp(logprob)
    # Normalize by length
    confidence = confidence ** (1 / len(logprobs))
    return confidence

@retry.retry(tries=3, delay=2)
def logprob_confidence(query, openai_client, model, temperature, seed):
    '''
    Given a query, generate a response along with the confidence score.
    The confidence score the product of each token's log probability normalized by the number of tokens.
    '''
    prompt = query
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost, logprobs = call_api(openai_client, model, prompt_messages, temperature=temperature, max_tokens=16, seed=seed, json_output=False)
    return prompt, response, cost, logprobs

@retry.retry(tries=3, delay=2)
def ensemble_confidence(query, num_prompts, openai_client, model, temperature, seed):
    '''
    Given a query, generate a response along with the confidence score.
    Try num_prompts different prompts and return the most frequent answer.
    The confidence score is the frequency of the most common answer.
    '''
    return 

@retry.retry(tries=3, delay=2)
def sruq_confidence(query, openai_client, model, seed):
    '''
    Given a query, generate a response along with the confidence score.
    The confidence score is calculated by a graph-based method.
    '''
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-opus-20240229', help='api engine; https://openai.com/api/')
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()

    with open("keys.json", "r") as f:
        keys = json.load(f)
    random.seed(args.seed)

    ANTH_KEY = keys["anthropic_key"]
    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    
    if "claude" in args.engine:
        client = anthropic.Anthropic(
            api_key=ANTH_KEY,
        )
    else:
        client = OpenAI(
            organization=ORG_ID,
            api_key=OAI_KEY
        )

    example_query = "What is the capital of China?"
    prompt, response, cost, logprobs = logprob_confidence(example_query, client, args.engine, temperature=0.7, seed=args.seed)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Cost: {cost}")
    print(f"Logprobs: {logprobs}")
    print(f"Confidence: {tokens_to_confidence(logprobs)}")
