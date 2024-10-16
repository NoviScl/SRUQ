from openai import OpenAI
import anthropic
from utils import call_api
import argparse
import json
import os
import random 
import retry
import numpy as np
from tqdm import tqdm 
from datasets import load_dataset


def load_gsm8k(test_size=100):
    '''
    Load the GSM8K dataset and clean up the data format.
    test_size: the number of test samples to sample
    '''

    dataset = load_dataset("gsm8k", "main")
    train = list(dataset["train"])
    test = list(dataset["test"])
    random.shuffle(train)
    random.shuffle(test)
    test = test[:test_size]

    for i in range(len(train)):
        train[i]["answer"] = train[i]["answer"].split("####")[-1].strip()
    for i in range(len(test)):
        test[i]["answer"] = test[i]["answer"].split("####")[-1].strip()
    print ("Train split size: ", len(train))
    print ("Test split size: ", len(test))
    
    return train, test

def brier_score(accuracy_lst, confidence_lst):
    '''
    Compute the brier score given the list of model predictions, gold answers, and the corresponding confidence scores.
    '''
    n = len(accuracy_lst)
    squared_errors = [(confidence - accuracy)**2 for confidence, accuracy in zip(confidence_lst, accuracy_lst)]
    brier_score = sum(squared_errors) / n
    
    return brier_score

def expected_calibration_error(accuracy_lst, confidence_lst, n_bins=10):

    n = len(accuracy_lst)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = [confidence_lst[i] > bin_lower and confidence_lst[i] <= bin_upper for i in range(n)]
        prop_in_bin = sum(in_bin) / n
        if prop_in_bin > 0:
            accuracy_in_bin = sum([accuracies[i] for i in range(n) if in_bin[i]]) / sum(in_bin)
            confidence_in_bin = sum([confidence_lst[i] for i in range(n) if in_bin[i]]) / sum(in_bin)
            ece += prop_in_bin * abs(accuracy_in_bin - confidence_in_bin)
    
    return ece

def tokens_to_confidence(logprobs_lst):
    '''
    Given a list of logprobs, calculate the confidence score.
    '''
    logprobs = [token.logprob for token in logprobs_lst]
    confidence = 1
    for logprob in logprobs:
        confidence *= np.exp(logprob)
    # Normalize by length
    confidence = confidence ** (1 / len(logprobs))
    return confidence

@retry.retry(tries=3, delay=2)
def logprob_confidence(query, train_set, openai_client, model, temperature, seed, demo_num=8):
    '''
    Given a query, generate a response along with the confidence score.
    The confidence score the product of each token's log probability normalized by the number of tokens.
    '''
    prompt = "Answer the following question. Directly give the final numerical answer, no need to explain or generate a full sentence. No punctuation or other symbols allowed, just output the number. Follow the format of the given examples.\n"
    ## sample demo examples
    demo_examples = random.sample(train_set, demo_num)
    for i in range(demo_num):
        prompt += f"Q: {demo_examples[i]['question']}\nA: {demo_examples[i]['answer']}\n\n"
    ## add the query to the prompt
    prompt += f"Q: {query}\nA: "
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost, logprobs = call_api(openai_client, model, prompt_messages, temperature=temperature, max_tokens=16, seed=seed, json_output=False)
    confidence = tokens_to_confidence(logprobs)
    
    return prompt, response, cost, confidence

@retry.retry(tries=3, delay=2)
def ensemble_confidence(query, train_set, openai_client, model, temperature, seed, num_prompts=10, demo_num=8):
    '''
    Given a query, generate a response along with the confidence score.
    Try num_prompts different prompts and return the most frequent answer.
    The confidence score is the frequency of the most common answer.
    '''
    responses = []
    costs = 0
    for _ in range(num_prompts):
        prompt, response, cost, _ = logprob_confidence(query, train_set, openai_client, model, temperature, seed, demo_num)
        responses.append(response)
        costs += cost
    
    # Count the frequency of each response
    response_counts = {}
    for response in responses:
        response_counts[response] = response_counts.get(response, 0) + 1
    
    # Find the most common response
    most_common_response = max(response_counts, key=response_counts.get)
    confidence = response_counts[most_common_response] / num_prompts
    
    return prompt, most_common_response, costs, confidence

@retry.retry(tries=3, delay=2)
def sruq_confidence(query, train_set, openai_client, model, temperature, seed, demo_num=8):
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

    random.seed(args.seed)

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

    # Load the GSM8K dataset
    train, test = load_gsm8k()
    
    # experiment on testset 
    predictions = []
    confidences = []
    accuracies = []
    costs = 0
    for i in tqdm(range(len(test))):
        query = test[i]["question"]
        gold_answer = test[i]["answer"]
        prompt, response, cost, confidence = logprob_confidence(query, train, client, args.engine, temperature=0.7, seed=args.seed)
        
        accuracy = 1 if response == gold_answer else 0
        accuracies.append(accuracy)
        predictions.append(response)
        confidences.append(confidence)
        costs += cost

    print ("Method: Logprob Confidence")
    # print ("Predictions: ", predictions)
    # print ("Gold Answers: ", [test[i]["answer"] for i in range(len(test))])
    # print ("Accuracies: ", accuracies)
    # print ("Confidences: ", confidences)
    print ("Accuracy: ", sum(accuracies) / len(accuracies))
    print ("Brier Score: ", brier_score(accuracies, confidences))
    print ("Expected Calibration Error: ", expected_calibration_error(accuracies, confidences))
    print ("Total Cost: ", costs)
    print ("----------------------------------------")

    # experiment on testset 
    predictions = []
    confidences = []
    accuracies = []
    costs = 0
    for i in tqdm(range(len(test))):
        query = test[i]["question"]
        gold_answer = test[i]["answer"]
        prompt, response, cost, confidence = ensemble_confidence(query, train, client, args.engine, temperature=0.7, seed=args.seed)
        
        accuracy = 1 if response == gold_answer else 0
        accuracies.append(accuracy)
        predictions.append(response)
        confidences.append(confidence)
        costs += cost

    print ("Method: Ensemble Confidence")
    print ("Predictions: ", predictions)
    print ("Gold Answers: ", [test[i]["answer"] for i in range(len(test))])
    print ("Accuracies: ", accuracies)
    print ("Confidences: ", confidences)
    print ("Accuracy: ", sum(accuracies) / len(accuracies))
    print ("Brier Score: ", brier_score(accuracies, confidences))
    print ("Expected Calibration Error: ", expected_calibration_error(accuracies, confidences))
    print ("Total Cost: ", costs)

   