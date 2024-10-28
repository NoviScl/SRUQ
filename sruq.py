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
import networkx as nx


def softmax(x):
    '''
    Compute the softmax of a list of numbers.
    '''
    return np.exp(x) / np.sum(np.exp(x))

def load_gsm8k(test_size=100, CoT=False):
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
        if CoT:
            train[i]["answer"] = train[i]["answer"].strip()
        else:
            train[i]["answer"] = train[i]["answer"].split("####")[-1].strip()
    for i in range(len(test)):
        test[i]["answer"] = test[i]["answer"].split("####")[-1].strip()
    print ("Train split size: ", len(train))
    print ("Test split size: ", len(test))
    
    return train, test


def answer_extractor(response, dataset="gsm8k"):
    '''
    Extract the answer from the response.
    '''
    if dataset == "gsm8k":
        response = response.split("####")[-1].strip()
        response = str(int(float(response)))
    
    return response.strip()
    

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
            accuracy_in_bin = sum([accuracy_lst[i] for i in range(n) if in_bin[i]]) / sum(in_bin)
            confidence_in_bin = sum([confidence_lst[i] for i in range(n) if in_bin[i]]) / sum(in_bin)
            ece += prop_in_bin * abs(accuracy_in_bin - confidence_in_bin)
    
    return ece


def pairwise_similarity(solution_a, solution_b, metric="char_jaccard"):
    '''
    Compute the pairwise similarity between two solutions.
    We can try various metrics to compute the similarity.
    '''
    solution_a = solution_a.strip().lower()
    solution_b = solution_b.strip().lower()
    similarity = 0
    if metric == "char_jaccard":
        similarity = len(set(solution_a).intersection(set(solution_b))) / len(set(solution_a).union(set(solution_b)))
    elif metric == "word_jaccard":
        solution_a = solution_a.split()
        solution_b = solution_b.split()
        similarity = len(set(solution_a).intersection(set(solution_b))) / len(set(solution_a).union(set(solution_b)))
    
    return similarity


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
def logprob_confidence(dataset, query, train_set, openai_client, model, temperature, seed, CoT=False, demo_num=8, extract_answer=True):
    '''
    Given a query, generate a response along with the confidence score.
    The confidence score is the product of each token's log probability normalized by the number of tokens.
    '''
    if not CoT: 
        prompt = "Answer the question. Directly give the final answer, no need to explain or generate a full sentence. No punctuation or other symbols needed, just output the answer itself. Follow the format of the given examples.\n"
    else:
        prompt = "Answer the question. Explain your reasoning step by step before giving the final answer. Follow the format of the given examples.\n"
    
    ## sample demo examples
    demo_examples = random.sample(train_set, demo_num)
    for i in range(demo_num):
        prompt += f"Q: {demo_examples[i]['question']}\nA: {demo_examples[i]['answer']}\n\n"
    ## add the query to the prompt
    prompt += f"Q: {query}\nA: "
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost, logprobs = call_api(openai_client, model, prompt_messages, temperature=temperature, max_tokens=1024, seed=seed, json_output=False)

    # print ("prompt: ", prompt)
    # print ("raw response: ", response)

    if extract_answer:
        response = answer_extractor(response, dataset)
    confidence = tokens_to_confidence(logprobs)
    
    return prompt, response, cost, confidence


@retry.retry(tries=3, delay=2)
def ensemble_confidence(dataset, query, train_set, openai_client, model, temperature, seed, num_prompts=5, CoT=False, demo_num=8):
    '''
    Given a query, generate a response along with the confidence score.
    Try num_prompts different prompts and return the most frequent answer.
    The confidence score is the frequency of the most common answer.
    '''
    responses = []
    costs = 0
    for _ in range(num_prompts):
        prompt, response, cost, _ = logprob_confidence(dataset, query, train_set, openai_client, model, temperature, seed, CoT, demo_num)
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


def graph(solutions, metric="jaccard", centrality="eigenvector"):
    # Create a similarity matrix (symmetric matrix with pairwise similarity weights)
    n = len(solutions)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i][j] = 1.0  # Self-similarity is always 1
            else:
                similarity = pairwise_similarity(solutions[i], solutions[j], metric)
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity  # Ensure symmetry

    # print ("solutions: ", solutions)
    # print ("similarity matrix: ", similarity_matrix)
    
    # Create a graph
    G = nx.Graph()

    # Add edges with weights based on the similarity matrix
    for i in range(len(solutions)):
        for j in range(i + 1, len(solutions)):
            G.add_edge(solutions[i], solutions[j], weight=similarity_matrix[i, j])

    # Compute centrality
    if centrality == "eigenvector":
        centrality = nx.eigenvector_centrality(G, weight='weight')
    elif centrality == "pagerank":
        centrality = nx.pagerank(G, weight='weight')
    
    # Sum up centrality for identical solutions
    unique_solutions = {}
    for solution, score in centrality.items():
        if solution in unique_solutions:
            unique_solutions[solution] += score
        else:
            unique_solutions[solution] = score
    
    # Normalize the confidence scores
    total_score = sum(unique_solutions.values())
    normalized_centrality = {sol: score / total_score for sol, score in unique_solutions.items()}
    
    # Update the centrality dictionary with normalized scores
    centrality = normalized_centrality

    # # Print the centrality of each solution
    # print("Centrality (Confidence Estimation):")
    # for solution, score in centrality.items():
    #     print(f"{solution}: {score:.4f}")

    # Find the most confident solution (highest centrality) and its confidence
    most_confident_solution = max(centrality, key=centrality.get)
    confidence = centrality[most_confident_solution]
    # print(f"\nMost confident solution: {most_confident_solution}")
    # print(f"Confidence score: {confidence:.4f}")
    
    return most_confident_solution, confidence


@retry.retry(tries=3, delay=2)
def sruq_confidence(dataset, query, train_set, openai_client, model, temperature, seed, num_prompts=5, CoT=False, demo_num=8):
    '''
    Given a query, generate a response along with the confidence score.
    The confidence score is calculated by a graph-based method.
    '''
    responses = []
    costs = 0
    for _ in range(num_prompts):
        prompt, response, cost, _ = logprob_confidence(dataset, query, train_set, openai_client, model, temperature, seed, CoT, demo_num, extract_answer=False)
        responses.append(response)
        costs += cost
    
    most_confident_solution, confidence = graph(responses)

    return prompt, most_confident_solution, costs, confidence


def experiment(client, test_size=100, dataset="gsm8k", CoT=False, method="logprob", engine="gpt-4o-mini", temperature=0.7, seed=2024, num_prompts=5, demo_num=8):
    '''
    Run the experiment on the given dataset.
    '''
    if dataset == "gsm8k":
        train, test = load_gsm8k(test_size=test_size, CoT=CoT)
    
    predictions = []
    confidences = []
    accuracies = []
    costs = 0

    for i in tqdm(range(len(test))):
        query = test[i]["question"]
        gold_answer = test[i]["answer"]
        if method == "logprob":
            prompt, response, cost, confidence = logprob_confidence(dataset, query, train, client, engine, temperature=temperature, seed=seed, CoT=CoT, demo_num=demo_num)
        elif method == "ensemble":
            prompt, response, cost, confidence = ensemble_confidence(dataset, query, train, client, engine, temperature=temperature, seed=seed, num_prompts=num_prompts, CoT=CoT, demo_num=demo_num)
        elif method == "sruq":
            prompt, response, cost, confidence = sruq_confidence(query, train, client, engine, temperature=temperature, seed=seed, num_prompts=num_prompts, demo_num=demo_num)

        response = answer_extractor(response, dataset)
        accuracy = 1 if response == gold_answer else 0
        accuracies.append(accuracy)
        predictions.append(response)
        confidences.append(confidence)
        costs += cost


    print ("Method: ", method)
    print ("Accuracy: ", sum(accuracies) / len(accuracies))
    print ("Brier Score: ", brier_score(accuracies, confidences))
    print ("Expected Calibration Error: ", expected_calibration_error(accuracies, confidences))
    print ("Total Cost: ", costs)
    print ("----------------------------------------")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4o-mini', help='api engine; https://openai.com/api/')
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()

    ## load the API models
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

    
    # # Experiment with logprob confidence
    print ("gsm8k; logprob; no CoT")
    experiment(client, test_size=1500, dataset="gsm8k", CoT=False, method="logprob", engine="gpt-4o-mini", temperature=0.7, seed=2024, demo_num=8)

    print ("gsm8k; logprob; CoT")
    experiment(client, test_size=1500, dataset="gsm8k", CoT=True, method="logprob", engine="gpt-4o-mini", temperature=0.7, seed=2024, demo_num=8)

    # # Experiment with ensemble confidence
    print ("gsm8k; ensemble; no CoT")
    experiment(client, test_size=1500, dataset="gsm8k", CoT=False, method="ensemble", engine="gpt-4o-mini", temperature=0.7, seed=2024, num_prompts=5, demo_num=8)

    print ("gsm8k; ensemble; CoT")
    experiment(client, test_size=1500, dataset="gsm8k", CoT=True, method="ensemble", engine="gpt-4o-mini", temperature=0.7, seed=2024, num_prompts=5, demo_num=8)

    # Experiment with SRUQ confidence
    # experiment(client, test_size=10, dataset="gsm8k", CoT=True, method="sruq", engine="gpt-4o-mini", temperature=0.7, seed=2024, demo_num=8)
