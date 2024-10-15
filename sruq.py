from openai import OpenAI
import anthropic
from utils import call_api
import argparse
import json
import os
import random 
import retry

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-opus-20240229', help='api engine; https://openai.com/api/')
    parser.add_argument('--paper_cache', type=str, default=None, required=True, help='cache file name for the retrieved papers')
    parser.add_argument('--idea_cache', type=str, default=None, required=True, help='where to store the generated ideas')
    parser.add_argument('--RAG', type=str, default="True", required=True, help='whether to do RAG for idea generation')
    parser.add_argument('--method', type=str, default='prompting', help='either prompting or finetuning')
    parser.add_argument('--grounding_k', type=int, default=10, help='how many papers to use for grounding')
    parser.add_argument('--ideas_n', type=int, default=5, help="how many ideas to generate")
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()


