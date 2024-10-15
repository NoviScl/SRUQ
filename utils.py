import os
import json
import random 

def calc_price(model, usage):
    if "claude" in model:
        return (0.015 * usage.input_tokens + 0.075 * usage.output_tokens) / 1000.0
    if model == "gpt-4-1106-preview" or model == "gpt-4-0125-preview":
        return (0.01 * usage.prompt_tokens + 0.03 * usage.completion_tokens) / 1000.0
    if model == "gpt-4":
        return (0.03 * usage.prompt_tokens + 0.06 * usage.completion_tokens) / 1000.0
    if (model == "gpt-3.5-turbo") or (model == "gpt-3.5-turbo-1106"):
        return (0.0015 * usage.prompt_tokens + 0.002 * usage.completion_tokens) / 1000.0
    if model == "gpt-4o":
        return (0.005 * usage.prompt_tokens + 0.015 * usage.completion_tokens) / 1000.0

def call_api(client, model, prompt_messages, temperature=1.0, max_tokens=100, seed=2024, json_output=False):
    if "claude" in model:
        if json_output:
            prompt = prompt_messages[0]["content"] + " Directly output the JSON dict with no additional text (avoid the presence of newline characters (\"\n\") and unescaped double quotes within the string so that we can call json.loads() on the output later)."
            prompt_messages = [{"role": "user", "content": prompt}]
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=prompt_messages
        )
        cost = calc_price(model, message.usage)
        response = message.content[0].text
    else:   
        response_format = {"type": "json_object"} if json_output else {"type": "text"}
        completion = client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            response_format=response_format
        )
        cost = calc_price(model, completion.usage)
        response = completion.choices[0].message.content.strip()
    
    return response, cost

def call_api_claude(client, model, prompt_messages, temperature=1.0, max_tokens=100):
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=prompt_messages
    )
    cost = calc_price(model, message.usage)
    response = message.content[0].text

    return response, cost

def cache_output(output, file_name):
    if file_name.endswith(".txt"):
        ## store GPT4 output into a txt file
        with open(file_name, "w") as f:
            f.write(output)
    elif file_name.endswith(".json"):
        ## store GPT4 output into a json file
        with open(file_name, "w") as f:
            json.dump(output, f, indent=4)
    return 


