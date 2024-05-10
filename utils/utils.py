
from utils.load_arxiv import get_raw_text_arxiv
from utils.load_cora import get_raw_text_cora
from utils.load_pubmed import get_raw_text_pubmed
from utils.load_arxiv_2023 import get_raw_text_arxiv_2023
from utils.load_products import get_raw_text_products
from HQprompt import promptGenerate

import time
import torch
import numpy as np
import json
import os
import openai
from time import sleep
from random import randint
import threading
import requests
import datetime

def load_data(dataset, use_text=False, seed=0):
    """
    Load data based on the dataset name.

    Parameters:
        dataset (str): Name of the dataset to be loaded. Options are "cora", "pubmed", "arxiv", "arxiv_2023", and "product".
        use_text (bool, optional): Whether to use text data. Default is False.
        seed (int, optional): Random seed for data loading. Default is 0.

    Returns:
        Tuple: Loaded data and text information.

    Raises:
        ValueError: If the dataset name is not recognized.
    """

    if dataset == "cora":
        data, text = get_raw_text_cora(use_text, seed)
    elif dataset == "pubmed":
        data, text = get_raw_text_pubmed(use_text, seed)
    elif dataset == "arxiv":
        data, text = get_raw_text_arxiv(use_text)
    elif dataset == "arxiv_2023":
        data, text = get_raw_text_arxiv_2023(use_text)
    elif dataset == "product":
        data, text = get_raw_text_products(use_text)
    else:
        raise ValueError("Dataset must be one of: cora, pubmed, arxiv")
    return data, text

def sample_test_nodes(data, text, sample_size, dataset):
    """
    Randomly sample test nodes for evaluation.

    Parameters:
        data: Graph data object.
        text: Textual information associated with nodes.
        sample_size (int): Number of test nodes to sample.
        dataset (str): Name of the dataset being used.

    Returns:
        list: Indices of sampled test nodes.
    """

    np.random.seed(42)
    test_indices = np.where(data.test_mask.numpy())[0]

    # Sample 2 times the sample size
    # node_indices = sample_test_nodes(data, 2 * sample_size)
    sampled_indices_double = np.random.choice(test_indices, size=2*sample_size, replace=False)

    # Filter out the indices of nodes with title "NA\n"
    sampled_indices = [node_idx for i, node_idx in enumerate(sampled_indices_double) 
                if text['title'][node_idx] != "NA\n"]
    sampled_indices = sampled_indices[:sample_size]

    # sanity check
    count = 0
    for node_idx in sampled_indices:
        if text['title'][node_idx] == "NA\n":
            count += 1
    assert count == 0
    assert len(sampled_indices) == sample_size

    return sampled_indices

def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, max_tokens=500):
    """
    Get completion from the OpenAI API based on the given messages.

    Parameters:
        messages (list): Messages to be sent to the OpenAI API.
        model (str, optional): The name of the model to be used. Default is "gpt-3.5-turbo".
        temperature (float, optional): Sampling temperature. Default is 0.
        max_tokens (int, optional): Maximum number of tokens for the response. Default is 500.

    Returns:
        str: The content of the completion message.
    """
    #OPENAI_API_BASE_URL
    #url = "https://freegpt35.tomleung.cn/v1/chat/completions"
    url = "https://api.openai-hk.com/v1/chat/completions"

    headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer hk-2q6mjv1000004445c0d228e1e0755764cd8179f916784d37"
        }

    data = {
            "max_tokens": max_tokens,
            "model": model,
            "temperature": temperature,
            "top_p": 1,
            "presence_penalty": 1,
            'messages':messages
        }
    response = requests.post(url, headers=headers, data=json.dumps(data).encode('utf-8') )
    rc = response.content
    cont = json.loads(rc)
    #cont = ast.literal_eval(str(rc.decode('utf-8')))#["content"]
    return  cont["choices"][0]['message']['content'] #response.choices[0].message["content"]



def process_all_paper(paras_prom):
    node_index_list = paras_prom[0]
    i = 0
    count = 0
    wrong_indexes = []
    wrong_list = []
    result_container = []
    t1 = datetime.datetime.strptime(time.ctime(),"%a %b %d %H:%M:%S %Y")
    for i in range(len(node_index_list)):
        print(f"Processing index {node_index_list[i]}...")
        node_index = node_index_list[i]
        paras_prom[0] = [node_index]
        result = LLMsInfer(paras_prom)
        #result_container[0] = result

    return count/len(node_index_list), wrong_indexes

def process_and_compare_predictions(paras_prom, timeout=60):
    """
    Process and compare predictions for a list of node indices.

    Parameters:
 
    Returns:
        tuple: The first element is the accuracy of the predictions, and the second is a list of wrong indexes.
    """
    node_index_list = paras_prom[0]
    i = 0
    count = 0
    wrong_indexes = []
    wrong_list = []
    base_sleep_time = 1 # Starting sleep time
    max_sleep_time = 0.2  # Maximum sleep time
    t1 = datetime.datetime.strptime(time.ctime(),"%a %b %d %H:%M:%S %Y")

    while i < len(node_index_list):
        t2 = datetime.datetime.strptime(time.ctime(),"%a %b %d %H:%M:%S %Y") 
        timestamp = (t2-t1).total_seconds()  
        if timestamp >=180:
            break
        retries = 0
        while True:  # Infinite loop for retries
            result_container = [None]  # List to store the result of the threaded function
            exception_container = [None]  # List to store exceptions if any
            if (i%3==0) & (i>2)==1:
                sleep(max_sleep_time+0.1)
            # Function to run in the thread
            def thread_target():
                try:
                    print(f"Processing index {node_index_list[i]}...")
                    node_index = node_index_list[i]
                    new_para = [[node_index]]+paras_prom[1:]
                    result = LLMsInfer(new_para)
                    #print(result)
                    result_container[0] = result
                except Exception as e:
                    exception_container[0] = e
            
            # Start the function in a separate thread
            thread = threading.Thread(target=thread_target)
            thread.start()
            thread.join(timeout=timeout)
        

            if result_container[0] is not None:
                count += result_container[0]
                print(f"Prediction: {result_container[0]}")
                if result_container[0] == 0:  # If the prediction is wrong, save the index
                    wrong_indexes.append(node_index_list[i])
                    wrong_list.append(i)
                i += 1
                break  # Exit the retry loop once the processing is successful
            
            # If there was an exception or timeout
            else:
                if exception_container[0]:  # If there was an exception
                    print(f"An error occurred at index {i}: {exception_container[0]}")
                else:  # If there was a timeout
                    print(f"Function timed out at index {i}")
                retries += 1
                sleep_time = min(base_sleep_time * (2 ** retries) + randint(0, 1000) / 1000, max_sleep_time)
                print(f"Retrying in {sleep_time} seconds...")
                sleep(sleep_time)

    print("Accuracy:", count/len(node_index_list))
    print("Wrong indexes:", wrong_indexes)
    print("Wrong list:",wrong_list)
    print("Wrong indexes length:", len(wrong_indexes))
    assert len(wrong_indexes) == len(node_index_list) - count

    return count/len(node_index_list), wrong_indexes



def LLMsInfer(paras_prom,print_out=True):
    ## 单线程
    text = paras_prom[2]
    message = promptGenerate(tuple(paras_prom))
    
    if print_out:
        print(message[0]['content'], end="\n\n")
        print(message[1]['content'], end="\n\n")

    ideal_answer = text['label'][paras_prom[0][0]]
    
    print("Ideal_answer:", ideal_answer, end="\n\n")
    
    # Get completion message and print
    ### 运行GPT
    response = get_completion_from_messages(message)   
    if print_out:
        print(response)
    
    prediction = response if response is not None else ""

    if prediction is not None:
        print("Prediction: ", prediction)
        # Compare the prediction with ideal_answer
        print("Is prediction correct? ", prediction == ideal_answer, end="\n\n")
        
        return int(prediction == ideal_answer)
    else:
        print("No valid prediction could be made.")

    return
