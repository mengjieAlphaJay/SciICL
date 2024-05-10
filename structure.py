import numpy as np
import os
import openai
from utils.utils import process_and_compare_predictions, load_data, sample_test_nodes,LLMsInfer
import sys
from HQprompt import promptGenerate
import requests
requests.packages.urllib3.disable_warnings()
def parameterCreate(parameters):
    include_options = False # set to true to include options in the prompt for arxiv datasets
    dataset_name,samplesize,moding,load_embedding = parameters[0],parameters[1],parameters[2],parameters[3]
    data, text = load_data(dataset_name, use_text=True, seed=42)

    #options = set(text['label'])

    if dataset_name == "cora":
        sample_size = len(data.test_id)
    else:
        sample_size = 100

    sample_size = samplesize

    node_indices = sample_test_nodes(data, text, sample_size, dataset_name)
    idx_list = list(range(sample_size))
    if moding == 'all':
        node_index_list = [node_indices[idx] for idx in idx_list]
    else:
        node_index_list = [node_indices[idx] for idx in wridx]

    max_papers_1 = 10
    max_papers_2 = 5

    #nodelist2 = [node_index_list[w] for w in  wridx[::2]]
    #print(f"{nodelist2 = }")
    #print(node_index_list)
    return (node_index_list, data, text, dataset_name,max_papers_1,max_papers_2,load_embedding)


def conductExper(experName, wridx = []):
    ## return experiment-log,acc,wrong_list
    (node_index_list, data, text, dataset_name,max_papers_1,max_papers_2,load_embedding) = parameterCreate(['cora',5,'all',True])
    #print(node_index_list)
    
    paras = [node_index_list, data, text, dataset_name,load_embedding,max_papers_1]
    #print(paras[0])
    
    f = open('results/'+dataset_name+"__"+experName+'.log', 'a')
    sys.stdout = f
    print("experName .... ",experName)
    #paras[0] = [paras[0][1]
    #out = LLMsInfer(paras)
    accuracy, wrong_indexes_list = process_and_compare_predictions(paras)
    
    # return 0
    return accuracy, wrong_indexes_list


#prompt = promptGenerate(node_index_list, data, text, dataset_name,load_embedding,max_papers_1)
#print(prompt)
conductExper("24.4.17-2100-sciemd-size_5-test")
#conductExper("test")