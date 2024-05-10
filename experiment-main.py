import numpy as np
import os
import openai
from utils.utils import process_and_compare_predictions, load_data, sample_test_nodes, map_arxiv_labels,createIndex
import sys
from txtai.embeddings import Embeddings


## wrong index --- 
'''
###### Arxiv2023
1. z-shot: wrong_index1 = [7, 8, 14, 17, 24, 27, 28, 31, 34, 44, 52, 66, 67, 68, 69, 75, 77, 79, 83, 85, 87, 98, 99]

2. 
'''
#wrong_index1 = [7, 8, 14, 17, 24, 27, 28, 31, 34, 44, 52, 66, 67, 68, 69, 75, 77, 79, 83, 85, 87, 98, 99]

'''
###### cora
1. z-shot: wrong_index1 = [0, 2, 4, 10, 11, 12, 16, 17, 24, 27, 30, 31, 32, 33, 35, 36, 41, 44, 47, 48, 50, 51, 55, 56, 58, 59, 62, 63, 64, 66, 67, 70, 72, 86, 90, 95, 96]
2. 
'''
wrong_index1 = [0, 2, 4, 10, 11, 12, 16, 17, 24, 27, 30, 31, 32, 33, 35, 36, 41, 44, 47, 48, 50, 51, 55, 56, 58, 59, 62, 63, 64, 66, 67, 70, 72, 86, 90, 95, 96]


'''
### parameters sets
##
'''
openai.api_key  = 'sk-JBAu6m906G5wetULzKlG71ttlImaFd2aPaRKXw05w7VvchFI' #'hk-7qb12k10000044454e95619ef6fd3616840e6d1b18f22734'#'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'#os.environ['OPENAI_API_KEY']
os.environ['OPENAI_API_BASE_URL'] = 'https://openkey.cloud'

# dataset_name = "arxiv_2023"
# dataset_name = "pubmed"
dataset_name = "cora"
# dataset_name = "arxiv"
# dataset_name = "product"

# sample size ---------------------------
# 


''''
## zero-shot

mode = "ego"
zero_shot_CoT=False
hop=1
few_shot=False
include_abs=True
include_label=False
'''

'''
## few-shot
mode = "ego"
zero_shot_CoT=False
hop=1
include_abs=True
include_label=False
few_shot=True
'''


## one-hop title + label
mode = "neighbors"
zero_shot_CoT=True
few_shot=False
hop=1
include_label = True
include_abs = True
load_embedding = True

##########################################

def conductExper(experName, samplesize,moding, wridx = []):
    f = open('results/'+dataset_name+"__"+experName+'.log', 'a')
    sys.stdout = f
    print("experName .... ",experName)
    if dataset_name == "arxiv" or dataset_name == "arxiv_2023":
        source = "arxiv"
    else:
        source = dataset_name

    # use_ori_arxiv_label=False # only for using original Arxiv identifier in system prompting for ogbn-arxiv
    arxiv_style="subcategory" # "identifier", "natural language"
    include_options = False # set to true to include options in the prompt for arxiv datasets

    data, text = load_data(dataset_name, use_text=True, seed=42)

    if source == "arxiv" and arxiv_style != "subcategory":
        text = map_arxiv_labels(data, text, source, arxiv_style)

    options = set(text['label'])


    if dataset_name == "arxiv_2023" or dataset_name == "cora":
        sample_size = len(data.test_id)
    else:
        sample_size = 1000

    sample_size = samplesize

    node_indices = sample_test_nodes(data, text, sample_size, dataset_name)

    idx_list = list(range(sample_size))
    if moding == 'all':
        node_index_list = [node_indices[idx] for idx in idx_list]
    else:
        node_index_list = [node_indices[idx] for idx in wridx]
      
    if dataset_name == "product":
        max_papers_1 = 40
        max_papers_2 = 10
    else:
        max_papers_1 = 20
        max_papers_2 = 5

    #nodelist2 = [node_index_list[w] for w in  wridx[::2]]
    #print(f"{nodelist2 = }")
    if load_embedding:
        embeddings = Embeddings()
        embeddings.load("core_index")
    else:
        embeddings = createIndex(text)
        embeddings.save("core_index")
    use_attention = True
    accuracy, wrong_indexes_list = process_and_compare_predictions(node_index_list, data, text, embeddings,dataset_name=dataset_name, abstract_len=400,source=source, mode=mode, max_papers_1=max_papers_1, max_papers_2=max_papers_2, hop=hop,  arxiv_style=arxiv_style, include_abs=include_abs, include_options=include_options, zero_shot_CoT=zero_shot_CoT, few_shot=few_shot, options=options,use_attention=use_attention)
    #accuracy, wrong_indexes_list = process_and_compare_predictions(nodelist2[:6], data, text, dataset_name=dataset_name, source=source, mode=mode, max_papers_1=max_papers_1, max_papers_2=max_papers_2, hop=hop,  include_label=include_label, arxiv_style=arxiv_style, include_abs=include_abs, include_options=include_options, zero_shot_CoT=zero_shot_CoT, few_shot=few_shot, options=options)
    #print(nodelist2)
    return accuracy, wrong_indexes_list

#conductExper("one_hop_title+label_odd", wrong_index1)
#conductExper("cora_zero_wrong37_hop1lables", 100,wrong_index1)
conductExper("100_wrong37_scibert_add_negative+abs",100,'wrong',wrong_index1)