import numpy as np
import os
import openai
import sys
from txtai.embeddings import Embeddings
from prompts import generate_system_prompt
import re
import torch
from selective_context import SelectiveContext
sc = SelectiveContext(model_type='gpt2', lang='en')
##########################################
import os
os.environ['CURL_CA_BUNDLE'] = ''
def get_subgraph(node_idx, edge_index, hop=1):
    """
    Get subgraph around a specific node up to a certain hop.

    Parameters:
        node_idx (int): Index of the node.
        edge_index (torch.Tensor): Edge index tensor.
        hop (int, optional): Number of hops around the node to consider. Default is 1.

    Returns:
        list: Lists of nodes for each hop distance.
    """

    current_nodes = torch.tensor([node_idx])
    all_hops = []

    for _ in range(hop):
        mask = torch.isin(edge_index[0], current_nodes) | torch.isin(edge_index[1], current_nodes)
        
        # Add both the source and target nodes involved in the edges 
        new_nodes = torch.unique(torch.cat((edge_index[0][mask], edge_index[1][mask])))

        # Remove the current nodes to get only the new nodes added in this hop
        diff_nodes_set = set(new_nodes.numpy()) - set(current_nodes.numpy())
        diff_nodes = torch.tensor(list(diff_nodes_set))  
        
        all_hops.append(diff_nodes.tolist())

        # Update current nodes for the next iteration
        current_nodes = torch.unique(torch.cat((current_nodes, new_nodes)))

    return all_hops

def createIndex(text):
    alltxt = []
    for i in range(len(text['title'])):
        alltxt.append(text['title'][i]+"|| "+text['abs'][i])
    # Create embeddings model, backed by sentence-transformers & transformers
    embeddings = Embeddings(path="allenai/scibert_scivocab_uncased")
    # Index the list of text
    embeddings.index(alltxt)

    return embeddings

def promptGenerate(paras,include_abs=True):
    node_indices, data, text, source,load_embedding,max_papers_1 = paras
    #node_index_list, data, text, dataset_name,load_embedding
    """
        Returns:
        Depending on the 'return_message' flag, either prints the prompt and ideal answer or returns a list of messages.
    """
    for node_index in node_indices:
        # Initial setup for neighbors mode
        title = text['title'][node_index]
        prompt_str = f"Title: {title}\n"
        
        # Include abstract if required
        if include_abs:
            abstract = text['abs'][node_index]
            prompt_str = prompt_str+f"Abstract: {abstract}\n"
        
        sys_prompt_str = generate_system_prompt(source)
        all_hops = get_subgraph(node_index, data.edge_index, hop=1)
        
        # Check for test nodes
        if data.train_mask[node_index] or data.val_mask[node_index]:
            print('node indices should only contain test nodes!!')

        paras = (node_index,text, data,source,load_embedding,max_papers_1)
        prompt_str += collect_neighbors(paras)

        # Finalize prompt for neighbors mode
        prompt_str += "Do not give any reasoning or logic for your answer.\nAnswer: \n\n"
        
        # Return the message
        return [{'role':'system', 'content': sys_prompt_str}, {'role':'user', 'content': f"{prompt_str}"}]


def collect_neighbors(paras):
    return embed_neigh(paras)


def compress(text,ratio = 0.5):
    context, reduced_content = sc(text, reduce_ratio = ratio)
    return context


def embed_neigh(paras,abstract_len=400, include_label=True):
    (node_index, text, data,source,load_embedding,max_paper_1) = paras
    """
    ##Handle neighbors when attention is not used.

    Parameters:
        node_index (int): Index of the target node.
        text: Textual information of the node.
        all_hops (list): List of all neighbor nodes up to a certain hop.
        data: Graph data object.
        abstract_len (int): Length of the abstract to consider.
        include_label (bool): Whether to include labels.
        dataset (str): Name of the dataset being used.

    Returns:
        str: String containing information about standard neighbors.
    """
    if load_embedding:
        embeddings = Embeddings()
        embeddings.load("core_index")
    else:
        embeddings = createIndex(text)
        embeddings.save("core_index")
    #max_paper_1 = 3
    all_neighbor = embeddings.search(text['title'][node_index]+text['abs'][node_index], max_paper_1)
    neibor_indices = [all_neighbor[i][0] for i in range(max_paper_1)]
    prompt_str = ""
    Target_word = "Paper"
    prompt_str += f"The following are related papers to this paper, please consider the content of these papers and making a judgment.\n"
    #prompt_str += f"The following are related papers to this paper and unrelated but confusing papers,"+"please combine the content of these two types of papers and carefully experience the difference between them before finally making a judgment.\n"
    neigh_list = []
    label_list = []
    for i, neighbor_idx in enumerate(neibor_indices):
        init_str = ""
        if i==0:
            #prompt_str += f"It has the following relevant papers:\n"
            continue
        #if 7<i<15:
            #break
        #    continue
        #if i==16:
            #prompt_str += f"It has the following papers that are less relevant but can easily be judged as similar:\n"
        #    continue
        neighbor_title = text['title'][neighbor_idx]
        init_str += f"{Target_word} {i+1}:  <{neighbor_title}.   "#f"{Target_word} {i+1} title: {neighbor_title}\n"
        
        if abstract_len > 0:
            neighbor_abstract = text['abs'][neighbor_idx]
            #prompt_str += f"{Target_word} {i+1} abstract: {neighbor_abstract[:abstract_len]}\n"
            cleaned_text = re.sub(r"\r?\n|\r", "", neighbor_abstract[:abstract_len])
            #init_str += f"Abstract: {compress(cleaned_text[9:])}>\n"#f"Abstract: {compress(cleaned_text)}>\n"
            init_str += f"Abstract: {cleaned_text[9:]}>\n"#f"Abstract: {compress(cleaned_text)}>\n"

        if include_label and (data.train_mask[neighbor_idx] or data.val_mask[neighbor_idx]):
            label = text['label'][neighbor_idx]
            init_str += f"Label: {label}\n"
            label_list.append(init_str)
        else:
            neigh_list.append(init_str)
    prompt_str += "".join(label_list)
    prompt_str += "".join(neigh_list)
    return prompt_str