import numpy as np
from utils.utils import process_and_compare_predictions, load_data, sample_test_nodes,LLMsInfer
import sys
from HQprompt import promptGenerate
import requests
requests.packages.urllib3.disable_warnings()
import yaml



class linkInContextAgent:
    def __init__(self):
        # read YAML
        with open('example.yaml', 'r') as file:
            parameters = yaml.safe_load(file)
        ssize = parameters['samplesize']
        self.load_embedding = parameters['load_embedding']
        dataname = parameters['dataset_name']
        self.LLM = parameters['LLM']
        self.maxp1 = parameters['max_papers'][0]
        self.stratage = parameters['stratage']
        self.abs_len = parameters['abstract_len']
        retry = parameters['retry']

        ## load data
        data, text = load_data(dataname, use_text=True, seed=42)
        if ssize == "all":
            self.sample_size = len(data.test_id)
        else:
            self.sample_size = self.ssize
        ##
        if retry == 'all':
            self.node_index_list = [node_indices[idx] for idx in idx_list]
        else:
            self.node_index_list = [node_indices[idx] for idx in wridx]
        return
    

class Paper:
    def __init__(self,text):
        self.title = text['title'][node_index]
        self.abs = text['abs']



class Prompter:
    def __init__(self,para) -> None:
        self.stratage = para['stratage']
        self.neibor_indices = []
        pass
    
    def retriever(self):
        if self.stratage = 'knn':
            return self.embed_knn()
        else:
            return None

    def standICL(self):
        return
    
    def hybridv1(self):
        instruct = f"The following are related papers to this paper, please consider the content of these papers and making a judgment.\n"
        Target_word = "Paper"
        neigh_list = []
        label_list = []
        for i, neighbor_idx in enumerate(self.neibor_indices):
            init_str = ""
            if i==0:
                continue

            neighbor_title = text['title'][neighbor_idx]
            init_str += f"{Target_word} {i+1}:  <{neighbor_title}.   "#f"{Target_word} {i+1} title: {neighbor_title}\n"
            neighbor_abstract = text['abs'][neighbor_idx]
            #prompt_str += f"{Target_word} {i+1} abstract: {neighbor_abstract[:abstract_len]}\n"
            cleaned_text = re.sub(r"\r?\n|\r", "", neighbor_abstract[:abstract_len])
            #init_str += f"Abstract: {compress(cleaned_text[9:])}>\n"#f"Abstract: {compress(cleaned_text)}>\n"
            init_str += f"Abstract: {cleaned_text[9:]}>\n"#f"Abstract: {compress(cleaned_text)}>\n"

            if (data.train_mask[neighbor_idx] or data.val_mask[neighbor_idx]):
                label = text['label'][neighbor_idx]
                init_str += f"Label: {label}\n"
                label_list.append(init_str)
            else:
                neigh_list.append(init_str)
        return
    
    def wholePrompt(self):
        return

    
    def embed_knn(self):
        if load_embedding:
            embeddings = Embeddings()
            embeddings.load("core_index")
        else:
            embeddings = createIndex(text)
            embeddings.save("core_index")
        #max_paper_1 = 3
        all_neighbor = embeddings.search(text['title'][node_index]+text['abs'][node_index], max_paper_1)
        self.neibor_indices = [all_neighbor[i][0] for i in range(max_paper_1)]
        return neibor_indices