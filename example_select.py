

all_neighbor = embeddings.search(text['title'][node_index]+text['abs'][node_index], 20)
neibor_indices = [all_neighbor[i][0] for i in range(20)]



def promptGenerate(node_indices, data, text, mode, dataset, source, hop=1, max_papers_1=20, max_papers_2=10, 
                  abstract_len=0, print_prompt=True, include_label=False, return_message=False, 
                  arxiv_style=False, include_options=False, include_abs=False, zero_shot_CoT=False, 
                  few_shot=False, use_attention=False):
    """
    Main function to get node information based on various modes and options.

    Parameters:
        node_indices (list): List of node indices to consider.
        data: Graph data object.
        text: Textual information associated with nodes.
        mode (str): Mode of operation, either 'neighbors' or 'ego'.
        dataset (str): Name of the dataset being used.
        source (str): Source of the data.
        hop (int, optional): Number of hops to consider. Default is 1.
        max_papers_1 (int, optional): Maximum number of papers for the first hop. Default is 20.
        max_papers_2 (int, optional): Maximum number of papers for the second hop. Default is 10.
        abstract_len (int, optional): Length of the abstract to consider. Default is 0.
        print_prompt (bool, optional): Whether to print the prompt. Default is True.
        include_label (bool, optional): Whether to include labels. Default is False.
        return_message (bool, optional): Whether to return the message. Default is False.
        arxiv_style (bool, optional): Whether to use arXiv style for labels. Default is False.
        include_options (bool, optional): Whether to include options in the system prompt. Default is False.
        include_abs (bool, optional): Whether to include abstracts. Default is False.
        zero_shot_CoT (bool, optional): Whether to use zero-shot CoT. Default is False.
        few_shot (bool, optional): Whether to use few-shot learning. Default is False.
        use_attention (bool, optional): Whether to use attention. Default is False.

    Returns:
        Depending on the 'return_message' flag, either prints the prompt and ideal answer or returns a list of messages.
    """

    # Initial setup for neighbors mode
    title = text['title'][node_index]
    prompt_str = f"Title: {title}\n"
    
    # Include abstract if required
    if include_abs:
        abstract = text['abs'][node_index]
        prompt_str = f"Abstract: {abstract}\n" + prompt_str
    
    sys_prompt_str = generate_system_prompt(source, arxiv_style=arxiv_style, include_options=include_options)
    all_hops = get_subgraph(node_index, data.edge_index, hop)
    
    # Check for test nodes
    if data.train_mask[node_index] or data.val_mask[node_index]:
        print('node indices should only contain test nodes!!')

    
    collect_neighbors()

    # Finalize prompt for neighbors mode
    prompt_str += "Do not give any reasoning or logic for your answer.\nAnswer: \n\n"
    
    # Return the message
    return [{'role':'system', 'content': sys_prompt_str}, {'role':'user', 'content': f"{prompt_str}"}]
    


def collect_neighbors():
    if use_attention:
        prompt_str += handle_important_neighbors(node_index, text, dataset, all_hops, data, abstract_len, include_label, max_papers_1)
    else:
        prompt_str += handle_standard_neighbors(node_index, text, all_hops, data, hop, max_papers_1, max_papers_2, 
                                                abstract_len, include_label, dataset)
    return


