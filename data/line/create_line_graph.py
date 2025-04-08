import networkx as nx
import random
import os
import argparse
import numpy

def generate_line_graph(num_nodes):
    # Create an empty directed graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for i in range(num_nodes):
        G.add_node(i)
    
    # Add edges to create a line graph (each node connected to the next)
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1)
        
    return G

def get_reachable_nodes(G, target_node):  
    # Get all nodes that can reach the target_node in a line graph
    # In a line graph, all nodes with index less than target_node can reach it
    return list(range(target_node))

def obtain_reachability(G):
    reachability = {}  
    pairs = 0
    for node in G.nodes():  
        reachability[node] = get_reachable_nodes(G, node)
        pairs += len(reachability[node])
    return reachability, pairs

def path_from_to(source_node, target_node):
    # In a line graph, the path is always sequential
    return list(range(source_node, target_node + 1))

def create_dataset(num_paths_per_pair):
    train_set = []
    test_set = []
    train_num_per_pair = max(num_paths_per_pair, 1)
    
    for target_node in range(num_nodes):
        for source_node in range(target_node):
            if data[source_node][target_node] == 1:
                # For line graph, always add direct path
                path = path_from_to(source_node, target_node)
                train_set.append([source_node, target_node] + path)
                
                # Add more paths if needed (all identical in a line graph)
                for _ in range(train_num_per_pair - 1):
                    train_set.append([source_node, target_node] + path)
            
            if data[source_node][target_node] == -1:
                path = path_from_to(source_node, target_node)
                test_set.append([source_node, target_node] + path)
                    
    return train_set, test_set

def add_x(train_set, test_set):
    cnt = 0
    for target_node in range(num_nodes):
        for source_node in range(target_node):
            if source_node not in reachability[target_node]:
                cnt += 1

    # For a line graph, we'll add some unreachable cases even though technically
    # all lower nodes can reach higher nodes
    prob_in_test = 0.1
    prob_in_train = 0.1
    train_repeat = max(int(len(train_set) * 0.15 / cnt if cnt > 0 else 1), 1)

    # For line graph, simulate some unreachable cases
    for target_node in range(num_nodes):
        for source_node in range(target_node):
            if random.random() < 0.05:  # 5% chance to mark as unreachable
                coin = random.random()
                if coin < prob_in_train:
                    for _ in range(train_repeat):
                        train_set.append([source_node, target_node, 'x'])
                        
                elif coin > 1 - prob_in_test:
                    test_set.append([source_node, target_node, 'x'])

    return train_set, test_set

def obtain_stats(dataset):
    max_len = 0
    pairs = set()

    for data in dataset:
        max_len = max(max_len, len(data))
        pairs.add((data[0], data[-1] if data[-1] != 'x' else data[1]))

    len_stats = [0] * (max_len + 1) 
    
    for data in dataset:
        length = len(data)
        len_stats[length] += 1
        
    print('number of source target pairs:', len(pairs))
    for ii in range(3, len(len_stats)):
        print(f'There are {len_stats[ii]} paths with length {ii-3}')

def format_data(data):
    return f"{data[0]} {data[1]} " + ' '.join(str(num) for num in data[2:]) + '\n'
        
def write_dataset(dataset, file_name):
    with open(file_name, "w") as file:
        for data in dataset:
            file.write(format_data(data))

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Generate a line graph with the given parameters.')  
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes in the graph')  
    parser.add_argument('--chance_in_train', type=float, default=0.5, help='Chance of a pair being in the training set')  
    parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths per pair nodes in training dataset')  

    args = parser.parse_args()  

    num_nodes = args.num_nodes
    chance_in_train = args.chance_in_train
    num_of_paths = args.num_of_paths

    # Generate a line graph
    line_graph = generate_line_graph(num_nodes)
    reachability, feasible_pairs = obtain_reachability(line_graph)

    folder_name = os.path.join(os.path.dirname(__file__), f'{num_nodes}')
    if not os.path.exists(folder_name):  
        os.makedirs(folder_name)

    data = numpy.zeros([num_nodes, num_nodes])
    for target_node in range(num_nodes):
        cnt = 0  # to avoid some target not appear in training dataset
        for source_node in range(target_node):
            # In a line graph, all lower nodes can reach higher nodes
            if random.random() < chance_in_train or cnt < 1:
                data[source_node][target_node] = 1
                cnt += 1 
            else:
                data[source_node][target_node] = -1
                
    train_set, test_set = create_dataset(num_of_paths)
    
    # Add some unreachable cases for variety
    # train_set, test_set = add_x(train_set, test_set)
        
    obtain_stats(train_set)
    print('number of test samples:', len(test_set))

    write_dataset(train_set, os.path.join(os.path.dirname(__file__), f'{num_nodes}/train_{num_of_paths}.txt'))
    write_dataset(test_set, os.path.join(os.path.dirname(__file__), f'{num_nodes}/test.txt'))
    nx.write_graphml(line_graph, os.path.join(os.path.dirname(__file__), f'{num_nodes}/path_graph.graphml'))