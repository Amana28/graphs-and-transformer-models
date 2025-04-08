import networkx as nx
import random
import os
import argparse
import numpy

def generate_circle_graph(num_nodes):
    # Create an empty directed graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for i in range(num_nodes):
        G.add_node(i)
    
    # Add edges to create a line graph (each node connected to the next)
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1)
        
    # Add the final edge to complete the circle
    G.add_edge(num_nodes - 1, 0)
        
    return G

def get_reachable_nodes(G, target_node):  
    # In a circle graph, all nodes can reach all other nodes
    # Return all nodes except the target itself
    return [node for node in G.nodes() if node != target_node]

def obtain_reachability(G):
    reachability = {}  
    pairs = 0
    for node in G.nodes():  
        reachability[node] = get_reachable_nodes(G, node)
        pairs += len(reachability[node])
    return reachability, pairs

def path_from_to(source_node, target_node, num_nodes):
    # In a circle, we have two potential paths: clockwise and counterclockwise
    # Choose the shorter one
    
    # Clockwise path
    if source_node < target_node:
        clockwise_path = list(range(source_node, target_node + 1))
        clockwise_length = target_node - source_node + 1
    else:
        clockwise_path = list(range(source_node, num_nodes)) + list(range(0, target_node + 1))
        clockwise_length = num_nodes - source_node + target_node + 1
        
    # Counterclockwise path
    if target_node < source_node:
        counter_path = list(range(source_node, -1, -1)) + list(range(num_nodes - 1, target_node - 1, -1))
        counter_length = source_node + 1 + (num_nodes - 1 - target_node)
    else:
        counter_path = list(range(source_node, -1, -1)) + list(range(num_nodes - 1, target_node - 1, -1))
        counter_length = source_node + 1 + (num_nodes - 1 - target_node)
    
    # Choose the shorter path
    if clockwise_length <= counter_length:
        return clockwise_path
    else:
        return counter_path

def create_dataset(num_paths_per_pair):
    train_set = []
    test_set = []
    train_num_per_pair = max(num_paths_per_pair, 1)
    
    for target_node in range(num_nodes):
        for source_node in range(num_nodes):
            if source_node != target_node:  # Skip self-loops
                if data[source_node][target_node] == 1:
                    # Add the path
                    path = path_from_to(source_node, target_node, num_nodes)
                    train_set.append([source_node, target_node] + path)
                    
                    # Add more paths if needed
                    for _ in range(train_num_per_pair - 1):
                        train_set.append([source_node, target_node] + path)
                
                if data[source_node][target_node] == -1:
                    path = path_from_to(source_node, target_node, num_nodes)
                    test_set.append([source_node, target_node] + path)
                    
    return train_set, test_set

def add_x(train_set, test_set):
    # In a circle graph, all nodes can reach all other nodes,
    # so we'll add some artificial unreachable cases
    prob_in_test = 0.1
    prob_in_train = 0.1
    train_repeat = 3  # Number of repetitions for training examples
    
    # Limit the number of unreachable cases to add
    num_to_add = int(len(train_set) * 0.15)
    added = 0
    
    # Simulate some unreachable cases
    for target_node in range(num_nodes):
        for source_node in range(num_nodes):
            if source_node != target_node and added < num_to_add:
                if random.random() < 0.05:  # 5% chance to mark as unreachable
                    added += 1
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
    parser = argparse.ArgumentParser(description='Generate a circle graph with the given parameters.')  
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes in the graph')  
    parser.add_argument('--chance_in_train', type=float, default=0.5, help='Chance of a pair being in the training set')  
    parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths per pair nodes in training dataset')  

    args = parser.parse_args()  

    num_nodes = args.num_nodes
    chance_in_train = args.chance_in_train
    num_of_paths = args.num_of_paths

    # Generate a circle graph
    circle_graph = generate_circle_graph(num_nodes)
    reachability, feasible_pairs = obtain_reachability(circle_graph)

    folder_name = os.path.join(os.path.dirname(__file__), f'{num_nodes}')
    if not os.path.exists(folder_name):  
        os.makedirs(folder_name)

    data = numpy.zeros([num_nodes, num_nodes])
    
    # For a circle graph, distribute training/testing pairs
    for target_node in range(num_nodes):
        for source_node in range(num_nodes):
            if source_node != target_node:  # Skip self-loops
                if random.random() < chance_in_train:
                    data[source_node][target_node] = 1
                else:
                    data[source_node][target_node] = -1
                
    train_set, test_set = create_dataset(num_of_paths)
    
    # Add some artificial unreachable cases for variety
    # train_set, test_set = add_x(train_set, test_set)
        
    obtain_stats(train_set)
    print('number of test samples:', len(test_set))

    write_dataset(train_set, os.path.join(os.path.dirname(__file__), f'{num_nodes}/train_{num_of_paths}.txt'))
    write_dataset(test_set, os.path.join(os.path.dirname(__file__), f'{num_nodes}/test.txt'))
    nx.write_graphml(circle_graph, os.path.join(os.path.dirname(__file__), f'{num_nodes}/path_graph.graphml'))