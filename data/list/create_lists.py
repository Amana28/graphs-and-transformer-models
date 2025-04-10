import os
import random
import argparse

def generate_random_list(min_value, max_value, min_length, max_length, is_sorted):
    """
    Generate a random list of integers with variable length.
    
    Args:
        min_value: Minimum value for generated numbers (inclusive)
        max_value: Maximum value for generated numbers (inclusive)
        min_length: Minimum length for the generated list
        max_length: Maximum length for the generated list
        is_sorted: Boolean flag indicating if the list should be sorted
    
    Returns:
        A random list of integers and its reversed version
    """
    # Determine the length of this particular list
    length = random.randint(min_length, max_length)
    
    # Generate the list with unique random numbers
    # If the range is smaller than requested length, limit to range size
    range_size = max_value - min_value + 1
    if length > range_size:
        length = range_size
        
    # Generate unique numbers
    rand_list = random.sample(range(min_value, max_value + 1), length)
    
    # Sort if required
    if is_sorted:
        rand_list.sort()
    
    # Create the reversed list
    reversed_list = list(reversed(rand_list))
    
    return rand_list, reversed_list

def format_list(rand_list, reversed_list):
    """Format the list as a string with a '%' separator."""
    return " ".join(map(str, rand_list)) + " % " + " ".join(map(str, reversed_list)) + "\n"

def generate_datasets(args):
    """Generate training and validation datasets according to the specified parameters."""
    all_lists = []
    
    # Generate the requested number of unique lists
    for _ in range(args.num_lists):
        rand_list, reversed_list = generate_random_list(
            args.min_value, 
            args.max_value, 
            args.min_length, 
            args.max_length,
            args.is_sorted
        )
        formatted_list = format_list(rand_list, reversed_list)
        all_lists.append(formatted_list)
    
    # Shuffle the generated lists
    random.shuffle(all_lists)
    
    # Split into training and validation sets
    train_size = int(args.num_lists * args.chance_in_train)
    train_lists = all_lists[:train_size]
    val_lists = all_lists[train_size:]
    
    # Repeat lists in training set as specified
    expanded_train_lists = []
    for list_item in train_lists:
        expanded_train_lists.extend([list_item] * args.num_list_copies)
    
    # Shuffle the expanded training set
    random.shuffle(expanded_train_lists)
    
    return expanded_train_lists, val_lists

def write_dataset(dataset, file_path):
    """Write the dataset to a file."""
    with open(file_path, "w") as file:
        for item in dataset:
            file.write(item)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a random list based on the given parameters')      
    parser.add_argument('--is_sorted', type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help='A boolean flag indicating sorted status')       
    parser.add_argument('--min_value', type=int, default=0, 
                        help='min value for generated numbers -- inclusive ')       
    parser.add_argument('--max_value', type=int, default=100, 
                        help='max value for generated numbers -- inclusive')     
    parser.add_argument('--min_length', type=int, default=1, 
                        help='min length for the generated lists')         
    parser.add_argument('--max_length', type=int, default=50, 
                        help='max length for the generated lists')         
    parser.add_argument('--num_list_copies', type=int, default=5, 
                        help='the number of times each list is repeated in the training data.')  
    parser.add_argument('--num_lists', type=int, default=10000, 
                        help='the total number of generated lists')       
    parser.add_argument('--chance_in_train', type=float, default=0.7, 
                        help='ratio of training set -- the rest is validation')  
    
    args = parser.parse_args()
    
    # Determine the folder path based on sorted status
    if args.is_sorted:
        folder_name = f"data/list/sorted/{args.min_value}-{args.max_value}"
    else:
        folder_name = f"data/list/unsorted/{args.min_value}-{args.max_value}"
    
    # Create directories if they don't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Generate datasets
    train_lists, val_lists = generate_datasets(args)
    
    # Write datasets to files
    train_file = os.path.join(folder_name, f'train_{args.num_list_copies}.txt')
    val_file = os.path.join(folder_name, 'test.txt')
    
    write_dataset(train_lists, train_file)
    write_dataset(val_lists, val_file)
    
    print(f"Generated:")
    print(f"- {len(train_lists)} training examples ({len(train_lists)/args.num_list_copies} unique lists × {args.num_list_copies} copies)")
    print(f"- {len(val_lists)} validation examples")
    print(f"Training data saved to: {train_file}")
    print(f"Validation data saved to: {val_file}")