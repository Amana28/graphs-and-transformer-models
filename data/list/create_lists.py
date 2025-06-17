import os
import random
import argparse

def generate_random_list(min_value, max_value, min_length, max_length, is_sorted, fixed_length=None, only_min_max_length=False):
    """
    Generate a random list of integers with variable or fixed length.
    
    Args:
        min_value: Minimum value for generated numbers (inclusive)
        max_value: Maximum value for generated numbers (inclusive)
        min_length: Minimum length for the generated list (used if fixed_length is None)
        max_length: Maximum length for the generated list (used if fixed_length is None)
        is_sorted: Boolean flag indicating if the list should be sorted
        fixed_length: If provided, all lists will have exactly this length
        only_min_max_length: If True, length will be randomly chosen between min_length and max_length only
                            (not any values in between)
    
    Returns:
        A random list of integers
    """
    # Determine the length of this particular list
    if fixed_length is not None:
        length = fixed_length
    elif only_min_max_length:
        # Choose randomly between min_length and max_length only
        length = random.choice([min_length, max_length])
    else:
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
    
    return rand_list

def apply_permutation(input_list, permutation_type="reversal", fixed_indices=None):
    """
    Apply a permutation to the input list based on the specified type.
    
    Args:
        input_list: The original list to permute
        permutation_type: Type of permutation to apply
                          "reversal" - simply reverse the list
                          "random" - apply a random permutation
                          "manual" - apply the specific permutation from the manual example
                          "custom" - compare n-2 and n-1 elements, append 1st or 2nd element based on comparison
        fixed_indices: Optional fixed permutation indices to use for "random" type
    
    Returns:
        The permuted list
    """
    if permutation_type == "reversal":
        # Simply reverse the list
        return list(reversed(input_list))
    
    elif permutation_type == "random":
        # For random permutation with fixed length, use the provided indices
        if fixed_indices is not None:
            return [input_list[i] for i in fixed_indices]
        else:
            # Fallback to truly random permutation if no fixed indices provided
            indices = list(range(len(input_list)))
            random.shuffle(indices)
            return [input_list[i] for i in indices]
    
    elif permutation_type == "manual":
        # The specific manual example permutation: σ(1)=2, σ(2)=6, σ(3)=5, σ(4)=4, σ(5)=3, σ(6)=1
        # Map: 1→2, 2→6, 3→5, 4→4, 5→3, 6→1
        # For arbitrary length lists, we'll extend this pattern cyclically
        
        # Define the mapping from the manual example (zero-indexed)
        manual_map = {0: 1, 1: 5, 2: 4, 3: 3, 4: 2, 5: 0}
        
        # Apply the mapping to each position, cycling for lists longer than 6
        permuted_list = []
        for i in range(len(input_list)):
            mapped_i = manual_map.get(i % 6, i)  # Cycle through the map for longer lists
            if mapped_i < len(input_list):
                permuted_list.append(input_list[mapped_i])
            else:
                # Fallback for edge cases when the mapping is out of bounds
                permuted_list.append(input_list[i])
                
        return permuted_list
    
    elif permutation_type == "custom":
        # Custom permutation: compare n-2 and n-1 elements, append 1st or 2nd element
        if len(input_list) < 2:
            # If list has less than 2 elements, just return a copy
            return input_list.copy()
        
        # Start with the original list
        result = input_list.copy()
        
        # Compare n-2 and n-1 elements (last two elements)
        n_minus_2 = input_list[-2]  # n-2 element
        n_minus_1 = input_list[-1]  # n-1 element
        
        # Add 1st element if n-2 < n-1, otherwise add 2nd element
        if n_minus_2 < n_minus_1:
            result.append(input_list[0])  # 1st element
        else:
            if len(input_list) >= 2:
                result.append(input_list[1])  # 2nd element
            else:
                result.append(input_list[0])  # Fallback if only 1 element
        
        return result
    
    else:
        # Default to returning the original list
        return input_list

def format_list(rand_list, permuted_list, include_separator=True):
    """Format the list as a string with an optional '%' separator."""
    if include_separator:
        return " ".join(map(str, rand_list)) + " % " + " ".join(map(str, permuted_list)) + "\n"
    else:
        return " ".join(map(str, permuted_list)) + "\n"

def generate_datasets(args, fixed_indices=None):
    """Generate training and validation datasets according to the specified parameters."""
    all_lists = []
    
    # Generate the requested number of unique lists
    for _ in range(args.num_lists):
        rand_list = generate_random_list(
            args.min_value, 
            args.max_value, 
            args.min_length, 
            args.max_length,
            args.is_sorted,
            args.fixed_length,
            args.only_min_max_length
        )
        
        # Apply the selected permutation
        permuted_list = apply_permutation(rand_list, args.permutation_type, fixed_indices)
        
        formatted_list = format_list(rand_list, permuted_list, args.include_separator)
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

def save_indices(indices, file_path):
    """Save the permutation indices to a file for reference."""
    with open(file_path, "w") as file:
        file.write(",".join(map(str, indices)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a random list based on the given parameters')      
    parser.add_argument('--is_sorted', type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help='A boolean flag indicating sorted status')       
    parser.add_argument('--min_value', type=int, default=0, 
                        help='min value for generated numbers -- inclusive ')       
    parser.add_argument('--max_value', type=int, default=100, 
                        help='max value for generated numbers -- inclusive')     
    parser.add_argument('--min_length', type=int, default=1, 
                        help='min length for the generated lists (when using variable length)')         
    parser.add_argument('--max_length', type=int, default=50, 
                        help='max length for the generated lists (when using variable length)')
    parser.add_argument('--fixed_length', type=int, default=None,
                        help='If provided, all lists will be this exact length (overrides min/max length)')
    parser.add_argument('--only_min_max_length', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='If True, randomly choose between min_length and max_length only (not values in between)')
    parser.add_argument('--num_list_copies', type=int, default=1, 
                        help='the number of times each list is repeated in the training data.')  
    parser.add_argument('--num_lists', type=int, default=10000, 
                        help='the total number of generated lists')       
    parser.add_argument('--chance_in_train', type=float, default=0.7, 
                        help='ratio of training set -- the rest is validation')  
    parser.add_argument('--permutation_type', type=str, default="reversal",
                        choices=["reversal", "random", "manual", "custom"],
                        help='Type of permutation to apply (reversal, random, manual, or custom)')
    parser.add_argument('--include_separator', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Include % separator between original sequence and permutation (default: True)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Determine the folder path based on sorted status and length type
    length_type = f"fixed{args.fixed_length}" if args.fixed_length is not None else "variable"
    perm_type = args.permutation_type
    
    if args.is_sorted:
        folder_name = f"data/list/sorted/{length_type}/{args.min_value}-{args.max_value}/{perm_type}"
    else:
        folder_name = f"data/list/unsorted/{length_type}/{args.min_value}-{args.max_value}/{perm_type}"
    
    # Create directories if they don't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # For random permutation type with fixed length, generate a single set of indices
    fixed_indices = None
    if args.permutation_type == "random" and args.fixed_length is not None:
        # Generate a random permutation of the fixed length
        fixed_indices = list(range(args.fixed_length))
        random.shuffle(fixed_indices)
        
        # Save the indices to a file for reference
        indices_file = os.path.join(folder_name, "random_indices.txt")
        save_indices(fixed_indices, indices_file)
        print(f"Generated and saved random permutation indices to {indices_file}")
    
    # Generate datasets
    train_lists, val_lists = generate_datasets(args, fixed_indices)
    
    # Write datasets to files
    train_file = os.path.join(folder_name, f'train_{args.num_list_copies}.txt')
    val_file = os.path.join(folder_name, 'test.txt')
    
    write_dataset(train_lists, train_file)
    write_dataset(val_lists, val_file)
    
    # Save the permutation indices in a separate file
    if args.permutation_type == "random" and fixed_indices is not None:
        permutation_file = os.path.join(folder_name, "permutation_indices.txt")
        with open(permutation_file, "w") as f:
            # Save indices with their positions for better readability
            for i, idx in enumerate(fixed_indices):
                f.write(f"{i} -> {idx}\n")
            
            # Also save as comma-separated on a single line
            f.write("\n# Comma-separated format:\n")
            f.write(",".join(map(str, fixed_indices)))
        
        print(f"Detailed permutation mapping saved to: {permutation_file}")
    
    # Print summary message with length information
    length_info = f"fixed length of {args.fixed_length}" if args.fixed_length is not None else f"variable length ({args.min_length}-{args.max_length})"
    if args.only_min_max_length and args.fixed_length is None:
        length_info = f"only {args.min_length} or {args.max_length} length"
    sort_info = "sorted" if args.is_sorted else "unsorted"
    
    print(f"Generated {sort_info} lists with {length_info} using {perm_type} permutation:")
    print(f"- {len(train_lists)} training examples ({len(train_lists)/args.num_list_copies} unique lists × {args.num_list_copies} copies)")
    print(f"- {len(val_lists)} validation examples")
    if args.permutation_type == "random" and fixed_indices is not None:
        print(f"- Using fixed random permutation indices: {fixed_indices[:10]}...")
    print(f"Training data saved to: {train_file}")
    print(f"Validation data saved to: {val_file}")