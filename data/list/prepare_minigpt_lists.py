import os
import pickle
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='Create the dataset based on the given parameters.')  
parser.add_argument('--min_value', type=int, default=0, help='Min value of numbers in the lists')
parser.add_argument('--max_value', type=int, default=100, help='Max value of numbers in the lists')
parser.add_argument('--is_sorted', type=bool, default=True, help='Whether the lists are sorted')
parser.add_argument('--num_list_copies', type=int, default=5, help='Number of copies of each list in training data')
args = parser.parse_args()  

min_value = args.min_value
max_value = args.max_value
is_sorted = args.is_sorted
num_list_copies = args.num_list_copies

# Define paths with list type subdirectory
list_type = "sorted" if is_sorted else "unsorted"
base_dir = os.path.join("data", "list", list_type, f'{min_value}-{max_value}')
output_dir = base_dir

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define file paths
train_file_path = os.path.join(base_dir, f'train_{num_list_copies}.txt')
val_file_path = os.path.join(base_dir, 'test.txt')

print(f"Loading training data from: {train_file_path}")
print(f"Loading validation data from: {val_file_path}")

try:
    with open(train_file_path, 'r') as f:
        train_data = f.read()
    print(f"Length of train dataset in characters: {len(train_data):,}")

    with open(val_file_path, 'r') as f:
        val_data = f.read()
    print(f"Length of val dataset in characters: {len(val_data):,}")

    all_data = train_data + val_data
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure you've generated the text files first.")
    exit(1)

def find_characters(data_string):
    """Find all unique tokens in the data string."""
    # For list task, we need to handle numbers and special characters like '%'
    pattern = r'\d+|%|\S'
    matches = re.findall(pattern, data_string)
    return set(matches)

def process_reasoning(s, stoi, block_size):
    """Process text into tokens with padding."""
    split_text = s.split('\n')
    ret = []
    for st in split_text:
        if st != "":
            enc_str = encode(st, stoi) + [1]  # Add newline token
            ret += enc_str + [0] * (block_size + 1 - len(enc_str))  # Add padding
    return ret

def get_block_size(s, stoi):
    """Calculate the maximum block size needed."""
    split_text = s.split('\n')
    bs = 0
    for st in split_text:
        if st != "":
            enc_str = encode(st, stoi) + [1]  # Add newline token
            bs = max(bs, len(enc_str))
    return bs

def encode_string(s, stonum):
    """Encode a string to a list of integers."""
    ss = s.split(" ")
    encoded_string = [stonum[ch] for ch in ss]
    return encoded_string

def decode_string(l, numtos):
    """Decode a list of integers to a string."""
    dec = ""
    for i in l:
        dec = dec + numtos[i] + " "
    return dec[:-1]

# Get all unique characters that occur in the text
chars = sorted(list(find_characters(all_data)))
print("All unique characters:", ' '.join(chars))

# Create a mapping from characters to integers
stoi = {}  # String to index
itos = {}  # Index to string
idx = 0

# Special tokens first
stoi['[PAD]'] = idx
itos[idx] = '[PAD]'
idx += 1

stoi['\n'] = idx
itos[idx] = '\n'
idx += 1

# Add all other tokens including numbers and '%'
for ch in chars:
    if ch not in stoi:  # Skip if already added
        stoi[ch] = idx
        itos[idx] = ch
        idx += 1

vocab_size = len(stoi)
print(f"Vocabulary size: {vocab_size}")

# Encoder and decoder functions using the mappings
def encode(s, stoi):
    return encode_string(s, stoi)

def decode(l, itos):
    return decode_string(l, itos)

# Calculate block size (round up to nearest multiple of 32)
block_size = (max(get_block_size(train_data, stoi), get_block_size(val_data, stoi)) // 32 + 1) * 32
print(f"The block size is {block_size}")

# Process the data
train_ids = process_reasoning(train_data, stoi, block_size)
val_ids = process_reasoning(val_data, stoi, block_size)

print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# Export to binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

# Define output files
train_output = os.path.join(output_dir, f'train_{num_list_copies}.bin')
val_output = os.path.join(output_dir, 'val.bin')

print(f"Saving training data to: {train_output}")
print(f"Saving validation data to: {val_output}")

train_ids.tofile(train_output)
val_ids.tofile(val_output)

# Save metadata
meta = {
    'block_size': block_size,
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'min_value': min_value,
    'max_value': max_value,
    'is_sorted': is_sorted,
    'simple_format': True,  # Keep this for compatibility
    'unreachable': False,   # Keep this for compatibility
}

meta_output = os.path.join(output_dir, 'meta.pkl')
print(f"Saving metadata to: {meta_output}")

print("String to index mapping:")
print(stoi)
print("Index to string mapping:")
print(itos)

with open(meta_output, 'wb') as f:
    pickle.dump(meta, f)

print(f"Processing complete for {'sorted' if is_sorted else 'unsorted'} lists with values from {min_value} to {max_value}.")