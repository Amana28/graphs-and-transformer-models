import pickle

# Specify the path to your pickle file
file_path = 'data/list/sorted/0-100/meta.pkl'

# Open and read the pickle file
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Display the contents
print(data)