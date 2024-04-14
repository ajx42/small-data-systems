import random
import numpy as np

INPUT_FILE = "The_Fellowship_Of_The_Ring.txt"
OUTPUT_FILE = "benchmark_input.txt"

# Get random string from input file with length in words
def get_request(file_name, length):
    with open(file_name, 'r') as file:
        content = file.read()
    words = content.split()

    assert(length< len(words))

    start_index = random.randint(0, len(words) - length)
    selected_words = words[start_index:start_index + length]
    selected_string = ' '.join(selected_words)

    return selected_string

# generate n requests with lengths from a zipf distribution
request_lens = []
n = 10
alpha = 1.3
min = 20
max = 200

while len(request_lens) < n:
    length = np.random.zipf(alpha, 1)[0]
    if length > min and length < max:
        request_lens.append(length)

requests = []
for l in request_lens:
    requests.append(get_request(INPUT_FILE, l))

# save to file
with open(OUTPUT_FILE, 'w') as file:
        for request in requests:
            file.write(request + '\n')