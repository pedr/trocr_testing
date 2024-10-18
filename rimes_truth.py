
import pandas as pd
from datetime import datetime
import time
import io

start = time.time()

def time_to_str(time):
    return str(time.hour) + ':' + str(time.minute) + ':' + str(time.second)

def log(message): 
    date = datetime.now()
    print(time_to_str(date) + ': ' + message)

file = open('rimes_truth.txt', 'w')

parquet_file = '../images/RIMES-2011-line/data/validation.parquet'
truth_data = pd.read_parquet(parquet_file)

count = 0

# Iterate over the DataFrame rows
for index, row in truth_data.iterrows():
    if count > 100:
        break
    count += 1
    text = row['text']
    
    file.write(text + '\n')

print('Total time: ', time.time() - start)
