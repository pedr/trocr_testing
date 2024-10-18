from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
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

model_name = 'large'
log('Loading model ' + model_name)

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-' + model_name + '-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-' + model_name + '-handwritten')

file = open('rimes_result_' + model_name + '.txt', 'w')

parquet_file = '../images/RIMES-2011-line/data/test.parquet'
truth_data = pd.read_parquet(parquet_file)

log('Starting inference...')

count = 0

# Iterate over the DataFrame rows
for index, row in truth_data.iterrows():
    if count > 100:
        break
    count += 1
    image = row['image']
    
    img = Image.open(io.BytesIO(image['bytes'])).convert("RGB").resize((384, 384))
    pixel_values = processor(images=img, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    log(generated_text)
    file.write(generated_text + '\n')

print('Total time: ', time.time() - start)
