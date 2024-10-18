from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import os
from datetime import datetime
import pdb
import time

start = time.time()
def time_to_str(time):
    return str(time.hour) + ':' + str(time.minute) + ':' + str(time.second)

def log(message): 
    date = datetime.now()
    print(time_to_str(date) + ': ' + message)

model_name = 'small'
log('Loading model' + model_name + '...')

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-' + model_name + '-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-' + model_name + '-handwritten')

file = open('iam_result_'+  model_name + '.txt', 'w')

truth_file = open('../images/IAM/gt_test.txt', 'r')

log('Starting inference...')

count = 0;

for line in truth_file: 
    if count > 100:
        break
    count = count + 1
    image = line.split(None, 1)[0]
    img = Image.open('../images/IAM/image/' + image).convert("RGB").resize((384, 384))
    pixel_values = processor(images=img, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    log(generated_text)
    file.write(generated_text + '\n')


print('Total time: ', time.time() - start)