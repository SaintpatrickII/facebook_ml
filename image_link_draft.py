import json
import pandas as pd
import numpy as np

image_str = '/Users/paddy/Desktop/AiCore/facebook_ml/images_table.json'
iamge_json = '/Users/paddy/Desktop/AiCore/facebook_ml/images.json'

print(type(image_str))

with open(image_str, "rb") as fin:
    json_in_str = json.load(fin)

with open('images.json', mode='w') as f:
    json.dump(json_in_str, f, indent=1)


df_image_uuid = pd.read_json('/Users/paddy/Desktop/AiCore/facebook_ml/images.json')
print(df_image_uuid.head())