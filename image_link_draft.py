import json

image_str = '/Users/paddy/Desktop/AiCore/facebook_ml/images_table.json'

print(type(image_str))

with open(image_str, "rb") as fin:
    json_in_str = json.load(fin)

with open('images.json', mode='w') as f:
    json.dump(json_in_str, f, indent=1)


# print(type(image.json))

# print(len(image_json[1]))
# for k, v in image_json.items():
    # print(v)
    # break