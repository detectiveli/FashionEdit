import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-xxxx"), )
# client.base_url = "xxx"
import base64
import os
from retry import retry
from PIL import Image, ImageDraw

def encode_image(file_path):
    with open(file_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    return base64_image

@retry(tries=10)
def find_diff(file_path, file_path2):
    try:
        prompt = """Detect the three detailed differences of the clothes between the two images and return JSON style. For each result, the discription should only contain the difference of the first image, and give the bounding box of the box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."""
        base64_image = encode_image(file_path)
        base64_image2 = encode_image(file_path2)

        response = client.chat.completions.create(
            model="gpt-4.1-mini", 
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    },
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image2}"}
                    }
                ]}
            ],
            response_format={"type": "json_object"},
            temperature=0.,
        )

        ans = json.loads(response.choices[0].message.content)['differences']
        return ans
    except:
        raise ValueError

def save_box(discription, path, name):
    image_orig = Image.open(path)
    image = Image.open(path)
    width, height = image.size
    item_0 = 0
    for bounding_box in discription:
        abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
        top_left = (abs_x1, abs_y1)
        bottom_right = (abs_x2, abs_y2)
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            [top_left, bottom_right],
            outline="red",    
            width=3,         
        )
        cropped_image = image_orig.crop((abs_x1, abs_y1, abs_x2, abs_y2))
        cropped_image.save('./diff/' + file.split(".")[0]+ "_" + str(item_0) + ".png") 
        item_0 += 1
    image.save('./diff/' + name)

json_data = {}
for root, dirs, files in os.walk("./resize_gen"):
    for file in tqdm(files):
        discription = find_diff("./resize_orig/"+file, "./resize_gen/"+file)
        save_box(discription, "./resize_orig/"+file, file)
        json_data[file] = str(discription)
            
import json 
with open('./diff.json', 'w') as file:
    json.dump(json_data, file, ensure_ascii=False)