import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-xxx"), )
# client.base_url = "xxx"
import base64
import os
from retry import retry
from PIL import Image, ImageDraw
import ast

def encode_image(file_path):
    with open(file_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    return base64_image

@retry(tries=10)
def change_text(file_path, text):
    try:
        prompt = "Please edit the image based on the following description:" + text
        base64_image1 = encode_image(file_path)
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image1}",
                        },
                    ],
                }
            ],
            tools=[{"type": "image_generation"}],
        )

        image_generation_calls = [
            output
            for output in response.output
            if output.type == "image_generation_call"
        ]

        image_data = [output.result for output in image_generation_calls]
        return image_data[0]
    except:
        raise ValueError 

with open('diff.json', 'r') as file:
    diff_data = json.load(file)

json_data = {}
counter = 0
index = 1 # step 0,1,2
input_folder = "./text_one"
output_folder = "./text_two"
os.makedirs(output_folder, exist_ok=True)
for root, dirs, files in os.walk("./resize_gen"):
    for file in tqdm(files[:1000]):
        for i in range(1):
            discription = diff_data[file]
            data = ast.literal_eval(discription)
            image = change_text(input_folder+file, data[i+index]['description'])
            if image:
                with open(output_folder+file, "wb") as f:
                    f.write(base64.b64decode(image))