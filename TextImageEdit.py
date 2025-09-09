import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-xxx"), )
client.base_url = "xxx"
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
def change_m(file_path, file_path2, text):
    try:
        prompt = "Please edit the first image based on the following description and the second reference image:" + text
        base64_image1 = encode_image(file_path)
        base64_image2 = encode_image(file_path2)
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
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image2}",
                        }
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
index = 1 # step 1,2,3
input_folder = "./m_one"
output_folder = "./m_two"
for root, dirs, files in os.walk("./m_one"):
    for file in tqdm(files):
        for i in range(1):
            discription = diff_data[file]
            data = ast.literal_eval(discription)
            image = change_m(input_folder+file, "./diff/"+file.split(".")[0]+"_"+str(i+index)+".png", data[i+index]['description'])
            if image:
                with open(output_folder+file, "wb") as f:
                    f.write(base64.b64decode(image))