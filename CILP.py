import json
import os
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import cv2

gen_img_dir = 'm_three/' # generated
visdial_dir = 'resize_orig/' # gt

if __name__ == "__main__":
    # Load CLIP model.
    device = 'cuda'
    model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.to(device)

    all_scores = []
    score_PSNR= []
    num_images = 0

    for root, dirs, files in os.walk(gen_img_dir):
        for file in tqdm(files[:1000]):
            gt_img_path = os.path.join(visdial_dir, file)
            gen_img_path = os.path.join(gen_img_dir, file)

            with open(gt_img_path, 'rb') as f:
                img = Image.open(f)
                img1 = cv2.imread(gt_img_path)
                inputs = clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                gt_feat = clip_model.get_image_features(**inputs)

            # # Compute generated image features.
            with open(gen_img_path, 'rb') as f:
                img = Image.open(f)
                img2 = cv2.imread(gen_img_path)
                inputs = clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                image_feat = clip_model.get_image_features(**inputs)

            # # Compute cosine similarity.
            score = ((image_feat / image_feat.norm()) @ (gt_feat / gt_feat.norm()).T).item()
            all_scores.append(score)
            if score > 0.9:
                num_images += 1
            psnr_score = cv2.PSNR(cv2.resize(img1, (256,256)), cv2.resize(img2, (256,256)))
            score_PSNR.append(psnr_score)

    score = np.mean(all_scores)
    psnr_score = np.mean(score_PSNR)
    print('CLIP similarity:', score)
    print('PSNR:', psnr_score)
    print('Number of images with similarity > 0.9:', num_images)