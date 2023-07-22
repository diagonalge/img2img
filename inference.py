import json
import torch
from PIL import Image
import numpy as np
import cv2
import time
from loader import Loader

def img2img(img, PROMPT, pipe, NEG_PROMPT = "", CFG = 7.5, SEED = 1, STEPS = 30, STRENGTH = 0.8):
    img = _read_image(img)
    
    if SEED==1:
        img = pipe(
            prompt = PROMPT,
            image = img,
            negative_prompt = NEG_PROMPT,
            guidance_scale = CFG,
            num_inference_steps = STEPS,
            strength = STRENGTH
        ).images[0]
    else:
        generator = torch.Generator("cuda").manual_seed(SEED)
        img = pipe(
            prompt = PROMPT,
            image = img,
            negative_prompt = NEG_PROMPT,
            guidance_scale = CFG,
            num_inference_steps = STEPS,
            generator = generator,
            strength = STRENGTH
        ).images[0]

    torch.cuda.empty_cache()
    img = np.array(img)[:, :, ::-1]
    img = cv2.imencode('.png', img)[1].tobytes()
    return img

def _read_image(img):
    image_bytes = img.file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image
