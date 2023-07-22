import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
model_id_or_path = "zhyemmmm/MeinaMix"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype = torch.float16)
pipe = pipe.to(device)
pipe.load_textual_inversion("./baddream.pt", token = "baddream")
pipe.load_textual_inversion("./easyneg.pt", token = "easyneg")
pipe.load_textual_inversion("./fastneg.pt", token = "fastneg")

image = Image.open("./image.jpg")


prompt = "A (happy) guy with kid on beach, ((cute)), adorable, high quality"
neg_prompt = "baddream, easyneg, fastneg, ((closed eyes))"


image = pipe(prompt=prompt, negative_prompt = neg_prompt, image=image, strength=0.5, guidance_scale=6).images[0]
image.save("result2.png")