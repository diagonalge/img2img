from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from inference import img2img
from typing import Optional
from utils import get_zip_response
from resize import resize
from loader import Loader
import shutil

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_loader = Loader()


def read_image(img):
    image_bytes = img.file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = resize(image)
    return image

@app.post("/img2img")
def infer(
    img: UploadFile,
    prompt: str = Form(),
    model: str = Form(),
    neg_prompt: Optional[str] = Form(' '),
    cfg: Optional[float] = Form(5),
    seed: Optional[int] = Form(1),
    steps: Optional[int] = Form(30),
    strength: Optional[float] = Form(50)
):
    pipe = _loader.get_pipe(Loader.Module.IMAGE_TO_IMAGE, model)
    strength = 1.3-(((strength- 0) / (100 - 0) ) * (1 - 0.3) + 0.3)
    img = img2img(img, prompt, pipe, neg_prompt, cfg, seed, steps, strength)
    return Response(img)


if __name__=="__main__":
    import uvicorn
    uvicorn.run("server:app", port=5000, host="0.0.0.0", workers=1)