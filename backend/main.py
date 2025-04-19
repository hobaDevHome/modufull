# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from io import BytesIO
import torch
from diffusers import StableDiffusionImg2ImgPipeline

app = FastAPI()
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]
# Allow React dev server (port 3000) to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins= origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "dreamlike-art/dreamlike-photoreal-2.0",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded.")

@app.post("/generate")
async def generate_image(prompt: str = Form(...), image: UploadFile = File(...)):
    image_bytes = await image.read()
    input_image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((512, 512))

    output = pipe(prompt=prompt, image=input_image, strength=0.75, guidance_scale=7.5).images[0]
    output_path = "output.png"
    output.save(output_path)
    return FileResponse(output_path, media_type="image/png")
