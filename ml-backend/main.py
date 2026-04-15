import asyncio
import io

import torch
from contextlib import asynccontextmanager
from diffusers import StableDiffusionImg2ImgPipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from aggregator import aggregate_activities
from prompt_builder import build_prompt


MODEL_ID       = "runwayml/stable-diffusion-v1-5"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.float16 if DEVICE == "cuda" else torch.float32
IMAGE_W        = 512
IMAGE_H        = 768
STRENGTH       = 0.45
GUIDANCE_SCALE = 7.5
NUM_STEPS      = 28


pipe: StableDiffusionImg2ImgPipeline | None = None
gen_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    print(f"Loading {MODEL_ID} on {DEVICE} ({DTYPE})...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
        variant="fp16" if DEVICE == "cuda" else None,
    ).to(DEVICE)
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("xFormers enabled")
        except Exception:
            print("xFormers not available, continuing without it")
    print("Model ready")
    yield
    del pipe


app = FastAPI(lifespan=lifespan)



def _generate_sync(image: Image.Image, prompt: str, negative_prompt: str) -> bytes:
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=STRENGTH,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_STEPS,
        )
    buf = io.BytesIO()
    result.images[0].save(buf, format="JPEG", quality=95)
    return buf.getvalue()



@app.post("/generate")
async def generate(
    photo: UploadFile = File(...),
    gender: str = Form(...),                  # 'male' | 'female'
    age: int = Form(...),
    height_cm: int = Form(...),
    weight_kg: float = Form(...),
    avg_steps_per_day: int = Form(...),
    avg_calories_per_day: float = Form(...),
    activity_type: str = Form(...),           # 'running' | 'walking' | 'strength' | 'cycling'
    period_months: int = Form(...),           # 1 | 3 | 6
    goal: str = Form(...),                    # 'lose_weight' | 'gain_muscle' | 'maintain'
):
    raw = await photo.read()
    try:
        source = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    source = source.resize((IMAGE_W, IMAGE_H), Image.LANCZOS)

    stats = aggregate_activities(
        avg_steps_per_day, avg_calories_per_day, activity_type,
        period_months, weight_kg, height_cm, goal,
    )
    prompt, negative_prompt, mode = build_prompt(gender, age, height_cm, stats, period_months)

    async with gen_lock:
        loop = asyncio.get_event_loop()
        image_bytes = await loop.run_in_executor(
            None, _generate_sync, source, prompt, negative_prompt
        )

    return Response(
        content=image_bytes,
        media_type="image/jpeg",
        headers={"X-Mode": mode},  # 'gain_weight' | 'motivate' | 'real_progress'
    )
