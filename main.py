import os
import uuid
import torch
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import uvicorn

# ======================================
# ⚙️ CONFIGURAÇÕES GERAIS
# ======================================

API_KEY = "SUA_CHAVE_SECRETA_AQUI"
MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
MAX_WIDTH = 768
MAX_HEIGHT = 768
MAX_STEPS = 60

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

app = FastAPI(title="AI Image Generator PRO")

# ======================================
# 🚀 CARREGAMENTO DO MODELO
# ======================================

print("🚀 Carregando modelo...")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    revision="fp16"
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.enable_xformers_memory_efficient_attention()

pipe.safety_checker = None
pipe.requires_safety_checker = False

print("✅ Modelo carregado!")

# ======================================
# 📦 MODELO DE REQUISIÇÃO
# ======================================

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    width: int = 512
    height: int = 512
    steps: int = 40
    guidance: float = 7.5
    seed: int | None = None

# ======================================
# 🔐 SEGURANÇA
# ======================================

def verificar_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")

# ======================================
# 🎨 ENDPOINT DE GERAÇÃO
# ======================================

@app.post("/gerar")
def gerar_imagem(data: GenerateRequest, x_api_key: str = Header(None)):

    verificar_api_key(x_api_key)

    # 🔒 Validações de segurança
    if data.width > MAX_WIDTH or data.height > MAX_HEIGHT:
        raise HTTPException(status_code=400, detail="Resolução muito alta")

    if data.steps > MAX_STEPS:
        raise HTTPException(status_code=400, detail="Steps muito alto")

    try:
        refinamento = (
            "RAW photo, ultra realistic, 8k, high detailed skin, "
            "cinematic lighting, depth of field, sharp focus, DSLR, "
            "photorealistic, masterpiece, best quality"
        )

        negativo = (
            "deformed hands, extra fingers, bad anatomy, low quality, "
            "blurry, jpeg artifacts, watermark"
        )

        prompt_final = f"{data.prompt}, {refinamento}"

        # 🎲 Seed opcional
        if data.seed is not None:
            generator = torch.Generator("cuda").manual_seed(data.seed)
        else:
            generator = None

        image = pipe(
            prompt=prompt_final,
            negative_prompt=negativo,
            width=data.width,
            height=data.height,
            num_inference_steps=data.steps,
            guidance_scale=data.guidance,
            generator=generator
        ).images[0]

        os.makedirs("outputs", exist_ok=True)
        file_name = f"outputs/{uuid.uuid4()}.png"
        image.save(file_name)

        torch.cuda.empty_cache()

        return {
            "status": "success",
            "file": file_name,
            "width": data.width,
            "height": data.height,
            "steps": data.steps,
            "guidance": data.guidance,
            "seed": data.seed
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail="GPU sem memória")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================================
# 🚀 START SERVIDOR
# ======================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )