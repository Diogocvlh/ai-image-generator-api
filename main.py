import os
import logging
import secrets
import uuid
import torch
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from dotenv import load_dotenv
import uvicorn

# ======================================
# ⚙️ CONFIGURAÇÕES GERAIS
# ======================================

load_dotenv()

API_KEY = os.environ.get("API_KEY", "").strip()
if len(API_KEY) < 32 or API_KEY == "SUA_CHAVE_SECRETA_AQUI":
    raise RuntimeError("Configure API_KEY com uma chave aleatória de pelo menos 32 caracteres.")

MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
MAX_WIDTH = 768
MAX_HEIGHT = 768
MAX_STEPS = 60
logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    raise RuntimeError("Esta versão exige uma GPU NVIDIA e uma instalação do PyTorch com CUDA.")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

app = FastAPI(title="AI Image Generator API", version="0.1.0")

# ======================================
# 🚀 CARREGAMENTO DO MODELO
# ======================================

print("🚀 Carregando modelo...")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
if os.environ.get("ENABLE_XFORMERS", "false").lower() == "true":
    pipe.enable_xformers_memory_efficient_attention()

print("✅ Modelo carregado!")

# ======================================
# 📦 MODELO DE REQUISIÇÃO
# ======================================

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    width: int = Field(512, ge=64, le=MAX_WIDTH, multiple_of=8)
    height: int = Field(512, ge=64, le=MAX_HEIGHT, multiple_of=8)
    steps: int = Field(40, ge=1, le=MAX_STEPS)
    guidance: float = Field(7.5, ge=1, le=20, allow_inf_nan=False)
    seed: int | None = Field(None, ge=0, le=2**63 - 1)

# ======================================
# 🔐 SEGURANÇA
# ======================================

def verificar_api_key(x_api_key: str | None = Header(None)):
    if not x_api_key or not secrets.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="API Key inválida")

# ======================================
# 🎨 ENDPOINT DE GERAÇÃO
# ======================================

@app.post("/gerar")
def gerar_imagem(data: GenerateRequest, x_api_key: str | None = Header(None)):

    verificar_api_key(x_api_key)

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

    except Exception:
        logger.exception("Falha ao gerar imagem")
        raise HTTPException(status_code=500, detail="Não foi possível gerar a imagem.")


# ======================================
# 🚀 START SERVIDOR
# ======================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False
    )
