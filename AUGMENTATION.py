import os

# Твой блок путей
USER_HOME = "/home/nshevtsova"
os.environ['HF_HOME'] = f"{USER_HOME}/.cache/huggingface"
os.environ['XDG_CACHE_HOME'] = f"{USER_HOME}/.cache"
os.environ['MPLCONFIGDIR'] = f"{USER_HOME}/.config/matplotlib"


# Создаем папки, чтобы не было Permission Denied
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)


import torch
import pandas as pd
import os
import random
from tqdm import tqdm
from pathlib import Path
from diffusers import StableDiffusionPipeline
from PIL import Image
from glob import glob

# --- КОНФИГУРАЦИЯ ---
LORA_WEIGHTS = "/home/nshevtsova/LoRa/lora_output/org_run/at-step00016000.safetensors"
DATASET_LORA_DIR = "/home/nshevtsova/datasets/bcn_org"
ORIGINAL_METADATA_PATH = "/home/nshevtsova/metadata_clean.csv"
OUT_DIR = Path("/home/nshevtsova/synthetic_dataset")
OUT_IMG_DIR = OUT_DIR / "org_run"

os.makedirs(OUT_IMG_DIR, exist_ok=True)

# Процент генерации
GEN_PERCENT = 0.20 

# Маппинг (как в твоем коде)
DIAGNOSIS_MAP = {
    "Solar or actinic keratosis": "AK",
    "Basal cell carcinoma": "BCC",
    "Seborrheic keratosis": "BKL",
    "Solar lentigo": "BKL",
    "Dermatofibroma": "DF",
    "Melanoma metastasis": "MEL",
    "Melanoma, NOS": "MEL",
    "Nevus": "NV",
    "Squamous cell carcinoma, NOS": "SCC",
}

PROMPTS_CACHE = {}

def load_prompts_for_class(cls):
    if cls in PROMPTS_CACHE:
        return PROMPTS_CACHE[cls]
    class_dir = os.path.join(DATASET_LORA_DIR, cls)
    if not os.path.exists(class_dir):
        return []
    txt_files = glob(os.path.join(class_dir, "*.txt"))
    prompts = []
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            p = f.read().strip()
            if p: prompts.append(p)
    PROMPTS_CACHE[cls] = prompts
    return prompts

# --- ПОДГОТОВКА МОДЕЛИ ---
print("Loading Model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.load_lora_weights(LORA_WEIGHTS)

# --- РАСЧЕТ КОЛИЧЕСТВА ---
df_orig = pd.read_csv(ORIGINAL_METADATA_PATH)
df_orig["class"] = df_orig["diagnosis_3"].map(DIAGNOSIS_MAP)
df_orig = df_orig.dropna(subset=["class"])

stats = df_orig["class"].value_counts()
print("\nOriginal dataset stats:")
print(stats)

# --- ГЛАВНЫЙ ЦИКЛ ГЕНЕРАЦИИ ---
synthetic_metadata = []

negative_prompt = "clinical photo, patient skin, ruler, green, text, hair, cosmos, blue, abstract, oversaturated, vibrant colors, electric colors, neon red, painting, illustration, cartoon, number, watermark, low quality, blurry"
  
for cls in stats.index:
    target_count = int(stats[cls] * GEN_PERCENT)
    print(f"\n Generating {target_count} images for class: {cls}")
    
    prompts_pool = load_prompts_for_class(cls)
    
    for i in tqdm(range(target_count)):
        # Выбор промпта
        if prompts_pool:
            prompt = random.choice(prompts_pool)
        else:
            prompt = f"dx_{cls.lower()}, {REV_DIAGNOSIS_MAP[cls].lower()}, dermoscopy image"

        # Генерация
        with torch.no_grad():
            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=8.0,
                cross_attention_kwargs={"scale": 0.7}
            ).images[0]

        # Сохранение файла
        synth_id = f"SYNTH_{cls}_{i:05d}"
        img_filename = f"{synth_id}.jpg"
        image.save(OUT_IMG_DIR / img_filename)

        # Создание записи для метадаты
        new_row = {
            "isic_id": synth_id,
            "attribution": "Synthetic (Stable Diffusion + LoRA)",
            "copyright_license": "CC-BY-NC",
            "age_approx": None,
            "anatom_site_1": "synthetic",
            "anatom_site_2": None,
            "anatom_site_general": None,
            "anatom_site_special": None,
            "concomitant_biopsy": False,
            "diagnosis_1": cls,
            "diagnosis_2": cls,
            "diagnosis_3": cls,
            "diagnosis_confirm_type": "synthetic_generation",
            "image_type": "dermoscopic",
            "lesion_id": f"SYNTH_LESION_{cls}_{i:05d}",
            "melanocytic": True if cls in ["MEL", "NV"] else False,
            "sex": None
        }
        synthetic_metadata.append(new_row)

# --- СОХРАНЕНИЕ МЕТАДАННЫХ ---
df_synth = pd.DataFrame(synthetic_metadata)
df_synth.to_csv(OUT_IMG_DIR / "metadata_synth.csv", index=False)

print(f"\n Done! Generated {len(df_synth)} images.")
print(f"Metadata saved to: {OUT_IMG_DIR / 'metadata_synth.csv'}")