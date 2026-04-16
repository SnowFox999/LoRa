import os

# Твой блок путей
USER_HOME = "/home/nshevtsova"
os.environ['HF_HOME'] = f"{USER_HOME}/.cache/huggingface"
os.environ['XDG_CACHE_HOME'] = f"{USER_HOME}/.cache"
os.environ['MPLCONFIGDIR'] = f"{USER_HOME}/.config/matplotlib"
os.environ['HF_TOKEN'] = "" 
MY_CACHE_DIR = os.path.join(USER_HOME, ".cache", "clip")
os.makedirs(MY_CACHE_DIR, exist_ok=True)


# Создаем папки, чтобы не было Permission Denied
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import gc

from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModelWithProjection, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
import torch.nn.functional as F
from torchvision import models, transforms

from torchmetrics.image.fid import FrechetInceptionDistance
from prdc import compute_prdc
import open_clip
from open_clip import create_model_from_pretrained, get_tokenizer
import random

from cleanfid import fid as clean_fid
import tempfile
import shutil


#METADATA = Path("/home/nshevtsova/datasets/BCN_org/metadata.csv")
#IMG_DIR = Path("/home/nshevtsova/datasets/BCN_org")

METADATA = Path("/home/nshevtsova/metadata_clean.csv")
IMG_DIR = Path("/home/nshevtsova/BCN_clean")

DATASET_LORA_DIR = "/home/nshevtsova/datasets/bio_clip"

run_name = "18_run" # МЕНЯЕШЬ ЗДЕСЬ

LOG_DIR = Path("/home/nshevtsova/metrics")
LOG_FILE_PATH = LOG_DIR / f"{run_name}_metrics.txt"

MIN_REAL = 100
N_GEN = 100

df = pd.read_csv(METADATA)


pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.load_lora_weights(
    f"lora_output/{run_name}", 
    weight_name="last.safetensors"
)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()


import random
from glob import glob

# Словарь для хранения загруженных промптов по классам
PROMPTS_CACHE = {}

def load_prompts_for_class(cls):
    """
    Загружает промпты, обеспечивая одинаковый порядок файлов.
    """
    if cls in PROMPTS_CACHE:
        return PROMPTS_CACHE[cls]
    
    class_dir = os.path.join(DATASET_LORA_DIR, cls)
    if not os.path.exists(class_dir):
        return []

    # КРИТИЧНО: sorted() гарантирует, что список промптов всегда будет в одном порядке
    txt_files = sorted(glob(os.path.join(class_dir, "*.txt")))
    
    prompts = []
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
                if prompt:
                    prompts.append(prompt)
        except Exception as e:
            print(f"Ошибка чтения файла {txt_file}: {e}")
            
    PROMPTS_CACHE[cls] = prompts
    return prompts

def generate_images(pipe, cls, n, seed=1908):
    """
    Генерируем картинки: промпты выбираются строго по seed, 
    но сами изображения остаются 'случайными'.
    """
    prompts_pool = load_prompts_for_class(cls)
    
    # Создаем локальный рандом, привязанный к seed. 
    # Он будет отвечать ТОЛЬКО за выбор промптов в этой функции.
    prompt_random = random.Random(seed)
    
    negative = "clinical photo, patient skin, ruler, green, text, hair, watermark, low quality, blurry"
    #negative_prompt = "clinical photo, patient skin, ruler, green, text, hair, cosmos, blue, abstract, oversaturated, vibrant colors, electric colors, neon red, painting, illustration, cartoon, number, watermark, low quality, blurry"
 
    images = []
    used_prompts = []

    for i in range(n):
        # Теперь .choice() всегда будет выбирать один и тот же промпт при одинаковом seed
        if prompts_pool:
            current_prompt = prompt_random.choice(prompts_pool)
        else:
            current_prompt = f"dx_{cls.lower()}, dermoscopy image"

        with torch.no_grad(), torch.amp.autocast('cuda'):
            img = pipe(
                current_prompt,
                negative_prompt=negative,
                num_images_per_prompt=1,
                guidance_scale=8,        
                num_inference_steps=30,    
                cross_attention_kwargs={"scale": 0.7}
            ).images[0]

        images.append(img)
        used_prompts.append(current_prompt)

    return images, used_prompts


def load_real_images(df, image_dir, cls, n=5):
    df = df.sort_values("isic_id") 
    df_cls = df[df["class"] == cls]

    if len(df_cls) == 0:
        return [], []

    # Фиксируем random_state для воспроизводимости
    df_cls = df_cls.sample(n=min(n, len(df_cls)), random_state=1908)

    images = []
    ids = []

    for isic_id in df_cls["isic_id"]:
        img_path = Path(image_dir) / f"{isic_id}.jpg"

        if img_path.exists():
            images.append(Image.open(img_path).convert("RGB"))
            ids.append(isic_id) # Сохраняем ID, чтобы найти потом .txt

    print(f"Loaded images → real: {len(images)} for class {cls}")
    return images, ids

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

fid = FrechetInceptionDistance(feature=2048).to(device)

def pil_to_uint8_tensor(img):
    # PIL → uint8 tensor [3,H,W]
    x = np.array(img, dtype=np.uint8)
    x = torch.from_numpy(x).permute(2, 0, 1)  # HWC → CHW
    return x

def compute_fid(real_imgs, fake_imgs):
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    for img in real_imgs:
        x = pil_to_uint8_tensor(img).unsqueeze(0).to(device)
        fid_metric.update(x, real=True)

    for img in fake_imgs:
        x = pil_to_uint8_tensor(img).unsqueeze(0).to(device)
        fid_metric.update(x, real=False)

    fid_value = fid_metric.compute().item()
    return fid_value

# для FID
inception_feat = models.inception_v3(
    weights=models.Inception_V3_Weights.IMAGENET1K_V1,
    transform_input=False
).to(device).eval()
inception_feat.fc = torch.nn.Identity()


def extract_features(images):
    feats = []
    preprocess = transforms.Compose([
        # Сначала уменьшаем короткую сторону до 299
        transforms.Resize(299), 
        # Вырезаем центральный квадрат 299x299 (без искажений пропорций)
        transforms.CenterCrop(299), 
        transforms.ToTensor(),
        # InceptionV3 ожидает нормализацию ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for img in images:
            x = preprocess(img).unsqueeze(0).to(device)
            f = inception_feat(x)[0].detach().cpu().numpy()
            feats.append(f)

    return np.vstack(feats)


def compute_prd(real_imgs, fake_imgs):
    real_feats = extract_features(real_imgs)
    fake_feats = extract_features(fake_imgs)

    return compute_prdc(
        real_features=real_feats,
        fake_features=fake_feats,
        nearest_k=5
    )


model_id = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

print("Loading BiomedCLIP...")
model, preprocess = create_model_from_pretrained(model_id)
tokenizer = get_tokenizer(model_id)

model.to(device)
model.eval()

context_length = 256 # Специфично для BiomedCLIP

def clip_similarity(images, texts):
    # Используем глобальные объекты из твоей ячейки инициализации
    global model, tokenizer, preprocess, device, context_length
    
    # 1. Токенизация списка текстов
    # Проверяем, что texts — это список строк
    text_tokens = tokenizer(texts, context_length=context_length).to(device)
    
    # 2. Подготовка картинок батчем (используем твое имя 'preprocess')
    # Это превратит список PIL-картинок в один тензор [100, 3, 224, 224]
    image_input = torch.stack([preprocess(img) for img in images]).to(device)
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        # Получаем эмбеддинги
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
        
        # Нормализация для расчета Cosine Similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Считаем сходство пар (image[i] vs text[i])
        # .sum(dim=-1) по нормализованным векторам — это и есть косинусное сходство
        similarities = (image_features * text_features).sum(dim=-1)
        
    return similarities.mean().item()

# Вспомогательная функция для записи в файл и вывода в консоль
def log_and_print(message):
    print(message)
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# Инициализируем файл (очищаем старый, если был)
with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
    f.write(f"=== METRICS FOR RUN: {run_name} ===\n")

results = []

DIAGNOSIS_MAP = {
    "Seborrheic keratosis": "BKL",
    "Solar lentigo": "BKL",
    "Melanoma metastasis": "MEL",
    "Melanoma, NOS": "MEL",
    "Dermatofibroma": "DF",
    "Solar or actinic keratosis": "AK",
    "Basal cell carcinoma": "BCC",
    "Nevus": "NV",
    "Squamous cell carcinoma, NOS": "SCC",
}

df = pd.read_csv(METADATA)
df["class"] = df["diagnosis_3"].map(DIAGNOSIS_MAP)
df = df.dropna(subset=["class"])
classes = sorted(df["class"].unique())

log_and_print(f"Found {len(classes)} classes: {classes}")

for cls in classes:
    log_and_print(f"\n" + "="*30)
    log_and_print(f"=== Class {cls} ===")
    log_and_print("="*30)

    real_imgs, real_ids = load_real_images(
        df=df,
        image_dir=IMG_DIR,
        cls=cls,
        n=N_GEN
    )

    if len(real_imgs) < 2:
        log_and_print(f"[SKIP] Not enough real images for {cls}")
        continue

    real_prompts = []
    trigger = f"dx_{cls.lower()}, "
    for isic_id in real_ids:
        txt_path = os.path.join(DATASET_LORA_DIR, cls, f"{isic_id}.txt")
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                p = f.read().strip()
                real_prompts.append(p.replace(trigger, "").strip())
        except FileNotFoundError:
            log_and_print(f"Внимание: Текст для {isic_id} не найден по пути {txt_path}")
            real_prompts.append(f"dermoscopy image of {cls}")

    # Генерация (использует нашу функцию с фиксированным выбором промптов)
    fake_imgs, fake_prompts = generate_images(pipe, cls, n=N_GEN, seed=42)

    clean_prompts = [p.replace(trigger, "") for p in fake_prompts]

    # Расчет метрик
    fid = compute_fid(real_imgs, fake_imgs)
    prd = compute_prd(real_imgs, fake_imgs)
    clip_sim = clip_similarity(fake_imgs, clean_prompts)
    clip_sim_real = clip_similarity(real_imgs, real_prompts)

    results.append({
        "class": cls,
        "n_real": len(real_imgs),
        "fid": fid,
        "precision": prd["precision"],
        "recall": prd["recall"],
        "density": prd["density"],     
        "coverage": prd["coverage"],
        "clip": clip_sim,
        "clip_real": clip_sim_real,
    })

    # Формируем красивый отчет для записи
    metrics_report = (
        f"FID: {fid:.2f}\n"
        f"Precision: {prd['precision']:.3f}\n"
        f"Density: {prd['density']:.3f}\n"
        f"Recall: {prd['recall']:.3f}\n"
        f"Coverage: {prd['coverage']:.3f}\n"
        f"CLIP (Fake): {clip_sim:.3f}\n"
        f"CLIP (Real): {clip_sim_real:.3f}\n"
    )
    log_and_print(metrics_report)

    del fake_imgs
    del real_imgs # Они тоже больше не нужны
    del prd       # Результат уже в логах
    gc.collect()  # Принудительный сборщик мусора Python
    torch.cuda.empty_cache()

log_and_print("\n=== ALL CLASSES FINISHED ===")
