from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from skimage import color

import matplotlib
matplotlib.use("Agg")          # backend non interactif pour serveur
import matplotlib.pyplot as plt

# --- IMPORTS PROJET ---
from preproc.LAB import rgb_to_lab, extraire_L_et_A_B
from preproc.Normalisation_tf import normalize_L_AB, denormalize_L_AB_np

# --- CHARGEMENT DU GÉNÉRATEUR ---
GENERATOR_PATH = "model_trained_colab.keras"  # toujours vérifier que c'est le bon model entrainé !!!!!!!!!!!!!
generator = tf.keras.models.load_model(GENERATOR_PATH, compile=False)

app = FastAPI(title="Colorizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Colorizer API - OK", "model": "GAN 256x256 LAB"}


# ---------------------------------------------------------
#  PREPROCESSING : PIL → LAB → L normalisé (-1 à 1)
# ---------------------------------------------------------
def preprocess_image_for_model(img_pil: Image.Image) -> np.ndarray:
    """
    Convertit une image PIL en :
      L_normalisé ∈ (1, 256, 256, 1)
    """
    # Resize en 256×256 pour Colorizer 2
    img_pil = img_pil.resize((256, 256))

    # RGB → LAB
    lab_img = color.rgb2lab(np.array(img_pil))

    # Séparation L et AB
    L, _ = extraire_L_et_A_B(lab_img)

    # Normalisation du L dans [-1, 1]
    L = (L / 50.0) - 1.0

    # reshape batch (1,256,256,1)
    L = L.astype("float32")
    L = np.expand_dims(L, axis=(0, -1))

    return L

# ---------------------------------------------------------
#  POST PROCESSING : AB prédits → LAB → RGB
# ---------------------------------------------------------
def postprocess_ab_to_rgb(L_input: np.ndarray, AB_pred: np.ndarray) -> Image.Image:
    """
    Version API alignée avec test_model_output.py → couleurs plus belles.
    """
    # --- Paramètre de saturation (même que dans le script test) ---
    SAT_FACTOR = 1.25  # On peut monter à 1.3 si tu veux encore plus de couleur

    # enlever batch
    L = L_input[0, :, :, 0]             # (256,256)
    AB = AB_pred[0]                     # (256,256,2)

    # 🔥 boost de saturation
    AB = np.clip(AB * SAT_FACTOR, -1.0, 1.0)

    # dénormalisation L → [0,100]
    L_rescaled = (L + 1.0) * 50.0

    # dénormalisation AB → [-128,128]
    AB_rescaled = AB * 128.0

    # reconstruction LAB
    lab = np.zeros((256,256,3), dtype=np.float32)
    lab[:,:,0] = L_rescaled
    lab[:,:,1] = AB_rescaled[:,:,0]
    lab[:,:,2] = AB_rescaled[:,:,1]

    # LAB → RGB float [0,1]
    rgb = color.lab2rgb(lab)

    # conversion en uint8
    rgb = (np.clip(rgb, 0, 1) * 255).astype("uint8")

    return Image.fromarray(rgb)

# ---------------------------------------------------------
#   ENDPOINT : /colorize_montage  → seule image GAN
# ---------------------------------------------------------
@app.post("/colorize_montage", response_class=Response)
async def colorize_montage(file: UploadFile = File(...)):
    """
    Retourne uniquement l'image colorisée par le GAN (PNG 256x256)
    """

    # 1) lire l’image uploadée
    content = await file.read()
    img_pil = Image.open(io.BytesIO(content)).convert("RGB")

    # 2) préprocess : L normalisé comme au training
    L_input = preprocess_image_for_model(img_pil)   # (1,256,256,1)

    # 3) prédiction GAN
    AB_pred = generator.predict(L_input)

    # 4) reconstruction couleur (même postprocess que ton script de test)
    img_colorized = postprocess_ab_to_rgb(L_input, AB_pred)  # PIL Image 256x256

    # 5) retour en PNG
    buf = io.BytesIO()
    img_colorized.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")

# ----------------------------------------------------------------------------------------------------
#   ENDPOINT DEBUG : /colorize_montage_debug → triple image uniquement pour nous et pas l'utilisateur
# ----------------------------------------------------------------------------------------------------
@app.post("/colorize_montage_debug", response_class=Response)
async def colorize_montage_debug(file: UploadFile = File(...)):
    """
    [DEBUG] Retourne un montage :
        [ Entrée L | Sortie GAN | Originale ]
    """

    content = await file.read()
    img_pil = Image.open(io.BytesIO(content)).convert("RGB")
    original_256 = img_pil.resize((256, 256))

    L_input = preprocess_image_for_model(img_pil)   # (1,256,256,1)
    AB_pred = generator.predict(L_input)
    img_colorized = postprocess_ab_to_rgb(L_input, AB_pred)

    # reconstruire L pour affichage N&B
    L = L_input[0, :, :, 0]
    L_rescaled = (L + 1.0) * 50.0
    lab = np.zeros((256, 256, 3), dtype=np.float32)
    lab[:, :, 0] = L_rescaled
    L_rgb = color.lab2rgb(lab)
    L_rgb = (np.clip(L_rgb, 0, 1) * 255).astype("uint8")
    img_L = Image.fromarray(L_rgb)

    # montage horizontal
    montage = Image.new("RGB", (256 * 3, 256))
    montage.paste(img_L, (0, 0))
    montage.paste(img_colorized, (256, 0))
    montage.paste(original_256, (512, 0))

    buf = io.BytesIO()
    montage.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")

