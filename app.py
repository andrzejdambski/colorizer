from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import numpy as np
from PIL import Image
import io
import tensorflow as tf

from skimage import color

# --- IMPORTS PROJET ---
from preproc.LAB import rgb_to_lab, extraire_L_et_A_B
from preproc.Normalisation_tf import normalize_L_AB, denormalize_L_AB_np

# --- CHARGEMENT DU GÉNÉRATEUR ---
GENERATOR_PATH = "model/generator_pix2pix_colorizer.keras"  # toujours vérifier que c'est le bon fichier !!!!!!!!!!!!!
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
    return {"message": "Colorizer API - OK", "model": "GAN 64x64 LAB"}


# ---------------------------------------------------------
#  PREPROCESSING : PIL → LAB → L normalisé (-1 à 1)
# ---------------------------------------------------------
def preprocess_image_for_model(img_pil: Image.Image) -> np.ndarray:
    """
    Convertit une image PIL en :
      L_normalisé ∈ (1, 64, 64, 1)
    """
    img_pil = img_pil.resize((64, 64))
    img_np = np.array(img_pil) / 255.0

    # RGB → LAB
    lab_img = color.rgb2lab(img_np)

    # Séparation L et AB
    L, _ = extraire_L_et_A_B(lab_img)

    # Normalisation L ∈ [-1,1]
    L = (L / 50.0) - 1.0

    # reshape batch
    L = L.astype("float32")
    L = np.expand_dims(L, axis=(0, -1))  # (1,64,64,1)
    return L


# ---------------------------------------------------------
#  POST PROCESSING : AB prédits → LAB → RGB
# ---------------------------------------------------------
def postprocess_ab_to_rgb(L_input: np.ndarray, AB_pred: np.ndarray) -> Image.Image:
    """
    Reconstruit l'image RGB à partir de :
       - L original (1,64,64,1)
       - AB prédits normalisés (1,64,64,2)
    """
    # enlever batch
    L = L_input[0, :, :, 0]          # (64,64)
    AB = AB_pred[0]                 # (64,64,2)

    # dénormalisation
    L_rescaled = (L + 1) * 50.0
    AB_rescaled = AB * 128.0

    # reconstruire LAB → RGB
    lab_stack = np.stack([L_rescaled, AB_rescaled[:, :, 0], AB_rescaled[:, :, 1]], axis=-1)
    rgb = color.lab2rgb(lab_stack)
    rgb = np.clip(rgb * 255, 0, 255).astype("uint8")

    return Image.fromarray(rgb)


# ---------------------------------------------------------
#   ENDPOINT PRINCIPAL : /colorize
# ---------------------------------------------------------
@app.post("/colorize", response_class=Response)
async def colorize_image(file: UploadFile = File(...)):
    """
    Envoie une image (RGB ou N&B),
    Retourne l'image colorisée (PNG).
    """
    # lire image
    content = await file.read()
    img_pil = Image.open(io.BytesIO(content)).convert("RGB")

    # preprocess L
    L_input = preprocess_image_for_model(img_pil)

    # prédiction AB
    AB_pred = generator.predict(L_input)

    # reconstruction RGB
    img_out = postprocess_ab_to_rgb(L_input, AB_pred)

    # conversion en bytes
    buf = io.BytesIO()
    img_out.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
