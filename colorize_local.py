import io
import sys
import requests
from PIL import Image

BASE_URL = "https://colorizer-api-847420607839.europe-west1.run.app"

def colorize_image(input_path: str, output_path: str):
    # 1. Lire l'image en binaire
    with open(input_path, "rb") as f:
        files = {"file": (input_path, f, "image/jpeg")}  # ou image/png, peu importe

        # 2. Envoyer la requête POST /colorize
        url = f"{BASE_URL}/colorize"
        response = requests.post(url, files=files)

    # 3. Vérifier la réponse
    if response.status_code != 200:
        print("Erreur de l'API :", response.status_code, response.text)
        return

    # 4. Convertir le contenu binaire en image PIL
    image_bytes = io.BytesIO(response.content)
    img = Image.open(image_bytes).convert("RGB")

    # 5. Sauvegarder le résultat
    img.save(output_path)
    print(f"Image colorisée sauvegardée dans : {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage : python colorize_local.py input.jpg output.png")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    colorize_image(input_path, output_path)
