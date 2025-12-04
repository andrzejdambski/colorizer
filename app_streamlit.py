import io
import requests
import streamlit as st
from PIL import Image

BASE_URL = "https://colorizer-api-847420607839.europe-west1.run.app"

st.set_page_config(page_title="Colorizer GAN", page_icon="🎨", layout="centered")

st.title("🎨 Colorizer GAN – API de prod")
st.write("Envoie une image (N&B ou couleur), l'API la colorise avec le modèle GAN 64x64 LAB.")

# Zone d'upload
uploaded_file = st.file_uploader(
    "Choisis une image (.jpg ou .png)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Afficher l'image source
    st.subheader("Image d'origine")
    original_img = Image.open(uploaded_file)
    st.image(original_img, use_column_width=True)

    if st.button("Coloriser l'image 🚀"):
        with st.spinner("Appel à l'API en cours..."):
            # Préparer les données multipart/form-data
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

            try:
                response = requests.post(f"{BASE_URL}/colorize", files=files, timeout=60)
            except Exception as e:
                st.error(f"Erreur lors de l'appel API : {e}")
            else:
                if response.status_code != 200:
                    st.error(
                        f"Erreur de l'API ({response.status_code}) : {response.text}"
                    )
                else:
                    # Convertir la réponse binaire en image
                    image_bytes = io.BytesIO(response.content)
                    colorized_img = Image.open(image_bytes)

                    st.subheader("Image colorisée (depuis Cloud Run)")
                    st.image(colorized_img, use_column_width=True)

                    # Bouton de téléchargement
                    buf = io.BytesIO()
                    colorized_img.save(buf, format="PNG")
                    st.download_button(
                        label="💾 Télécharger l'image colorisée",
                        data=buf.getvalue(),
                        file_name="colorized.png",
                        mime="image/png",
                    )
