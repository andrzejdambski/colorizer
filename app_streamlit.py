import io
import requests
import streamlit as st
from PIL import Image

BASE_URL = "https://colorizer-api-847420607839.europe-west1.run.app"

st.set_page_config(page_title="Colorizer GAN", page_icon="🎨", layout="centered")

st.title("🎨 Colorizer GAN – API de prod")
st.write("Envoie une image (N&B ou couleur), l'API la colorise avec le modèle GAN 256×256 LAB.")

# Zone d’upload
uploaded_file = st.file_uploader(
    "Choisis une image (.jpg ou .png)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Afficher l'image source
    st.subheader("Image d'origine")
    original_img = Image.open(uploaded_file)
    st.image(original_img, use_column_width=True)

    # ---------- Bouton principal : image GAN SEULE ----------
    if st.button("Coloriser l'image 🚀", key="btn_gan_only"):
        with st.spinner("Appel à l'API (GAN) en cours..."):
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

            try:
                # 👉 endpoint qui renvoie UNIQUEMENT l'image GAN
                response = requests.post(f"{BASE_URL}/colorize_montage", files=files, timeout=120)
            except Exception as e:
                st.error(f"Erreur lors de l'appel API : {e}")
            else:
                if response.status_code != 200:
                    st.error(
                        f"Erreur de l'API ({response.status_code}) : {response.text}"
                    )
                else:
                    image_bytes = io.BytesIO(response.content)
                    colorized_img = Image.open(image_bytes)

                    st.subheader("Image colorisée (sortie GAN)")
                    st.image(colorized_img, use_column_width=True)

                    # 👉 Bouton de téléchargement : SEULEMENT le GAN
                    buf = io.BytesIO()
                    colorized_img.save(buf, format="PNG")
                    st.download_button(
                        label="💾 Télécharger l'image colorisée",
                        data=buf.getvalue(),
                        file_name="colorized_gan.png",
                        mime="image/png",
                    )

    # ---------- Debug : montage triple (L | GAN | Original) ----------
    with st.expander("🔬 Mode debug : voir montage (L | GAN | Original)", expanded=False):
        st.write(
            "Cette vue est seulement pour vérifier visuellement le comportement du modèle. "
            "Elle n'est pas destinée aux utilisateurs finaux."
        )

        if st.button("Générer le montage debug", key="btn_debug_montage"):
            with st.spinner("Génération du montage debug..."):
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }

                try:
                    # 👉 endpoint debug qui renvoie le triple montage
                    response = requests.post(f"{BASE_URL}/colorize_montage_debug", files=files, timeout=120)
                except Exception as e:
                    st.error(f"Erreur lors de l'appel API (debug) : {e}")
                else:
                    if response.status_code != 200:
                        st.error(
                            f"Erreur de l'API (debug) ({response.status_code}) : {response.text}"
                        )
                    else:
                        image_bytes = io.BytesIO(response.content)
                        montage_img = Image.open(image_bytes)

                        st.image(montage_img, use_column_width=True)
