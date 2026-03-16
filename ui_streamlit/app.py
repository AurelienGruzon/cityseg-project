from __future__ import annotations

import io
from pathlib import Path
from typing import List

import requests
import streamlit as st
from PIL import Image

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="CitySeg Demo", layout="wide")

# Ajuste cette URL si besoin.
DEFAULT_API_URL = "https://cityseg-api-hwhqavd3a2arhfd5.francecentral-01.azurewebsites.net"
DEFAULT_IMAGES_DIR = Path("samples/leftImg8bit/val")
DEFAULT_MASKS_DIR = Path("samples/gtFine/val")


# -----------------------------
# Helpers
# -----------------------------
def find_cityscapes_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []
    return sorted(images_dir.rglob("*_leftImg8bit.png"))


def get_mask_path_from_image(image_path: Path, images_root: Path, masks_root: Path) -> Path:
    relative = image_path.relative_to(images_root)
    mask_name = image_path.name.replace("_leftImg8bit.png", "_gtFine_color.png")
    return masks_root / relative.parent / mask_name


def call_prediction_api(api_url: str, image_bytes: bytes, mode: str = "overlay") -> Image.Image:
    url = f"{api_url.rstrip('/')}/predict"
    response = requests.post(
        url,
        params={"mode": mode},
        files={"file": ("image.png", image_bytes, "image/png")},
        timeout=120,
    )
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Configuration")
api_url = st.sidebar.text_input("URL de l'API", value=DEFAULT_API_URL)
images_dir_str = st.sidebar.text_input("Dossier images", value=str(DEFAULT_IMAGES_DIR))
masks_dir_str = st.sidebar.text_input("Dossier masks réels", value=str(DEFAULT_MASKS_DIR))
predict_mode = st.sidebar.selectbox(
    "Mode de prédiction",
    options=["overlay", "mask"],
    index=0,
    help="overlay = image + segmentation, mask = segmentation seule",
)

images_dir = Path(images_dir_str)
masks_dir = Path(masks_dir_str)


# -----------------------------
# Main UI
# -----------------------------
st.title("CitySeg — Démo de segmentation")
st.write(
    "Sélectionne une image Cityscapes, appelle l'API de prédiction, puis compare l'image réelle, le mask réel et le mask prédit."
)

# Health check simple
with st.expander("Vérification rapide de l'API"):
    if st.button("Tester /health"):
        try:
            r = requests.get(f"{api_url.rstrip('/')}/health", timeout=30)
            st.success(f"API joignable — statut HTTP {r.status_code} — réponse: {r.text}")
        except Exception as exc:
            st.error(f"Impossible de joindre l'API: {exc}")

image_paths = find_cityscapes_images(images_dir)

if not image_paths:
    st.warning(
        "Aucune image trouvée. Vérifie les chemins dans la barre latérale. "
        "La webapp attend les images Cityscapes locales pour afficher l'image réelle et le mask réel."
    )
    st.stop()

# IDs lisibles pour l'utilisateur
image_options = {
    p.name.replace("_leftImg8bit.png", ""): p
    for p in image_paths
}
selected_id = st.selectbox("ID de l'image", options=list(image_options.keys()))
selected_image_path = image_options[selected_id]
selected_mask_path = get_mask_path_from_image(selected_image_path, images_dir, masks_dir)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Image réelle")
    real_image = load_image(selected_image_path)
    st.image(real_image, width='stretch')
    st.caption(str(selected_image_path))

with col2:
    st.subheader("Mask réel")
    if selected_mask_path.exists():
        real_mask = load_image(selected_mask_path)
        st.image(real_mask, width='stretch')
        st.caption(str(selected_mask_path))
    else:
        st.info("Mask réel introuvable pour cette image.")

with col3:
    st.subheader("Mask prédit")
    if st.button("Lancer la prédiction", type="primary"):
        try:
            image_bytes = selected_image_path.read_bytes()
            with st.spinner("Appel à l'API en cours..."):
                pred_img = call_prediction_api(api_url, image_bytes, mode=predict_mode)
            st.image(pred_img, width='stretch')
            st.success("Prédiction terminée.")
        except Exception as exc:
            st.error(f"Erreur pendant l'appel API: {exc}")
    else:
        st.info("Clique sur « Lancer la prédiction ».")


# -----------------------------
# Upload libre (optionnel)
# -----------------------------
st.divider()
st.subheader("Tester une image libre")
uploaded_file = st.file_uploader("Uploader une image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.read()
    up_col1, up_col2 = st.columns(2)

    with up_col1:
        st.write("Image envoyée")
        st.image(Image.open(io.BytesIO(uploaded_bytes)).convert("RGB"), width='stretch')

    with up_col2:
        st.write("Résultat API")
        if st.button("Prédire l'image uploadée"):
            try:
                with st.spinner("Prédiction en cours..."):
                    pred_uploaded = call_prediction_api(api_url, uploaded_bytes, mode=predict_mode)
                st.image(pred_uploaded, width='stretch')
            except Exception as exc:
                st.error(f"Erreur pendant la prédiction: {exc}")


st.divider()
st.caption(
    "Démo Streamlit du projet CitySeg. L'application consomme l'API FastAPI déployée sur Azure."
)
