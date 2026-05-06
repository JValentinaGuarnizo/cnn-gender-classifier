import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Conv2D, Dense

matplotlib.use("Agg")

IMG_SIZE = (128, 128)
MODEL_PATH = "models/model.keras"
CLASS_NAMES = ["female", "male"]


@st.cache_resource(show_spinner="Cargando modelo…")
def load_models():
    model = tf.keras.models.load_model(MODEL_PATH)

    conv_layers = [l for l in model.layers if isinstance(l, Conv2D)]
    last_conv = conv_layers[-1]

    logit_layer = Dense(1, name="logit_out")
    logit_out = logit_layer(model.layers[-2].output)
    logit_model = tf.keras.Model(inputs=model.input, outputs=logit_out)
    logit_model.get_layer("logit_out").set_weights(model.layers[-1].get_weights())

    grad_logit_model = tf.keras.Model(
        inputs=model.input,
        outputs=[last_conv.output, logit_model.output],
    )
    return model, logit_model, grad_logit_model


def preprocess(img_pil: Image.Image) -> np.ndarray:
    img = img_pil.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def compute_saliency(logit_model, img: np.ndarray) -> np.ndarray:
    img_var = tf.Variable(img[np.newaxis])
    with tf.GradientTape() as tape:
        logit = logit_model(img_var, training=False)
        score = logit[:, 0]  # slice must be inside the tape context
    grads = tape.gradient(score, img_var)
    saliency = tf.reduce_max(tf.abs(grads[0]), axis=-1).numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


def compute_gradcam(grad_logit_model, img: np.ndarray) -> np.ndarray:
    img_var = tf.Variable(img[np.newaxis])
    with tf.GradientTape() as tape:
        conv_out, logit = grad_logit_model(img_var, training=False)
        score = logit[:, 0]  # slice must be inside the tape context
    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.maximum(tf.reduce_sum(conv_out[0] * pooled, axis=-1), 0).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cv2.resize(cam, IMG_SIZE)


def to_heatmap(gray: np.ndarray, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    colored = cv2.applyColorMap(np.uint8(255 * gray), colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay(img_01: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(np.uint8(255 * img_01), 1 - alpha, heatmap_rgb, alpha, 0)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CNN — Clasificador de Género", layout="centered")
st.title("Clasificador de Género con CNN")
st.markdown(
    "Sube una imagen de un rostro. El modelo predice **male / female** "
    "y genera los mapas de interpretabilidad **Saliency** y **Grad-CAM**."
)

uploaded = st.file_uploader("Selecciona una imagen (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img_pil = Image.open(uploaded)
    st.image(img_pil, caption="Imagen cargada", use_container_width=True)

    model, logit_model, grad_logit_model = load_models()

    with st.spinner("Procesando…"):
        img_arr = preprocess(img_pil)
        prob_male = float(model.predict(img_arr[np.newaxis], verbose=0)[0, 0])
        prob_female = 1.0 - prob_male
        predicted = CLASS_NAMES[int(prob_male > 0.5)]
        confidence = max(prob_male, prob_female)

        saliency = compute_saliency(logit_model, img_arr)
        gradcam = compute_gradcam(grad_logit_model, img_arr)

    # --- Prediction ---
    st.subheader("Predicción")
    col1, col2 = st.columns(2)
    col1.metric("Female", f"{prob_female * 100:.1f}%")
    col2.metric("Male", f"{prob_male * 100:.1f}%")

    st.markdown(f"**Clase predicha:** `{predicted}` — confianza `{confidence * 100:.1f}%`")
    st.progress(prob_male)
    st.caption("← Female (0 %)　　　　　Male (100 %) →")

    # --- Interpretability ---
    st.subheader("Mapas de Interpretabilidad")

    sal_heat = to_heatmap(saliency, cv2.COLORMAP_HOT)
    cam_heat = to_heatmap(gradcam, cv2.COLORMAP_JET)
    sal_over = overlay(img_arr, sal_heat)
    cam_over = overlay(img_arr, cam_heat)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, panel, title in zip(
        axes,
        [sal_heat, sal_over, cam_heat, cam_over],
        ["Saliency Map", "Saliency Overlay", "Grad-CAM", "Grad-CAM Overlay"],
    ):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "**Saliency:** píxeles con mayor influencia sobre el logit. "
        "**Grad-CAM:** regiones activadas en la última capa convolucional."
    )
