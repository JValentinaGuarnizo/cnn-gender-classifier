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

APPLE_CSS = """
<style>
/* ── Fonts ── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue",
                 "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ── Page background ── */
.stApp {
    background-color: #f5f5f7;
}

/* ── Centre column width ── */
.main .block-container {
    max-width: 740px;
    padding-top: 2.8rem;
    padding-bottom: 3rem;
}

/* ── Hero title ── */
h1 {
    font-size: 2.3rem !important;
    font-weight: 700 !important;
    color: #1d1d1f !important;
    letter-spacing: -0.6px;
    line-height: 1.2 !important;
    margin-bottom: 0.2rem !important;
}

/* ── Section headings ── */
h2, h3 {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: #1d1d1f !important;
    letter-spacing: -0.2px;
    margin-top: 1.6rem !important;
}

/* ── Body copy ── */
p, li, .stMarkdown p {
    color: #6e6e73;
    font-size: 1.08rem;
    line-height: 1.65;
}

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    border: 1.5px dashed #d2d2d7 !important;
    border-radius: 18px !important;
    background: #ffffff !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #0071e3 !important;
    background: #f0f7ff !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border-radius: 18px;
    padding: 1.4rem 1.6rem !important;
    box-shadow: 0 2px 14px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] > div {
    color: #6e6e73 !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] > div {
    color: #1d1d1f !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div {
    background-color: #e5e5ea !important;
    border-radius: 10px;
    height: 7px !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #34aadc 0%, #0071e3 100%) !important;
    border-radius: 10px;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: 16px;
}

/* ── Caption ── */
.stCaption, small {
    color: #8e8e93 !important;
    font-size: 0.88rem !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #0071e3 !important; }

/* ── Matplotlib figure background ── */
.stPlot { border-radius: 16px; overflow: hidden; }
</style>
"""


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
        score = logit[:, 0]
    grads = tape.gradient(score, img_var)
    saliency = tf.reduce_max(tf.abs(grads[0]), axis=-1).numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


def compute_gradcam(grad_logit_model, img: np.ndarray) -> np.ndarray:
    img_var = tf.Variable(img[np.newaxis])
    with tf.GradientTape() as tape:
        conv_out, logit = grad_logit_model(img_var, training=False)
        score = logit[:, 0]
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


def prediction_card(predicted: str, confidence: float, prob_female: float, prob_male: float) -> str:
    if predicted == "female":
        label, icon, accent = "Female", "♀", "#ff375f"
    else:
        label, icon, accent = "Male", "♂", "#0071e3"

    bar_female = int(prob_female * 100)
    bar_male = int(prob_male * 100)

    return f"""
    <div style="
        background:#ffffff;
        border-radius:18px;
        padding:1.6rem 2rem;
        box-shadow:0 2px 14px rgba(0,0,0,0.07);
        margin:1rem 0 1.4rem;
    ">
        <div style="display:flex;align-items:center;gap:1.1rem;margin-bottom:1.2rem;">
            <span style="font-size:2.2rem;line-height:1;">{icon}</span>
            <div>
                <p style="margin:0;color:#6e6e73;font-size:0.83rem;text-transform:uppercase;
                           letter-spacing:0.08em;font-weight:500;">Clase predicha</p>
                <p style="margin:0;color:{accent};font-size:1.65rem;font-weight:700;
                           letter-spacing:-0.4px;">{label}</p>
            </div>
            <div style="margin-left:auto;text-align:right;">
                <p style="margin:0;color:#6e6e73;font-size:0.83rem;text-transform:uppercase;
                           letter-spacing:0.08em;font-weight:500;">Confianza</p>
                <p style="margin:0;color:#1d1d1f;font-size:1.65rem;font-weight:700;">{confidence*100:.1f}%</p>
            </div>
        </div>
        <div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:0.3rem;">
            <span style="font-size:0.87rem;color:#6e6e73;width:52px;">Female</span>
            <div style="flex:1;height:7px;background:#e5e5ea;border-radius:10px;overflow:hidden;">
                <div style="width:{bar_female}%;height:100%;
                             background:linear-gradient(90deg,#ff6b9d,#ff375f);
                             border-radius:10px;"></div>
            </div>
            <span style="font-size:0.9rem;font-weight:600;color:#ff375f;width:42px;text-align:right;">{prob_female*100:.1f}%</span>
        </div>
        <div style="display:flex;gap:0.5rem;align-items:center;">
            <span style="font-size:0.87rem;color:#6e6e73;width:52px;">Male</span>
            <div style="flex:1;height:7px;background:#e5e5ea;border-radius:10px;overflow:hidden;">
                <div style="width:{bar_male}%;height:100%;
                             background:linear-gradient(90deg,#34aadc,#0071e3);
                             border-radius:10px;"></div>
            </div>
            <span style="font-size:0.9rem;font-weight:600;color:#0071e3;width:42px;text-align:right;">{prob_male*100:.1f}%</span>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CNN — Clasificador de Género", layout="centered")
st.markdown(APPLE_CSS, unsafe_allow_html=True)

st.title("Clasificador de Género")
st.markdown(
    "Sube una imagen de un rostro. El modelo predice **female / male** "
    "y genera los mapas de interpretabilidad **Saliency** y **Grad-CAM**."
)

st.markdown("<br>", unsafe_allow_html=True)

uploaded = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"],
                            label_visibility="collapsed")

if uploaded is not None:
    img_pil = Image.open(uploaded)

    col_img, _ = st.columns([1, 0.01])
    with col_img:
        st.image(img_pil, caption=uploaded.name, use_container_width=True)

    model, logit_model, grad_logit_model = load_models()

    with st.spinner("Analizando imagen…"):
        img_arr = preprocess(img_pil)
        prob_male = float(model.predict(img_arr[np.newaxis], verbose=0)[0, 0])
        prob_female = 1.0 - prob_male
        predicted = CLASS_NAMES[int(prob_male > 0.5)]
        confidence = max(prob_male, prob_female)

        saliency = compute_saliency(logit_model, img_arr)
        gradcam = compute_gradcam(grad_logit_model, img_arr)

    # --- Prediction card ---
    st.markdown("### Predicción")
    st.markdown(
        prediction_card(predicted, confidence, prob_female, prob_male),
        unsafe_allow_html=True,
    )

    # --- Interpretability ---
    st.markdown("### Mapas de Interpretabilidad")

    sal_heat = to_heatmap(saliency, cv2.COLORMAP_HOT)
    cam_heat = to_heatmap(gradcam, cv2.COLORMAP_JET)
    sal_over = overlay(img_arr, sal_heat)
    cam_over = overlay(img_arr, cam_heat)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.patch.set_facecolor("#ffffff")
    for ax, panel, title in zip(
        axes,
        [sal_heat, sal_over, cam_heat, cam_over],
        ["Saliency Map", "Saliency Overlay", "Grad-CAM", "Grad-CAM Overlay"],
    ):
        ax.imshow(panel)
        ax.set_title(title, fontsize=9, fontweight="500", color="#1d1d1f", pad=8)
        ax.axis("off")
    plt.tight_layout(pad=1.2)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "**Saliency:** píxeles con mayor influencia sobre el logit.  "
        "**Grad-CAM:** regiones activadas en la última capa convolucional."
    )
