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

CSS = """
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue",
                 "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
}
.stApp { background-color: #f5f5f7; }

.main .block-container {
    max-width: 900px;
    padding-top: 2.5rem;
    padding-bottom: 3.5rem;
}

h1 {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: #1d1d1f !important;
    letter-spacing: -0.6px;
    line-height: 1.2 !important;
    margin-bottom: 0.15rem !important;
}
h2, h3 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #1d1d1f !important;
    letter-spacing: -0.2px;
    margin-top: 1.5rem !important;
}
p, li, .stMarkdown p { color: #6e6e73; font-size: 1.05rem; line-height: 1.65; }

/* File uploader */
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

/* Images */
[data-testid="stImage"] img { border-radius: 14px; }

/* Caption */
.stCaption, small { color: #8e8e93 !important; font-size: 0.88rem !important; }

/* Spinner */
.stSpinner > div { border-top-color: #0071e3 !important; }

/* Tabs — segmented control estilo macOS */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #e5e5ea;
    border-radius: 14px;
    padding: 4px;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px !important;
    padding: 0.42rem 1.15rem !important;
    color: #6e6e73 !important;
    font-size: 0.91rem !important;
    font-weight: 500 !important;
    border: none !important;
    transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #1d1d1f !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important;
}

/* Status box */
[data-testid="stStatusWidget"] {
    border-radius: 14px !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border-radius: 14px !important;
    border: 1px solid #e5e5ea !important;
}
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


def make_figure(panels: list, titles: list) -> plt.Figure:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(n * 4.0, 4.2))
    axes = list(axes) if n > 1 else [axes]
    fig.patch.set_facecolor("#ffffff")
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=9.5, fontweight="600", color="#1d1d1f", pad=9)
        ax.axis("off")
    plt.tight_layout(pad=1.4)
    return fig


def prediction_card(predicted: str, confidence: float, prob_female: float, prob_male: float) -> str:
    if predicted == "female":
        label, icon, accent = "Femenino", "♀", "#ff375f"
    else:
        label, icon, accent = "Masculino", "♂", "#0071e3"

    bar_f = int(prob_female * 100)
    bar_m = int(prob_male * 100)

    return f"""
    <div style="
        background:#ffffff;
        border-radius:20px;
        padding:1.8rem 2.2rem;
        box-shadow:0 2px 20px rgba(0,0,0,0.08);
        margin:1rem 0 1.6rem;
    ">
        <div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1.5rem;">
            <div style="
                width:56px;height:56px;border-radius:50%;
                background:{accent}18;
                display:flex;align-items:center;justify-content:center;
                font-size:1.9rem;line-height:1;
            ">{icon}</div>
            <div>
                <p style="margin:0;color:#8e8e93;font-size:0.80rem;text-transform:uppercase;
                           letter-spacing:0.09em;font-weight:500;">Clase predicha</p>
                <p style="margin:0;color:{accent};font-size:1.8rem;font-weight:700;
                           letter-spacing:-0.5px;line-height:1.15;">{label}</p>
            </div>
            <div style="
                margin-left:auto;text-align:right;
                background:{accent}10;border-radius:16px;
                padding:0.7rem 1.1rem;
            ">
                <p style="margin:0;color:#8e8e93;font-size:0.80rem;text-transform:uppercase;
                           letter-spacing:0.09em;font-weight:500;">Confianza</p>
                <p style="margin:0;color:{accent};font-size:1.8rem;font-weight:700;
                           line-height:1.15;">{confidence * 100:.1f}%</p>
            </div>
        </div>

        <div style="display:flex;gap:0.6rem;align-items:center;margin-bottom:0.55rem;">
            <span style="font-size:0.86rem;color:#6e6e73;width:74px;">Femenino</span>
            <div style="flex:1;height:8px;background:#f0f0f5;border-radius:10px;overflow:hidden;">
                <div style="width:{bar_f}%;height:100%;
                             background:linear-gradient(90deg,#ff6b9d,#ff375f);
                             border-radius:10px;"></div>
            </div>
            <span style="font-size:0.92rem;font-weight:600;color:#ff375f;
                         width:46px;text-align:right;">{prob_female * 100:.1f}%</span>
        </div>
        <div style="display:flex;gap:0.6rem;align-items:center;">
            <span style="font-size:0.86rem;color:#6e6e73;width:74px;">Masculino</span>
            <div style="flex:1;height:8px;background:#f0f0f5;border-radius:10px;overflow:hidden;">
                <div style="width:{bar_m}%;height:100%;
                             background:linear-gradient(90deg,#34aadc,#0071e3);
                             border-radius:10px;"></div>
            </div>
            <span style="font-size:0.92rem;font-weight:600;color:#0071e3;
                         width:46px;text-align:right;">{prob_male * 100:.1f}%</span>
        </div>
    </div>
    """


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CNN — Clasificador de Género", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Clasificador de Género")
st.markdown(
    "Sube una foto de un rostro y la red neuronal predecirá el género, "
    "mostrando también **cómo llegó a esa conclusión** mediante mapas de calor."
)

with st.expander("¿Cómo funciona esta app?"):
    st.markdown("""
Una **red neuronal convolucional (CNN)** aprende a reconocer patrones visuales en rostros.
Junto a la predicción, generamos dos tipos de mapas que revelan qué zonas del rostro
influyeron más en la decisión:

**Saliency Map** — Pinta los píxeles que más cambiaron la predicción del modelo.
Es como el "resaltador" interno: áreas brillantes = alta influencia.

**Grad-CAM** — Muestra las regiones que más activaron la última capa convolucional.
Es más preciso espacialmente que el Saliency Map.

Los colores **cálidos** (rojo / amarillo) indican alta importancia;
los **fríos** (azul / morado) indican menor relevancia para la predicción.
    """)

st.markdown("<br>", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Arrastra una imagen o haz clic para seleccionar",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded is not None:
    img_pil = Image.open(uploaded)

    # Small preview + filename, side by side
    col_prev, col_info = st.columns([1, 2], gap="large")
    with col_prev:
        st.image(img_pil, caption=uploaded.name, use_container_width=True)
    with col_info:
        st.markdown("**Imagen cargada correctamente.**")
        w, h = img_pil.size
        st.markdown(
            f"- Resolución original: **{w} × {h} px**\n"
            f"- Formato procesado por la CNN: **128 × 128 px**\n\n"
            "Haz clic en **Analizar** cuando estés listo."
        )
        run = st.button("Analizar imagen", type="primary", use_container_width=True)

    if run:
        model, logit_model, grad_logit_model = load_models()

        with st.status("Analizando imagen...", expanded=True) as status:
            st.write("Preprocesando imagen...")
            img_arr = preprocess(img_pil)

            st.write("Ejecutando clasificación con la CNN...")
            prob_male = float(model.predict(img_arr[np.newaxis], verbose=0)[0, 0])
            prob_female = 1.0 - prob_male
            predicted = CLASS_NAMES[int(prob_male > 0.5)]
            confidence = max(prob_male, prob_female)

            st.write("Generando mapas de interpretabilidad...")
            saliency = compute_saliency(logit_model, img_arr)
            gradcam = compute_gradcam(grad_logit_model, img_arr)

            status.update(label="Analisis completo", state="complete", expanded=False)

        # ── Prediction card ────────────────────────────────────────────────────
        st.markdown("### Prediccion del modelo")
        st.html(prediction_card(predicted, confidence, prob_female, prob_male))

        # ── Interpretability maps ──────────────────────────────────────────────
        st.markdown("### Interpretabilidad visual")
        st.caption(
            "La imagen original se muestra al mismo tamaño que los mapas "
            "para facilitar la comparacion."
        )

        original_rgb = np.uint8(255 * img_arr)
        sal_heat = to_heatmap(saliency, cv2.COLORMAP_HOT)
        cam_heat = to_heatmap(gradcam, cv2.COLORMAP_JET)
        sal_over = overlay(img_arr, sal_heat)
        cam_over = overlay(img_arr, cam_heat)

        tab_gen, tab_sal, tab_cam = st.tabs(
            ["Vista General", "Saliency Map", "Grad-CAM"]
        )

        with tab_gen:
            st.caption(
                "Original · Saliency Overlay · Grad-CAM Overlay — todos al mismo tamano."
            )
            fig = make_figure(
                [original_rgb, sal_over, cam_over],
                ["Original", "Saliency Overlay", "Grad-CAM Overlay"],
            )
            st.pyplot(fig)
            plt.close(fig)

        with tab_sal:
            st.caption(
                "Los pixeles mas brillantes son los que mas influyeron en la decision del modelo."
            )
            fig = make_figure(
                [sal_heat, sal_over],
                ["Saliency Map", "Overlay sobre imagen"],
            )
            st.pyplot(fig)
            plt.close(fig)

        with tab_cam:
            st.caption(
                "Las regiones en rojo / amarillo son las que mas activo la ultima "
                "capa convolucional."
            )
            fig = make_figure(
                [cam_heat, cam_over],
                ["Grad-CAM", "Overlay sobre imagen"],
            )
            st.pyplot(fig)
            plt.close(fig)
