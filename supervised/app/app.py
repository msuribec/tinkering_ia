import streamlit as st

st.set_page_config(
    page_title="Taller 02 - Supervisado | EAFIT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1a3a5c;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #3a6ea5;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .author-box {
        background: linear-gradient(135deg, #1a3a5c 0%, #3a6ea5 100%);
        color: white !important;
        border-radius: 10px;
        padding: 1.2rem 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .author-box * { color: white !important; }
    .divider   { border-top: 2px solid #e2e8f0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎓 Taller 02 — Aprendizaje Supervisado</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Fundamentos, Experimentación y Despliegue de Modelos</div>',
    unsafe_allow_html=True
)

# ── Author / Course Info ──────────────────────────────────────────────────────
st.markdown("""
<div class="author-box">
    <div style="font-size:1.1rem; font-weight:700; margin-bottom:0.5rem;">
        👥 Equipo de Trabajo
    </div>
    <div style="font-size:1.0rem;">
        Javier Daza Olivella &nbsp;·&nbsp; Pablo Jimeno Juca &nbsp;·&nbsp; María Sofía Uribe
    </div>
    <div style="margin-top:0.8rem; font-size:0.95rem; opacity:0.9;">
        📚 Maestría en Ciencia de los Datos &nbsp;|&nbsp;
        🏛️ Universidad EAFIT — Periodo 2026‑1<br>
        👨‍🏫 Docente: Jorge Iván Padilla‑Buriticá
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Navigation Guide ──────────────────────────────────────────────────────────
st.markdown("### 🗺️ Guía de Navegación")
col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "**📊 Regresión**\n\n"
        "Pipeline completo de regresión sobre el dataset local de costos médicos (Insurance). "
        "EDA, preprocesamiento, entrenamiento de 5 modelos, comparación de métricas R², MAE y RMSE."
    )

with col2:
    st.info(
        "**📈 Clasificación**\n\n"
        "Pipeline de clasificación multiclase sobre el dataset local Digits. "
        "5 modelos con GridSearchCV, Matriz de Confusión 10×10, "
        "F1-Score macro y Feature Importance de píxeles."
    )

with col3:
    st.info(
        "**🚀 Dashboard**\n\n"
        "Tablero interactivo de predicciones. Predicción individual con "
        "visualización del dígito 8×8, predicción por lote (CSV), importancia de features y "
        "evidencia de pruebas para los mejores modelos."
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Datasets ──────────────────────────────────────────────────────────────────
st.markdown("### 📦 Datasets Utilizados")

col_d1, col_d2 = st.columns(2)

with col_d1:
    st.info(
        "**📊 Medical Cost Insurance**\n\n"
        "- 1,338 registros · 7 variables\n"
        "- Target: `charges` (costo médico en USD)\n"
        "- Fuente local: `data/raw/insurance.csv`\n"
        "- Features: age, sex, BMI, children, smoker, region\n"
        "- Tarea: **Regresión**"
    )

with col_d2:
    st.info(
        "**📈 Digits**\n\n"
        "- 1,797 registros · 64 features (píxeles 8×8)\n"
        "- Target: `target` — 0 a 9 (10 clases)\n"
        "- Fuente local: `data/raw/digits.csv`\n"
        "- Features: pixel_0 … pixel_63 (intensidad 0–16)\n"
        "- Tarea: **Clasificación Multiclase**"
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Methodology ───────────────────────────────────────────────────────────────
st.markdown("### 🔬 Metodología")

st.info(
    "Cada pipeline sigue el proceso estándar de aprendizaje supervisado:\n\n"
    "1. **Carga y exploración de datos (EDA)** — estadísticos, distribuciones, correlaciones\n"
    "2. **Preprocesamiento** — imputación, encoding, escalamiento\n"
    "3. **División train/test 80/20** — con semilla aleatoria para reproducibilidad\n"
    "4. **Entrenamiento con validación cruzada 5-fold** — para estimar el error de generalización\n"
    "5. **Optimización de hiperparámetros** — GridSearchCV sobre parámetros clave\n"
    "6. **Evaluación en conjunto de prueba** — métricas finales sobre datos no vistos\n"
    "7. **Comparación de modelos** — tablas y gráficas comparativas\n"
    "8. **Despliegue interactivo** — dashboard de predicciones en tiempo real"
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Universidad EAFIT · Maestría en Ciencia de los Datos · 2026-1 · "
    "Taller 02 — Aprendizaje Supervisado"
    "</div>",
    unsafe_allow_html=True
)
