"""
Taller 02 – Dashboard Interactivo
Aprendizaje Supervisado: Regresión y Clasificación
Universidad EAFIT – 2026-1

Uso:
    streamlit run app/app.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
import streamlit as st

# ── Configuración de página ──────────────────────────────────
st.set_page_config(
    page_title="Taller 02 – Aprendizaje Supervisado | EAFIT",
    page_icon="🎓",
    layout="wide",
)

# ── Constantes de rutas ──────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")

REG_MODEL_PATH  = os.path.join(MODELS_DIR, "regression_pipeline.pkl")
CLF_MODEL_PATH  = os.path.join(MODELS_DIR, "classification_pipeline.pkl")
REG_META_PATH   = os.path.join(MODELS_DIR, "regression_metadata.json")
CLF_META_PATH   = os.path.join(MODELS_DIR, "classification_metadata.json")
FI_REG_PATH     = os.path.join(DATA_DIR, "feature_importance_regression.png")
FI_CLF_PATH     = os.path.join(DATA_DIR, "feature_importance_classification.png")


# ── Helpers ──────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def add_features_regression(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bmi_smoker"] = df["bmi"] * (df["smoker"] == "yes").astype(int)
    df["age2"] = df["age"] ** 2
    return df


def add_features_classification(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    return df


# ── Header ───────────────────────────────────────────────────
st.title("🎓 Taller 02 – Aprendizaje Supervisado")
st.caption("Maestría en Ciencia de Datos | Universidad EAFIT – 2026-1")
st.divider()

tab1, tab2 = st.tabs(["📈 Regresión – Seguros Médicos", "🔵 Clasificación – Churn"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — REGRESIÓN
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.header("Predicción de Costos Médicos (Medical Cost Insurance)")
    st.markdown(
        "Predice el costo médico anual en USD basado en características "
        "demográficas y de estilo de vida del asegurado."
    )

    reg_model = load_model(REG_MODEL_PATH)
    reg_meta  = load_json(REG_META_PATH)

    # ── Métricas del modelo ──────────────────────────────────
    if reg_meta:
        c1, c2, c3 = st.columns(3)
        c1.metric("Modelo", reg_meta.get("model_name", "—"))
        c2.metric("Test R²", f"{reg_meta.get('test_r2', 0):.4f}")
        c3.metric("Test MAE", f"${reg_meta.get('test_mae', 0):,.0f}")
        st.divider()

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.subheader("Ingrese los datos del asegurado")

        age      = st.slider("Edad", 18, 90, 35)
        sex      = st.selectbox("Sexo", ["male", "female"])
        bmi      = st.slider("BMI", 10.0, 60.0, 25.0, step=0.5)
        children = st.slider("Número de hijos", 0, 5, 0)
        smoker   = st.selectbox("¿Fumador?", ["no", "yes"])
        region   = st.selectbox("Región", ["northeast", "northwest", "southeast", "southwest"])

        predict_reg = st.button("🔮 Predecir costo médico", type="primary")

    with col_result:
        st.subheader("Resultado")

        if predict_reg:
            if reg_model is None:
                st.warning(
                    "⚠️ Modelo no encontrado. Ejecuta primero el notebook "
                    "`notebooks/01_regression.ipynb` para entrenar y guardar el modelo."
                )
            else:
                input_df = pd.DataFrame([{
                    "age": age, "sex": sex, "bmi": bmi,
                    "children": children, "smoker": smoker, "region": region
                }])
                input_fe = add_features_regression(input_df)
                prediction = reg_model.predict(input_fe)[0]

                st.success(f"💰 Costo médico estimado: **${prediction:,.2f} USD**")
                st.info(
                    f"Un {'fumador' if smoker == 'yes' else 'no fumador'} de {age} años "
                    f"con BMI {bmi:.1f} en la región {region} tendría un costo estimado de "
                    f"**${prediction:,.0f}/año**."
                )

                # Factores de riesgo
                st.subheader("Factores de riesgo identificados")
                risk_items = []
                if smoker == "yes":
                    risk_items.append("🚬 **Fumador** – factor de mayor impacto en el costo")
                if bmi >= 30:
                    risk_items.append(f"⚖️ **Obesidad** (BMI={bmi:.1f} ≥ 30)")
                if age >= 50:
                    risk_items.append(f"👴 **Edad avanzada** ({age} años)")
                if not risk_items:
                    risk_items.append("✅ Sin factores de riesgo mayores identificados")
                for item in risk_items:
                    st.markdown(f"- {item}")
        else:
            st.info("⬅️ Complete el formulario y presione **Predecir**.")

    # ── Feature Importance ───────────────────────────────────
    st.divider()
    st.subheader("📊 Importancia de Características (Feature Importance)")
    if os.path.exists(FI_REG_PATH):
        st.image(FI_REG_PATH, use_column_width=True)
    else:
        st.info("Gráfico disponible después de ejecutar el notebook de regresión.")

    # ── Predicción por lote ──────────────────────────────────
    st.divider()
    st.subheader("📂 Predicción por Lote (CSV)")
    st.markdown(
        "Sube un CSV con columnas: `age, sex, bmi, children, smoker, region` "
        "para obtener predicciones masivas."
    )
    uploaded_reg = st.file_uploader("Subir CSV para regresión", type="csv", key="reg_csv")

    if uploaded_reg:
        batch_df = pd.read_csv(uploaded_reg)
        required = {"age", "sex", "bmi", "children", "smoker", "region"}
        if not required.issubset(set(batch_df.columns)):
            st.error(f"El CSV debe tener las columnas: {required}")
        elif reg_model is None:
            st.warning("Modelo no disponible.")
        else:
            batch_fe = add_features_regression(batch_df)
            batch_df["predicted_charges"] = reg_model.predict(batch_fe)
            st.dataframe(batch_df)
            csv_out = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar predicciones", csv_out,
                               "predicciones_seguros.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════
# TAB 2 — CLASIFICACIÓN
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.header("Predicción de Churn de Clientes (Telco)")
    st.markdown(
        "Predice la probabilidad de que un cliente abandone el servicio de telecomunicaciones."
    )

    clf_model = load_model(CLF_MODEL_PATH)
    clf_meta  = load_json(CLF_META_PATH)

    # ── Métricas del modelo ──────────────────────────────────
    if clf_meta:
        c1, c2, c3 = st.columns(3)
        c1.metric("Modelo", clf_meta.get("model_name", "—"))
        c2.metric("Test AUC-ROC", f"{clf_meta.get('test_auc', 0):.4f}")
        c3.metric("Test F1", f"{clf_meta.get('test_f1', 0):.4f}")
        st.divider()

    col_form2, col_result2 = st.columns([1, 1])

    with col_form2:
        st.subheader("Datos del cliente")

        gender            = st.selectbox("Género", ["Male", "Female"])
        senior_citizen    = st.selectbox("¿Ciudadano mayor?", [0, 1], format_func=lambda x: "Sí" if x else "No")
        partner           = st.selectbox("¿Tiene pareja?", ["Yes", "No"])
        dependents        = st.selectbox("¿Tiene dependientes?", ["Yes", "No"])
        tenure            = st.slider("Meses de antigüedad", 0, 72, 12)
        phone_service     = st.selectbox("Servicio telefónico", ["Yes", "No"])
        multiple_lines    = st.selectbox("Múltiples líneas", ["No phone service", "No", "Yes"])
        internet_service  = st.selectbox("Servicio de internet", ["DSL", "Fiber optic", "No"])
        online_security   = st.selectbox("Seguridad online", ["No internet service", "No", "Yes"])
        online_backup     = st.selectbox("Respaldo online", ["No internet service", "No", "Yes"])
        device_protection = st.selectbox("Protección de dispositivo", ["No internet service", "No", "Yes"])
        tech_support      = st.selectbox("Soporte técnico", ["No internet service", "No", "Yes"])
        streaming_tv      = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        streaming_movies  = st.selectbox("Streaming películas", ["No internet service", "No", "Yes"])
        contract          = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Facturación sin papel", ["Yes", "No"])
        payment_method    = st.selectbox("Método de pago", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges   = st.slider("Cargos mensuales (USD)", 0.0, 120.0, 65.0, step=0.5)
        total_charges     = st.number_input("Cargos totales (USD)", 0.0, 10000.0,
                                            value=float(tenure * monthly_charges))

        predict_clf = st.button("🔮 Predecir Churn", type="primary")

    with col_result2:
        st.subheader("Resultado")

        if predict_clf:
            if clf_model is None:
                st.warning(
                    "⚠️ Modelo no encontrado. Ejecuta primero el notebook "
                    "`notebooks/02_classification.ipynb`."
                )
            else:
                input_data = {
                    "gender": gender, "SeniorCitizen": senior_citizen,
                    "Partner": partner, "Dependents": dependents,
                    "tenure": tenure, "PhoneService": phone_service,
                    "MultipleLines": multiple_lines, "InternetService": internet_service,
                    "OnlineSecurity": online_security, "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection, "TechSupport": tech_support,
                    "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
                    "Contract": contract, "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }
                input_df  = pd.DataFrame([input_data])
                input_fe  = add_features_classification(input_df)
                churn_prob = clf_model.predict_proba(input_fe)[0][1]
                churn_pred = clf_model.predict(input_fe)[0]

                # Gauge visual
                if churn_prob >= 0.6:
                    color, label, icon = "red", "ALTO RIESGO de Churn", "🔴"
                elif churn_prob >= 0.35:
                    color, label, icon = "orange", "RIESGO MODERADO de Churn", "🟠"
                else:
                    color, label, icon = "green", "BAJO RIESGO de Churn", "🟢"

                st.metric(f"{icon} Probabilidad de Churn", f"{churn_prob*100:.1f}%")

                # Barra de probabilidad
                fig, ax = plt.subplots(figsize=(5, 1.2))
                ax.barh(0, churn_prob, color=color, height=0.5)
                ax.barh(0, 1 - churn_prob, left=churn_prob, color="#e0e0e0", height=0.5)
                ax.set_xlim(0, 1)
                ax.set_yticks([])
                ax.set_xlabel("Probabilidad")
                ax.axvline(0.5, color='black', linestyle='--', lw=1)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                if churn_pred == 1:
                    st.error(f"**Predicción: {label}**")
                    st.markdown("**Recomendaciones de retención:**")
                    if contract == "Month-to-month":
                        st.markdown("- 📋 Ofrecer descuento por migrar a contrato anual")
                    if monthly_charges > 70:
                        st.markdown("- 💲 Revisar plan tarifario – cargos por encima del promedio")
                    if tenure < 12:
                        st.markdown("- 🤝 Programa de fidelización para clientes nuevos")
                else:
                    st.success(f"**Predicción: {label}**")
        else:
            st.info("⬅️ Complete el formulario y presione **Predecir Churn**.")

    # ── Feature Importance ───────────────────────────────────
    st.divider()
    st.subheader("📊 Importancia de Características (Feature Importance)")
    if os.path.exists(FI_CLF_PATH):
        st.image(FI_CLF_PATH, use_column_width=True)
    else:
        st.info("Gráfico disponible después de ejecutar el notebook de clasificación.")

    # ── Predicción por lote ──────────────────────────────────
    st.divider()
    st.subheader("📂 Predicción por Lote (CSV)")
    st.markdown(
        "Sube un CSV con las mismas columnas del dataset Telco Churn "
        "(sin `customerID` y sin `Churn`) para predicción masiva."
    )
    uploaded_clf = st.file_uploader("Subir CSV para clasificación", type="csv", key="clf_csv")

    if uploaded_clf:
        batch_df = pd.read_csv(uploaded_clf)
        if clf_model is None:
            st.warning("Modelo no disponible.")
        else:
            try:
                if "TotalCharges" in batch_df.columns:
                    batch_df["TotalCharges"] = pd.to_numeric(
                        batch_df["TotalCharges"], errors="coerce"
                    )
                    batch_df["TotalCharges"].fillna(
                        batch_df["TotalCharges"].median(), inplace=True
                    )
                batch_fe = add_features_classification(batch_df)
                batch_df["churn_probability"] = clf_model.predict_proba(batch_fe)[:, 1]
                batch_df["churn_prediction"]  = clf_model.predict(batch_fe)
                st.dataframe(batch_df)
                csv_out = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar predicciones", csv_out,
                                   "predicciones_churn.csv", "text/csv")
            except Exception as e:
                st.error(f"Error al procesar el CSV: {e}")

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption(
    '"In God we trust, all others must bring data." – W. Edwards Deming  |  '
    "Universidad EAFIT · Inteligencia Artificial ECA&I · 2026-1"
)
