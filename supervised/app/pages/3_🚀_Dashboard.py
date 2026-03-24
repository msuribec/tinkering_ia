import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score,
)

from utils.data_loader import load_insurance, load_digits_data
from utils.preprocessing import preprocess_insurance, preprocess_digits

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .page-title  { font-size:2.2rem; font-weight:800; color:#1a3a5c; }
    .section-hdr {
        font-size:1.3rem; font-weight:700; color:#8e44ad;
        border-bottom:2px solid #8e44ad; padding-bottom:0.3rem; margin:1rem 0 0.7rem;
    }
    .pred-box {
        background:linear-gradient(135deg,#1a3a5c,#3a6ea5);
        color:white; border-radius:12px; padding:1.5rem 2rem; text-align:center;
        margin:1rem 0;
    }
    .pred-value { font-size:3rem; font-weight:900; }
    .pred-label { font-size:1rem; opacity:0.85; margin-top:0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🚀 Dashboard Interactivo de Predicciones</div>',
            unsafe_allow_html=True)
st.markdown("Predicción individual, por lote y análisis de modelos en tiempo real")
st.markdown("---")

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Configuración")
task = st.sidebar.radio("Tarea", ["📊 Regresión — Seguros Médicos",
                                   "📈 Clasificación — Reconocimiento de Dígitos"])

REG_MODELS = ["Linear Regression", "Ridge Regression", "Lasso Regression",
              "Random Forest", "Gradient Boosting"]
CLF_MODELS = ["Logistic Regression", "KNN", "Decision Tree",
              "Random Forest", "Gradient Boosting"]

is_regression  = task.startswith("📊")
model_options  = REG_MODELS if is_regression else CLF_MODELS
selected_model = st.sidebar.selectbox(
    "Modelo",
    model_options,
    index=model_options.index("Random Forest"),
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Info del Dataset")
if is_regression:
    st.sidebar.markdown(
        "**Medical Cost Insurance (local)**\n"
        "- 1,338 registros\n- 8 features\n- Target: charges (USD)"
    )
else:
    st.sidebar.markdown(
        "**Digits (local)**\n"
        "- 1,797 registros\n- 64 features (píxeles 8×8)\n- Target: dígito (0-9)"
    )

# ── DATA & MODEL (cached) ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_prep_regression():
    df = load_insurance()
    X, y, feat_names = preprocess_insurance(df)
    return X, y, feat_names, df


@st.cache_data(show_spinner=False)
def load_and_prep_classification():
    df, feature_names = load_digits_data()
    X, y, feat_names = preprocess_digits(df, feature_names)
    return X, y, feat_names, df


@st.cache_data(show_spinner=False)
def train_regression_model(model_name):
    X, y, feat_names, _ = load_and_prep_regression()
    X_arr = X.values
    y_arr = y.values
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42,
    )
    sc = StandardScaler()
    if model_name == "Linear Regression":
        m = LinearRegression()
        m.fit(sc.fit_transform(X_train), y_train)
        preds = m.predict(sc.transform(X_test))
        model_obj = ('scaled', m, sc)
        fi = None
    elif model_name == "Ridge Regression":
        m = Ridge(alpha=10)
        m.fit(sc.fit_transform(X_train), y_train)
        preds = m.predict(sc.transform(X_test))
        model_obj = ('scaled', m, sc)
        fi = None
    elif model_name == "Lasso Regression":
        m = Lasso(alpha=10, max_iter=10000)
        m.fit(sc.fit_transform(X_train), y_train)
        preds = m.predict(sc.transform(X_test))
        model_obj = ('scaled', m, sc)
        fi = None
    elif model_name == "Random Forest":
        m = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        model_obj = ('raw', m, None)
        fi = m.feature_importances_
    else:
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      learning_rate=0.1, random_state=42)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        model_obj = ('raw', m, None)
        fi = m.feature_importances_

    r2   = r2_score(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return model_obj, feat_names, fi, r2, mae, rmse, X_test, y_test, preds


@st.cache_data(show_spinner=False)
def train_classification_model(model_name):
    X, y, feat_names, _ = load_and_prep_classification()
    X_arr = X.values.astype(float)
    y_arr = y.values
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr,
    )
    if model_name == "Logistic Regression":
        pipe = Pipeline([('sc', StandardScaler()),
                         ('clf', LogisticRegression(C=1, max_iter=2000, random_state=42))])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        model_obj = ('pipe', pipe, None)
        fi = None
    elif model_name == "KNN":
        pipe = Pipeline([('sc', StandardScaler()),
                         ('clf', KNeighborsClassifier(n_neighbors=5))])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        model_obj = ('pipe', pipe, None)
        fi = None
    elif model_name == "Decision Tree":
        m = DecisionTreeClassifier(max_depth=10, random_state=42)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        model_obj = ('raw', m, None)
        fi = m.feature_importances_
    elif model_name == "Random Forest":
        m = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        model_obj = ('raw', m, None)
        fi = m.feature_importances_
    else:
        m = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                       learning_rate=0.1, random_state=42)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        model_obj = ('raw', m, None)
        fi = m.feature_importances_

    acc      = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average='macro')
    return model_obj, feat_names, fi, acc, f1_macro, X_test, y_test, preds


def _predict_reg(model_obj, x):
    kind, m, sc = model_obj
    return m.predict(sc.transform(x) if kind == 'scaled' else x)[0]


def _predict_clf(model_obj, x):
    _, m, _ = model_obj
    return int(m.predict(x)[0])


# ── Load / train selected model ──────────────────────────────────────────────
if is_regression:
    with st.spinner(f"Entrenando {selected_model} (regresión)…"):
        model_obj, feat_names, fi_vals, r2, mae, rmse, X_test, y_test, preds_test = \
            train_regression_model(selected_model)
else:
    with st.spinner(f"Entrenando {selected_model} (clasificación)…"):
        model_obj, feat_names, fi_vals, acc, f1_macro, \
            X_test, y_test, preds_test = \
            train_classification_model(selected_model)

# ── Metric banner ────────────────────────────────────────────────────────────
if is_regression:
    c1, c2, c3 = st.columns(3)
    c1.metric("Test R²",   f"{r2:.4f}")
    c2.metric("Test MAE",  f"${mae:,.0f}")
    c3.metric("Test RMSE", f"${rmse:,.0f}")
else:
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("macro-F1", f"{f1_macro:.4f}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
tab_ind, tab_batch, tab_fi, tab_ev = st.tabs([
    "🎯 Predicción Individual",
    "📂 Predicción por Lote",
    "📊 Feature Importance",
    "🧪 Evidencia de Pruebas",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — INDIVIDUAL PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab_ind:
    st.markdown('<div class="section-hdr">Predicción Individual</div>', unsafe_allow_html=True)
    st.caption("Evidencia Parte 3: formulario para predicción individual con el mismo preprocesamiento usado en entrenamiento.")

    if is_regression:
        st.markdown("Ajusta los valores del asegurado:")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            age      = st.slider("Edad", 18, 64, 35)
            bmi      = st.slider("BMI", 15.0, 55.0, 28.0, 0.1)
            children = st.slider("Número de hijos", 0, 5, 1)
        with col_i2:
            sex    = st.selectbox("Sexo", ["male", "female"])
            smoker = st.selectbox("¿Fumador?", ["no", "yes"])
            region = st.selectbox("Región", ["northeast", "northwest", "southeast", "southwest"])
        input_df = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region,
        }])
        x_vec, _, _ = preprocess_insurance(input_df, require_target=False)

        if st.button("Predecir Costo Médico", type="primary"):
            pred_val = _predict_reg(model_obj, x_vec.values)
            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">Costo Médico Estimado</div>
                <div class="pred-value">${pred_val:,.0f} USD</div>
                <div class="pred-label">Modelo: {selected_model}</div>
            </div>
            """, unsafe_allow_html=True)
            if smoker == 'yes':
                st.warning("El hábito tabáquico incrementa significativamente el costo estimado.")
            if bmi > 35:
                st.info("Un BMI alto se asocia con costos médicos mayores.")

    else:
        # ── Digits individual prediction ──────────────────────────────────
        st.markdown("Selecciona una muestra del conjunto de prueba para predecir el dígito.")

        sample_idx = st.slider("Índice de muestra (test set)", 0, len(X_test) - 1, 0)
        sample_pixels = X_test[sample_idx].reshape(8, 8)

        fig_digit = px.imshow(
            sample_pixels,
            color_continuous_scale='gray_r',
            title=f"Muestra #{sample_idx} — imagen 8×8",
        )
        fig_digit.update_layout(
            height=300, width=300,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig_digit.update_xaxes(showticklabels=False)
        fig_digit.update_yaxes(showticklabels=False)

        col_img, col_pred = st.columns([1, 2])
        with col_img:
            st.plotly_chart(fig_digit, use_container_width=True)

        with col_pred:
            if st.button("Predecir Dígito", type="primary"):
                x_sample = X_test[sample_idx:sample_idx+1]
                pred_digit = _predict_clf(model_obj, x_sample)
                true_digit = int(y_test[sample_idx])
                correct = pred_digit == true_digit
                box_color = ("linear-gradient(135deg,#1a7a4a,#2e9e6a)"
                             if correct
                             else "linear-gradient(135deg,#7b1818,#c0392b)")
                icon = "✅" if correct else "❌"
                st.markdown(f"""
                <div class="pred-box" style="background:{box_color};">
                    <div class="pred-label">Dígito Predicho</div>
                    <div class="pred-value">{pred_digit} {icon}</div>
                    <div class="pred-label">
                        Dígito real: {true_digit} | Modelo: {selected_model}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if correct:
                    st.success(f"Predicción correcta: el modelo identificó el dígito **{pred_digit}**.")
                else:
                    st.error(f"Predicción incorrecta: predijo **{pred_digit}**, el valor real es **{true_digit}**.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="section-hdr">Predicción por Lote (CSV)</div>', unsafe_allow_html=True)
    st.caption("Evidencia Parte 3: carga de CSV, predicción en lote y descarga del archivo de resultados.")

    if is_regression:
        expected_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        sample_df = pd.DataFrame({
            'age': [25, 45, 60],
            'sex': ['male', 'female', 'male'],
            'bmi': [22.5, 31.0, 28.7],
            'children': [0, 2, 3],
            'smoker': ['no', 'yes', 'no'],
            'region': ['northeast', 'southwest', 'northwest'],
        })
    else:
        X_digits_sample, _, fn_digits, _ = load_and_prep_classification()
        expected_cols = fn_digits
        sample_df = pd.DataFrame(X_test[:5], columns=fn_digits).round(4)

    st.markdown(f"**Columnas esperadas ({len(expected_cols)}):** "
                f"{', '.join(expected_cols[:8])}{'…' if len(expected_cols) > 8 else ''}")

    sample_csv = sample_df.to_csv(index=False)
    st.download_button("Descargar CSV de ejemplo", data=sample_csv,
                       file_name="sample_input.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.markdown(f"Archivo cargado: **{df_upload.shape[0]} filas, {df_upload.shape[1]} columnas**")
            st.dataframe(df_upload.head(), use_container_width=True)

            if is_regression:
                df_proc_up, _, _ = preprocess_insurance(df_upload, require_target=False)
                x_batch = df_proc_up[feat_names].values
                kind, m, sc = model_obj
                preds_batch = m.predict(sc.transform(x_batch) if kind == 'scaled' else x_batch)
                df_upload['predicted_charges'] = preds_batch.round(2)
                st.dataframe(df_upload, use_container_width=True)

            else:
                for col in feat_names:
                    if col not in df_upload.columns:
                        df_upload[col] = 0.0
                x_batch = df_upload[feat_names].values.astype(float)
                _, m_clf, _ = model_obj
                labels = m_clf.predict(x_batch)
                df_upload['predicted_digit'] = labels
                st.dataframe(df_upload, use_container_width=True)

                digit_counts = pd.Series(labels).value_counts().sort_index()
                fig_b = px.bar(
                    x=digit_counts.index, y=digit_counts.values,
                    labels={'x': 'Dígito predicho', 'y': 'Cantidad'},
                    title='Distribución de dígitos predichos',
                    color=digit_counts.values, color_continuous_scale='Teal',
                )
                st.plotly_chart(fig_b, use_container_width=True)

            result_csv = df_upload.to_csv(index=False)
            st.download_button("Descargar resultados CSV", data=result_csv,
                               file_name="predictions_output.csv", mime="text/csv",
                               type="primary")

        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
    else:
        st.info("Sube un CSV con las columnas esperadas para obtener predicciones en lote. "
                "Descarga el CSV de ejemplo arriba para ver el formato correcto.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════
with tab_fi:
    st.markdown('<div class="section-hdr">Importancia de Características</div>', unsafe_allow_html=True)
    st.caption("Evidencia Parte 3: visualización de feature importance para modelos basados en árboles.")

    if fi_vals is not None:
        fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': fi_vals})
        fi_df = fi_df.sort_values('Importance', ascending=False)

        fig_fi_d = px.bar(
            fi_df.head(10), x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Purples',
            title=f'Top 10 Features más Importantes — {selected_model}',
        )
        fig_fi_d.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig_fi_d, use_container_width=True)

        with st.expander("Ver todas las features"):
            st.dataframe(fi_df.reset_index(drop=True), use_container_width=True)

        top5     = fi_df.head(5)
        rest_sum = fi_df.iloc[5:]['Importance'].sum()
        pie_df   = pd.concat([
            top5[['Feature', 'Importance']],
            pd.DataFrame([{'Feature': 'Otras features', 'Importance': rest_sum}]),
        ])
        fig_pie_fi = px.pie(
            pie_df, values='Importance', names='Feature',
            title='Distribución de Importancia (Top 5 vs Resto)',
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_pie_fi, use_container_width=True)

    else:
        st.info(f"El modelo **{selected_model}** no expone importancia de features directamente. "
                "Selecciona **Random Forest**, **Gradient Boosting** o **Decision Tree** "
                "para ver la importancia de características.")

        # Show coefficients for linear/logistic models
        _, m_inner, _ = model_obj
        actual_m = m_inner.named_steps.get('clf', m_inner) if hasattr(m_inner, 'named_steps') else m_inner
        if hasattr(actual_m, 'coef_'):
            coefs = actual_m.coef_
            # For multiclass, take mean absolute value across classes
            if coefs.ndim > 1:
                coef_vals = np.mean(np.abs(coefs), axis=0)
            else:
                coef_vals = coefs.flatten()
            coef_df = pd.DataFrame({'Feature': feat_names, 'Importancia media |coef|': coef_vals})
            coef_df = coef_df.sort_values('Importancia media |coef|', ascending=False)
            fig_coef = px.bar(
                coef_df.head(15), x='Importancia media |coef|', y='Feature', orientation='h',
                color='Importancia media |coef|', color_continuous_scale='Purples',
                title=f'Importancia media |coeficiente| — {selected_model}',
            )
            fig_coef.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)
            st.plotly_chart(fig_coef, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVIDENCIA DE PRUEBAS
# ════════════════════════════════════════════════════════════════════════════
with tab_ev:
    st.markdown('<div class="section-hdr">Evidencia de Pruebas (5 Casos Documentados)</div>',
                unsafe_allow_html=True)
    st.caption("Evidencia Parte 3: cinco casos documentados con entradas, salida real y predicción del modelo.")
    st.markdown(f"**Modelo:** {selected_model} &nbsp;|&nbsp; **Conjunto:** Test (20%)")

    if is_regression:
        c1e, c2e, c3e = st.columns(3)
        c1e.metric("R² Test",   f"{r2:.4f}")
        c2e.metric("MAE Test",  f"${mae:,.0f}")
        c3e.metric("RMSE Test", f"${rmse:,.0f}")
    else:
        c1e, c2e = st.columns(2)
        c1e.metric("Accuracy", f"{acc:.4f}")
        c2e.metric("macro-F1", f"{f1_macro:.4f}")

    st.markdown("---")
    st.markdown("#### 5 Casos de Ejemplo con Inputs, Real y Predicho")

    if is_regression:
        errors   = np.abs(preds_test - y_test)
        idx_good = list(np.argsort(errors)[:2])
        idx_bad  = list(np.argsort(errors)[-2:])
        idx_med  = [np.argsort(errors)[len(y_test) // 2]]
        five_idx = (idx_good + idx_bad + idx_med)[:5]

        X_re, y_re, fn_re, _ = load_and_prep_regression()
        for ci, idx in enumerate(five_idx, 1):
            real_v = y_test[idx]
            pred_v = preds_test[idx]
            err    = pred_v - real_v
            pct    = 100 * abs(err) / real_v
            st.markdown(
                f"**Caso {ci}** — Real: **${real_v:,.0f}** | "
                f"Predicho: **${pred_v:,.0f}** | Error: **${err:+,.0f}** ({pct:.1f}%)"
            )
            with st.expander(f"Ver inputs del Caso {ci}"):
                st.json({fn_re[i]: round(float(X_test[idx][i]), 3) for i in range(len(fn_re))})

    else:
        # Show 5 test cases with 8x8 digit image and predicted vs real
        five_idx = list(range(min(5, len(X_test))))
        cols_ev = st.columns(5)
        for ci, (col_widget, idx) in enumerate(zip(cols_ev, five_idx), 1):
            real_v = int(y_test[idx])
            pred_v = int(preds_test[idx])
            correct = pred_v == real_v
            pixels = X_test[idx].reshape(8, 8)
            fig_ev = px.imshow(
                pixels,
                color_continuous_scale='gray_r',
                title=f"Real:{real_v} Pred:{pred_v} {'✅' if correct else '❌'}",
            )
            fig_ev.update_layout(
                height=160,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            fig_ev.update_xaxes(showticklabels=False)
            fig_ev.update_yaxes(showticklabels=False)
            col_widget.plotly_chart(fig_ev, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Detalles de los 5 casos")
        for ci, idx in enumerate(five_idx, 1):
            real_v = int(y_test[idx])
            pred_v = int(preds_test[idx])
            correct = pred_v == real_v
            icon = "✅ Correcto" if correct else "❌ Incorrecto"
            st.write(f"**Caso {ci}** — Dígito real: **{real_v}** | Predicho: **{pred_v}** | {icon}")

    st.markdown("---")
    st.markdown("#### Distribución del Error / Predicciones en el Conjunto de Prueba")

    if is_regression:
        errors_all = preds_test - y_test
        fig_err = px.histogram(
            x=errors_all, nbins=40,
            title='Distribución del Error (Predicho − Real)',
            labels={'x': 'Error (USD)', 'count': 'Frecuencia'},
            color_discrete_sequence=['#8e44ad'],
        )
        fig_err.add_vline(x=0, line_dash='dash', line_color='red')
        st.plotly_chart(fig_err, use_container_width=True)

        fig_sc = px.scatter(
            x=y_test, y=preds_test,
            labels={'x': 'Valor Real (USD)', 'y': 'Predicción (USD)'},
            title='Real vs Predicho — Test Set',
            opacity=0.5, color_discrete_sequence=['#8e44ad'],
        )
        mn = min(y_test.min(), preds_test.min())
        mx = max(y_test.max(), preds_test.max())
        fig_sc.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines',
                                    line=dict(color='red', dash='dash'), name='Ideal'))
        st.plotly_chart(fig_sc, use_container_width=True)

    else:
        from sklearn.metrics import confusion_matrix as _cm
        digit_labels = [str(i) for i in range(10)]
        cm_ev = _cm(y_test, preds_test)
        fig_cm_ev = px.imshow(
            cm_ev, text_auto=True,
            labels=dict(x='Predicción', y='Valor Real'),
            x=digit_labels,
            y=digit_labels,
            color_continuous_scale='Purples',
            title='Matriz de Confusión — Test Set',
        )
        fig_cm_ev.update_layout(height=500)
        st.plotly_chart(fig_cm_ev, use_container_width=True)

        # Accuracy per digit class
        acc_per_class = []
        for d in range(10):
            mask = y_test == d
            if mask.sum() > 0:
                acc_d = (preds_test[mask] == d).mean()
                acc_per_class.append({'Dígito': str(d), 'Accuracy': acc_d})
        acc_df = pd.DataFrame(acc_per_class)
        fig_acc_cls = px.bar(
            acc_df, x='Dígito', y='Accuracy',
            title='Accuracy por clase de dígito',
            color='Accuracy', color_continuous_scale='Teal',
        )
        fig_acc_cls.update_layout(yaxis_range=[0, 1.05])
        st.plotly_chart(fig_acc_cls, use_container_width=True)
