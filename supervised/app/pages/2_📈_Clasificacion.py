import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
)
from sklearn.pipeline import Pipeline

from utils.data_loader import load_digits_data
from utils.preprocessing import preprocess_digits

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .page-title { font-size:2.2rem; font-weight:800; color:#1a3a5c; }
    .section-header {
        font-size:1.4rem; font-weight:700; color:#2e9e6a;
        border-bottom: 2px solid #2e9e6a; padding-bottom:0.3rem;
        margin:1.2rem 0 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📈 Clasificación — Reconocimiento de Dígitos (Digits)</div>',
            unsafe_allow_html=True)
st.markdown("**Dataset:** `data/raw/digits.csv` (derivado de `sklearn.datasets.load_digits()`) &nbsp;|&nbsp; "
            "**Target:** dígito manuscrito (0–9)")
st.markdown("---")

# ── 1. DATA LOADING ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">1. Carga de Datos</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_digits_data():
    return load_digits_data()


with st.spinner("Cargando dataset Digits desde archivo local…"):
    df_raw, feature_names = get_digits_data()

class_counts = df_raw['target'].value_counts().sort_index()
balance_str = f"min={class_counts.min()}, max={class_counts.max()}"
missing_values = int(df_raw[feature_names].isna().sum().sum())
balance_ratio = class_counts.max() / class_counts.min()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total muestras", f"{df_raw.shape[0]:,}")
col2.metric("Features (píxeles)", f"{len(feature_names)}")
col3.metric("Clases", "10 (0–9)")
col4.metric("Valores faltantes", str(missing_values))
col5.metric("Balance de clases", balance_str)

with st.expander("Ver muestra del dataset (primeras 10 filas)", expanded=False):
    st.dataframe(df_raw.head(10), use_container_width=True)

with st.expander("Estadísticos descriptivos (primeras 10 features)", expanded=False):
    st.dataframe(df_raw[feature_names[:10]].describe(), use_container_width=True)

# ── 2. EDA ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">2. Análisis Exploratorio de Datos (EDA)</div>',
            unsafe_allow_html=True)

tab_cls, tab_samples, tab_pix, tab_corr, tab_pca = st.tabs([
    "Distribución de clases",
    "Muestras del Dataset",
    "Distribución de Píxeles",
    "Correlación de píxeles",
    "PCA 2D",
])

with tab_cls:
    fig_bar = px.bar(
        x=class_counts.index, y=class_counts.values,
        labels={'x': 'Dígito', 'y': 'Cantidad de muestras'},
        title='Distribución de Clases — Dígitos 0 a 9',
        color=class_counts.values, color_continuous_scale='Teal',
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.success(f"Valores faltantes detectados en las 64 features: **{missing_values}**.")
    st.info("El dataset está muy bien balanceado: cada dígito tiene aproximadamente "
            "180 muestras. No se requieren técnicas de rebalanceo (SMOTE, class_weight).")
    st.caption(f"Chequeo de sesgo por clase: ratio máximo/mínimo = **{balance_ratio:.2f}**, "
               "lo que indica una distribución muy uniforme entre las 10 clases.")

with tab_samples:
    st.markdown("**10 muestras aleatorias del dataset (imagen 8×8)**")
    sample_rows = df_raw.sample(10, random_state=42)
    cols_img = st.columns(10)
    for col_widget, (_, row) in zip(cols_img, sample_rows.iterrows()):
        pixels = row[feature_names].values.reshape(8, 8)
        fig_hm = px.imshow(pixels, color_continuous_scale='gray_r',
                           title=f"Dígito: {int(row['target'])}")
        fig_hm.update_layout(
            margin=dict(l=0, r=0, t=25, b=0),
            height=120,
            coloraxis_showscale=False,
        )
        fig_hm.update_xaxes(showticklabels=False)
        fig_hm.update_yaxes(showticklabels=False)
        col_widget.plotly_chart(fig_hm, use_container_width=True)
    st.info("Cada imagen es una cuadrícula 8×8 de intensidades de píxel (0–16). "
            "Los patrones son claramente distinguibles a pesar de la baja resolución.")

with tab_pix:
    key_pixels = ['pixel_0', 'pixel_27', 'pixel_36', 'pixel_63']
    fig_vio = px.violin(
        df_raw, x='target', y=key_pixels[0],
        color='target', title=f'Intensidad de {key_pixels[0]} por clase',
        labels={'target': 'Dígito'},
        box=True, points=False,
    )
    st.plotly_chart(fig_vio, use_container_width=True)

    selected_px = st.selectbox("Selecciona un píxel para visualizar:", feature_names,
                               index=27)
    fig_vio2 = px.violin(
        df_raw, x='target', y=selected_px,
        color='target',
        title=f'Intensidad de {selected_px} por clase de dígito',
        labels={'target': 'Dígito'},
        box=True, points=False,
    )
    st.plotly_chart(fig_vio2, use_container_width=True)
    st.info("Los píxeles del centro de la imagen (ej. pixel_27, pixel_36) tienen mayor "
            "varianza entre clases, siendo más discriminativos que los píxeles de las esquinas.")

with tab_corr:
    @st.cache_data(show_spinner=False)
    def compute_pixel_correlation(_df, _feature_names, top_n=16):
        variances = _df[_feature_names].var().sort_values(ascending=False)
        top_pixels = variances.head(top_n).index.tolist()
        corr = _df[top_pixels].corr()
        return corr, variances.loc[top_pixels].reset_index()

    corr_df, top_var_df = compute_pixel_correlation(df_raw, feature_names)
    top_var_df.columns = ['Píxel', 'Varianza']

    fig_corr = px.imshow(
        corr_df,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Correlación entre los 16 píxeles con mayor varianza',
    )
    fig_corr.update_layout(height=650)
    st.plotly_chart(fig_corr, use_container_width=True)

    col_corr_1, col_corr_2 = st.columns([2, 1])
    with col_corr_1:
        st.info("La mayor parte de las correlaciones fuertes aparece entre píxeles vecinos "
                "de la zona central, donde se dibujan los trazos. Esto confirma que hay "
                "estructura espacial útil para la clasificación.")
    with col_corr_2:
        st.dataframe(top_var_df, use_container_width=True, hide_index=True)

with tab_pca:
    @st.cache_data(show_spinner=False)
    def compute_pca(_df, _feat_names):
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(_df[_feat_names].values)
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['Dígito'] = _df['target'].astype(str).values
        return pca_df, pca.explained_variance_ratio_

    pca_df, evr = compute_pca(df_raw, feature_names)
    fig_pca = px.scatter(
        pca_df, x='PC1', y='PC2', color='Dígito',
        title=f'PCA 2D — Varianza explicada: PC1={evr[0]:.1%}, PC2={evr[1]:.1%}',
        opacity=0.7,
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    st.info(f"Las dos primeras componentes principales explican el "
            f"{evr[0]+evr[1]:.1%} de la varianza total. Se observan agrupaciones "
            "claras para varios dígitos, aunque con cierto solapamiento entre clases "
            "similares (ej. 3/5/8).")

# ── 3. PREPROCESSING ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">3. Preprocesamiento</div>', unsafe_allow_html=True)

st.markdown("""
| Paso | Detalle |
|------|---------|
| Dataset origen | `data/raw/digits.csv` (derivado de `sklearn.datasets.load_digits()`) — 1,797 muestras, 64 features numéricas |
| Valores faltantes | Ninguno (0 valores nulos en las 64 features) |
| Encoding | No requerido (todas las variables son píxeles enteros 0–16) |
| Escalamiento | `StandardScaler` aplicado **dentro de cada Pipeline** — se ajusta solo sobre `X_train` para evitar *data leakage* |
| División train/test | 80% / 20% con `stratify=y` y `random_state=42` |
| Validación cruzada | `StratifiedKFold(n_splits=5)` preserva la proporción de clases en cada fold |
| Target | 0–9 (10 clases de dígitos) |
""")


@st.cache_data(show_spinner=False)
def get_digits_preprocessed():
    df, feat_names = get_digits_data()
    X, y, fn = preprocess_digits(df, feat_names)
    return X, y, fn


X, y, feat_names_pp = get_digits_preprocessed()
st.markdown(f"**Shape final:** X = {X.shape}, y = {y.shape}")

# ── 4. FEATURE ENGINEERING ──────────────────────────────────────────────────
st.markdown('<div class="section-header">4. Ingeniería de Características</div>', unsafe_allow_html=True)

st.markdown("Se utiliza la importancia de un **Random Forest preliminar** para identificar los "
            "píxeles más informativos antes del entrenamiento final.")


@st.cache_data(show_spinner=False)
def get_preliminary_fi(_X, _y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(_X.values, _y.values)
    fi_df = pd.DataFrame({'Feature': _X.columns, 'Importance': rf.feature_importances_})
    return fi_df.sort_values('Importance', ascending=False)


with st.spinner("Calculando importancia preliminar de features…"):
    fi_prelim = get_preliminary_fi(X, y)

fig_fi_pre = px.bar(
    fi_prelim.head(15), x='Importance', y='Feature', orientation='h',
    color='Importance', color_continuous_scale='Teal',
    title='Top 15 Píxeles más Importantes (RF preliminar)',
)
fig_fi_pre.update_layout(yaxis={'categoryorder': 'total ascending'}, height=430)
st.plotly_chart(fig_fi_pre, use_container_width=True)

st.info("Los píxeles más importantes se concentran en la zona central de la imagen 8×8, "
        "donde la tinta de los dígitos varía más entre clases. Los píxeles de las esquinas "
        "suelen tener importancia cercana a cero.")

# ── 5. MODEL TRAINING ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">5. Entrenamiento de Modelos</div>', unsafe_allow_html=True)

st.markdown("""
Se entrenan **5 modelos** con `StratifiedKFold(5)` y `GridSearchCV`.
División train/test: **80% / 20%** con `stratify=y` y `random_state=42`.
Métrica de optimización en CV: **accuracy**.
""")


@st.cache_data(show_spinner=False)
def train_classifiers(_X, _y):
    X_arr = _X.values.astype(float)
    y_arr = _y.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # ── Logistic Regression
    lr_pipe = Pipeline([
        ('sc', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, random_state=42)),
    ])
    gs_lr = GridSearchCV(lr_pipe, {'clf__C': [0.01, 0.1, 1, 10]},
                         cv=skf, scoring='accuracy', n_jobs=-1)
    gs_lr.fit(X_train, y_train)
    best_lr = gs_lr.best_estimator_
    y_pred_lr = best_lr.predict(X_test)
    cv_acc_lr = cross_val_score(best_lr, X_train, y_train, cv=skf, scoring='accuracy')
    results['Logistic Regression'] = dict(
        model=best_lr, y_pred=y_pred_lr,
        cv_acc_mean=cv_acc_lr.mean(), cv_acc_std=cv_acc_lr.std(),
        accuracy=accuracy_score(y_test, y_pred_lr),
        f1_macro=f1_score(y_test, y_pred_lr, average='macro'),
        f1_weighted=f1_score(y_test, y_pred_lr, average='weighted'),
        cm=confusion_matrix(y_test, y_pred_lr),
        best_params=gs_lr.best_params_,
    )

    # ── KNN
    knn_pipe = Pipeline([
        ('sc', StandardScaler()),
        ('clf', KNeighborsClassifier()),
    ])
    gs_knn = GridSearchCV(knn_pipe,
                          {'clf__n_neighbors': [3, 5, 7],
                           'clf__weights': ['uniform', 'distance']},
                          cv=skf, scoring='accuracy', n_jobs=-1)
    gs_knn.fit(X_train, y_train)
    best_knn = gs_knn.best_estimator_
    y_pred_knn = best_knn.predict(X_test)
    cv_acc_knn = cross_val_score(best_knn, X_train, y_train, cv=skf, scoring='accuracy')
    results['KNN'] = dict(
        model=best_knn, y_pred=y_pred_knn,
        cv_acc_mean=cv_acc_knn.mean(), cv_acc_std=cv_acc_knn.std(),
        accuracy=accuracy_score(y_test, y_pred_knn),
        f1_macro=f1_score(y_test, y_pred_knn, average='macro'),
        f1_weighted=f1_score(y_test, y_pred_knn, average='weighted'),
        cm=confusion_matrix(y_test, y_pred_knn),
        best_params=gs_knn.best_params_,
    )

    # ── Decision Tree
    gs_dt = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        {'max_depth': [5, 10, 15, None],
         'criterion': ['gini', 'entropy']},
        cv=skf, scoring='accuracy', n_jobs=-1,
    )
    gs_dt.fit(X_train, y_train)
    best_dt = gs_dt.best_estimator_
    y_pred_dt = best_dt.predict(X_test)
    cv_acc_dt = cross_val_score(best_dt, X_train, y_train, cv=skf, scoring='accuracy')
    results['Decision Tree'] = dict(
        model=best_dt, y_pred=y_pred_dt,
        cv_acc_mean=cv_acc_dt.mean(), cv_acc_std=cv_acc_dt.std(),
        accuracy=accuracy_score(y_test, y_pred_dt),
        f1_macro=f1_score(y_test, y_pred_dt, average='macro'),
        f1_weighted=f1_score(y_test, y_pred_dt, average='weighted'),
        cm=confusion_matrix(y_test, y_pred_dt),
        best_params=gs_dt.best_params_,
        feature_importances=best_dt.feature_importances_,
    )

    # ── Random Forest
    gs_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {'n_estimators': [100, 200],
         'max_depth': [None, 20]},
        cv=skf, scoring='accuracy', n_jobs=-1,
    )
    gs_rf.fit(X_train, y_train)
    best_rf = gs_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    cv_acc_rf = cross_val_score(best_rf, X_train, y_train, cv=skf, scoring='accuracy')
    results['Random Forest'] = dict(
        model=best_rf, y_pred=y_pred_rf,
        cv_acc_mean=cv_acc_rf.mean(), cv_acc_std=cv_acc_rf.std(),
        accuracy=accuracy_score(y_test, y_pred_rf),
        f1_macro=f1_score(y_test, y_pred_rf, average='macro'),
        f1_weighted=f1_score(y_test, y_pred_rf, average='weighted'),
        cm=confusion_matrix(y_test, y_pred_rf),
        best_params=gs_rf.best_params_,
        feature_importances=best_rf.feature_importances_,
    )

    # ── Gradient Boosting
    gs_gb = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        {'n_estimators': [100],
         'max_depth': [3, 5],
         'learning_rate': [0.1]},
        cv=skf, scoring='accuracy', n_jobs=-1,
    )
    gs_gb.fit(X_train, y_train)
    best_gb = gs_gb.best_estimator_
    y_pred_gb = best_gb.predict(X_test)
    cv_acc_gb = cross_val_score(best_gb, X_train, y_train, cv=skf, scoring='accuracy')
    results['Gradient Boosting'] = dict(
        model=best_gb, y_pred=y_pred_gb,
        cv_acc_mean=cv_acc_gb.mean(), cv_acc_std=cv_acc_gb.std(),
        accuracy=accuracy_score(y_test, y_pred_gb),
        f1_macro=f1_score(y_test, y_pred_gb, average='macro'),
        f1_weighted=f1_score(y_test, y_pred_gb, average='weighted'),
        cm=confusion_matrix(y_test, y_pred_gb),
        best_params=gs_gb.best_params_,
        feature_importances=best_gb.feature_importances_,
    )

    return results, y_test, X_test


with st.spinner("Entrenando 5 clasificadores…"):
    clf_results, y_test_global, X_test_global = train_classifiers(X, y)

st.success("Entrenamiento completado.")

with st.expander("Mejores hiperparámetros por GridSearchCV"):
    for name, res in clf_results.items():
        st.write(f"**{name}:** {res['best_params']}")

# ── 6. RESULTS ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">6. Resultados y Evaluación</div>', unsafe_allow_html=True)

rows_c = []
for name, res in clf_results.items():
    rows_c.append({
        'Modelo': name,
        'CV Accuracy (mean)': f"{res['cv_acc_mean']:.4f}",
        'CV Accuracy (std)': f"±{res['cv_acc_std']:.4f}",
        'Test Accuracy': f"{res['accuracy']:.4f}",
        'Test macro-F1': f"{res['f1_macro']:.4f}",
        'Test weighted-F1': f"{res['f1_weighted']:.4f}",
    })
df_metrics_c = pd.DataFrame(rows_c)
st.dataframe(df_metrics_c, use_container_width=True)

# Grouped bar chart accuracy vs macro-F1
model_names_c = list(clf_results.keys())
fig_metrics = go.Figure()
fig_metrics.add_trace(go.Bar(
    name='Test Accuracy',
    x=model_names_c,
    y=[clf_results[m]['accuracy'] for m in model_names_c],
    marker_color='#3a6ea5',
))
fig_metrics.add_trace(go.Bar(
    name='Test macro-F1',
    x=model_names_c,
    y=[clf_results[m]['f1_macro'] for m in model_names_c],
    marker_color='#2e9e6a',
))
fig_metrics.update_layout(
    barmode='group', title='Accuracy vs macro-F1 por Modelo (Test Set)',
    yaxis_range=[0.7, 1.02], height=400,
)
st.plotly_chart(fig_metrics, use_container_width=True)

best_clf_name = max(clf_results, key=lambda k: clf_results[k]['accuracy'])
best_clf_res = clf_results[best_clf_name]
st.markdown(f"**Mejor modelo (por Accuracy): {best_clf_name}** — "
            f"Accuracy = {best_clf_res['accuracy']:.4f} | macro-F1 = {best_clf_res['f1_macro']:.4f}")

# 10x10 Confusion Matrix
st.markdown("#### Matriz de Confusión (10×10) — Mejor Modelo")
cm = best_clf_res['cm']
digit_labels = [str(i) for i in range(10)]
fig_cm = px.imshow(
    cm, text_auto=True,
    labels=dict(x='Predicción', y='Valor Real'),
    x=digit_labels,
    y=digit_labels,
    color_continuous_scale='Blues',
    title=f'Matriz de Confusión — {best_clf_name}',
)
fig_cm.update_layout(height=500)
st.plotly_chart(fig_cm, use_container_width=True)

# Feature importance of best tree model
fi_model = None
for name in [best_clf_name, 'Random Forest', 'Gradient Boosting', 'Decision Tree']:
    if name in clf_results and 'feature_importances' in clf_results[name]:
        fi_model = name
        break

if fi_model:
    st.markdown(f"#### Importancia de Píxeles (Top 20) — {fi_model}")
    fi_vals = clf_results[fi_model]['feature_importances']
    fi_df = pd.DataFrame({'Feature': feat_names_pp, 'Importance': fi_vals})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(20)
    fig_fi_c = px.bar(
        fi_df, x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='Greens',
        title=f'Top 20 Píxeles más Importantes — {fi_model}',
    )
    fig_fi_c.update_layout(yaxis={'categoryorder': 'total ascending'}, height=550)
    st.plotly_chart(fig_fi_c, use_container_width=True)

# ── 7. CONCLUSIONS ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">7. Conclusiones</div>', unsafe_allow_html=True)

st.success(
    f"**Resumen de resultados — Digits Classification (10 clases):**\n\n"
    f"1. **Mejor modelo:** {best_clf_name} con Accuracy = {best_clf_res['accuracy']:.4f} "
    f"y macro-F1 = {best_clf_res['f1_macro']:.4f} en el conjunto de prueba.\n\n"
    "2. **Dataset balanceado:** Las 10 clases tienen aproximadamente el mismo número de muestras, "
    "por lo que Accuracy y macro-F1 son métricas igualmente válidas.\n\n"
    "3. **Píxeles discriminativos:** Los píxeles del centro de la imagen 8×8 aportan mayor "
    "información para distinguir dígitos, mientras los de las esquinas tienen importancia mínima.\n\n"
    "4. **Modelos lineales (LR)** logran alta accuracy gracias al StandardScaler + regularización. "
    "Los modelos de ensamble capturan interacciones no-lineales entre píxeles.\n\n"
    "5. **Confusiones frecuentes:** Los dígitos más confundidos son aquellos con trazos similares "
    "(ej. 3/5, 4/9, 7/1), lo cual es coherente con la percepción humana.\n\n"
    f"6. **Recomendación:** Para producción se recomendaría {best_clf_name} con los hiperparámetros "
    "encontrados por GridSearchCV, con posible mejora usando SVM-RBF o CNN para mayor accuracy."
)
