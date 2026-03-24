import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from utils.data_loader import load_insurance
from utils.preprocessing import preprocess_insurance

# ── Page config ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .page-title { font-size:2.2rem; font-weight:800; color:#1a3a5c; }
    .section-header {
        font-size:1.4rem; font-weight:700; color:#3a6ea5;
        border-bottom: 2px solid #3a6ea5; padding-bottom:0.3rem;
        margin:1.2rem 0 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📊 Regresión — Costos de Seguro Médico</div>',
            unsafe_allow_html=True)
st.markdown("**Dataset:** `data/raw/insurance.csv` &nbsp;|&nbsp; **Target:** `charges` (USD)")
st.markdown("---")

# ── 1. DATA LOADING ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">1. Carga de Datos</div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_insurance_data():
    return load_insurance()

with st.spinner("Cargando dataset de seguros médicos desde archivo local…"):
    df_raw = get_insurance_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros", f"{df_raw.shape[0]:,}")
col2.metric("Variables", df_raw.shape[1])
col3.metric("Fumadores", f"{(df_raw['smoker']=='yes').sum():,}")
col4.metric("Costo Promedio", f"${df_raw['charges'].mean():,.0f}")

with st.expander("Ver muestra del dataset (primeras 10 filas)", expanded=False):
    st.dataframe(df_raw.head(10), use_container_width=True)

with st.expander("Estadísticos descriptivos", expanded=False):
    st.dataframe(df_raw.describe(), use_container_width=True)

# ── 2. EDA ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">2. Análisis Exploratorio de Datos (EDA)</div>',
            unsafe_allow_html=True)

tab_dist, tab_corr, tab_out, tab_num, tab_cat = st.tabs([
    "Distribución charges", "Correlación", "Outliers",
    "Numéricas vs charges", "Variables categóricas"
])

with tab_dist:
    col_h, col_s = st.columns(2)
    with col_h:
        fig = px.histogram(df_raw, x='charges', nbins=50, color_discrete_sequence=['#3a6ea5'],
                           title='Distribución de Costos Médicos',
                           labels={'charges': 'Costo (USD)', 'count': 'Frecuencia'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_s:
        stats = df_raw['charges'].describe()
        st.markdown("**Estadísticos de `charges`**")
        for k, v in stats.items():
            st.write(f"- **{k}**: ${v:,.2f}")
        st.write(f"- **Asimetría (skewness)**: {df_raw['charges'].skew():.3f}")
        st.write(f"- **Curtosis**: {df_raw['charges'].kurtosis():.3f}")
        st.info("La distribución de `charges` presenta una **asimetría positiva** "
                "significativa, impulsada por fumadores que tienen costos sustancialmente más altos. "
                "Esto es importante para la elección del modelo y la transformación de la variable objetivo.")

with tab_corr:
    df_enc = df_raw.copy()
    df_enc['sex'] = (df_enc['sex'] == 'male').astype(int)
    df_enc['smoker'] = (df_enc['smoker'] == 'yes').astype(int)
    df_enc = pd.get_dummies(df_enc, columns=['region'])
    corr = df_enc.corr()
    fig_corr = px.imshow(
        corr, text_auto='.2f', color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title='Matriz de Correlación (variables numéricas y codificadas)'
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.info("**Hallazgos:** `smoker` tiene la correlación más alta con `charges` "
            "(~0.79), seguido por `age` (~0.30) y `bmi` (~0.20). "
            "La región y el sexo muestran correlaciones débiles con el costo.")

with tab_out:
    num_cols = ['age', 'bmi', 'children', 'charges']
    fig_box = make_subplots(rows=1, cols=4, subplot_titles=num_cols)
    colors = ['#3a6ea5', '#e07b39', '#2e9e6a', '#c0392b']
    for i, col in enumerate(num_cols):
        fig_box.add_trace(
            go.Box(y=df_raw[col], name=col, marker_color=colors[i], showlegend=False),
            row=1, col=i+1
        )
    fig_box.update_layout(title='Boxplots — Detección de Outliers', height=400)
    st.plotly_chart(fig_box, use_container_width=True)
    # IQR summary
    st.markdown("**Resumen de Outliers (método IQR)**")
    rows = []
    for col in num_cols:
        Q1, Q3 = df_raw[col].quantile(0.25), df_raw[col].quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((df_raw[col] < Q1 - 1.5*IQR) | (df_raw[col] > Q3 + 1.5*IQR)).sum()
        rows.append({'Variable': col, 'Q1': round(Q1,2), 'Q3': round(Q3,2),
                     'IQR': round(IQR,2), 'Outliers': n_out,
                     '% Outliers': f"{100*n_out/len(df_raw):.1f}%"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab_num:
    num_feat = ['age', 'bmi', 'children']
    for feat in num_feat:
        fig_sc = px.scatter(
            df_raw, x=feat, y='charges', color='smoker',
            color_discrete_map={'yes': '#c0392b', 'no': '#3a6ea5'},
            title=f'{feat} vs charges (color = smoker)',
            labels={feat: feat.capitalize(), 'charges': 'Costo (USD)'},
            opacity=0.6
        )
        fig_sc.update_traces(marker=dict(size=5))
        st.plotly_chart(fig_sc, use_container_width=True)
    st.info("En los scatter plots se observa claramente la **segmentación por hábito tabáquico**: "
            "los fumadores conforman una nube de puntos con costos entre 2× y 5× mayores. "
            "La relación `bmi`→`charges` es más evidente en fumadores con IMC alto.")

with tab_cat:
    cat_features = ['sex', 'smoker', 'region']
    for feat in cat_features:
        agg = df_raw.groupby(feat)['charges'].agg(['mean', 'median', 'count']).reset_index()
        agg.columns = [feat, 'Media', 'Mediana', 'Conteo']
        fig_bar = px.bar(
            agg, x=feat, y='Media', text='Conteo',
            title=f'Costo Médico Promedio por {feat}',
            color='Media', color_continuous_scale='Blues',
            labels={feat: feat.capitalize(), 'Media': 'Costo Promedio (USD)'}
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

# ── 3. PREPROCESSING ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">3. Preprocesamiento</div>', unsafe_allow_html=True)

st.markdown("""
Los pasos de preprocesamiento aplicados al dataset son:

| Paso | Variable(s) | Transformación |
|------|-------------|----------------|
| Label Encoding | `sex` | male→1, female→0 |
| Binary Encoding | `smoker` | yes→1, no→0 |
| One-Hot Encoding | `region` | 4 categorías fijas → 3 dummies (`northwest`, `southeast`, `southwest`) |
| Sin escalamiento | numéricas | Los árboles no lo requieren; Ridge/Lasso lo aplican internamente via Pipeline |
""")

st.caption("El mismo esquema de columnas se reutiliza en entrenamiento, predicción individual y carga masiva por CSV.")

@st.cache_data(show_spinner=False)
def get_preprocessed():
    df = get_insurance_data()
    X, y, feat_names = preprocess_insurance(df)
    return X, y, feat_names

X, y, feat_names = get_preprocessed()

col_b, col_a = st.columns(2)
with col_b:
    st.markdown("**Antes del preprocesamiento**")
    st.dataframe(df_raw.head(5), use_container_width=True)
with col_a:
    st.markdown("**Después del preprocesamiento**")
    df_show = pd.concat([X, y], axis=1)
    st.dataframe(df_show.head(5), use_container_width=True)

st.markdown(f"**Shape final:** X = {X.shape}, y = {y.shape}")
st.markdown(f"**Columnas:** {', '.join(feat_names)}")

# ── 4. FEATURE ENGINEERING / IMPORTANCE ────────────────────────────────────
st.markdown('<div class="section-header">4. Importancia de Características (Random Forest inicial)</div>',
            unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_initial_importance(_X, _y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(_X, _y)
    imp = pd.DataFrame({'Feature': feat_names, 'Importance': rf.feature_importances_})
    return imp.sort_values('Importance', ascending=False)

with st.spinner("Calculando importancia de features…"):
    imp_df = get_initial_importance(X, y)

fig_imp = px.bar(
    imp_df, x='Importance', y='Feature', orientation='h',
    title='Importancia de Características (Random Forest — Gini Impurity)',
    color='Importance', color_continuous_scale='Blues'
)
fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=350)
st.plotly_chart(fig_imp, use_container_width=True)

st.info("**smoker** domina ampliamente la importancia (>50%), confirmando lo visto en el EDA. "
        "**age** y **bmi** son las siguientes features más relevantes. "
        "Las variables de región aportan poco al poder predictivo del modelo.")

# ── 5. MODEL TRAINING ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">5. Entrenamiento de Modelos</div>', unsafe_allow_html=True)

st.markdown("""
Se entrenan **5 modelos** con validación cruzada 5-fold y búsqueda de hiperparámetros.
División train/test: **80% / 20%** con `random_state=42`.
""")

@st.cache_data(show_spinner=False)
def train_all_models(_X, _y):
    X_arr = _X.values
    y_arr = _y.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # ── Linear Regression
    lr = LinearRegression()
    cv_r2 = cross_val_score(lr, X_train_sc, y_train, cv=5, scoring='r2')
    lr.fit(X_train_sc, y_train)
    y_pred_train = lr.predict(X_train_sc)
    y_pred_test  = lr.predict(X_test_sc)
    results['Linear Regression'] = {
        'model': lr, 'scaler': scaler,
        'cv_r2_mean': cv_r2.mean(), 'cv_r2_std': cv_r2.std(),
        'train_r2':  r2_score(y_train, y_pred_train),
        'test_r2':   r2_score(y_test,  y_pred_test),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae':  mean_absolute_error(y_test,  y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse':  np.sqrt(mean_squared_error(y_test,  y_pred_test)),
        'y_test': y_test, 'y_pred_test': y_pred_test,
        'best_params': {},
    }

    # ── Ridge
    ridge_pipe = Pipeline([('scaler', StandardScaler()), ('model', Ridge())])
    gs_ridge = GridSearchCV(ridge_pipe,
                            {'model__alpha': [0.01, 0.1, 1, 10, 100, 500]},
                            cv=5, scoring='r2', n_jobs=-1)
    gs_ridge.fit(X_train, y_train)
    best_ridge = gs_ridge.best_estimator_
    y_pred_train_r = best_ridge.predict(X_train)
    y_pred_test_r  = best_ridge.predict(X_test)
    cv_ridge = cross_val_score(best_ridge, X_train, y_train, cv=5, scoring='r2')
    results['Ridge Regression'] = {
        'model': best_ridge, 'scaler': None,
        'cv_r2_mean': cv_ridge.mean(), 'cv_r2_std': cv_ridge.std(),
        'train_r2':  r2_score(y_train, y_pred_train_r),
        'test_r2':   r2_score(y_test,  y_pred_test_r),
        'train_mae': mean_absolute_error(y_train, y_pred_train_r),
        'test_mae':  mean_absolute_error(y_test,  y_pred_test_r),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train_r)),
        'test_rmse':  np.sqrt(mean_squared_error(y_test,  y_pred_test_r)),
        'y_test': y_test, 'y_pred_test': y_pred_test_r,
        'best_params': gs_ridge.best_params_,
    }

    # ── Lasso
    lasso_pipe = Pipeline([('scaler', StandardScaler()), ('model', Lasso(max_iter=10000))])
    gs_lasso = GridSearchCV(lasso_pipe,
                            {'model__alpha': [0.1, 1, 10, 50, 100, 500]},
                            cv=5, scoring='r2', n_jobs=-1)
    gs_lasso.fit(X_train, y_train)
    best_lasso = gs_lasso.best_estimator_
    y_pred_train_l = best_lasso.predict(X_train)
    y_pred_test_l  = best_lasso.predict(X_test)
    cv_lasso = cross_val_score(best_lasso, X_train, y_train, cv=5, scoring='r2')
    results['Lasso Regression'] = {
        'model': best_lasso, 'scaler': None,
        'cv_r2_mean': cv_lasso.mean(), 'cv_r2_std': cv_lasso.std(),
        'train_r2':  r2_score(y_train, y_pred_train_l),
        'test_r2':   r2_score(y_test,  y_pred_test_l),
        'train_mae': mean_absolute_error(y_train, y_pred_train_l),
        'test_mae':  mean_absolute_error(y_test,  y_pred_test_l),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train_l)),
        'test_rmse':  np.sqrt(mean_squared_error(y_test,  y_pred_test_l)),
        'y_test': y_test, 'y_pred_test': y_pred_test_l,
        'best_params': gs_lasso.best_params_,
    }

    # ── Random Forest
    gs_rf = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [None, 10, 20],
         'min_samples_split': [2, 5]},
        cv=5, scoring='r2', n_jobs=-1
    )
    gs_rf.fit(X_train, y_train)
    best_rf = gs_rf.best_estimator_
    y_pred_train_rf = best_rf.predict(X_train)
    y_pred_test_rf  = best_rf.predict(X_test)
    cv_rf = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='r2')
    results['Random Forest'] = {
        'model': best_rf, 'scaler': None,
        'cv_r2_mean': cv_rf.mean(), 'cv_r2_std': cv_rf.std(),
        'train_r2':  r2_score(y_train, y_pred_train_rf),
        'test_r2':   r2_score(y_test,  y_pred_test_rf),
        'train_mae': mean_absolute_error(y_train, y_pred_train_rf),
        'test_mae':  mean_absolute_error(y_test,  y_pred_test_rf),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train_rf)),
        'test_rmse':  np.sqrt(mean_squared_error(y_test,  y_pred_test_rf)),
        'y_test': y_test, 'y_pred_test': y_pred_test_rf,
        'best_params': gs_rf.best_params_,
        'feature_importances': best_rf.feature_importances_,
    }

    # ── Gradient Boosting
    gs_gb = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [3, 5],
         'learning_rate': [0.05, 0.1]},
        cv=5, scoring='r2', n_jobs=-1
    )
    gs_gb.fit(X_train, y_train)
    best_gb = gs_gb.best_estimator_
    y_pred_train_gb = best_gb.predict(X_train)
    y_pred_test_gb  = best_gb.predict(X_test)
    cv_gb = cross_val_score(best_gb, X_train, y_train, cv=5, scoring='r2')
    results['Gradient Boosting'] = {
        'model': best_gb, 'scaler': None,
        'cv_r2_mean': cv_gb.mean(), 'cv_r2_std': cv_gb.std(),
        'train_r2':  r2_score(y_train, y_pred_train_gb),
        'test_r2':   r2_score(y_test,  y_pred_test_gb),
        'train_mae': mean_absolute_error(y_train, y_pred_train_gb),
        'test_mae':  mean_absolute_error(y_test,  y_pred_test_gb),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train_gb)),
        'test_rmse':  np.sqrt(mean_squared_error(y_test,  y_pred_test_gb)),
        'y_test': y_test, 'y_pred_test': y_pred_test_gb,
        'best_params': gs_gb.best_params_,
        'feature_importances': best_gb.feature_importances_,
    }

    split_info = {'X_train': X_train, 'X_test': X_test,
                  'y_train': y_train, 'y_test': y_test}
    return results, split_info

with st.spinner("Entrenando modelos con GridSearchCV + 5-fold CV… (puede tardar ~60 s)"):
    results, split_info = train_all_models(X, y)

st.success("Entrenamiento completado.")

# Show best params
with st.expander("Mejores hiperparámetros encontrados por GridSearchCV"):
    for name, res in results.items():
        if res['best_params']:
            st.write(f"**{name}:** {res['best_params']}")

# ── 6. MODEL COMPARISON ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">6. Comparación de Modelos</div>', unsafe_allow_html=True)

# Metrics table
rows = []
for name, res in results.items():
    rows.append({
        'Modelo': name,
        'CV R² (mean)': f"{res['cv_r2_mean']:.4f}",
        'CV R² (std)': f"±{res['cv_r2_std']:.4f}",
        'Train R²': f"{res['train_r2']:.4f}",
        'Test R²': f"{res['test_r2']:.4f}",
        'Train MAE': f"${res['train_mae']:,.0f}",
        'Test MAE': f"${res['test_mae']:,.0f}",
        'Train RMSE': f"${res['train_rmse']:,.0f}",
        'Test RMSE': f"${res['test_rmse']:,.0f}",
    })
df_metrics = pd.DataFrame(rows)
st.dataframe(df_metrics, use_container_width=True)

# Bar chart — Test R²
fig_r2 = px.bar(
    x=list(results.keys()),
    y=[res['test_r2'] for res in results.values()],
    color=[res['test_r2'] for res in results.values()],
    color_continuous_scale='Blues',
    title='R² en Conjunto de Prueba por Modelo',
    labels={'x': 'Modelo', 'y': 'R² Test'}
)
fig_r2.update_layout(showlegend=False, yaxis_range=[0, 1])
for i, (name, res) in enumerate(results.items()):
    fig_r2.add_annotation(x=name, y=res['test_r2']+0.01,
                          text=f"{res['test_r2']:.4f}", showarrow=False,
                          font=dict(size=11))
st.plotly_chart(fig_r2, use_container_width=True)

# Bar chart — Test RMSE
fig_rmse = px.bar(
    x=list(results.keys()),
    y=[res['test_rmse'] for res in results.values()],
    color=[res['test_rmse'] for res in results.values()],
    color_continuous_scale='Reds',
    title='RMSE en Conjunto de Prueba por Modelo (menor es mejor)',
    labels={'x': 'Modelo', 'y': 'RMSE Test (USD)'}
)
fig_rmse.update_layout(showlegend=False)
st.plotly_chart(fig_rmse, use_container_width=True)

# Scatter actual vs predicted — best model
best_name = max(results, key=lambda k: results[k]['test_r2'])
best_res   = results[best_name]
st.markdown(f"**Mejor modelo: {best_name}** (Test R² = {best_res['test_r2']:.4f})")

fig_scatter = px.scatter(
    x=best_res['y_test'], y=best_res['y_pred_test'],
    labels={'x': 'Valor Real (USD)', 'y': 'Predicción (USD)'},
    title=f'Real vs Predicho — {best_name}',
    opacity=0.6, color_discrete_sequence=['#3a6ea5']
)
min_val = min(best_res['y_test'].min(), best_res['y_pred_test'].min())
max_val = max(best_res['y_test'].max(), best_res['y_pred_test'].max())
fig_scatter.add_trace(
    go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
               mode='lines', line=dict(color='red', dash='dash'),
               name='Predicción Perfecta')
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Feature importance of best model
if 'feature_importances' in best_res:
    imp_best = pd.DataFrame({
        'Feature': feat_names,
        'Importance': best_res['feature_importances']
    }).sort_values('Importance', ascending=False)
    fig_fi = px.bar(
        imp_best, x='Importance', y='Feature', orientation='h',
        title=f'Importancia de Características — {best_name}',
        color='Importance', color_continuous_scale='Blues'
    )
    fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'}, height=350)
    st.plotly_chart(fig_fi, use_container_width=True)

# Overfitting analysis
st.markdown("#### Análisis de Sobreajuste (Overfitting)")
fig_ov = go.Figure()
model_names = list(results.keys())
train_r2s = [results[m]['train_r2'] for m in model_names]
test_r2s  = [results[m]['test_r2']  for m in model_names]
fig_ov.add_trace(go.Bar(name='Train R²', x=model_names, y=train_r2s,
                        marker_color='#3a6ea5'))
fig_ov.add_trace(go.Bar(name='Test R²', x=model_names, y=test_r2s,
                        marker_color='#e07b39'))
fig_ov.update_layout(barmode='group', title='Train vs Test R² — Detección de Sobreajuste',
                     yaxis_range=[0, 1.05])
st.plotly_chart(fig_ov, use_container_width=True)

# ── 7. CONCLUSIONS ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">7. Conclusiones</div>', unsafe_allow_html=True)

st.success(
    f"**Resumen de resultados:**\n\n"
    f"1. **Mejor modelo:** {best_name} con R² = {best_res['test_r2']:.4f} y "
    f"RMSE = ${best_res['test_rmse']:,.0f} USD en el conjunto de prueba.\n\n"
    "2. **Variable más importante:** `smoker` domina la predicción, representando más del 50% "
    "de la importancia total. Los fumadores tienen costos médicos significativamente más altos.\n\n"
    "3. **Lineal vs No-lineal:** Los modelos de ensamble (Random Forest, Gradient Boosting) superan "
    "a los modelos lineales porque capturan las interacciones no-lineales (ej: bmi×smoker).\n\n"
    "4. **Regularización:** Ridge y Lasso ofrecen mejoras menores sobre Regresión Lineal, "
    "dado que el dataset no tiene multicolinealidad severa.\n\n"
    "5. **Overfitting:** Random Forest muestra la mayor diferencia Train-Test R², "
    "mientras Gradient Boosting mantiene mayor generalización gracias al learning_rate.\n\n"
    f"6. **Recomendación:** Para producción se recomendaría {best_name} con los hiperparámetros "
    "encontrados por GridSearchCV, monitoreando periódicamente la distribución de `charges`."
)
