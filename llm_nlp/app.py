"""
Parcial – NLP y LLMs Avanzado | EAFIT 2026-1
Partes 02, 03 y 04: Laboratorio de Parámetros, Métricas de Similitud, Agente
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import json
import re
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP & LLMs Lab – EAFIT",
    page_icon="🧠",
    layout="wide",
)

# ── API client factory ────────────────────────────────────────────────────────

def get_client(provider: str, api_key: str):
    """Retorna el cliente de API según el proveedor seleccionado."""
    if provider == "Groq":
        from groq import Groq
        return Groq(api_key=api_key)
    elif provider == "OpenAI":
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Proveedor no soportado: {provider}")


def generate_response(client, provider: str, prompt: str, model: str,
                      temperature: float, top_p: float, max_tokens: int,
                      frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0,
                      messages: list = None):
    """
    Genera texto usando Groq o OpenAI con parámetros personalizados.
    Retorna (content, usage).
    """
    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    # frequency_penalty / presence_penalty solo en OpenAI-compatible
    if provider in ("OpenAI", "Groq"):
        kwargs["frequency_penalty"] = frequency_penalty
        kwargs["presence_penalty"] = presence_penalty

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content, response.usage


# ── Groq model catalog ────────────────────────────────────────────────────────
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

# ── Sidebar: API config ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 Configuración API")
    provider = st.selectbox("Proveedor", ["Groq", "OpenAI"])
    api_key = st.text_input("API Key", type="password",
                             value=st.secrets.get("GROQ_API_KEY", "") if provider == "Groq"
                             else st.secrets.get("OPENAI_API_KEY", ""))
    model_list = GROQ_MODELS if provider == "Groq" else OPENAI_MODELS
    model = st.selectbox("Modelo", model_list)
    st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab2, tab3, tab4 = st.tabs([
    "■ Laboratorio de Parámetros",
    "■ Métricas de Similitud",
    "■ Agente Especializado",
])

# ═══════════════════════════════════════════════════════════════════════════════
# PARTE 02 – Laboratorio de Parámetros
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("Laboratorio de Sintonización de Parámetros")
    st.caption("Experimenta con los hiperparámetros de generación y observa su efecto.")

    # ── Panel de control ──────────────────────────────────────────────────────
    with st.expander("⚙️ Panel de Control Interactivo", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            temp = st.slider("Temperatura", 0.0, 2.0, 0.7, 0.05,
                             help="Creatividad vs. determinismo")
            top_p = st.slider("Top-p (nucleus)", 0.0, 1.0, 0.9, 0.05,
                              help="Masa de probabilidad acumulada")
        with c2:
            top_k = st.number_input("Top-k", 1, 100, 40,
                                    help="Vocabulario efectivo por paso")
            max_tok = st.slider("Max tokens", 50, 2048, 512, 50,
                                help="Longitud máxima de la respuesta")
        with c3:
            freq_pen = st.slider("Frequency penalty", 0.0, 2.0, 0.0, 0.1,
                                 help="Penalización por repetición de tokens frecuentes")
            pres_pen = st.slider("Presence penalty", 0.0, 2.0, 0.0, 0.1,
                                 help="Penalización por aparición previa de tokens")

    prompt_single = st.text_area(
        "Prompt personalizado",
        value="Explica el concepto de atención en transformers.",
        height=80,
    )

    if st.button("Generar respuesta individual", key="btn_single"):
        if not api_key:
            st.error("Ingresa tu API Key en la barra lateral.")
        else:
            with st.spinner("Generando…"):
                try:
                    client = get_client(provider, api_key)
                    text, usage = generate_response(
                        client, provider, prompt_single, model,
                        temp, top_p, max_tok, freq_pen, pres_pen,
                    )
                    st.markdown("**Respuesta:**")
                    st.write(text)
                    st.caption(f"Tokens — entrada: {usage.prompt_tokens} | "
                               f"salida: {usage.completion_tokens}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Experimento comparativo ───────────────────────────────────────────────
    st.divider()
    st.subheader("Experimento Comparativo (4 configuraciones)")

    FIXED_PROMPT = "Explica el concepto de atención en transformers."
    CONFIGS = [
        {"temp": 0.1, "top_p": 0.9, "label": "T=0.1 / p=0.9"},
        {"temp": 1.5, "top_p": 0.9, "label": "T=1.5 / p=0.9"},
        {"temp": 0.1, "top_p": 0.3, "label": "T=0.1 / p=0.3"},
        {"temp": 1.5, "top_p": 0.3, "label": "T=1.5 / p=0.3"},
    ]

    st.info(f"**Prompt fijo:** {FIXED_PROMPT}")

    if st.button("Ejecutar experimento comparativo", key="btn_compare"):
        if not api_key:
            st.error("Ingresa tu API Key en la barra lateral.")
        else:
            client = get_client(provider, api_key)
            results = []
            progress = st.progress(0, text="Generando respuestas…")
            for i, cfg in enumerate(CONFIGS):
                try:
                    t0 = time.time()
                    text, usage = generate_response(
                        client, provider, FIXED_PROMPT, model,
                        cfg["temp"], cfg["top_p"], 400,
                    )
                    elapsed = time.time() - t0
                    tokens = text.split()
                    unique_tokens = set(t.lower() for t in tokens)
                    ttr = len(unique_tokens) / len(tokens) if tokens else 0
                    results.append({
                        "label": cfg["label"],
                        "text": text,
                        "n_tokens": usage.completion_tokens,
                        "ttr": round(ttr, 3),
                        "latency": round(elapsed, 2),
                    })
                except Exception as e:
                    results.append({"label": cfg["label"], "text": f"Error: {e}",
                                    "n_tokens": 0, "ttr": 0, "latency": 0})
                progress.progress((i + 1) / len(CONFIGS),
                                  text=f"Configuración {i+1}/4…")
            progress.empty()

            st.session_state["compare_results"] = results

    if "compare_results" in st.session_state:
        results = st.session_state["compare_results"]

        # Mostrar respuestas en columnas paralelas
        cols = st.columns(4)
        for col, r in zip(cols, results):
            with col:
                st.markdown(f"**{r['label']}**")
                st.caption(f"Tokens: {r['n_tokens']} | TTR: {r['ttr']}")
                st.text_area("", r["text"], height=250, key=f"res_{r['label']}")

        # Gráficas Plotly
        df = pd.DataFrame(results)
        fig1 = px.bar(df, x="label", y="n_tokens", color="label",
                      title="Longitud en tokens por configuración",
                      labels={"n_tokens": "Tokens", "label": "Config"},
                      color_discrete_sequence=px.colors.qualitative.Plotly)
        fig1.update_layout(showlegend=False)

        fig2 = px.bar(df, x="label", y="ttr", color="label",
                      title="Diversidad léxica (Type-Token Ratio)",
                      labels={"ttr": "TTR", "label": "Config"},
                      color_discrete_sequence=px.colors.qualitative.Safe)
        fig2.update_layout(showlegend=False)

        col_g1, col_g2 = st.columns(2)
        col_g1.plotly_chart(fig1, use_container_width=True)
        col_g2.plotly_chart(fig2, use_container_width=True)

        st.subheader("Observaciones del estudiante")
        st.text_area(
            "Documenta aquí el efecto observado de temperatura y top-p sobre las respuestas:",
            height=150,
            key="observations_lab",
            placeholder="Ej: Con T=1.5 las respuestas son más creativas pero menos precisas…",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PARTE 03 – Métricas de Similitud y Evaluación Automática
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("Métricas de Similitud y Evaluación Automática")
    st.caption("Compara cuantitativamente un texto de referencia con la salida del LLM.")

    col_ref, col_gen = st.columns(2)
    with col_ref:
        reference_text = st.text_area(
            "Texto de referencia (ground truth)",
            height=150,
            placeholder="Escribe o pega aquí la respuesta esperada…",
        )
    with col_gen:
        eval_prompt = st.text_area(
            "Prompt enviado al LLM",
            value="Explica el mecanismo de atención (attention) en los modelos Transformer.",
            height=150,
        )

    if st.button("Evaluar", key="btn_eval"):
        if not api_key:
            st.error("Ingresa tu API Key en la barra lateral.")
        elif not reference_text.strip():
            st.error("Ingresa un texto de referencia.")
        else:
            # 1. Generar respuesta candidata
            with st.spinner("Generando respuesta candidata…"):
                try:
                    client = get_client(provider, api_key)
                    generated_text, usage = generate_response(
                        client, provider, eval_prompt, model,
                        0.3, 0.9, 512,
                    )
                    st.session_state["eval_generated"] = generated_text
                    st.session_state["eval_reference"] = reference_text
                    st.session_state["eval_prompt_used"] = eval_prompt
                except Exception as e:
                    st.error(f"Error generando respuesta: {e}")
                    st.stop()

            st.markdown("**Respuesta generada:**")
            st.write(st.session_state["eval_generated"])

            # 2. Calcular métricas
            scores = {}

            # ── Cosine similarity ──────────────────────────────────────────
            with st.spinner("Calculando Similitud Coseno…"):
                try:
                    from sentence_transformers import SentenceTransformer
                    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
                    @st.cache_resource
                    def load_sbert():
                        return SentenceTransformer("all-MiniLM-L6-v2")
                    sbert = load_sbert()
                    embs = sbert.encode([reference_text,
                                         st.session_state["eval_generated"]])
                    cosine = float(cos_sim([embs[0]], [embs[1]])[0][0])
                    scores["Cosine"] = round(cosine, 4)
                except Exception as e:
                    scores["Cosine"] = None
                    st.warning(f"Cosine error: {e}")

            # ── BLEU ──────────────────────────────────────────────────────
            with st.spinner("Calculando BLEU…"):
                try:
                    import nltk
                    nltk.download("punkt", quiet=True)
                    nltk.download("punkt_tab", quiet=True)
                    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                    ref_tokens = reference_text.lower().split()
                    hyp_tokens = st.session_state["eval_generated"].lower().split()
                    sf = SmoothingFunction().method1
                    bleu = sentence_bleu([ref_tokens], hyp_tokens,
                                         smoothing_function=sf)
                    scores["BLEU"] = round(bleu, 4)
                except Exception as e:
                    scores["BLEU"] = None
                    st.warning(f"BLEU error: {e}")

            # ── ROUGE-L ───────────────────────────────────────────────────
            with st.spinner("Calculando ROUGE-L…"):
                try:
                    from rouge_score import rouge_scorer
                    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                    rouge_result = scorer.score(reference_text,
                                                 st.session_state["eval_generated"])
                    scores["ROUGE-L"] = round(rouge_result["rougeL"].fmeasure, 4)
                except Exception as e:
                    scores["ROUGE-L"] = None
                    st.warning(f"ROUGE error: {e}")

            # ── BERTScore ─────────────────────────────────────────────────
            with st.spinner("Calculando BERTScore (puede tardar)…"):
                try:
                    from bert_score import score as bert_score_fn
                    P, R, F1 = bert_score_fn(
                        [st.session_state["eval_generated"]],
                        [reference_text],
                        lang="es",
                        verbose=False,
                    )
                    scores["BERTScore"] = round(float(F1[0]), 4)
                except Exception as e:
                    scores["BERTScore"] = None
                    st.warning(f"BERTScore error: {e}")

            # ── LLM-as-Judge ──────────────────────────────────────────────
            with st.spinner("Ejecutando LLM-as-Judge…"):
                try:
                    system_judge = (
                        "Eres un evaluador experto en NLP. Evalúa la respuesta generada "
                        "comparándola con la referencia. Responde ÚNICAMENTE en JSON con "
                        "este esquema: "
                        '{"score": <1-10>, "veracidad": <1-10>, "coherencia": <1-10>, '
                        '"relevancia": <1-10>, "fortalezas": "<texto>", "debilidades": "<texto>"}'
                    )
                    user_judge = (
                        f"REFERENCIA: {reference_text}\n\n"
                        f"RESPUESTA GENERADA: {st.session_state['eval_generated']}\n\n"
                        f"PROMPT ORIGINAL: {eval_prompt}"
                    )
                    judge_msgs = [
                        {"role": "system", "content": system_judge},
                        {"role": "user", "content": user_judge},
                    ]
                    judge_text, _ = generate_response(
                        client, provider, "", model,
                        0.0, 0.9, 512,
                        messages=judge_msgs,
                    )
                    # Extract JSON robustly
                    json_match = re.search(r'\{.*\}', judge_text, re.DOTALL)
                    if json_match:
                        judge_json = json.loads(json_match.group())
                    else:
                        judge_json = json.loads(judge_text)
                    st.session_state["judge_json"] = judge_json
                    scores["LLM-Judge"] = judge_json.get("score", 0) / 10
                except Exception as e:
                    scores["LLM-Judge"] = None
                    st.warning(f"LLM-Judge error: {e}")

            st.session_state["eval_scores"] = scores

    # ── Display results ───────────────────────────────────────────────────────
    if "eval_scores" in st.session_state:
        scores = st.session_state["eval_scores"]
        st.divider()
        st.subheader("Resultados de las métricas")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Cosine Similarity", scores.get("Cosine", "N/A"))
        m2.metric("BLEU", scores.get("BLEU", "N/A"))
        m3.metric("ROUGE-L", scores.get("ROUGE-L", "N/A"))
        m4.metric("BERTScore F1", scores.get("BERTScore", "N/A"))
        m5.metric("LLM-Judge (/10)",
                  round(scores.get("LLM-Judge", 0) * 10, 1)
                  if scores.get("LLM-Judge") is not None else "N/A")

        # LLM-Judge details
        if "judge_json" in st.session_state:
            j = st.session_state["judge_json"]
            with st.expander("Detalle LLM-as-Judge"):
                col_j1, col_j2, col_j3 = st.columns(3)
                col_j1.metric("Veracidad", j.get("veracidad"))
                col_j2.metric("Coherencia", j.get("coherencia"))
                col_j3.metric("Relevancia", j.get("relevancia"))
                st.markdown(f"**Fortalezas:** {j.get('fortalezas', '')}")
                st.markdown(f"**Debilidades:** {j.get('debilidades', '')}")

        # Radar chart
        valid = {k: v for k, v in scores.items() if v is not None}
        if valid:
            categories = list(valid.keys())
            values = list(valid.values())
            # Normalize all to [0,1]
            values_norm = [min(max(v, 0), 1) for v in values]

            fig_radar = go.Figure(go.Scatterpolar(
                r=values_norm + [values_norm[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Métricas",
                line_color="royalblue",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Radar de Métricas (normalizadas 0–1)",
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTE 04 – Agente Especializado con Métricas de Producción
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("Agente Especializado – Tutor de Machine Learning")

    # ── Controls inside the tab ───────────────────────────────────────────────
    with st.expander("⚙️ Parámetros del agente", expanded=False):
        col_ap1, col_ap2 = st.columns(2)
        agent_temp = col_ap1.slider("Temperatura", 0.0, 2.0, 0.5, 0.05,
                                     key="agent_temp")
        agent_max_tok = col_ap2.slider("Max tokens", 50, 2048, 600, 50,
                                        key="agent_max_tok")

    # Pricing (USD per 1K tokens) – Groq llama-3.3-70b approx public pricing
    PRICE_IN = 0.00059 / 1000   # per token
    PRICE_OUT = 0.00079 / 1000  # per token

    SYSTEM_PROMPT = (
        "Eres un Tutor experto en Machine Learning y Ciencia de Datos. "
        "Tu nombre es 'MLBot'. Explicas conceptos complejos de forma clara, "
        "con ejemplos prácticos y analogías. Solo respondes preguntas relacionadas "
        "con ML, estadística, programación en Python/R, y temas afines. "
        "Si te preguntan algo fuera de tu dominio, rediriges amablemente la conversación. "
        "Usas lenguaje técnico apropiado pero accesible."
    )

    # ── Session state init ────────────────────────────────────────────────────
    if "agent_history" not in st.session_state:
        st.session_state["agent_history"] = []  # list of {role, content}
    if "agent_metrics" not in st.session_state:
        st.session_state["agent_metrics"] = []  # list of metric dicts per turn

    # ── Clear button ──────────────────────────────────────────────────────────
    if st.button("🗑 Limpiar conversación"):
        st.session_state["agent_history"] = []
        st.session_state["agent_metrics"] = []
        st.rerun()

    # ── Render chat history ───────────────────────────────────────────────────
    for msg in st.session_state["agent_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input("Escribe tu pregunta sobre ML…")

    if user_input:
        if not api_key:
            st.error("Ingresa tu API Key en la barra lateral.")
        else:
            # Show user message
            with st.chat_message("user"):
                st.write(user_input)
            st.session_state["agent_history"].append(
                {"role": "user", "content": user_input}
            )

            # Build messages with system prompt
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + \
                       st.session_state["agent_history"]

            # Generate response with timing
            with st.chat_message("assistant"):
                with st.spinner("MLBot está pensando…"):
                    try:
                        client = get_client(provider, api_key)
                        t_start = time.time()
                        response_text, usage = generate_response(
                            client, provider, "", model,
                            agent_temp, 0.9, agent_max_tok,
                            messages=messages,
                        )
                        latency = time.time() - t_start

                        st.write(response_text)
                        st.session_state["agent_history"].append(
                            {"role": "assistant", "content": response_text}
                        )

                        # ── Production metrics ────────────────────────────
                        completion_tokens = usage.completion_tokens
                        prompt_tokens = usage.prompt_tokens
                        tps = completion_tokens / latency if latency > 0 else 0
                        cost = (prompt_tokens * PRICE_IN +
                                completion_tokens * PRICE_OUT)

                        # LLM-Judge auto-eval
                        judge_score = None
                        try:
                            judge_sys = (
                                "Eres un evaluador. Puntúa la respuesta del asistente "
                                "del 1 al 10 según veracidad, coherencia y relevancia. "
                                "Responde SOLO con JSON: {\"score\": <número>}"
                            )
                            judge_usr = (
                                f"PREGUNTA: {user_input}\n"
                                f"RESPUESTA: {response_text}"
                            )
                            judge_msgs = [
                                {"role": "system", "content": judge_sys},
                                {"role": "user", "content": judge_usr},
                            ]
                            j_text, _ = generate_response(
                                client, provider, "", model,
                                0.0, 0.9, 50,
                                messages=judge_msgs,
                            )
                            j_match = re.search(r'\{.*\}', j_text, re.DOTALL)
                            if j_match:
                                judge_score = json.loads(j_match.group()).get("score")
                        except Exception:
                            pass

                        # Store turn metrics
                        turn_num = len(st.session_state["agent_metrics"]) + 1
                        st.session_state["agent_metrics"].append({
                            "Turno": turn_num,
                            "Latencia (s)": round(latency, 2),
                            "TPS": round(tps, 1),
                            "Tokens entrada": prompt_tokens,
                            "Tokens salida": completion_tokens,
                            "Costo USD": round(cost, 6),
                            "LLM-Judge": judge_score,
                        })

                        # Display metrics for this turn
                        with st.expander("📊 Métricas de este turno"):
                            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                            mc1.metric("Latencia", f"{latency:.2f}s")
                            mc2.metric("TPS", f"{tps:.1f}")
                            mc3.metric("Tokens entrada", prompt_tokens)
                            mc4.metric("Tokens salida", completion_tokens)
                            mc5.metric("Costo USD", f"${cost:.6f}")
                            if judge_score is not None:
                                st.metric("LLM-Judge", f"{judge_score}/10")

                    except Exception as e:
                        st.error(f"Error: {e}")

    # ── Metrics history chart ─────────────────────────────────────────────────
    if st.session_state["agent_metrics"]:
        st.divider()
        st.subheader("Historial de Métricas por Turno")
        df_metrics = pd.DataFrame(st.session_state["agent_metrics"])

        tab_lat, tab_tps, tab_judge, tab_cost = st.tabs(
            ["Latencia", "TPS", "LLM-Judge", "Costo"]
        )

        def line_chart(df, y_col, title, color="royalblue"):
            fig = px.line(df, x="Turno", y=y_col, markers=True,
                          title=title)
            fig.update_traces(line_color=color)
            return fig

        tab_lat.plotly_chart(
            line_chart(df_metrics, "Latencia (s)", "Latencia por turno (s)",
                       "tomato"),
            use_container_width=True,
        )
        tab_tps.plotly_chart(
            line_chart(df_metrics, "TPS", "Tokens por segundo", "green"),
            use_container_width=True,
        )
        if df_metrics["LLM-Judge"].notna().any():
            tab_judge.plotly_chart(
                line_chart(df_metrics.dropna(subset=["LLM-Judge"]),
                           "LLM-Judge", "Puntuación LLM-Judge (1-10)",
                           "goldenrod"),
                use_container_width=True,
            )
        tab_cost.plotly_chart(
            line_chart(df_metrics, "Costo USD", "Costo acumulado por turno (USD)",
                       "purple"),
            use_container_width=True,
        )

        with st.expander("Ver tabla completa de métricas"):
            st.dataframe(df_metrics, use_container_width=True)
