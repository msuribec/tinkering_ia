import streamlit as st
import google.generativeai as genai
import pandas as pd
from gtts import gTTS
import json
import re
from PIL import Image
import io
from typing import Optional
import plotly.express as px

default_categories = [
    "Food & Groceries",
    "Transport",
    "Entertainment",
    "Health & Beauty",
    "Household",
    "Clothing",
    "Utilities",
    "Other",
]

GTTS_SUPPORTED_LANGS = {
    "af", "sq", "ar", "hy", "bn", "bs", "ca", "hr", "cs", "da", "nl",
    "en", "eo", "et", "tl", "fi", "fr", "de", "el", "gu", "hi", "hu",
    "is", "id", "it", "ja", "jw", "kn", "km", "ko", "la", "lv", "mk",
    "ml", "mr", "my", "ne", "no", "pl", "pt", "ro", "ru", "sr", "si",
    "sk", "es", "su", "sw", "sv", "ta", "te", "th", "tr", "uk", "ur",
    "vi", "cy", "zh-cn", "zh-tw",
}

st.set_page_config(
    page_title="My-Expense Auditor",
    page_icon="💰",
    layout="centered",
)

st.title("💰 My-Expense Auditor")
st.markdown(
    "*Upload a receipt and your expense categories — get an instant breakdown and a savings tip by voice.*"
)

# ── Green primary buttons ─────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #16a34a; border-color: #16a34a; color: white;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #15803d; border-color: #15803d; color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session-state bootstrap ───────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "widget_seed": 0,
        "categories_approved": False,
        "categories_signature": "",
        "approved_categories": [],
        "receipt_history": [],  # list[dict] — one entry per analyzed receipt
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()
widget_seed = st.session_state["widget_seed"]


def restart_streamlit_app() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
    next_seed = st.session_state["widget_seed"] + 1
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state["widget_seed"] = next_seed
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.header("Set up API Key 🗝️")
    api_key = st.text_input(
        "Paste your Google Gemini API Key",
        key=f"api_key_{widget_seed}",
        type="password",
        placeholder="AIza...",
        help=(
            "Free at aistudio.google.com — no credit card needed.\n\n"
            "How to get a free key:\n"
            "1. Go to Google AI Studio: https://aistudio.google.com\n"
            "2. Sign in with Google\n"
            "3. Click Get API Key → Create API Key\n"
            "4. Paste it above"
        ),
    )

    categories_file = None
    if api_key:
        st.markdown("---")
        st.header("Load your categories file 📂")
        categories_file = st.file_uploader(
            "Your categories file (TXT or CSV)",
            type=["csv", "txt"],
            key=f"categories_file_{widget_seed}",
        )
    else:
        st.caption("Enter your API key to unlock category upload.")

    st.markdown("---")
    if st.button("End session", use_container_width=True):
        restart_streamlit_app()


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_categories(file) -> list[str]:
    content = file.read().decode("utf-8")
    file.seek(0)
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(io.StringIO(content))
        for col in df.columns:
            if col.strip().lower() in ("category", "categoria", "categories", "categorias"):
                return df[col].dropna().str.strip().tolist()
        # No recognised column name — fall back to first column
        return df.iloc[:, 0].dropna().str.strip().tolist()
    return [line.strip() for line in content.splitlines() if line.strip()]


def pick_supported_model() -> Optional[str]:
    preferred = [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-lite",
        "models/gemini-2.0-flash",
        "models/gemini-1.5-flash",
    ]
    try:
        models = list(genai.list_models())
    except Exception:
        return None
    supported = {
        m.name
        for m in models
        if hasattr(m, "supported_generation_methods")
        and "generateContent" in m.supported_generation_methods
    }
    for candidate in preferred:
        if candidate in supported:
            return candidate
    for name in supported:
        if "gemini" in name and "vision" not in name:
            return name
    return None


def analyze_receipt(image_bytes: bytes, categories: list[str]) -> dict:
    chosen_model = pick_supported_model() or "models/gemini-2.0-flash"
    model = genai.GenerativeModel(chosen_model)
    pil_image = Image.open(io.BytesIO(image_bytes))
    categories_block = "\n".join(f"  - {c}" for c in categories)
    prompt = f"""You are a personal finance auditor. Analyze the receipt in this image.

USER'S EXPENSE CATEGORIES:
{categories_block}

TASKS:
1. Extract every line item (product/service name and price).
2. Identify the vendor/store name.
3. Identify the purchase date (if visible).
4. Detect the currency symbol used in the receipt.
5. Assign each item to the closest matching category from the list above.
   Use "Other" only when no category is even remotely appropriate.
6. Sum the amounts per category.
7. Write ONE short, specific, actionable savings tip based on where most money was spent.
   Write the tip in the SAME LANGUAGE as the text on the receipt.

Return ONLY a valid JSON object — no markdown fences, no extra text.
Schema:
{{
  "vendor":          "string",
  "date":            "string or Unknown",
  "currency":        "string (e.g. $, €, COP)",
  "items": [
    {{"name": "string", "price": 0.00, "category": "string"}}
  ],
  "total":            0.00,
  "category_totals": {{"Category Name": 0.00}},
  "savings_tip":     "string",
  "tip_language":    "ISO 639-1 code, e.g. es or en"
}}"""
    response = model.generate_content([prompt, pil_image])
    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def generate_audio(tip: str, lang_code: str) -> tuple[bytes, str]:
    lang = lang_code if lang_code in GTTS_SUPPORTED_LANGS else "es"
    tts = gTTS(text=tip, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read(), lang


def render_receipt_result(entry: dict) -> None:
    """Render one analyzed receipt inside an assistant chat bubble."""
    data = entry["data"]
    currency = data.get("currency", "$")

    c1, c2, c3 = st.columns(3)
    c1.metric("🏪 Vendor", data.get("vendor", "Unknown"))
    c2.metric("📅 Date", data.get("date", "Unknown"))
    c3.metric("💵 Total", f"{currency} {data.get('total', 0):.2f}")

    st.markdown("**📋 Itemised Expenses**")
    items = data.get("items", [])
    if items:
        df_items = pd.DataFrame(items)
        df_items.columns = ["Item", "Price", "Category"]
        df_items["Price"] = df_items["Price"].apply(lambda x: f"{currency} {x:.2f}")
        st.dataframe(df_items, use_container_width=True, hide_index=True)

    cat_totals: dict = data.get("category_totals", {})
    if cat_totals:
        st.markdown("**📊 Spending by Category**")
        df_cat = pd.DataFrame(
            list(cat_totals.items()), columns=["Category", "Amount"]
        ).sort_values("Amount", ascending=False)
        fig = px.bar(
            df_cat,
            x="Category",
            y="Amount",
            labels={"Amount": f"Amount ({currency})"},
            color="Category",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    tip = data.get("savings_tip", "")
    if tip:
        st.markdown("**💡 Savings Tip**")
        st.info(f"🎯 {tip}")
        if entry.get("audio_bytes"):
            st.audio(entry["audio_bytes"], format="audio/mp3")
            st.caption("🔊 Listen to your personalised savings tip")


# ── Categories gate ───────────────────────────────────────────────────────────
if not api_key:
    st.header("Step 1 — Set up an API key 🔑")
    st.info("Enter your Gemini API key in the sidebar.")

categories: list[str] = []
categories_valid = False

if api_key and not categories_file:
    st.header("Step 2 — Upload your categories file in the sidebar.")
    st.caption("`.txt` with one category per line, or `.csv` with a `category` column.")
    st.info("If you don't have a custom file, click **Use default categories** below.")
    st.markdown("The default categories are:")
    st.markdown("- " + "\n- ".join(default_categories))

    if st.session_state.categories_signature != "__default__":
        st.session_state.categories_approved = False
        st.session_state.approved_categories = []

    if not st.session_state.categories_approved:
        if st.button("✅ Use default expense categories", type="primary", use_container_width=True):
            st.session_state.categories_approved = True
            st.session_state.categories_signature = "__default__"
            st.session_state.approved_categories = default_categories
            st.rerun()
        st.info("Please upload a categories file or use the default categories to continue.")
    else:
        categories = default_categories
        categories_valid = True
        st.session_state.approved_categories = categories
        st.success("Using default categories. Continue with receipt upload below.")

elif api_key and categories_file:
    try:
        categories = parse_categories(categories_file)
    except Exception as exc:
        st.error(f"Could not read categories file: {exc}")
        categories = []

    if categories:
        current_signature = f"{categories_file.name}:{categories_file.size}"
        if st.session_state.categories_signature != current_signature:
            st.session_state.categories_signature = current_signature
            st.session_state.categories_approved = False
            st.session_state.approved_categories = []

        st.markdown(f"**{len(categories)} categories loaded from file:**")
        for c in categories:
            st.markdown(f"- {c}")

        categories_valid = True
        if not st.session_state.categories_approved:
            if st.button("✅ Approve Categories", type="primary", use_container_width=True):
                st.session_state.categories_approved = True
                st.session_state.approved_categories = categories
                st.rerun()
            st.info("Please approve these categories to continue.")
        else:
            st.session_state.approved_categories = categories
            st.success("Categories approved. Continue with receipt upload below.")

if not (api_key and categories_valid and st.session_state.categories_approved):
    st.stop()

genai.configure(api_key=api_key)
categories = st.session_state.approved_categories

# ── Main tabs ─────────────────────────────────────────────────────────────────
st.divider()
tab_chat, tab_dash = st.tabs(["📨 Receipts", "📊 Dashboard"])

# ── Tab 1: Chat interface ─────────────────────────────────────────────────────
with tab_chat:
    # Render history
    for entry in st.session_state.receipt_history:
        with st.chat_message("user"):
            st.image(entry["image_bytes"], width=260)
        with st.chat_message("assistant"):
            render_receipt_result(entry)

    st.divider()
    st.markdown("#### Upload a new receipt")
    receipt_file = st.file_uploader(
        "Photo of your receipt or invoice (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"receipt_file_{widget_seed}_{len(st.session_state.receipt_history)}",
    )

    if receipt_file:
        st.image(receipt_file, caption="Preview", width=260)
        if st.button("🔍 Analyze Receipt", type="primary", use_container_width=True):
            with st.spinner("Reading receipt with Gemini Vision…"):
                try:
                    data = analyze_receipt(receipt_file.getvalue(), categories)
                except json.JSONDecodeError:
                    st.error("The model returned an unexpected response. Please try again.")
                    st.stop()
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.stop()

            tip = data.get("savings_tip", "")
            audio_bytes: bytes | None = None
            if tip:
                with st.spinner("Generating audio tip…"):
                    audio_bytes, _ = generate_audio(tip, data.get("tip_language", "es"))

            st.session_state.receipt_history.append(
                {
                    "image_bytes": receipt_file.getvalue(),
                    "data": data,
                    "audio_bytes": audio_bytes,
                }
            )
            st.rerun()

# ── Tab 2: Dashboard ──────────────────────────────────────────────────────────
with tab_dash:
    history = st.session_state.receipt_history

    if not history:
        st.info("No receipts analyzed yet. Analyze some receipts in the **Receipts** tab first.")
    else:
        # Build aggregated data
        currency = history[-1]["data"].get("currency", "$")

        # KPI values
        grand_total = sum(e["data"].get("total", 0) for e in history)
        combined_cats: dict[str, float] = {}
        for e in history:
            for cat, amt in e["data"].get("category_totals", {}).items():
                combined_cats[cat] = combined_cats.get(cat, 0) + amt
        top_category = max(combined_cats, key=combined_cats.get) if combined_cats else "—"

        k1, k2, k3 = st.columns(3)
        k1.metric("🧾 Receipts analyzed", len(history))
        k2.metric("💰 Grand total", f"{currency} {grand_total:.2f}")
        k3.metric("📌 Top category", top_category)

        st.divider()

        # 1. Pie — spending share by category
        if combined_cats:
            st.subheader("Spending by Category (all receipts)")
            df_pie = pd.DataFrame(
                list(combined_cats.items()), columns=["Category", "Amount"]
            )
            fig_pie = px.pie(
                df_pie,
                names="Category",
                values="Amount",
                color_discrete_sequence=px.colors.qualitative.Safe,
                hole=0.35,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=True, margin=dict(t=30, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()

        # 2. Bar — total per receipt
        st.subheader("Total per Receipt")
        receipt_labels = [
            f"{e['data'].get('vendor', 'Unknown')} ({e['data'].get('date', '?')})"
            for e in history
        ]
        receipt_totals = [e["data"].get("total", 0) for e in history]
        df_totals = pd.DataFrame({"Receipt": receipt_labels, "Total": receipt_totals})
        fig_bar = px.bar(
            df_totals,
            x="Receipt",
            y="Total",
            labels={"Total": f"Total ({currency})"},
            color="Receipt",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text_auto=".2f",
        )
        fig_bar.update_layout(showlegend=False, margin=dict(t=20, b=60))
        fig_bar.update_xaxes(tickangle=-25)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # 3. Stacked bar — per-receipt breakdown by category
        st.subheader("Category Breakdown per Receipt")
        rows = []
        for label, entry in zip(receipt_labels, history):
            for cat, amt in entry["data"].get("category_totals", {}).items():
                rows.append({"Receipt": label, "Category": cat, "Amount": amt})
        if rows:
            df_stack = pd.DataFrame(rows)
            fig_stack = px.bar(
                df_stack,
                x="Receipt",
                y="Amount",
                color="Category",
                labels={"Amount": f"Amount ({currency})"},
                color_discrete_sequence=px.colors.qualitative.Safe,
                barmode="stack",
            )
            fig_stack.update_layout(margin=dict(t=20, b=60))
            fig_stack.update_xaxes(tickangle=-25)
            st.plotly_chart(fig_stack, use_container_width=True)
