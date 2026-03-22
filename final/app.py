import streamlit as st
import google.generativeai as genai
import pandas as pd
from gtts import gTTS
import json
import re
from PIL import Image
import io
from typing import Optional

default_categories = [
        "Food & Groceries",
        "Transport",
        "Entertainment",
        "Health & Beauty",
        "Household",
        "Clothing",
        "Utilities",
        "Other"
]

st.set_page_config(
    page_title="My-Expense Auditor",
    page_icon="💰",
    layout="centered",
)

st.title("💰 My-Expense Auditor")
st.markdown("*Upload a receipt and your expense categories — get an instant breakdown and a savings tip by voice.*")

# ── Sidebar: API key ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.35rem;">
            <h3 style="margin: 0;">⚙️ Configuration</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.header("Set up API Key 🗝️")
    api_key = st.text_input(
        "Paste your Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help=(
            "Free at aistudio.google.com — no credit card needed.\n\n"
            "How to get a free key:\n"
            "1. Go to Google AI Studio: https://aistudio.google.com\n"
            "2. Sign in with Google\n"
            "3. Click Get API Key -> Create API Key\n"
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
        )
    else:
        st.caption("Enter your API key to unlock category upload.")

# ── Helper: parse categories file ────────────────────────────────────────────

def parse_categories(file) -> list[str]:
    content = file.read().decode("utf-8")
    file.seek(0)
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(io.StringIO(content))
        for col in df.columns:
            if col.strip().lower() in ("category", "categoria", "categories", "categorias"):
                return df[col].dropna().str.strip().tolist()
        return df.iloc[:, 0].dropna().str.strip().tolist()
    return [line.strip() for line in content.splitlines() if line.strip()]


def pick_supported_model() -> Optional[str]:
    """
    Pick the best available Gemini model that supports generateContent.
    Returns the model name in "models/..." format or None if discovery fails.
    """
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


# ── Helper: call Gemini Vision ────────────────────────────────────────────────

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
    # Strip markdown code fences if the model added them
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


# ── Main flow: categories approval gate ───────────────────────────────────────
st.markdown(
    """
    <style>
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #16a34a;
        border-color: #16a34a;
        color: white;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #15803d;
        border-color: #15803d;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not api_key:
    st.header("Step 1 — Set up an api key 🔑")
    st.info("Enter your Gemini API key in the sidebar")

if "categories_approved" not in st.session_state:
    st.session_state.categories_approved = False
if "categories_signature" not in st.session_state:
    st.session_state.categories_signature = ""
if "approved_categories" not in st.session_state:
    st.session_state.approved_categories = []


categories: list[str] = []
categories_valid = False
use_default_categories = False


if api_key and not categories_file:
    st.header("Step 2 - Upload your categories file in the sidebar.")
    st.caption("You can upload a `.txt` with one category per line, or `.csv` with a `category` column.")
    st.info("If you don't have a custom categories file, you can download click the 'Use default' button")
    st.markdown("The default categories are: \n Food & Groceries\nTransport\nEntertainment\nHealth & Beauty\n"
        "Household\nClothing\nUtilities\nOther")
    
    if st.button("✅ Use default expense categories", type="primary", use_container_width=True):
        categories_valid = True
        categories = default_categories
        
elif api_key and categories_file:
    if categories:

        current_signature = f"{categories_file.name}:{categories_file.size}"
        if st.session_state.categories_signature != current_signature:
            st.session_state.categories_signature = current_signature
            st.session_state.categories_approved = False
            st.session_state.approved_categories = []

        st.markdown(f"**{len(categories)} categories where loaded from the file:**")
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

    else:
        try:
            categories = parse_categories(categories_file)
        except Exception as exc:
            st.error(f"Could not read categories file: {exc}")
            categories = []


if not (api_key and categories_valid and st.session_state.categories_approved):
    st.stop()

genai.configure(api_key=api_key)

# ── Step 2: receipt image ─────────────────────────────────────────────────────
st.divider()
st.header("Step 2 — Upload your receipt 📷")
receipt_file = st.file_uploader(
    "Photo of your receipt or invoice (JPG / PNG / WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
)

if receipt_file:
    categories = st.session_state.approved_categories

    col_img, col_cats = st.columns([1, 1])
    with col_img:
        st.image(receipt_file, caption="Your receipt", use_container_width=True)
    with col_cats:
        st.markdown(f"**Approved categories ({len(categories)}):**")
        for c in categories:
            st.markdown(f"- {c}")

    st.divider()

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

        st.success("Receipt analyzed!")

        # ── Metrics ───────────────────────────────────────────────────────────
        currency = data.get("currency", "$")
        c1, c2, c3 = st.columns(3)
        c1.metric("🏪 Vendor", data.get("vendor", "Unknown"))
        c2.metric("📅 Date", data.get("date", "Unknown"))
        c3.metric("💵 Total", f"{currency} {data.get('total', 0):.2f}")

        st.divider()

        # ── Itemised table ────────────────────────────────────────────────────
        st.subheader("📋 Itemised Expenses")
        items = data.get("items", [])
        if items:
            df_items = pd.DataFrame(items)
            df_items.columns = ["Item", "Price", "Category"]
            df_items["Price"] = df_items["Price"].apply(lambda x: f"{currency} {x:.2f}")
            st.dataframe(df_items, use_container_width=True, hide_index=True)
        else:
            st.info("No line items were extracted.")

        st.divider()

        # ── Category breakdown ────────────────────────────────────────────────
        st.subheader("📊 Spending by Category")
        cat_totals: dict = data.get("category_totals", {})
        if cat_totals:
            df_cat = (
                pd.DataFrame(list(cat_totals.items()), columns=["Category", "Amount"])
                .sort_values("Amount", ascending=False)
            )
            st.bar_chart(df_cat.set_index("Category"), use_container_width=True)
            df_cat_display = df_cat.copy()
            df_cat_display["Amount"] = df_cat_display["Amount"].apply(
                lambda x: f"{currency} {x:.2f}"
            )
            st.dataframe(df_cat_display, use_container_width=True, hide_index=True)
        else:
            st.info("No category totals available.")

        st.divider()

        # ── Savings tip (text + audio) ────────────────────────────────────────
        tip = data.get("savings_tip", "")
        if tip:
            st.subheader("💡 Savings Tip")
            st.info(f"🎯 {tip}")

            lang = data.get("tip_language", "es")
            # gTTS only accepts certain codes; default to Spanish if unknown
            if lang not in ("af","sq","ar","hy","bn","bs","ca","hr","cs","da","nl",
                            "en","eo","et","tl","fi","fr","de","el","gu","hi","hu",
                            "is","id","it","ja","jw","kn","km","ko","la","lv","mk",
                            "ml","mr","my","ne","no","pl","pt","ro","ru","sr","si",
                            "sk","es","su","sw","sv","ta","te","th","tr","uk","ur",
                            "vi","cy","zh-cn","zh-tw"):
                lang = "es"

            with st.spinner("Generating audio…"):
                tts = gTTS(text=tip, lang=lang, slow=False)
                audio_buf = io.BytesIO()
                tts.write_to_fp(audio_buf)
                audio_buf.seek(0)

            st.audio(audio_buf, format="audio/mp3")
            st.caption("🔊 Listen to your personalised savings tip")
else:
    st.info("👆 Upload a receipt image to analyze it.")
