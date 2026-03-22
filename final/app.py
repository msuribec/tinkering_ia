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
    "*Upload a receipt and your expense categories — extract expenses, edit them, and generate a savings tip by voice.*"
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
        "editing_receipt_index": None,
        "history_category_suggestions": [],
        "history_purchase_tips": [],
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


def clear_history_suggestions() -> None:
    st.session_state["history_category_suggestions"] = []
    st.session_state["history_purchase_tips"] = []


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_categories_export_csv(categories: list[str]) -> bytes:
    return dataframe_to_csv_bytes(pd.DataFrame({"category": categories}))


def build_spending_export_csv(history: list[dict]) -> bytes:
    rows = []
    for index, entry in enumerate(history, start=1):
        data = entry.get("data", {})
        vendor = data.get("vendor", "Unknown")
        date = data.get("date", "Unknown")
        currency = data.get("currency", "$")
        receipt_total = float(data.get("total", 0) or 0)

        for item in data.get("items", []):
            rows.append(
                {
                    "receipt_index": index,
                    "vendor": vendor,
                    "date": date,
                    "currency": currency,
                    "item": item.get("name", ""),
                    "price": float(item.get("price", 0) or 0),
                    "category": item.get("category", ""),
                    "receipt_total": receipt_total,
                }
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "receipt_index",
            "vendor",
            "date",
            "currency",
            "item",
            "price",
            "category",
            "receipt_total",
        ],
    )
    return dataframe_to_csv_bytes(df)


def build_history_summary(history: list[dict]) -> dict:
    receipts = []
    stores = []
    items = []
    category_totals: dict[str, float] = {}

    for index, entry in enumerate(history, start=1):
        data = entry.get("data", {})
        vendor = data.get("vendor", "Unknown")
        stores.append(vendor)
        receipt_items = data.get("items", [])

        receipts.append(
            {
                "receipt_index": index,
                "vendor": vendor,
                "date": data.get("date", "Unknown"),
                "currency": data.get("currency", "$"),
                "total": float(data.get("total", 0) or 0),
                "items_count": len(receipt_items),
            }
        )

        for item in receipt_items:
            item_name = item.get("name", "")
            item_category = item.get("category", "")
            item_price = float(item.get("price", 0) or 0)
            items.append(
                {
                    "receipt_index": index,
                    "vendor": vendor,
                    "item": item_name,
                    "category": item_category,
                    "price": item_price,
                }
            )
            category_totals[item_category] = round(category_totals.get(item_category, 0.0) + item_price, 2)

    top_stores = pd.Series(stores).value_counts().head(10).to_dict() if stores else {}

    return {
        "receipts": receipts,
        "items": items,
        "top_stores": top_stores,
        "category_totals": category_totals,
    }


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
    st.header("Export data")
    approved_categories = st.session_state.approved_categories
    categories_csv = build_categories_export_csv(approved_categories)
    st.download_button(
        "Download categories CSV",
        data=categories_csv,
        file_name="expense_categories.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=not st.session_state.categories_approved,
    )
    if not st.session_state.categories_approved:
        st.caption("Approve categories to enable category export.")

    spending_history = st.session_state.receipt_history
    spending_csv = build_spending_export_csv(spending_history) if spending_history else b""
    st.download_button(
        "Download spending CSV",
        data=spending_csv,
        file_name="spending_data.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=not spending_history,
    )
    if not spending_history:
        st.caption("Analyze at least one receipt to export spending data.")

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
  "category_totals": {{"Category Name": 0.00}}
}}"""
    response = model.generate_content([prompt, pil_image])
    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def generate_savings_tip(receipt_data: dict) -> dict:
    chosen_model = pick_supported_model() or "models/gemini-2.0-flash"
    model = genai.GenerativeModel(chosen_model)
    receipt_payload = json.dumps(
        {
            "vendor": receipt_data.get("vendor", "Unknown"),
            "date": receipt_data.get("date", "Unknown"),
            "currency": receipt_data.get("currency", "$"),
            "items": receipt_data.get("items", []),
            "total": receipt_data.get("total", 0),
            "category_totals": receipt_data.get("category_totals", {}),
        },
        ensure_ascii=False,
    )
    prompt = f"""You are a personal finance auditor. Use the receipt data below to write a concise savings tip.

RECEIPT DATA:
{receipt_payload}

TASKS:
1. Review the spending pattern in this receipt.
2. Write ONE short, specific, actionable savings tip based on where most money was spent.
3. Write the tip in the same language as the receipt data when it is reasonably clear. If the language is unclear, write it in Spanish.

Return ONLY a valid JSON object — no markdown fences, no extra text.
Schema:
{{
  "savings_tip": "string",
  "tip_language": "ISO 639-1 code, e.g. es or en"
}}"""
    response = model.generate_content(prompt)
    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def generate_category_suggestions(history: list[dict], categories: list[str]) -> list[dict]:
    chosen_model = pick_supported_model() or "models/gemini-2.0-flash"
    model = genai.GenerativeModel(chosen_model)
    history_summary = json.dumps(build_history_summary(history), ensure_ascii=False)
    categories_payload = json.dumps(categories, ensure_ascii=False)
    prompt = f"""You are a personal finance auditor. Review the purchase history and the user's current categories.

CURRENT CATEGORIES:
{categories_payload}

PURCHASE HISTORY SUMMARY:
{history_summary}

TASKS:
1. Suggest up to 5 NEW categories that do not already exist in the current categories list.
2. Base the suggestions on recurring items, stores, and spending patterns from the history.
3. Keep each category concise and practical for a budgeting app.
4. For each suggestion, include a short reason mentioning the relevant items or stores.
5. If the current categories are already sufficient, return an empty list.

Return ONLY a valid JSON object — no markdown fences, no extra text.
Schema:
{{
  "suggested_categories": [
    {{"category": "string", "reason": "string"}}
  ]
}}"""
    response = model.generate_content(prompt)
    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    return data.get("suggested_categories", [])


def generate_history_tips(history: list[dict]) -> dict:
    chosen_model = pick_supported_model() or "models/gemini-2.0-flash"
    model = genai.GenerativeModel(chosen_model)
    history_summary = json.dumps(build_history_summary(history), ensure_ascii=False)
    prompt = f"""You are a personal finance auditor. Review the user's full purchase history and write practical savings tips.

PURCHASE HISTORY SUMMARY:
{history_summary}

TASKS:
1. Review the overall spending patterns across all receipts.
2. Write 3 short, specific, actionable savings tips.
3. Base the tips on repeated stores, recurring item types, or dominant spending categories.
4. Keep each tip to one sentence.
5. Write the tips in the same language as the purchase history when it is reasonably clear. If unclear, write them in Spanish.

Return ONLY a valid JSON object — no markdown fences, no extra text.
Schema:
{{
  "tips": ["string", "string", "string"],
  "tip_language": "ISO 639-1 code, e.g. es or en"
}}"""
    response = model.generate_content(prompt)
    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    return {
        "tips": [tip.strip() for tip in data.get("tips", []) if isinstance(tip, str) and tip.strip()],
        "tip_language": str(data.get("tip_language", "es")).strip() or "es",
    }


def generate_audio(tip: str, lang_code: str) -> tuple[bytes, str]:
    lang = lang_code if lang_code in GTTS_SUPPORTED_LANGS else "es"
    tts = gTTS(text=tip, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read(), lang


def normalize_edited_items(items_df: pd.DataFrame, categories: list[str]) -> tuple[list[dict], list[str]]:
    cleaned_items: list[dict] = []
    errors: list[str] = []

    for row_number, row in enumerate(items_df.to_dict("records"), start=1):
        raw_name = row.get("name", "")
        raw_category = row.get("category", "")
        raw_price = row.get("price", None)

        name = "" if pd.isna(raw_name) else str(raw_name).strip()
        category = "" if pd.isna(raw_category) else str(raw_category).strip()
        price_missing = raw_price is None or pd.isna(raw_price)

        if not name and not category and price_missing:
            continue

        if not name:
            errors.append(f"Row {row_number}: item name is required.")

        if category not in categories:
            errors.append(f"Row {row_number}: category must be one of the approved categories.")

        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            errors.append(f"Row {row_number}: price must be a valid number.")
            continue

        if price < 0:
            errors.append(f"Row {row_number}: price cannot be negative.")

        if name and category in categories and price >= 0:
            cleaned_items.append(
                {
                    "name": name,
                    "price": round(price, 2),
                    "category": category,
                }
            )

    if not cleaned_items:
        errors.append("Add at least one valid item before saving.")

    return cleaned_items, errors


def rebuild_receipt_data(
    original_data: dict,
    vendor: str,
    date: str,
    currency: str,
    items: list[dict],
) -> dict:
    category_totals: dict[str, float] = {}
    total = 0.0

    for item in items:
        price = round(float(item["price"]), 2)
        category = item["category"]
        total += price
        category_totals[category] = round(category_totals.get(category, 0.0) + price, 2)

    return {
        **original_data,
        "vendor": vendor.strip() or "Unknown",
        "date": date.strip() or "Unknown",
        "currency": currency.strip() or "$",
        "items": items,
        "total": round(total, 2),
        "category_totals": category_totals,
        "savings_tip": "",
        "tip_language": "",
    }


def render_receipt_editor(index: int, entry: dict, categories: list[str]) -> None:
    data = entry["data"]
    items_df = pd.DataFrame(data.get("items", []), columns=["name", "price", "category"])

    with st.expander("Edit receipt details", expanded=True):
        with st.form(f"edit_receipt_form_{index}"):
            vendor = st.text_input("Vendor", value=data.get("vendor", ""))
            date = st.text_input("Date", value=data.get("date", ""))
            currency = st.text_input("Currency", value=data.get("currency", "$"))
            edited_items_df = st.data_editor(
                items_df,
                key=f"receipt_items_editor_{index}",
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "name": st.column_config.TextColumn("Item"),
                    "price": st.column_config.NumberColumn(
                        "Price",
                        min_value=0.0,
                        step=0.01,
                        format="%.2f",
                    ),
                    "category": st.column_config.SelectboxColumn("Category", options=categories),
                },
            )

            save_col, cancel_col = st.columns(2)
            save_clicked = save_col.form_submit_button(
                "Save changes",
                type="primary",
                use_container_width=True,
            )
            cancel_clicked = cancel_col.form_submit_button(
                "Cancel",
                use_container_width=True,
            )

        if cancel_clicked:
            st.session_state.editing_receipt_index = None
            st.rerun()

        if save_clicked:
            cleaned_items, errors = normalize_edited_items(edited_items_df, categories)
            if errors:
                for error in errors:
                    st.error(error)
                return

            updated_data = rebuild_receipt_data(data, vendor, date, currency, cleaned_items)
            st.session_state.receipt_history[index]["data"] = updated_data
            st.session_state.receipt_history[index]["audio_bytes"] = None
            st.session_state.editing_receipt_index = None
            clear_history_suggestions()
            st.rerun()


def render_receipt_result(index: int, entry: dict, categories: list[str]) -> None:
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
    actions_col, tip_action_col = st.columns(2)
    is_editing = st.session_state.editing_receipt_index == index
    button_label = "Close editor" if is_editing else "Edit receipt"
    if actions_col.button(button_label, key=f"edit_receipt_button_{index}", use_container_width=True):
        st.session_state.editing_receipt_index = None if is_editing else index
        st.rerun()

    if not tip:
        if tip_action_col.button(
            "Generate tip based on receipt",
            key=f"generate_tip_button_{index}",
            use_container_width=True,
        ):
            with st.spinner("Generating savings tip..."):
                try:
                    tip_data = generate_savings_tip(data)
                    tip_text = tip_data.get("savings_tip", "").strip()
                    tip_language = tip_data.get("tip_language", "es").strip() or "es"
                    if not tip_text:
                        st.error("The model did not return a savings tip. Please try again.")
                        return
                    audio_bytes, detected_lang = generate_audio(tip_text, tip_language)
                except json.JSONDecodeError:
                    st.error("The model returned an unexpected tip response. Please try again.")
                    return
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    return

            st.session_state.receipt_history[index]["data"]["savings_tip"] = tip_text
            st.session_state.receipt_history[index]["data"]["tip_language"] = detected_lang
            st.session_state.receipt_history[index]["audio_bytes"] = audio_bytes
            st.rerun()
    else:
        tip_action_col.empty()

    if is_editing:
        render_receipt_editor(index, entry, categories)

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
            clear_history_suggestions()
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
                clear_history_suggestions()
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
tab_chat, tab_dash, tab_suggestions = st.tabs(["📨 Receipts", "📊 Dashboard", "💡 Suggestions"])

# ── Tab 1: Chat interface ─────────────────────────────────────────────────────
with tab_chat:
    # Render history
    for index, entry in enumerate(st.session_state.receipt_history):
        with st.chat_message("user"):
            st.image(entry["image_bytes"], width=260)
        with st.chat_message("assistant"):
            render_receipt_result(index, entry, categories)

    st.divider()
    st.markdown("#### Upload a new receipt")
    receipt_file = st.file_uploader(
        "Photo of your receipt or invoice (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"receipt_file_{widget_seed}_{len(st.session_state.receipt_history)}",
    )

    if receipt_file:
        st.image(receipt_file, caption="Preview", width=260)
        if st.button("🔍 Extract data from receipt", type="primary", use_container_width=True):
            with st.spinner("Extracting receipt data with Gemini Vision..."):
                try:
                    data = analyze_receipt(receipt_file.getvalue(), categories)
                except json.JSONDecodeError:
                    st.error("The model returned an unexpected response. Please try again.")
                    st.stop()
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.stop()

            data["savings_tip"] = ""
            data["tip_language"] = ""

            st.session_state.receipt_history.append(
                {
                    "image_bytes": receipt_file.getvalue(),
                    "data": data,
                    "audio_bytes": None,
                }
            )
            clear_history_suggestions()
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

# ── Tab 3: Suggestions ────────────────────────────────────────────────────────
with tab_suggestions:
    history = st.session_state.receipt_history

    if not history:
        st.info("No receipts analyzed yet. Analyze some receipts in the **Receipts** tab first.")
    else:
        st.subheader("Suggestions from your full history")
        st.caption("Use your complete receipt history to discover better categories and broader saving opportunities.")

        suggest_col, tip_col = st.columns(2)

        if suggest_col.button(
            "Suggest new categories",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Reviewing your history to suggest new categories..."):
                try:
                    st.session_state.history_category_suggestions = generate_category_suggestions(history, categories)
                except json.JSONDecodeError:
                    st.error("The model returned an unexpected response for category suggestions. Please try again.")
                except Exception as exc:
                    st.error(f"Error: {exc}")

        if tip_col.button(
            "Generate history tips",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Reviewing your purchase history for savings tips..."):
                try:
                    tip_payload = generate_history_tips(history)
                    tip_language = tip_payload.get("tip_language", "es")
                    history_tip_entries = []
                    for tip_text in tip_payload.get("tips", []):
                        audio_bytes, detected_lang = generate_audio(tip_text, tip_language)
                        history_tip_entries.append(
                            {
                                "tip": tip_text,
                                "audio_bytes": audio_bytes,
                                "tip_language": detected_lang,
                            }
                        )
                    st.session_state.history_purchase_tips = history_tip_entries
                except json.JSONDecodeError:
                    st.error("The model returned an unexpected response for history tips. Please try again.")
                except Exception as exc:
                    st.error(f"Error: {exc}")

        st.divider()

        st.markdown("**Suggested New Categories**")
        category_suggestions = st.session_state.history_category_suggestions
        if category_suggestions:
            for suggestion in category_suggestions:
                category_name = str(suggestion.get("category", "")).strip()
                reason = str(suggestion.get("reason", "")).strip()
                if category_name:
                    st.markdown(f"**{category_name}**")
                    if reason:
                        st.caption(reason)
        else:
            st.caption("No category suggestions generated yet.")

        st.divider()

        st.markdown("**Tips Based on All Purchase History**")
        history_tips = st.session_state.history_purchase_tips
        if history_tips:
            for tip_entry in history_tips:
                tip_text = tip_entry.get("tip", "") if isinstance(tip_entry, dict) else str(tip_entry)
                st.info(f"🎯 {tip_text}")
                if isinstance(tip_entry, dict) and tip_entry.get("audio_bytes"):
                    st.audio(tip_entry["audio_bytes"], format="audio/mp3")
        else:
            st.caption("No history tips generated yet.")
