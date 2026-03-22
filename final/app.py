import streamlit as st
import google.generativeai as genai
import pandas as pd
from gtts import gTTS
import calendar
import json
import re
from PIL import Image
import io
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go
from rag import ReceiptVectorStore, answer_question



SPENDING_EXPORT_COLUMNS = [
    "receipt_index",
    "vendor",
    "date",
    "currency",
    "item",
    "price",
    "category",
    "receipt_total",
]

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
        "category_budgets": {},
        "selected_budget_month": "",
        "vector_store": None,
        "rag_chat_history": [],
        "rag_receipt_count": 0,
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


def sync_category_budgets(categories: list[str]) -> None:
    current_budgets = st.session_state.get("category_budgets", {})
    st.session_state["category_budgets"] = {
        category: round(float(current_budgets.get(category, 0.0) or 0.0), 2)
        for category in categories
    }


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
        columns=SPENDING_EXPORT_COLUMNS,
    )
    return dataframe_to_csv_bytes(df)


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


def parse_uploaded_spending_history(file) -> list[dict]:
    content = file.read().decode("utf-8")
    file.seek(0)

    try:
        df = pd.read_csv(io.StringIO(content))
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    missing_columns = [column for column in SPENDING_EXPORT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    df = df[SPENDING_EXPORT_COLUMNS].copy()
    for column in ["vendor", "date", "currency", "item", "category"]:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["vendor"] = df["vendor"].replace("", "Unknown")
    df["date"] = df["date"].replace("", "Unknown")
    df["currency"] = df["currency"].replace("", "$")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["receipt_total"] = pd.to_numeric(df["receipt_total"], errors="coerce")

    invalid_price_rows = df.index[df["price"].isna()].tolist()
    if invalid_price_rows:
        rows_text = ", ".join(str(row + 2) for row in invalid_price_rows)
        raise ValueError(f"Invalid price values found in CSV rows: {rows_text}")

    blank_item_rows = df.index[df["item"] == ""].tolist()
    if blank_item_rows:
        rows_text = ", ".join(str(row + 2) for row in blank_item_rows)
        raise ValueError(f"Blank item values found in CSV rows: {rows_text}")

    blank_category_rows = df.index[df["category"] == ""].tolist()
    if blank_category_rows:
        rows_text = ", ".join(str(row + 2) for row in blank_category_rows)
        raise ValueError(f"Blank category values found in CSV rows: {rows_text}")

    imported_entries = []
    for _, receipt_df in df.groupby("receipt_index", sort=False):
        first_row = receipt_df.iloc[0]
        items = [
            {
                "name": row["item"],
                "price": round(float(row["price"]), 2),
                "category": row["category"],
            }
            for _, row in receipt_df.iterrows()
        ]
        data = rebuild_receipt_data(
            {},
            str(first_row["vendor"]),
            str(first_row["date"]),
            str(first_row["currency"]),
            items,
        )
        imported_entries.append(
            {
                "image_bytes": None,
                "source": "imported_csv",
                "data": data,
                "audio_bytes": None,
            }
        )

    if not imported_entries:
        raise ValueError("The uploaded CSV does not contain any receipt rows.")

    return imported_entries


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


def parse_receipt_date(date_value: str) -> Optional[pd.Timestamp]:
    if date_value is None:
        return None

    text = str(date_value).strip()
    if not text or text.lower() in {"unknown", "none", "null", "nan", "nat"}:
        return None

    attempts = [
        {"format": "mixed", "dayfirst": False},
        {"format": "mixed", "dayfirst": True},
        {"dayfirst": False},
        {"dayfirst": True},
    ]
    for kwargs in attempts:
        try:
            parsed = pd.to_datetime(text, errors="coerce", **kwargs)
        except (TypeError, ValueError):
            continue
        if pd.notna(parsed):
            return pd.Timestamp(parsed).normalize()
    return None


def build_budget_source_data(history: list[dict]) -> dict:
    item_rows = []
    receipt_rows = []
    skipped_receipts = 0

    for index, entry in enumerate(history, start=1):
        data = entry.get("data", {})
        parsed_date = parse_receipt_date(data.get("date", ""))
        if parsed_date is None:
            skipped_receipts += 1
            continue

        receipt_rows.append(
            {
                "receipt_index": index,
                "parsed_date": parsed_date,
                "vendor": data.get("vendor", "Unknown"),
                "currency": data.get("currency", "$"),
                "total": float(data.get("total", 0) or 0),
            }
        )

        for item in data.get("items", []):
            item_rows.append(
                {
                    "receipt_index": index,
                    "parsed_date": parsed_date,
                    "vendor": data.get("vendor", "Unknown"),
                    "currency": data.get("currency", "$"),
                    "item": item.get("name", ""),
                    "category": item.get("category", "Other"),
                    "price": float(item.get("price", 0) or 0),
                }
            )

    items_df = pd.DataFrame(
        item_rows,
        columns=[
            "receipt_index",
            "parsed_date",
            "vendor",
            "currency",
            "item",
            "category",
            "price",
        ],
    )
    receipts_df = pd.DataFrame(
        receipt_rows,
        columns=["receipt_index", "parsed_date", "vendor", "currency", "total"],
    )
    return {
        "items_df": items_df,
        "receipts_df": receipts_df,
        "skipped_receipts": skipped_receipts,
    }


def get_default_budget_month(receipts_df: pd.DataFrame) -> str:
    if not receipts_df.empty:
        latest_date = receipts_df["parsed_date"].max()
        return latest_date.strftime("%Y-%m")
    return pd.Timestamp.today().strftime("%Y-%m")


def ensure_selected_budget_month(receipts_df: pd.DataFrame) -> None:
    selected_budget_month = st.session_state.get("selected_budget_month", "")
    if re.fullmatch(r"\d{4}-\d{2}", selected_budget_month or ""):
        return
    st.session_state["selected_budget_month"] = get_default_budget_month(receipts_df)


def build_month_budget_analytics(
    budget_source: dict,
    category_budgets: dict[str, float],
    selected_month_id: str,
) -> dict:
    receipts_df = budget_source["receipts_df"]
    items_df = budget_source["items_df"]
    month_period = pd.Period(selected_month_id, freq="M")

    budget_series = pd.Series(category_budgets, dtype="float64")
    if budget_series.empty:
        budget_series = pd.Series(dtype="float64")

    if not receipts_df.empty:
        month_receipts_df = receipts_df[
            receipts_df["parsed_date"].dt.to_period("M") == month_period
        ].copy()
    else:
        month_receipts_df = receipts_df.copy()

    if not items_df.empty:
        month_items_df = items_df[
            items_df["parsed_date"].dt.to_period("M") == month_period
        ].copy()
    else:
        month_items_df = items_df.copy()

    actual_series = (
        month_items_df.groupby("category")["price"].sum()
        if not month_items_df.empty
        else pd.Series(dtype="float64")
    )
    all_categories = list(dict.fromkeys(budget_series.index.tolist() + actual_series.index.tolist()))
    budget_series = budget_series.reindex(all_categories, fill_value=0.0)
    actual_series = actual_series.reindex(all_categories, fill_value=0.0)

    category_df = pd.DataFrame(
        {
            "Category": budget_series.index.tolist(),
            "Budget": budget_series.values,
            "Actual": actual_series.values,
        }
    )
    if category_df.empty:
        category_df = pd.DataFrame(columns=["Category", "Budget", "Actual"])
    category_df["Variance"] = category_df["Actual"] - category_df["Budget"]
    category_df["Variance Status"] = category_df["Variance"].apply(
        lambda value: "Overspent" if value > 0 else "Within budget"
    )

    stacked_df = category_df.melt(
        id_vars="Category",
        value_vars=["Budget", "Actual"],
        var_name="Scenario",
        value_name="Amount",
    )

    daily_index = pd.date_range(
        month_period.start_time.normalize(),
        month_period.end_time.normalize(),
        freq="D",
    )
    if not month_items_df.empty:
        daily_actual_series = month_items_df.groupby("parsed_date")["price"].sum()
        daily_actual_series = daily_actual_series.reindex(daily_index, fill_value=0.0)
    else:
        daily_actual_series = pd.Series(0.0, index=daily_index)

    daily_df = pd.DataFrame(
        {
            "Date": daily_index,
            "Actual Daily": daily_actual_series.values,
        }
    )
    daily_df["Day"] = daily_df["Date"].dt.day

    total_budget = float(category_df["Budget"].sum()) if not category_df.empty else 0.0
    total_actual = float(category_df["Actual"].sum()) if not category_df.empty else 0.0
    days_in_month = len(daily_df) if not daily_df.empty else month_period.days_in_month
    daily_df["Ideal Cumulative"] = (
        total_budget * daily_df["Day"] / max(days_in_month, 1)
        if not daily_df.empty
        else 0.0
    )
    daily_df["Actual Cumulative"] = daily_df["Actual Daily"].cumsum()

    currency = "$"
    if not month_items_df.empty:
        currency = str(month_items_df["currency"].dropna().iloc[-1])
    elif not month_receipts_df.empty:
        currency = str(month_receipts_df["currency"].dropna().iloc[-1])

    return {
        "month_period": month_period,
        "month_receipts_df": month_receipts_df,
        "month_items_df": month_items_df,
        "category_df": category_df,
        "stacked_df": stacked_df,
        "daily_df": daily_df,
        "total_budget": round(total_budget, 2),
        "total_actual": round(total_actual, 2),
        "currency": currency,
    }


def build_budget_heatmap_figure(
    month_period: pd.Period,
    daily_df: pd.DataFrame,
    currency: str,
) -> go.Figure:
    spend_by_day = {
        int(row["Day"]): float(row["Actual Daily"])
        for _, row in daily_df.iterrows()
    }
    month_matrix = calendar.Calendar(firstweekday=0).monthdayscalendar(
        month_period.year,
        month_period.month,
    )

    z_values = []
    text_values = []
    for week in month_matrix:
        week_values = []
        week_text = []
        for day in week:
            if day == 0:
                week_values.append(None)
                week_text.append("")
            else:
                amount = spend_by_day.get(day, 0.0)
                week_values.append(amount)
                week_text.append(f"Day {day}<br>{currency} {amount:.2f}")
        z_values.append(week_values)
        text_values.append(week_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            y=[f"Week {index + 1}" for index in range(len(month_matrix))],
            text=text_values,
            hovertemplate="%{text}<extra></extra>",
            colorscale="YlGnBu",
            colorbar={"title": f"Spend ({currency})"},
            zmin=0,
        )
    )
    fig.update_layout(
        margin=dict(t=30, b=10),
        xaxis_title="Day of week",
        yaxis_title="Week of month",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


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
    st.header("Upload receipt history")
    receipt_history_file = st.file_uploader(
        "Import a spending history CSV",
        type=["csv"],
        key=f"receipt_history_file_{widget_seed}",
        disabled=not st.session_state.categories_approved,
    )
    import_history_clicked = st.button(
        "Import receipt history",
        use_container_width=True,
        disabled=not (st.session_state.categories_approved and receipt_history_file),
    )
    if not st.session_state.categories_approved:
        st.caption("Approve categories to enable receipt history import.")
    elif import_history_clicked and receipt_history_file is not None:
        try:
            imported_entries = parse_uploaded_spending_history(receipt_history_file)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state.receipt_history.extend(imported_entries)
            clear_history_suggestions()
            st.success(f"Imported {len(imported_entries)} receipt(s) from CSV.")

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


def render_receipt_editor(index: int, entry: dict, categories: list[str]) -> None:
    data = entry["data"]
    editable_categories = list(
        dict.fromkeys(categories + [str(item.get("category", "")).strip() for item in data.get("items", []) if str(item.get("category", "")).strip()])
    )
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
                    "category": st.column_config.SelectboxColumn("Category", options=editable_categories),
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
            cleaned_items, errors = normalize_edited_items(edited_items_df, editable_categories)
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
sync_category_budgets(categories)

# ── Main tabs ─────────────────────────────────────────────────────────────────
st.divider()
tab_chat, tab_dash, tab_suggestions, tab_budget, tab_search = st.tabs(
    ["📨 Receipts", "📊 Dashboard", "💡 Suggestions", "💸 Budget", "🔍 Receipt Search"]
)

# ── Tab 1: Chat interface ─────────────────────────────────────────────────────
with tab_chat:
    # Render history
    for index, entry in enumerate(st.session_state.receipt_history):
        with st.chat_message("user"):
            if entry.get("image_bytes"):
                st.image(entry["image_bytes"], width=260)
            else:
                st.info("Imported from CSV")
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

# ── Tab 4: Budget ─────────────────────────────────────────────────────────────
with tab_budget:
    history = st.session_state.receipt_history
    budget_source = build_budget_source_data(history)
    receipts_df = budget_source["receipts_df"]
    ensure_selected_budget_month(receipts_df)

    st.subheader("Monthly budget vs actual")
    st.caption("Set one shared monthly budget per category and compare it against any month of receipt history.")

    selected_budget_month = st.session_state["selected_budget_month"]
    selected_year = int(selected_budget_month[:4])
    selected_month_number = int(selected_budget_month[5:7])

    parseable_years = (
        {int(value.year) for value in receipts_df["parsed_date"]}
        if not receipts_df.empty
        else set()
    )
    year_options = sorted(
        parseable_years | {pd.Timestamp.today().year, selected_year},
        reverse=True,
    )

    picker_col, month_col = st.columns(2)
    selected_year = picker_col.selectbox(
        "Year",
        options=year_options,
        index=year_options.index(selected_year),
        key="budget_year_picker",
    )
    selected_month_number = month_col.selectbox(
        "Month",
        options=list(range(1, 13)),
        index=selected_month_number - 1,
        format_func=lambda month: calendar.month_name[month],
        key="budget_month_picker",
    )
    selected_budget_month = f"{selected_year}-{selected_month_number:02d}"
    st.session_state["selected_budget_month"] = selected_budget_month

    st.markdown("#### Shared monthly budget by category")
    st.caption("These amounts are reused for every month you select in this tab.")

    category_budgets = st.session_state["category_budgets"]
    with st.form("budget_amounts_form"):
        budget_inputs: dict[str, float] = {}
        columns = st.columns(2)
        for index, category in enumerate(categories):
            budget_inputs[category] = columns[index % 2].number_input(
                category,
                min_value=0.0,
                value=float(category_budgets.get(category, 0.0)),
                step=1.0,
                format="%.2f",
                key=f"budget_input_{category}",
            )
        budget_saved = st.form_submit_button(
            "Save budget amounts",
            type="primary",
            use_container_width=True,
        )

    if budget_saved:
        category_budgets = {
            category: round(float(amount), 2)
            for category, amount in budget_inputs.items()
        }
        st.session_state["category_budgets"] = category_budgets
    else:
        category_budgets = st.session_state["category_budgets"]

    analytics = build_month_budget_analytics(
        budget_source,
        category_budgets,
        selected_budget_month,
    )
    month_label = analytics["month_period"].start_time.strftime("%B %Y")
    usage_pct = (
        (analytics["total_actual"] / analytics["total_budget"]) * 100
        if analytics["total_budget"] > 0
        else 0.0
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("💼 Total budget", f"{analytics['currency']} {analytics['total_budget']:.2f}")
    k2.metric("💸 Actual spend", f"{analytics['currency']} {analytics['total_actual']:.2f}")
    k3.metric("📈 Budget used", f"{usage_pct:.1f}%")

    if budget_source["skipped_receipts"] > 0:
        st.warning(
            f"{budget_source['skipped_receipts']} receipt(s) were excluded from budget charts because their dates were missing or could not be parsed."
        )

    if not history:
        st.info("No receipts analyzed yet. Analyze some receipts in the **Receipts** tab first.")
    elif receipts_df.empty:
        st.info("No dated receipts are available for budget analysis yet. Upload or edit receipts with recognizable dates to unlock the charts.")
    elif analytics["month_receipts_df"].empty:
        st.info(f"No dated receipts are available for {month_label}. Pick another month or add receipts with dates in that month.")
    else:
        category_df = analytics["category_df"].copy()
        budget_vs_actual_df = category_df.melt(
            id_vars="Category",
            value_vars=["Budget", "Actual"],
            var_name="Type",
            value_name="Amount",
        )

        st.divider()
        st.subheader(f"Budget analysis for {month_label}")

        fig_budget_vs_actual = px.bar(
            budget_vs_actual_df,
            x="Category",
            y="Amount",
            color="Type",
            barmode="group",
            labels={"Amount": f"Amount ({analytics['currency']})"},
            color_discrete_map={"Budget": "#94a3b8", "Actual": "#16a34a"},
        )
        fig_budget_vs_actual.update_layout(margin=dict(t=20, b=60))
        fig_budget_vs_actual.update_xaxes(tickangle=-25)
        st.plotly_chart(fig_budget_vs_actual, use_container_width=True)

        st.divider()

        fig_stacked = px.bar(
            analytics["stacked_df"],
            x="Scenario",
            y="Amount",
            color="Category",
            labels={"Amount": f"Amount ({analytics['currency']})"},
            color_discrete_sequence=px.colors.qualitative.Safe,
            barmode="stack",
        )
        fig_stacked.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_stacked, use_container_width=True)

        st.divider()

        line_df = analytics["daily_df"].melt(
            id_vars=["Date"],
            value_vars=["Ideal Cumulative", "Actual Cumulative"],
            var_name="Series",
            value_name="Amount",
        )
        fig_line = px.line(
            line_df,
            x="Date",
            y="Amount",
            color="Series",
            labels={"Amount": f"Amount ({analytics['currency']})"},
            color_discrete_map={
                "Ideal Cumulative": "#94a3b8",
                "Actual Cumulative": "#2563eb",
            },
        )
        fig_line.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_line, use_container_width=True)

        st.divider()

        pie_df = category_df[category_df["Actual"] > 0][["Category", "Actual"]]
        if not pie_df.empty:
            fig_pie = px.pie(
                pie_df,
                names="Category",
                values="Actual",
                color_discrete_sequence=px.colors.qualitative.Safe,
                hole=0.35,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No actual spending was recorded in this month, so the spending breakdown pie chart is empty.")

        st.divider()

        fig_variance = px.bar(
            category_df,
            x="Category",
            y="Variance",
            color="Variance Status",
            labels={"Variance": f"Variance ({analytics['currency']})"},
            color_discrete_map={
                "Overspent": "#dc2626",
                "Within budget": "#16a34a",
            },
        )
        fig_variance.add_hline(y=0, line_dash="dash", line_color="#475569")
        fig_variance.update_layout(margin=dict(t=20, b=60), showlegend=False)
        fig_variance.update_xaxes(tickangle=-25)
        st.plotly_chart(fig_variance, use_container_width=True)

        st.divider()

        gauge_max = max(100.0, usage_pct + 10.0) if analytics["total_budget"] > 0 else 100.0
        gauge_title = "Total spending vs budget"
        if analytics["total_budget"] == 0:
            gauge_title = "Set a budget to track total usage"
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=usage_pct if analytics["total_budget"] > 0 else 0.0,
                number={"suffix": "%"},
                title={"text": gauge_title},
                gauge={
                    "axis": {"range": [0, gauge_max]},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [0, min(100.0, gauge_max)], "color": "#dbeafe"},
                        {"range": [100.0, gauge_max], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#dc2626", "width": 4},
                        "thickness": 0.75,
                        "value": 100.0,
                    },
                },
            )
        )
        fig_gauge.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()

        fig_heatmap = build_budget_heatmap_figure(
            analytics["month_period"],
            analytics["daily_df"],
            analytics["currency"],
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

# ── Tab 5: Receipt Search (RAG) ───────────────────────────────────────────────
with tab_search:
    history = st.session_state.receipt_history

    import google.generativeai as genai

    # Make sure you have your API key set up as an environment variable
    # or pass it to the client constructor.
    client = genai.Client()

    for model in client.models.list():
    print(model.name)


    if not history:
        st.info("No receipts yet. Analyze some receipts in the **Receipts** tab first.")
    else:
        # Lazy build / rebuild when receipt count changes
        if (
            st.session_state.vector_store is None
            or st.session_state.rag_receipt_count != len(history)
        ):
            with st.spinner(f"Indexing {len(history)} receipt(s)…"):
                store = ReceiptVectorStore()
                first_error: str | None = None
                for receipt in history:
                    try:
                        store.add_receipt(receipt)
                    except Exception as e:
                        if first_error is None:
                            first_error = str(e)

            if store.ntotal == 0:
                st.error(
                    f"Could not index any receipts. "
                    f"Error: {first_error or 'Unknown error'}. "
                    "Check that your API key has access to the Gemini embedding model "
                )
                # Reset so next render retries
                st.session_state.vector_store = None
                st.session_state.rag_receipt_count = 0
                st.stop()

            st.session_state.vector_store = store
            st.session_state.rag_receipt_count = len(history)
            if first_error:
                st.warning(f"Some receipts could not be indexed: {first_error}")

        vector_store = st.session_state.vector_store

        st.markdown("#### Ask about your receipts")

        # Render existing chat history
        for msg in st.session_state.rag_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        question = st.chat_input("E.g. How much did I spend on groceries?")
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.rag_chat_history.append({"role": "user", "content": question})

            chosen_model = pick_supported_model() or "models/gemini-2.0-flash"
            llm = genai.GenerativeModel(chosen_model)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = answer_question(question, vector_store, llm)
                    except Exception as exc:
                        answer = f"Error generating answer: {exc}"
                st.markdown(answer)
            st.session_state.rag_chat_history.append({"role": "assistant", "content": answer})

        # Clear button
        if st.session_state.rag_chat_history:
            if st.button("Clear chat history", use_container_width=False):
                st.session_state.rag_chat_history = []
                st.rerun()
