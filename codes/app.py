"""
app.py — Streamlit frontend for SmartFinance AI Assistant
COMP8420 Assignment 2: Large Language Models
Scaffold v1.0 — Initial functional version
"""

import streamlit as st
from dialogue_manager import run_financial_dialogue
from utils import validate_profile, format_currency, compute_surplus, compute_savings_rate

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SmartFinance AI Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — Settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    model_mode = st.selectbox(
        "🤖 Model Mode",
        options=["Mock Mode", "Local Model Placeholder"],
        help=(
            "Mock Mode returns a structured demo response based on your profile. "
            "Local Model Placeholder is reserved for future LLM integration."
        ),
    )

    language = st.selectbox(
        "🌐 Language",
        options=["English", "Bangla"],
        help="Select the language for the assistant's response.",
    )

    task_type = st.selectbox(
        "📋 Financial Task",
        options=[
            "Budget Planning",
            "Savings Strategy",
            "Debt Management",
            "Beginner Investment Guidance",
        ],
        help="Choose the type of financial guidance you need.",
    )

    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.markdown(
        "**COMP8420 — Assignment 2**  \n"
        "Smart Personal Finance Assistant  \n"
        "*Scaffold v1.0*"
    )
    st.markdown(
        "This tool collects your financial profile and generates "
        "personalised recommendations using an LLM backend."
    )

# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------

st.title("💰 SmartFinance AI Assistant")
st.markdown(
    "Enter your financial details below to receive a tailored recommendation. "
    "All data is processed locally and is not stored or transmitted anywhere."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Input Form
# ---------------------------------------------------------------------------

with st.form(key="finance_form"):
    st.subheader("📋 Your Financial Profile")
    st.caption(
        "Fill in the fields as accurately as possible. "
        "The quality of the recommendation depends on the accuracy of your inputs."
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**Personal Details**")
        age = st.number_input("Age", min_value=16, max_value=100, value=25, step=1)
        employment_status = st.selectbox(
            "Employment Status",
            options=[
                "Full-time Employed", "Part-time Employed", "Self-employed",
                "Student", "Unemployed", "Retired",
            ],
        )

        st.markdown("**Monthly Finances (AUD)**")
        monthly_income = st.number_input(
            "Monthly Income",
            min_value=0.0, value=4000.0, step=100.0,
            help="Total take-home income per month after tax.",
        )
        monthly_expenses = st.number_input(
            "Monthly Expenses",
            min_value=0.0, value=2500.0, step=100.0,
            help="Total monthly spending (rent, food, transport, bills, etc.).",
        )
        current_savings = st.number_input(
            "Current Savings",
            min_value=0.0, value=5000.0, step=500.0,
            help="Total amount currently in savings or accessible funds.",
        )

    with col2:
        st.markdown("**Debt & Risk**")
        current_debt = st.number_input(
            "Current Debt (AUD)",
            min_value=0.0, value=0.0, step=500.0,
            help="Total outstanding debt (credit cards, personal loans, student loans, etc.).",
        )
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            options=["Low", "Medium", "High"],
            help="Low = prefer safe options. Medium = balanced. High = comfortable with volatility.",
        )

        st.markdown("**Goals & Preferences**")
        financial_goal = st.text_input(
            "Financial Goal",
            placeholder="e.g., Save for a house deposit, pay off student loans...",
        )
        investment_horizon = st.selectbox(
            "Investment Horizon",
            options=["Short (< 2 years)", "Medium (2–5 years)", "Long (5+ years)"],
            help="How long before you need the money?",
        )
        extra_preferences = st.text_area(
            "Extra Preferences or Constraints",
            placeholder=(
                "e.g., No interest-based products, prefer ETFs, "
                "have 2 dependants, already have a mortgage..."
            ),
            height=112,
        )

    st.markdown("---")
    submitted = st.form_submit_button(
        "🚀 Generate Financial Advice", use_container_width=True
    )

# ---------------------------------------------------------------------------
# On Submit — Validate, Process, Display
# ---------------------------------------------------------------------------

if submitted:
    raw_profile = {
        "age": age,
        "employment_status": employment_status,
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "current_savings": current_savings,
        "current_debt": current_debt,
        "risk_tolerance": risk_tolerance,
        "financial_goal": financial_goal.strip(),
        "investment_horizon": investment_horizon,
        "extra_preferences": extra_preferences.strip(),
    }

    errors = validate_profile(raw_profile)
    if errors:
        for err in errors:
            st.error(f"⚠️ {err}")
        st.stop()

    with st.spinner("🤔 Analysing your financial profile…"):
        result = run_financial_dialogue(
            profile=raw_profile,
            task_type=task_type,
            mode=model_mode,
            language=language,
        )

    if "error" in result:
        st.error(f"Something went wrong: {result['error']}")
        st.stop()

    st.success("✅ Your personalised financial advice is ready!")
    st.markdown("---")

    # --- Profile Snapshot ---
    st.subheader("👤 Your Financial Snapshot")
    surplus      = compute_surplus(monthly_income, monthly_expenses)
    savings_rate = compute_savings_rate(monthly_income, monthly_expenses)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Monthly Income",   format_currency(monthly_income))
    col_b.metric("Monthly Expenses", format_currency(monthly_expenses))
    col_c.metric(
        "Monthly Surplus",
        format_currency(surplus),
        delta=f"{savings_rate:.1f}% savings rate",
        delta_color="normal" if surplus >= 0 else "inverse",
    )
    col_d.metric("Current Savings",  format_currency(current_savings))

    col_e, col_f, col_g, col_h = st.columns(4)
    col_e.metric("Current Debt",       format_currency(current_debt))
    col_f.metric("Risk Tolerance",     risk_tolerance)
    col_g.metric("Horizon",            investment_horizon.split("(")[0].strip())
    col_h.metric("Task",               task_type)

    st.markdown("---")

    # --- Recommendation ---
    st.subheader(f"💡 {task_type} Recommendation")
    st.markdown(result.get("recommendation", "_No recommendation available._"))
    st.markdown("---")

    # --- Action Steps ---
    st.subheader("✅ Action Steps")
    action_steps = result.get("action_steps", [])
    if isinstance(action_steps, list) and action_steps:
        for i, step in enumerate(action_steps, start=1):
            st.markdown(f"**{i}.** {step}")
    else:
        st.markdown(str(action_steps) or "_No action steps provided._")

    st.markdown("---")

    # --- Explanation ---
    st.subheader("🔍 Why This Advice Fits You")
    st.info(result.get("explanation", "_No explanation available._"))

    # --- Prompt Inspector (assignment evidence) ---
    with st.expander("🔧 View Generated Prompt  (LLM comparison / assignment evidence)"):
        from prompt_templates import build_structured_prompt
        preview = build_structured_prompt(raw_profile, task_type, language)
        st.markdown("**System Prompt:**")
        st.code(preview.get("system", ""), language="text")
        st.markdown("**User Prompt:**")
        st.code(preview.get("user", ""),   language="text")

    # --- Disclaimer ---
    st.markdown("---")
    st.warning(result.get(
        "disclaimer",
        "⚠️ This is AI-generated advice for educational purposes only. "
        "Please consult a qualified financial adviser before making any decisions.",
    ))