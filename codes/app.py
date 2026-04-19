"""
app.py — Streamlit frontend for the Smart Personal Finance Assistant
COMP8420 Assignment 2 — Large Language Models

Supports:
- Mock Mode: stable rule-based personalised responses
- TinyLlama (Selected Model): real open-source LLM inference
- English and Bangla language output
"""

import streamlit as st
from dialogue_manager import DialogueManager
from utils import safe_float

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Personal Finance Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — Settings and Model Info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Assistant Settings")
    st.divider()

    # --- Model selection ---
    st.markdown("### 🤖 Backend Model")

    mode_options = ["Mock Mode", "TinyLlama (Selected Model)"]
    mode_labels = {
        "Mock Mode": "📋 Mock Mode",
        "TinyLlama (Selected Model)": "🦙 TinyLlama (Selected Model)",
    }

    selected_mode = st.radio(
        "Choose backend mode",
        options=mode_options,
        format_func=lambda x: mode_labels[x],
        index=0,
        help=(
            "Mock Mode: instant, reliable rule-based responses — best for demos.\n\n"
            "TinyLlama: live LLM inference using the selected open-source model. "
            "Requires model to be downloaded. Output quality may vary."
        ),
    )

    if selected_mode == "Mock Mode":
        st.info(
            "**Mock Mode** generates structured responses using rule-based logic. "
            "Fast, stable, and reliable for all tasks.",
            icon="📋",
        )
    else:
        st.info(
            "**TinyLlama/TinyLlama-1.1B-Chat-v1.0** was selected after comparative evaluation "
            "against FLAN-T5. It runs locally via Hugging Face Transformers. "
            "First run requires model download (~2 GB). Output quality may vary "
            "on CPU-only systems.",
            icon="🦙",
        )

    st.divider()

    # --- Language selection ---
    st.markdown("### 🌐 Output Language")

    language_options = {
        "English": "English 🇬🇧",
        "Bangla": "বাংলা 🇧🇩",
    }
    selected_language = st.selectbox(
        "Choose language",
        options=list(language_options.keys()),
        format_func=lambda x: language_options[x],
        index=0,
        help=(
            "English: full structured output.\n\n"
            "Bangla (বাংলা): output is generated with Bangla-oriented text. "
            "TinyLlama may produce mixed English/Bangla output for small models — "
            "this is expected behaviour."
        ),
    )

    st.divider()

    # --- Task selection ---
    st.markdown("### 📋 Financial Task")

    task_options = [
        "Budget Planning",
        "Savings Strategy",
        "Debt Management",
        "Beginner Investment Guidance",
    ]
    selected_task = st.selectbox(
        "Select task type",
        options=task_options,
        index=0,
        help="Choose the type of financial guidance you need. Each task uses a dedicated prompt strategy.",
    )

    st.divider()
    st.markdown(
        "<small style='color: #888;'>COMP8420 Assignment 2 — LLM Finance Assistant<br>"
        "Model selected via comparative evaluation notebook.</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.markdown("# 💰 Smart Personal Finance Assistant")
st.markdown(
    "Enter your financial profile below to receive personalised advice. "
    "This assistant uses a selected open-source LLM backend evaluated on "
    "finance-specific test cases."
)
st.divider()

# ---------------------------------------------------------------------------
# Financial profile form
# ---------------------------------------------------------------------------

with st.form("finance_form", clear_on_submit=False):
    st.markdown("### 👤 Your Financial Profile")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=16,
            max_value=85,
            value=28,
            step=1,
            help="Your current age.",
        )

        employment_status = st.selectbox(
            "Employment Status",
            options=["Full-time employed", "Part-time employed", "Self-employed / Freelance",
                     "Student", "Unemployed", "Retired"],
            index=0,
            help="Your current employment situation.",
        )

        monthly_income = st.number_input(
            "Monthly Income (AUD $)",
            min_value=0,
            max_value=100_000,
            value=4_500,
            step=100,
            help="Your total monthly take-home income after tax.",
        )

        monthly_expenses = st.number_input(
            "Monthly Expenses (AUD $)",
            min_value=0,
            max_value=100_000,
            value=3_200,
            step=100,
            help="Your total monthly spending including rent, food, transport, and subscriptions.",
        )

    with col2:
        current_savings = st.number_input(
            "Current Savings (AUD $)",
            min_value=0,
            max_value=10_000_000,
            value=8_000,
            step=500,
            help="Your total savings across all accounts.",
        )

        current_debt = st.number_input(
            "Current Debt (AUD $)",
            min_value=0,
            max_value=10_000_000,
            value=5_000,
            step=500,
            help="Total outstanding debt (credit cards, personal loans, student loans, etc.).",
        )

        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Low", "Medium", "High"],
            value="Medium",
            help="How comfortable are you with investment risk and short-term losses?",
        )

        investment_horizon = st.selectbox(
            "Investment Horizon",
            options=[
                "Short (less than 2 years)",
                "Medium (2-5 years)",
                "Long (more than 5 years)",
            ],
            index=1,
            help="How long are you planning to keep funds invested or saved?",
        )

    st.markdown("#### 🎯 Goals & Preferences")

    col3, col4 = st.columns(2)

    with col3:
        financial_goal = st.text_input(
            "Financial Goal",
            value="Build an emergency fund and start investing",
            max_chars=150,
            help="Describe your main financial goal (e.g. 'buy a house in 5 years', 'pay off credit card debt').",
        )

    with col4:
        extra_preferences = st.text_input(
            "Extra Preferences (optional)",
            value="",
            max_chars=200,
            placeholder="e.g. prefer ethical investing, avoid interest-based products, have 2 dependants",
            help="Any preferences or constraints — ethical investing, family responsibilities, specific products to avoid.",
        )

    # --- Compute live snapshot preview ---
    income_val = safe_float(monthly_income)
    expenses_val = safe_float(monthly_expenses)
    surplus_val = income_val - expenses_val

    st.markdown("#### 📊 Financial Snapshot Preview")
    snap_col1, snap_col2, snap_col3, snap_col4 = st.columns(4)
    snap_col1.metric("Monthly Income", f"${income_val:,.0f}")
    snap_col2.metric("Monthly Expenses", f"${expenses_val:,.0f}")
    delta_label = f"${surplus_val:,.0f}/mo"
    snap_col3.metric(
        "Monthly Surplus",
        delta_label,
        delta=f"${surplus_val:,.0f}" if surplus_val >= 0 else f"-${abs(surplus_val):,.0f}",
        delta_color="normal" if surplus_val >= 0 else "inverse",
    )
    snap_col4.metric("Current Debt", f"${safe_float(current_debt):,.0f}")

    st.markdown("")
    submitted = st.form_submit_button(
        "💡 Generate Financial Advice",
        use_container_width=True,
        type="primary",
    )

# ---------------------------------------------------------------------------
# Result handling
# ---------------------------------------------------------------------------

if submitted:
    profile = {
        "age": age,
        "employment_status": employment_status,
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "current_savings": current_savings,
        "current_debt": current_debt,
        "risk_tolerance": risk_tolerance,
        "investment_horizon": investment_horizon,
        "financial_goal": financial_goal,
        "extra_preferences": extra_preferences if extra_preferences.strip() else "None stated",
    }

    # --- Validation ---
    if monthly_income <= 0:
        st.error("⚠️ Please enter a valid monthly income greater than zero.")
        st.stop()

    if monthly_expenses < 0:
        st.error("⚠️ Monthly expenses cannot be negative.")
        st.stop()

    if not financial_goal.strip():
        st.warning("💡 Tip: Adding a specific financial goal gives more personalised advice.")

    # --- Generate response ---
    with st.spinner(
        "🤖 TinyLlama is generating your advice… This may take 30–90 seconds on first run."
        if selected_mode == "TinyLlama (Selected Model)"
        else "🔄 Generating your personalised financial advice…"
    ):
        try:
            manager = DialogueManager()
            result = manager.generate_response(
                profile=profile,
                task_type=selected_task,
                mode=selected_mode,
                language=selected_language,
            )
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            st.stop()

    # --- Error check from backend ---
    if not result or isinstance(result, dict) and "error" in result:
        error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else "No result returned."
        st.error(f"❌ The assistant encountered an issue: {error_msg}")
        st.stop()

    # --- Display results ---
    st.divider()
    st.markdown("## 📋 Your Personalised Financial Advice")

    lang_note = " (বাংলা)" if selected_language == "Bangla" else ""
    st.caption(
        f"Task: **{selected_task}**{lang_note}  ·  "
        f"Mode: **{selected_mode}**  ·  "
        f"Language: **{selected_language}**"
    )

    # --- Recommendation ---
    recommendation = result.get("recommendation", "")
    if recommendation:
        st.markdown("### 💡 Recommendation")
        st.markdown(recommendation)
    else:
        st.warning("No recommendation was returned. Please try again.")

    st.divider()

    # --- Action Steps ---
    action_steps = result.get("action_steps", [])
    st.markdown("### ✅ Action Steps")

    if action_steps and isinstance(action_steps, list) and len(action_steps) > 0:
        useful_steps = [
            step for step in action_steps
            if isinstance(step, str) and step.strip() and len(step.strip().split()) >= 3
        ]
        if useful_steps:
            for i, step in enumerate(useful_steps, 1):
                st.markdown(f"**{i}.** {step.strip()}")
        else:
            st.info("Action steps were generated but may need review. Please check the recommendation text above for guidance.")
    elif isinstance(action_steps, str) and action_steps.strip():
        for line in action_steps.strip().splitlines():
            line = line.strip()
            if line:
                st.markdown(f"• {line}")
    else:
        st.info(
            "No structured action steps were returned. "
            "Please refer to the recommendation text for guidance."
        )

    st.divider()

    # --- Two-column layout for explanation and prompt ---
    res_col1, res_col2 = st.columns([3, 2])

    with res_col1:
        explanation = result.get("explanation", "")
        st.markdown("### 📖 Explanation")
        if explanation and len(explanation.strip().split()) >= 5:
            st.markdown(explanation)
        else:
            st.info("Explanation was not available in this response.")

    with res_col2:
        prompt_data = result.get("prompt_used") if isinstance(result, dict) else None
        with st.expander("🔍 Prompt Preview (Assignment Evidence)", expanded=False):
            if prompt_data:
                st.markdown("**Task Type:**")
                st.code(prompt_data.get("task_type", selected_task), language=None)

                st.markdown("**Zero-Shot Prompt:**")
                st.code(prompt_data.get("zero_shot", "N/A"), language=None)

                st.markdown("**Structured Prompt:**")
                st.code(prompt_data.get("structured", "N/A"), language=None)

                if prompt_data.get("few_shot"):
                    st.markdown("**Few-Shot Prompt:**")
                    st.code(prompt_data.get("few_shot", "N/A"), language=None)
            else:
                task_display = selected_task
                income_disp = f"${safe_float(monthly_income):,.0f}"
                expenses_disp = f"${safe_float(monthly_expenses):,.0f}"
                surplus_disp = f"${surplus_val:,.0f}"
                st.markdown("**Effective Prompt (reconstructed):**")
                st.code(
                    f"Task: {task_display}\n"
                    f"Profile: age {age}; employment {employment_status}; "
                    f"income {income_disp}/month; expenses {expenses_disp}/month; "
                    f"surplus {surplus_disp}/month; risk {risk_tolerance}; "
                    f"goal: {financial_goal}",
                    language=None,
                )
            st.caption("Prompt preview is included as assignment evidence of LLM prompting strategy.")

    st.divider()

    # --- Disclaimer ---
    disclaimer = result.get("disclaimer", "")
    if disclaimer and disclaimer.strip():
        st.markdown("### ⚠️ Disclaimer")
        st.warning(disclaimer)
    else:
        st.warning(
            "This information is for educational purposes only and is not personalised financial advice. "
            "Please consult a registered financial adviser before making financial decisions."
        )

    # --- Assignment footer note ---
    st.divider()
    st.caption(
        "🎓 **COMP8420 Assignment 2** — Smart Personal Finance Assistant  "
        "| TinyLlama/TinyLlama-1.1B-Chat-v1.0 selected as backend after comparative evaluation  "
        "| Hugging Face Transformers · Streamlit · Python"
    )