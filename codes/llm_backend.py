"""
llm_backend.py — LLM response generation for SmartFinance AI Assistant

Supports two modes:
- Mock Mode: structured, personalised demo response based on user profile data.
- TinyLlama (Selected Model): real TinyLlama inference path with safe fallback.

Main entry point: generate_financial_response(profile, prompt, mode, language)
"""

import re

from utils import (
    safe_float,
    format_currency,
    compute_surplus,
    compute_savings_rate,
    compute_debt_to_income_ratio,
)



try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except Exception:
    Translator = None
    GOOGLETRANS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional TinyLlama dependencies
# ---------------------------------------------------------------------------

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    TRANSFORMERS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Selected model from comparative evaluation
# ---------------------------------------------------------------------------

SELECTED_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

_TOKENIZER = None
_MODEL = None
_MODEL_LOAD_ERROR = None


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------

def _safe_text(value, default: str = "") -> str:
    """Return a clean string without raising exceptions."""
    try:
        text = str(value).strip()
        return text if text else default
    except Exception:
        return default


def _format_amount(value) -> str:
    """Format numeric values for prompt injection."""
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"


def _clean_generated_text(text: str) -> str:
    """Remove common junk that small models often echo back."""
    if not text:
        return ""

    cleaned = text.replace("<|assistant|>", " ").replace("<|user|>", " ").replace("<|system|>", " ")
    cleaned = re.sub(r"\[[^\]]{1,80}\]", "", cleaned)
    cleaned = re.sub(r"\{[^\}]{1,80}\}", "", cleaned)
    cleaned = re.sub(r"(?i)^instructions:\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"(?i)use this exact output structure:.*", "", cleaned)
    cleaned = re.sub(r"(?i)do not use placeholders.*", "", cleaned)
    cleaned = re.sub(r"(?i)do not repeat the profile.*", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _strip_label(line: str) -> str:
    return re.sub(r"^[\-\•\*\d\.\)\s]+", "", line).strip()


def _ensure_disclaimer_text(language: str) -> str:
    if language == "Bangla":
        return (
            "এই তথ্য কেবল শিক্ষামূলক উদ্দেশ্যে দেওয়া হয়েছে। ব্যক্তিগত আর্থিক সিদ্ধান্ত নেওয়ার আগে "
            "একজন নিবন্ধিত আর্থিক উপদেষ্টার সঙ্গে পরামর্শ করুন।"
        )
    return (
        "This information is for educational purposes only and is not personalised financial advice. "
        "Please consult a registered financial adviser before making financial decisions."
    )



def _translate_result_to_bangla(result: dict) -> dict:
    """
    Translate a structured English result dict into Bangla.
    Falls back safely if translation is unavailable.
    """
    if not GOOGLETRANS_AVAILABLE:
        return _build_bangla_mock_wrapper(result)

    try:
        translator = Translator()
        translated = dict(result)

        translated["recommendation"] = translator.translate(
            result.get("recommendation", ""), dest="bn"
        ).text

        translated["action_steps"] = [
            translator.translate(step, dest="bn").text
            for step in result.get("action_steps", [])
            if isinstance(step, str) and step.strip()
        ]

        translated["explanation"] = translator.translate(
            result.get("explanation", ""), dest="bn"
        ).text

        translated["disclaimer"] = translator.translate(
            result.get("disclaimer", ""), dest="bn"
        ).text

        return translated

    except Exception:
        return _build_bangla_mock_wrapper(result)




def _build_bangla_mock_wrapper(result: dict) -> dict:
    """Add a Bangla-oriented wrapper without relying on external translation."""
    wrapped = dict(result)
    wrapped["recommendation"] = (
        "**বাংলা মোড নোট:** সম্পূর্ণ বাংলা উত্তর সবসময় স্থিতিশীল নাও হতে পারে। নিচে ব্যবহারযোগ্য আর্থিক পরামর্শ দেওয়া হলো.\n\n"
        + wrapped.get("recommendation", "")
    )
    wrapped["explanation"] = (
        "এই ব্যাখ্যাটি আপনার আয়, ব্যয়, সঞ্চয়, ঋণ, লক্ষ্য এবং ঝুঁকি সহনশীলতার ভিত্তিতে তৈরি করা হয়েছে। "
        + wrapped.get("explanation", "")
    )
    wrapped["disclaimer"] = _ensure_disclaimer_text("Bangla")
    return wrapped


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_selected_model():
    """
    Lazily load TinyLlama only when needed.
    Prefers GPU when available and remains safe on CPU-only systems.
    """
    global _TOKENIZER, _MODEL, _MODEL_LOAD_ERROR

    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    if _MODEL_LOAD_ERROR is not None:
        raise RuntimeError(_MODEL_LOAD_ERROR)

    if not TRANSFORMERS_AVAILABLE:
        _MODEL_LOAD_ERROR = (
            "Transformers/PyTorch dependencies are not installed. "
            "Install: pip install transformers torch accelerate sentencepiece safetensors"
        )
        raise RuntimeError(_MODEL_LOAD_ERROR)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        _TOKENIZER = AutoTokenizer.from_pretrained(SELECTED_MODEL, use_fast=True)
        if _TOKENIZER.pad_token is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token
        _TOKENIZER.padding_side = "left"

        model_kwargs = {"torch_dtype": dtype}
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["low_cpu_mem_usage"] = True

        _MODEL = AutoModelForCausalLM.from_pretrained(SELECTED_MODEL, **model_kwargs)

        if device != "cuda":
            _MODEL = _MODEL.to(device)

        _MODEL.eval()

        if getattr(_TOKENIZER, "chat_template", None) is None:
            _TOKENIZER.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}\n{% endif %}"
                "{% if message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n{% endif %}"
                "{% if message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}\n{% endif %}"
                "{% endfor %}<|assistant|>\n"
            )

        return _TOKENIZER, _MODEL

    except Exception as e:
        _MODEL_LOAD_ERROR = f"Failed to load {SELECTED_MODEL}: {str(e)}"
        raise RuntimeError(_MODEL_LOAD_ERROR)


# ---------------------------------------------------------------------------
# Mock Mode — Task-specific response generators
# ---------------------------------------------------------------------------

def _budget_planning_mock(profile: dict) -> dict:
    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    surplus = compute_surplus(income, expenses)
    savings_rate = compute_savings_rate(income, expenses)
    goal = profile.get("financial_goal", "your financial goals")

    if savings_rate >= 20:
        framework = "50/30/20 rule (50% needs, 30% wants, 20% savings/investments)"
        budget_note = (
            f"Your current savings rate of {savings_rate:.1f}% is healthy. "
            "The 50/30/20 framework will help you maintain this discipline."
        )
    elif savings_rate >= 10:
        framework = "60/20/20 rule (60% needs, 20% wants, 20% savings)"
        budget_note = (
            f"Your current savings rate of {savings_rate:.1f}% is moderate. "
            "Tightening discretionary spending by 10% could significantly "
            "accelerate progress toward your goal."
        )
    else:
        framework = "Zero-based budgeting (every dollar assigned a purpose)"
        budget_note = (
            f"Your current savings rate is only {savings_rate:.1f}%, which leaves "
            "little room for saving or investing. Zero-based budgeting can help you "
            "identify and cut unnecessary expenses."
        )

    needs_target = income * 0.50
    wants_target = income * 0.30
    savings_target = income * 0.20

    recommendation = (
        f"Based on your monthly income of {format_currency(income)} and expenses of "
        f"{format_currency(expenses)}, your monthly surplus is {format_currency(surplus)}. "
        f"We recommend the **{framework}** to structure your budget and work towards "
        f"**{goal}**. {budget_note}"
    )

    action_steps = [
        "Track all spending for 30 days using a free budgeting app to get a clear picture of where your money goes.",
        f"Allocate {format_currency(needs_target)}/mo (50%) to essential needs (rent, food, transport, utilities).",
        f"Limit discretionary spending to {format_currency(wants_target)}/mo (30%).",
        f"Direct at least {format_currency(savings_target)}/mo (20%) to a dedicated savings or investment account on payday.",
        "Review and cancel unused subscriptions — a common source of budget leakage.",
        "Revisit your budget every month and adjust as income or expenses change.",
    ]

    health = "strong" if savings_rate >= 20 else "moderate" if savings_rate >= 10 else "limited"
    explanation = (
        f"The recommended budget framework is chosen because your monthly surplus of "
        f"{format_currency(surplus)} ({savings_rate:.1f}% of income) suggests {health} "
        f"financial headroom. A structured budget ensures consistent progress towards "
        f"{goal} while keeping lifestyle expenses in check."
    )

    return {
        "recommendation": recommendation,
        "action_steps": action_steps,
        "explanation": explanation,
    }


def _savings_strategy_mock(profile: dict) -> dict:
    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    savings = safe_float(profile.get("current_savings", 0))
    surplus = compute_surplus(income, expenses)
    goal = profile.get("financial_goal", "your savings goal")
    horizon = profile.get("investment_horizon", "Medium (2-5 years)")

    if "Long" in horizon:
        save_pct = 0.25
        vehicle = "a high-interest savings account for your emergency fund, then low-cost index ETFs"
    elif "Medium" in horizon:
        save_pct = 0.20
        vehicle = "a high-interest savings account and term deposits"
    else:
        save_pct = 0.15
        vehicle = "a dedicated high-interest savings account or offset account"

    monthly_save = surplus * save_pct if surplus > 0 else 0
    annual_save = monthly_save * 12
    emergency_fund_target = expenses * 3

    recommendation = (
        f"With a monthly surplus of {format_currency(surplus)}, we recommend saving at least "
        f"**{format_currency(monthly_save)}/month** ({save_pct * 100:.0f}% of surplus) into "
        f"{vehicle}. This will help you build toward {goal}. "
        f"Over 12 months, you could accumulate approximately {format_currency(annual_save)} "
        f"in new savings on top of your existing {format_currency(savings)}."
    )

    action_steps = [
        f"Build an emergency fund of {format_currency(emergency_fund_target)} (3 months of expenses) before investing.",
        "Open a dedicated savings account separate from your everyday spending account.",
        f"Set up an automatic transfer of {format_currency(monthly_save)} on payday ('pay yourself first').",
        f"Once the emergency fund is complete, direct additional savings to {vehicle}.",
        "Avoid dipping into savings for non-emergency purchases.",
        "Reassess your savings rate every 6 months as your income grows.",
    ]

    horizon_label = horizon.split("(")[0].strip().lower()
    explanation = (
        f"Your {horizon.lower()} investment horizon guides the choice of savings vehicle. "
        f"For a {horizon_label} horizon, liquidity and capital preservation are prioritised, "
        f"making {vehicle} appropriate. The {save_pct * 100:.0f}% target is calibrated to your "
        f"current surplus of {format_currency(surplus)}/month."
    )

    return {
        "recommendation": recommendation,
        "action_steps": action_steps,
        "explanation": explanation,
    }


def _debt_management_mock(profile: dict) -> dict:
    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    debt = safe_float(profile.get("current_debt", 0))
    surplus = compute_surplus(income, expenses)
    dti = compute_debt_to_income_ratio(debt, income)
    goal = profile.get("financial_goal", "become debt-free")

    if dti > 5:
        urgency, strategy, monthly_debt_pct = "high", "Debt Avalanche", 0.40
    elif dti > 2:
        urgency, strategy, monthly_debt_pct = "moderate", "Debt Snowball", 0.30
    else:
        urgency, strategy, monthly_debt_pct = "manageable", "balanced repayment", 0.20

    monthly_debt_payment = surplus * monthly_debt_pct if surplus > 0 else 0

    if debt == 0:
        recommendation = (
            "You currently have no reported debt — well done! "
            f"Focus on building your emergency fund and then working towards {goal}."
        )
        action_steps = [
            "Maintain a zero-debt policy by paying credit card balances in full each month.",
            "Redirect funds that would go to debt repayment into a savings or investment account.",
            "Build an emergency fund (3–6 months of expenses) to avoid future debt.",
        ]
        explanation = (
            "With no current debt, your financial foundation is solid. "
            "The focus should be on maintaining this position and building wealth steadily."
        )
    else:
        months_to_clear = (debt / monthly_debt_payment) if monthly_debt_payment > 0 else 999
        recommendation = (
            f"Your debt-to-income ratio is **{dti:.1f}x** monthly income, which is {urgency}. "
            f"We recommend the **{strategy}** method to work towards {goal}. "
            f"Allocating {format_currency(monthly_debt_payment)}/month to debt repayment "
            f"could clear your debt in approximately **{months_to_clear:.0f} months** "
            "(simplified estimate, excluding interest)."
        )
        action_steps = [
            "List all debts with their interest rates and minimum repayments.",
            "Pay the minimum on all debts every month to avoid penalties.",
            f"Allocate {format_currency(monthly_debt_payment)}/month as your extra repayment using the {strategy} approach.",
            "Contact lenders to ask about lower interest rates or hardship support if needed.",
            "Avoid taking on any new debt while paying down existing obligations.",
            "Once debt-free, redirect the repayment amount into savings or investments.",
        ]

        if urgency == "high":
            extra = "it minimises total interest paid over time."
        elif urgency == "moderate":
            extra = "it builds motivational momentum through early wins."
        else:
            extra = "your debt level allows a balanced approach without sacrificing savings."
        explanation = (
            f"A debt-to-income ratio of {dti:.1f}x monthly income indicates {urgency} debt pressure. "
            f"The {strategy} strategy is appropriate because {extra}"
        )

    return {
        "recommendation": recommendation,
        "action_steps": action_steps,
        "explanation": explanation,
    }


def _investment_guidance_mock(profile: dict) -> dict:
    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    debt = safe_float(profile.get("current_debt", 0))
    risk = profile.get("risk_tolerance", "Medium")
    horizon = profile.get("investment_horizon", "Medium (2-5 years)")
    goal = profile.get("financial_goal", "grow your wealth")
    age = safe_float(profile.get("age", 25))
    surplus = compute_surplus(income, expenses)

    risk_map = {
        "Low": {
            "vehicles": "high-interest savings accounts, government bonds, and term deposits",
            "allocation": "80% defensive assets and 20% growth assets",
            "note": "Capital preservation is the priority.",
        },
        "Medium": {
            "vehicles": "diversified ETFs and balanced funds",
            "allocation": "50% growth assets and 50% defensive assets",
            "note": "Balanced growth with moderate risk.",
        },
        "High": {
            "vehicles": "growth ETFs, diversified equity exposure, and growth-oriented listed funds",
            "allocation": "80% growth assets and 20% defensive assets",
            "note": "Higher potential returns come with higher short-term volatility.",
        },
    }

    selected_profile = risk_map.get(risk, risk_map["Medium"])
    monthly_inv = surplus * 0.20 if surplus > 0 else 0
    dti = compute_debt_to_income_ratio(debt, income)
    debt_warning = (
        " **Important:** Your current debt level suggests you should prioritise debt repayment before investing, "
        "as many debt interest rates exceed typical long-term investment returns."
        if dti > 3 else ""
    )

    recommendation = (
        f"As a beginner investor with **{risk.lower()} risk tolerance** and a "
        f"**{horizon.lower()} horizon**, we recommend starting with "
        f"**{selected_profile['vehicles']}**. "
        f"A suggested allocation is {selected_profile['allocation']}. "
        f"Consider investing {format_currency(monthly_inv)}/month consistently "
        f"to work towards {goal}.{debt_warning}"
    )

    action_steps = [
        "Ensure you have at least 3 months of expenses saved as an emergency fund before investing.",
        "Open a low-fee brokerage or investment account appropriate to your needs.",
        f"Start with {selected_profile['vehicles']} — beginner-friendly and diversified options.",
        f"Invest {format_currency(monthly_inv)}/month consistently using dollar-cost averaging.",
        f"Follow a {selected_profile['allocation']} portfolio approach.",
        "Reinvest distributions where appropriate to compound long-term returns.",
        "Review your portfolio every 6–12 months rather than reacting to short-term market moves.",
        "Use reputable educational resources before making major investment decisions.",
    ]

    age_note = "substantial" if age < 40 else "moderate"
    horizon_label = horizon.split("(")[0].strip().lower()
    explanation = (
        f"The recommended approach is tailored to your {risk.lower()} risk tolerance and "
        f"{horizon_label} investment horizon. At age {int(age)}, you have {age_note} time "
        f"to benefit from compounding. {selected_profile['note']} Starting with diversified "
        "investment options reduces single-asset risk, which is appropriate for a beginner investor."
    )

    return {
        "recommendation": recommendation,
        "action_steps": action_steps,
        "explanation": explanation,
    }


def _generate_mock_response(profile: dict, task_type: str, language: str) -> dict:
    """Route to the correct task-specific mock response generator."""
    generators = {
        "Budget Planning": _budget_planning_mock,
        "Savings Strategy": _savings_strategy_mock,
        "Debt Management": _debt_management_mock,
        "Beginner Investment Guidance": _investment_guidance_mock,
    }

    result = generators.get(task_type, _budget_planning_mock)(profile)
    result["disclaimer"] = (
        "This recommendation is generated for educational purposes as part of COMP8420 Assignment 2. "
        "It is not personalised financial advice. Please consult a registered financial adviser "
        "before making any financial decisions."
    )

    if language == "Bangla":
        result = _build_bangla_mock_wrapper(result)

    return result


# ---------------------------------------------------------------------------
# TinyLlama — Prompt builders
# ---------------------------------------------------------------------------

def _build_tinyllama_messages(prompt: dict, profile: dict, language: str) -> tuple:
    """
    Build concise, task-specific chat messages and a priming prefix for TinyLlama.
    The prompt is intentionally short to reduce instruction echoing.
    """
    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    savings = safe_float(profile.get("current_savings", 0))
    debt = safe_float(profile.get("current_debt", 0))
    surplus = compute_surplus(income, expenses)
    age = int(safe_float(profile.get("age", 25)))
    risk = _safe_text(profile.get("risk_tolerance", "Medium"), "Medium")
    goal = _safe_text(profile.get("financial_goal", "improve finances"), "improve finances")
    horizon = _safe_text(profile.get("investment_horizon", "Medium (2-5 years)"), "Medium (2-5 years)")
    employment = _safe_text(profile.get("employment_status", "Not stated"), "Not stated")
    extras = _safe_text(profile.get("extra_preferences", "None stated"), "None stated")
    task = _safe_text(prompt.get("task_type", "Budget Planning"), "Budget Planning")

    if language == "Bangla":
        language_instruction = (
            "Write the answer mostly in simple Bangla. Keep the section labels in English exactly as given. "
            "You may keep financial terms like budget, savings, debt, ETF, or risk in English if needed."
        )
    else:
        language_instruction = "Write the answer in clear English."

    system_prompt = (
        "You are a practical finance assistant. Use only the profile facts provided. "
        "Do not copy the prompt. Do not repeat the profile line by line. "
        "Do not use placeholders or square brackets. "
        "Write all four sections with real content. "
        "Make the advice specific, short, and useful."
    )

    user_content = (
        f"Task: {task}\n"
        f"Profile: age {age}; employment {employment}; income {_format_amount(income)}/month; "
        f"expenses {_format_amount(expenses)}/month; surplus {_format_amount(surplus)}/month; "
        f"savings {_format_amount(savings)}; debt {_format_amount(debt)}; risk {risk}; "
        f"goal {goal}; horizon {horizon}; extra preferences {extras}.\n"
        f"{language_instruction}\n"
        "Give practical personal finance advice with exactly these sections:\n"
        "RECOMMENDATION:\n"
        "ACTION STEPS:\n"
        "1.\n2.\n3.\n4.\n"
        "EXPLANATION:\n"
        "DISCLAIMER:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    priming = "RECOMMENDATION: Based on your current finances, "
    return messages, priming


def _build_tinyllama_repair_messages(prompt: dict, profile: dict, language: str) -> tuple:
    """
    Build a stricter repair prompt after a low-quality first pass.
    This prompt tells the model to rewrite the answer fully and correctly.
    """
    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    savings = safe_float(profile.get("current_savings", 0))
    debt = safe_float(profile.get("current_debt", 0))
    surplus = compute_surplus(income, expenses)
    age = int(safe_float(profile.get("age", 25)))
    risk = _safe_text(profile.get("risk_tolerance", "Medium"), "Medium")
    goal = _safe_text(profile.get("financial_goal", "improve finances"), "improve finances")
    horizon = _safe_text(profile.get("investment_horizon", "Medium (2-5 years)"), "Medium (2-5 years)")
    employment = _safe_text(profile.get("employment_status", "Not stated"), "Not stated")
    extras = _safe_text(profile.get("extra_preferences", "None stated"), "None stated")
    task = _safe_text(prompt.get("task_type", "Budget Planning"), "Budget Planning")

    if language == "Bangla":
        language_instruction = (
            "Rewrite the full answer mostly in simple Bangla. Keep section labels in English. "
            "Do not switch back to English except for short finance terms when necessary."
        )
    else:
        language_instruction = "Rewrite the full answer in clear English."

    system_prompt = (
        "The previous answer was low quality. Rewrite it fully. "
        "Use only the profile facts below. "
        "No placeholders. No copied instructions. No empty headings. "
        "Recommendation must be specific. Action steps must be useful. "
        "Explanation and disclaimer must be complete."
    )

    user_content = (
        f"Rewrite {task} advice for this profile:\n"
        f"age {age}; employment {employment}; income {_format_amount(income)}/month; "
        f"expenses {_format_amount(expenses)}/month; surplus {_format_amount(surplus)}/month; "
        f"savings {_format_amount(savings)}; debt {_format_amount(debt)}; risk {risk}; "
        f"goal {goal}; horizon {horizon}; extra preferences {extras}.\n"
        f"{language_instruction}\n"
        "Return all four sections only:\n"
        "RECOMMENDATION:\n"
        "ACTION STEPS:\n"
        "1.\n2.\n3.\n4.\n"
        "EXPLANATION:\n"
        "DISCLAIMER:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if language == "Bangla":
        priming = (
            f"RECOMMENDATION: আপনার মাসিক আয় {_format_amount(income)} এবং ব্যয় {_format_amount(expenses)} হওয়ায় "
        )
    else:
        priming = (
            f"RECOMMENDATION: Given your monthly income of {_format_amount(income)} and expenses of {_format_amount(expenses)}, "
        )

    return messages, priming


# ---------------------------------------------------------------------------
# TinyLlama — Output parser
# ---------------------------------------------------------------------------

def _parse_tinyllama_output(text: str, language: str = "English") -> dict:
    """
    Parse raw TinyLlama output into structured sections.
    Handles imperfect formatting and recovers useful content where possible.
    """
    text = _clean_generated_text(_safe_text(text))

    recommendation = ""
    action_steps = []
    explanation = ""
    disclaimer = ""

    section_pattern = re.compile(
        r"(?is)(RECOMMENDATION|ACTION\s*STEPS|EXPLANATION|DISCLAIMER)\s*:?\s*"
        r"(.*?)(?=(?:\n\s*(?:RECOMMENDATION|ACTION\s*STEPS|EXPLANATION|DISCLAIMER)\s*:)|\Z)"
    )

    sections = {}
    for label, content in section_pattern.findall(text):
        sections[label.upper().replace(" ", "_")] = content.strip()

    recommendation = sections.get("RECOMMENDATION", "")
    action_block = sections.get("ACTION_STEPS", "")
    explanation = sections.get("EXPLANATION", "")
    disclaimer = sections.get("DISCLAIMER", "")

    if action_block:
        for line in action_block.splitlines():
            line = _strip_label(line)
            if line and len(line.split()) >= 3 and "recommendation" not in line.lower():
                action_steps.append(line)

    if not action_steps:
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r"^\d+[\.\)]\s+", stripped):
                cleaned = _strip_label(stripped)
                if cleaned and len(cleaned.split()) >= 3:
                    action_steps.append(cleaned)

    if not recommendation or len(recommendation.split()) < 12:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            upper = para.upper()
            if not upper.startswith("ACTION") and not upper.startswith("EXPLANATION") and not upper.startswith("DISCLAIMER"):
                if len(para.split()) >= 12:
                    recommendation = para
                    break

    if recommendation:
        recommendation = re.split(r"\n\s*(?:ACTION\s*STEPS|EXPLANATION|DISCLAIMER)\s*:?", recommendation)[0].strip()

    if not explanation or len(explanation.split()) < 8:
        for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
            upper = para.upper()
            if "because" in para.lower() or "based on" in para.lower() or "কারণ" in para:
                if not upper.startswith("RECOMMENDATION") and not upper.startswith("ACTION"):
                    explanation = para
                    break

    if not disclaimer:
        for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
            low = para.lower()
            if "educational" in low or "financial advice" in low or "উপদেষ্টা" in para or "শিক্ষামূলক" in para:
                disclaimer = para
                break

    recommendation = recommendation.strip()
    explanation = explanation.strip()
    disclaimer = disclaimer.strip()

    if len(action_steps) > 6:
        action_steps = action_steps[:6]

    if not explanation:
        explanation = (
            "This advice is based on the user's income, expenses, debt, savings, financial goal, "
            "investment horizon, and risk tolerance."
            if language != "Bangla"
            else "এই পরামর্শটি ব্যবহারকারীর আয়, ব্যয়, ঋণ, সঞ্চয়, লক্ষ্য, সময়সীমা এবং ঝুঁকি সহনশীলতার ভিত্তিতে তৈরি।"
        )

    if not disclaimer:
        disclaimer = _ensure_disclaimer_text(language)

    return {
        "recommendation": recommendation,
        "action_steps": action_steps,
        "explanation": explanation,
        "disclaimer": disclaimer,
    }


# ---------------------------------------------------------------------------
# TinyLlama — Quality checker
# ---------------------------------------------------------------------------

def _is_low_quality_tinyllama_output(raw_text: str, parsed: dict) -> bool:
    """
    Return True if output is too poor to show directly.

    Balanced checks:
    - rejects placeholders, instruction echoes, empty sections, and trivial outputs
    - allows imperfect but usable answers to pass
    """
    text_lower = _safe_text(raw_text).lower()
    recommendation = _safe_text(parsed.get("recommendation", ""))
    explanation = _safe_text(parsed.get("explanation", ""))
    disclaimer = _safe_text(parsed.get("disclaimer", ""))
    action_steps = parsed.get("action_steps", [])

    placeholder_patterns = [
        r"\[[^\]]+\]",
        r"\bplaceholder\b",
        r"\byour name\b",
        r"\binsert\b",
        r"\bwrite 4\b",
        r"\bwrite the answer\b",
        r"\buse only the profile facts\b",
        r"\breturn all four sections\b",
        r"\bdo not copy the prompt\b",
    ]
    for pattern in placeholder_patterns:
        if re.search(pattern, text_lower):
            return True

    if len(text_lower.split()) < 40:
        return True

    if len(recommendation.split()) < 8:
        return True

    useful_steps = [
        step for step in action_steps
        if isinstance(step, str) and len(step.split()) >= 4 and "action steps" not in step.lower()
    ]
    if len(useful_steps) < 2:
        return True

    if len(explanation.split()) < 6:
        return True

    if len(disclaimer.split()) < 6:
        return True

    heading_only_patterns = [
        "recommendation:",
        "action steps:",
        "explanation:",
        "disclaimer:",
    ]
    for heading in heading_only_patterns:
        if text_lower.count(heading) >= 2 and len(text_lower.replace(heading, "").strip().split()) < 40:
            return True

    return False


# ---------------------------------------------------------------------------
# Selected Model — Real TinyLlama inference with quality check + repair
# ---------------------------------------------------------------------------

def _generate_selected_model_response(profile: dict, prompt: dict, language: str) -> dict:
    """
    Run real TinyLlama inference with one repair pass.
    Falls back to a structured mock response only as a last resort.
    """
    task_type = prompt.get("task_type", "Budget Planning")

    def _run_inference(messages: list, priming: str, temperature: float, top_p: float) -> str:
        tokenizer, model = _load_selected_model()

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        primed_formatted = formatted + priming

        inputs = tokenizer(
            primed_formatted,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=380,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        raw_completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _clean_generated_text(priming + raw_completion)

    try:
        messages, priming = _build_tinyllama_messages(prompt, profile, language)
        generated_text = _run_inference(messages, priming, temperature=0.55, top_p=0.9)
        parsed = _parse_tinyllama_output(generated_text, language)

        if _is_low_quality_tinyllama_output(generated_text, parsed):
            repair_messages, repair_priming = _build_tinyllama_repair_messages(prompt, profile, language)
            repaired_text = _run_inference(repair_messages, repair_priming, temperature=0.35, top_p=0.85)
            repaired_parsed = _parse_tinyllama_output(repaired_text, language)

            if not _is_low_quality_tinyllama_output(repaired_text, repaired_parsed):
                parsed = repaired_parsed
            else:
                fallback = _generate_mock_response(profile, task_type, language)
                fallback["recommendation"] = (
                    "**Generated using selected backend model: TinyLlama-1.1B-Chat**\n\n"
                    "The live model output on this run was incomplete or low quality, so a stable fallback response is shown below.\n\n"
                    + fallback["recommendation"]
                )
                return fallback

        parsed["recommendation"] = (
            "**Generated using selected backend model: TinyLlama-1.1B-Chat**\n\n"
            + parsed["recommendation"]
        )

        if language == "Bangla":
            # If the model output is not genuinely Bangla, translate the full structured result.
            rec_text = _safe_text(parsed.get("recommendation", ""))
            if not re.search(r"[\u0980-\u09FF]", rec_text):
                label = "**Generated using selected backend model: TinyLlama-1.1B-Chat**"
                bare_result = {
                    "recommendation": rec_text.replace(label, "").strip(),
                    "action_steps": parsed.get("action_steps", []),
                    "explanation": parsed.get("explanation", ""),
                    "disclaimer": parsed.get("disclaimer", ""),
                }

                translated = _translate_result_to_bangla(bare_result)
                translated["recommendation"] = label + "\n\n" + translated.get("recommendation", "")
                parsed = translated


        return parsed

    except Exception as e:
        fallback = _generate_mock_response(profile, task_type, language)
        fallback["recommendation"] = (
            "**TinyLlama inference was unavailable on this device, so the app used a stable fallback response.**\n\n"
            + fallback["recommendation"]
        )
        fallback["explanation"] = (
            fallback.get("explanation", "")
            + " TinyLlama was selected as the preferred backend model based on comparative evaluation, "
            + f"but live inference could not be completed on this run. Technical reason: {str(e)}"
        )
        return fallback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_financial_response(
    profile: dict,
    prompt: dict,
    mode: str = "Mock Mode",
    language: str = "English",
) -> dict:
    """
    Main entry point for generating a financial recommendation.

    Args:
        profile: The structured user financial profile dictionary.
        prompt: Structured prompt dict from prompt_templates.build_structured_prompt().
        mode: "Mock Mode" or "TinyLlama (Selected Model)".
        language: User selected language.

    Returns:
        A dict with keys: recommendation, action_steps, explanation, disclaimer.
    """
    try:
        if mode == "Mock Mode":
            return _generate_mock_response(
                profile,
                prompt.get("task_type", "Budget Planning"),
                language,
            )

        if mode == "TinyLlama (Selected Model)":
            return _generate_selected_model_response(profile, prompt, language)

        return {
            "error": (
                f"Unknown mode: '{mode}'. Choose 'Mock Mode' or "
                "'TinyLlama (Selected Model)'."
            )
        }

    except Exception as e:
        return {"error": f"Unexpected error in LLM backend: {str(e)}"}