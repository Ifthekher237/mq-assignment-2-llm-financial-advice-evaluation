"""
llm_backend.py — LLM response generation for SmartFinance AI Assistant

Supports two modes:
  - Mock Mode: structured, personalised demo response based on user profile data.
  - TinyLlama (Selected Model): real TinyLlama inference path with safe fallback.

Main entry point: generate_financial_response(profile, prompt, mode, language)
"""

from utils import (
    safe_float,
    format_currency,
    compute_surplus,
    compute_savings_rate,
    compute_debt_to_income_ratio,
)

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


def _load_selected_model():
    """
    Lazily load TinyLlama only when needed.
    This avoids slowing down Mock Mode and keeps the app stable.
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
        dtype = torch.float32

        _TOKENIZER = AutoTokenizer.from_pretrained(SELECTED_MODEL)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            SELECTED_MODEL,
            torch_dtype=dtype,
        )

        _MODEL = _MODEL.to("cpu")
        _MODEL.eval()

        if _TOKENIZER.chat_template is None:
            _TOKENIZER.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}</s>\n{% endif %}"
                "{% if message['role'] == 'user' %}<|user|>\n{{ message['content'] }}</s>\n{% endif %}"
                "{% if message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}</s>\n{% endif %}"
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
        result["disclaimer"] += (
            " | বিঃদ্রঃ বর্তমান সংস্করণে ইংরেজিতে পরামর্শ প্রদান করা হচ্ছে। "
            "ভবিষ্যতে বাংলা ভাষায় সম্পূর্ণ সমর্থন যোগ করা হবে।"
        )

    return result


# ---------------------------------------------------------------------------
# TinyLlama prompt + parsing
# ---------------------------------------------------------------------------

def _build_tinyllama_messages(prompt: dict, language: str):
    system_prompt = prompt.get(
        "system",
        "You are a responsible personal finance assistant. Provide educational guidance only."
    )

    user_prompt = prompt.get("user", "")

    if language == "Bangla":
        system_prompt += " Respond in Bangla if possible. Keep the advice simple and practical."

    user_prompt += (
        "\n\nPlease answer in this exact format:\n"
        "RECOMMENDATION: ...\n\n"
        "ACTION STEPS:\n"
        "1. ...\n"
        "2. ...\n"
        "3. ...\n\n"
        "EXPLANATION: ...\n\n"
        "DISCLAIMER: This is educational information only and not personalised financial advice."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _parse_tinyllama_output(text: str) -> dict:
    text = text.strip()

    recommendation = ""
    action_steps = []
    explanation = ""
    disclaimer = ""

    upper_text = text.upper()

    rec_idx = upper_text.find("RECOMMENDATION:")
    act_idx = upper_text.find("ACTION STEPS:")
    exp_idx = upper_text.find("EXPLANATION:")
    dis_idx = upper_text.find("DISCLAIMER:")

    if rec_idx != -1:
        end = act_idx if act_idx != -1 else len(text)
        recommendation = text[rec_idx + len("RECOMMENDATION:"):end].strip()

    if act_idx != -1:
        end = exp_idx if exp_idx != -1 else len(text)
        action_block = text[act_idx + len("ACTION STEPS:"):end].strip()
        for line in action_block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped[:2].isdigit() or stripped.startswith(("-", "•", "*")):
                action_steps.append(stripped.lstrip("0123456789.-•* ").strip())
            else:
                action_steps.append(stripped)

    if exp_idx != -1:
        end = dis_idx if dis_idx != -1 else len(text)
        explanation = text[exp_idx + len("EXPLANATION:"):end].strip()

    if dis_idx != -1:
        disclaimer = text[dis_idx + len("DISCLAIMER:"):].strip()

    if not recommendation:
        recommendation = text

    if not action_steps:
        action_steps = [
            "Review the recommendation carefully and adapt it to your personal circumstances.",
            "Consider discussing major financial decisions with a qualified financial adviser.",
        ]

    if not explanation:
        explanation = (
            "This response was generated by the selected TinyLlama model based on your financial profile and task type."
        )

    if not disclaimer:
        disclaimer = (
            "This is educational information only and not personalised financial advice."
        )

    return {
        "recommendation": recommendation,
        "action_steps": action_steps,
        "explanation": explanation,
        "disclaimer": disclaimer,
    }


# ---------------------------------------------------------------------------
# Selected Model — Real TinyLlama inference with fallback
# ---------------------------------------------------------------------------

def _generate_selected_model_response(profile: dict, prompt: dict, language: str) -> dict:
    """
    Run real TinyLlama inference.
    If model loading or generation fails, safely fall back to mock response.
    """
    task_type = prompt.get("task_type", "Budget Planning")

    try:
        tokenizer, model = _load_selected_model()
        messages = _build_tinyllama_messages(prompt, language)

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=280,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        parsed = _parse_tinyllama_output(generated_text)

        parsed["recommendation"] = (
            f"**Generated using selected backend model: TinyLlama-1.1B-Chat**\n\n"
            f"{parsed['recommendation']}"
        )

        return parsed

    except Exception as e:
        fallback = _generate_mock_response(profile, task_type, language)
        fallback["recommendation"] = (
            "**TinyLlama inference was unavailable on this device, so the app used a stable fallback response.**\n\n"
            + fallback["recommendation"]
        )
        fallback["explanation"] += (
            f" TinyLlama was still selected as the preferred backend model based on evaluation results, "
            f"but local inference could not be completed here. Technical reason: {str(e)}"
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
        profile:  The structured user financial profile dictionary.
        prompt:   Structured prompt dict from prompt_templates.build_structured_prompt().
        mode:     "Mock Mode" or "TinyLlama (Selected Model)".
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
        elif mode == "TinyLlama (Selected Model)":
            return _generate_selected_model_response(profile, prompt, language)
        else:
            return {
                "error": (
                    f"Unknown mode: '{mode}'. Choose 'Mock Mode' or "
                    "'TinyLlama (Selected Model)'."
                )
            }
    except Exception as e:
        return {"error": f"Unexpected error in LLM backend: {str(e)}"}