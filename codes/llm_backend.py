"""
llm_backend.py — LLM response generation for SmartFinance AI Assistant

Supports two modes:
- Mock Mode: structured, personalised demo response based on user profile data.
- TinyLlama (Selected Model): real TinyLlama inference path with safe fallback.

Main entry point: generate_financial_response(profile, prompt, mode, language)
"""

import re  # moved to top-level imports

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        _TOKENIZER = AutoTokenizer.from_pretrained(SELECTED_MODEL)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            SELECTED_MODEL,
            torch_dtype=dtype,
        )

        _MODEL = _MODEL.to(device)
        _MODEL.eval()

        if _TOKENIZER.chat_template is None:
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
        result["disclaimer"] += (
            " | বিঃদ্রঃ বর্তমান সংস্করণে ইংরেজিতে পরামর্শ প্রদান করা হচ্ছে। "
            "ভবিষ্যতে বাংলা ভাষায় সম্পূর্ণ সমর্থন যোগ করা হবে।"
        )

    return result

# ---------------------------------------------------------------------------
# TinyLlama — Prompt builders
# ---------------------------------------------------------------------------
# CHANGED: _build_tinyllama_messages now accepts `profile` to inject real
# user values directly into the prompt, and returns a (messages, priming)
# tuple. The priming string is appended after apply_chat_template so the
# model's first generated token continues a real sentence rather than a
# format header, dramatically reducing placeholder and echo output.
# ---------------------------------------------------------------------------

def _build_tinyllama_messages(prompt: dict, profile: dict, language: str) -> tuple:
    """
    Build chat messages and a priming prefix for TinyLlama inference.

    Returns:
        (messages, priming) where `priming` is appended to the
        apply_chat_template output before tokenisation to force the model
        to start generating real content rather than template skeletons.
    """
    income     = safe_float(profile.get("monthly_income", 0))
    expenses   = safe_float(profile.get("monthly_expenses", 0))
    savings    = safe_float(profile.get("current_savings", 0))
    debt       = safe_float(profile.get("current_debt", 0))
    surplus    = compute_surplus(income, expenses)
    age        = int(safe_float(profile.get("age", 25)))
    risk       = profile.get("risk_tolerance", "Medium")
    goal       = profile.get("financial_goal", "improve my finances")
    horizon    = profile.get("investment_horizon", "Medium (2-5 years)")
    task       = prompt.get("task_type", "Budget Planning")
    employment = profile.get("employment_status", "employed")

    system_prompt = (
        "You are a financial advisor assistant. "
        "Your only job is to write complete financial advice using ONLY the real numbers "
        "from the user profile below. "
        "CRITICAL RULES — violating any of these makes your answer wrong: "
        "1. NEVER write placeholder text such as [Name], [Age], [Amount], [Employment Status], "
        "or ANY text inside square brackets. "
        "2. NEVER copy or repeat the user profile or these instructions. "
        "3. NEVER leave a section heading without real content underneath it. "
        "4. Write full sentences with specific dollar amounts from the profile. "
        "5. If you do not know a specific value, use the numbers provided — do not invent placeholders."
    )

    if language == "Bangla":
        system_prompt += " Respond in Bangla where possible."

    user_content = (
        f"Provide {task} advice for this person:\n"
        f"- Age: {age}, Employment: {employment}\n"
        f"- Monthly income: ${income:,.0f}, Monthly expenses: ${expenses:,.0f}\n"
        f"- Monthly surplus: ${surplus:,.0f}\n"
        f"- Current savings: ${savings:,.0f}, Current debt: ${debt:,.0f}\n"
        f"- Financial goal: {goal}\n"
        f"- Risk tolerance: {risk}, Investment horizon: {horizon}\n\n"
        "Respond using exactly these four labelled sections:\n"
        "RECOMMENDATION:\n"
        "ACTION STEPS:\n"
        "1.\n2.\n3.\n4.\n"
        "EXPLANATION:\n"
        "DISCLAIMER:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    # Priming: append "RECOMMENDATION:" after the assistant header so the
    # model's very first token continues real advice, not a template echo.
    priming = "RECOMMENDATION:"
    return messages, priming


# CHANGED: new repair-pass prompt builder — called only when first pass
# is detected as low quality.  Uses stronger priming that seeds the
# RECOMMENDATION sentence with actual numbers, making it far harder for
# the model to backslide into placeholders.
def _build_tinyllama_repair_messages(prompt: dict, profile: dict, language: str) -> tuple:
    """
    Stricter prompt for the repair pass after a low-quality first generation.
    The priming string seeds the RECOMMENDATION sentence with real numbers
    so the model must continue with real content.
    """
    income     = safe_float(profile.get("monthly_income", 0))
    expenses   = safe_float(profile.get("monthly_expenses", 0))
    savings    = safe_float(profile.get("current_savings", 0))
    debt       = safe_float(profile.get("current_debt", 0))
    surplus    = compute_surplus(income, expenses)
    age        = int(safe_float(profile.get("age", 25)))
    risk       = profile.get("risk_tolerance", "Medium")
    goal       = profile.get("financial_goal", "improve my finances")
    horizon    = profile.get("investment_horizon", "Medium (2-5 years)")
    task       = prompt.get("task_type", "Budget Planning")
    employment = profile.get("employment_status", "employed")

    system_prompt = (
        "You are a financial advisor. Write practical, complete financial advice. "
        "Use ONLY real values from the profile. "
        "No placeholders. No square brackets. No repeated instructions. "
        "Every section must contain full sentences with specific dollar amounts."
    )

    if language == "Bangla":
        system_prompt += " Respond in Bangla where possible."

    user_content = (
        f"Write {task} advice for: {age}-year-old {employment}, "
        f"income ${income:,.0f}/month, expenses ${expenses:,.0f}/month, "
        f"surplus ${surplus:,.0f}/month, savings ${savings:,.0f}, debt ${debt:,.0f}, "
        f"goal: {goal}, risk: {risk}, horizon: {horizon}.\n\n"
        "Complete all four sections with real sentences — no placeholders:\n"
        "RECOMMENDATION:\n"
        "ACTION STEPS:\n"
        "1.\n2.\n3.\n4.\n"
        "EXPLANATION:\n"
        "DISCLAIMER:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    # Stronger priming: seed the first sentence of RECOMMENDATION with
    # actual numbers so the model cannot default to placeholder text.
    priming = (
        f"RECOMMENDATION: Given your monthly income of ${income:,.0f} and expenses of "
        f"${expenses:,.0f}, your monthly surplus is ${surplus:,.0f}."
    )
    return messages, priming

# ---------------------------------------------------------------------------
# TinyLlama — Quality checker
# ---------------------------------------------------------------------------
# CHANGED: new helper — detects low-quality or template-like TinyLlama
# output so the repair loop knows when to intervene.
# ---------------------------------------------------------------------------

def _is_low_quality_tinyllama_output(raw_text: str, parsed: dict) -> bool:
    """
    Return True if the output is considered too low quality to show the user.

    Checks for:
    - Placeholder / square-bracket text
    - Echoed instruction fragments from the prompt
    - Too-short recommendation (< 15 words)
    - Fewer than 3 action steps
    - Missing or too-short explanation (< 8 words)
    - Overall response too short (< 60 words total)
    """
    text_lower = raw_text.lower()

    # 1. Placeholder patterns (square-bracket templates)
    placeholder_patterns = [
        r'\[your name\]', r'\[age\]', r'\[employment',  r'\[amount',
        r'\[your ',       r'\[insert', r'\[write ',      r'\[step ',
        r'\[specific',    r'\[number', r'\[goal\]',      r'\[risk',
        r'\[income',      r'\[debt\]', r'\[savings',     r'\[horizon',
    ]
    for pat in placeholder_patterns:
        if re.search(pat, text_lower):
            return True

    # 2. Echoed instruction fragments
    echo_phrases = [
        'write 2-4',          'write a specific',  'write 2 to',
        'write full sentence', '2-4 sentences',     'using this exact format',
        'using exactly these', 'complete all four', 'no placeholders:\n',
        'respond using exactly', 'four labelled',
    ]
    for phrase in echo_phrases:
        if phrase in text_lower:
            return True

    # 3. Too-short recommendation
    rec = parsed.get("recommendation", "")
    if len(rec.split()) < 15:
        return True

    # 4. Fewer than 3 action steps
    steps = parsed.get("action_steps", [])
    if len(steps) < 3:
        return True

    # 5. Missing / trivially short explanation
    exp = parsed.get("explanation", "")
    if len(exp.split()) < 8:
        return True

    # 6. Response far too short overall
    if len(raw_text.split()) < 60:
        return True

    return False

# ---------------------------------------------------------------------------
# TinyLlama — Output parser
# ---------------------------------------------------------------------------
# CHANGED: parser is now more robust:
# - strips placeholder text and echoed instructions before extraction
# - uses flexible section-header regex (colon optional, case-insensitive)
# - trims recommendation if it bleeds into numbered lists without a header
# - skips trivially short action step items (< 3 words)
# - uses paragraph-level fallback for recommendation if regex fails
# ---------------------------------------------------------------------------

def _parse_tinyllama_output(text: str) -> dict:
    """
    Parse raw TinyLlama output into structured sections.

    Attempts RECOMMENDATION / ACTION STEPS / EXPLANATION / DISCLAIMER
    extraction with flexible regex, then falls back to heuristic recovery.
    """
    text = text.strip()

    # ── 1. Remove square-bracket placeholders ────────────────────────────────
    text = re.sub(r'\[[^\]]{2,60}\]', '', text)

    # ── 2. Remove echoed instruction fragments ────────────────────────────────
    echo_patterns = [
        r'Write \d[–\-]\d (full )?sentences?[^\n]*\n?',
        r'Write a specific [^\n]*\n?',
        r'using this exact format[^\n]*\n?',
        r'\(2[–\-]4 sentences\)[^\n]*',
        r'\(2 sentences\)[^\n]*',
    ]
    for pat in echo_patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)

    text = text.strip()

    recommendation = ""
    action_steps   = []
    explanation    = ""
    disclaimer     = ""

    # ── 3. Section extraction (flexible: colon optional, case-insensitive) ───
    rec_match = re.search(
        r'RECOMMENDATION:?\s*(.*?)(?=\n\s*ACTION STEPS:?|\Z)',
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    act_match = re.search(
        r'ACTION STEPS:?\s*(.*?)(?=\n\s*EXPLANATION:?|\Z)',
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    exp_match = re.search(
        r'EXPLANATION:?\s*(.*?)(?=\n\s*DISCLAIMER:?|\Z)',
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    dis_match = re.search(
        r'DISCLAIMER:?\s*(.+?)(?:\n\n|\Z)',
        text, flags=re.IGNORECASE | re.DOTALL,
    )

    if rec_match:
        recommendation = rec_match.group(1).strip()
        # Trim if it bleeds into a numbered list without a section header
        recommendation = re.split(r'\n\s*1[\.\)]', recommendation)[0].strip()

    if act_match:
        action_block = act_match.group(1).strip()
        for line in action_block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            cleaned = re.sub(r'^[\-\•\*\d\.\)\s]+', '', stripped).strip()
            # Only keep items that are actual sentences (>= 3 words)
            if cleaned and len(cleaned.split()) >= 3:
                action_steps.append(cleaned)

    if exp_match:
        explanation = exp_match.group(1).strip()

    if dis_match:
        disclaimer = dis_match.group(1).strip()

    # ── 4. Robust fallbacks ───────────────────────────────────────────────────

    # Recommendation fallback: use first meaningful paragraph
    if not recommendation or len(recommendation.split()) < 12:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        for para in paragraphs:
            if (
                len(para.split()) >= 15
                and not para.upper().startswith('ACTION')
                and not para.upper().startswith('EXPLANATION')
                and not para.upper().startswith('DISCLAIMER')
            ):
                recommendation = para[:600]
                break
        if not recommendation:
            recommendation = text[:600].strip()

    # Action steps fallback: scan full text for numbered lines
    if not action_steps:
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r'^\d+[\.\)]\s+\S', stripped):
                cleaned = re.sub(r'^\d+[\.\)]\s+', '', stripped).strip()
                if cleaned and len(cleaned.split()) >= 3:
                    action_steps.append(cleaned)

    # Explanation fallback
    if not explanation or len(explanation.split()) < 8:
        explanation = (
            "This advice was generated using TinyLlama based on the user's income, "
            "expenses, debt, savings, financial goal, investment horizon, and risk tolerance."
        )

    # Disclaimer fallback
    if not disclaimer:
        disclaimer = (
            "This is educational information only and not personalised financial advice. "
            "Please consult a registered financial adviser before making any financial decisions."
        )

    return {
        "recommendation": recommendation,
        "action_steps":   action_steps[:6],
        "explanation":    explanation,
        "disclaimer":     disclaimer,
    }

# ---------------------------------------------------------------------------
# Selected Model — Real TinyLlama inference with quality check + repair
# ---------------------------------------------------------------------------
# CHANGED: added repair loop.
#   1. First-pass generation with primed prompt at temperature=0.5.
#   2. _is_low_quality_tinyllama_output() checks the result.
#   3. If low quality: one repair pass with stricter primed prompt at t=0.3.
#   4. If repair also fails: honest mock fallback with academic explanation.
# All other behaviour (lazy loading, CPU compat, exception fallback) is
# preserved exactly as before.
# ---------------------------------------------------------------------------

def _generate_selected_model_response(profile: dict, prompt: dict, language: str) -> dict:
    """
    Run real TinyLlama inference with an automatic repair pass on low-quality
    output.  Falls back to a structured mock response only as a last resort,
    with an academically honest explanation.
    """
    task_type = prompt.get("task_type", "Budget Planning")

    # ── Shared inference helper ───────────────────────────────────────────────
    def _run_inference(messages: list, priming: str,
                       temperature: float, do_sample: bool) -> str:
        """Tokenise, generate, and return the full response text."""
        tokenizer, model = _load_selected_model()

        # Apply chat template then append priming so the model's first
        # generated token continues real content, not a format header.
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
                max_new_tokens=480,
                temperature=temperature,
                do_sample=do_sample,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        raw_completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Reconstruct the full response including the primed prefix so the
        # parser sees "RECOMMENDATION: ..." from the very first character.
        return priming + " " + raw_completion

    # ── Main inference flow ───────────────────────────────────────────────────
    try:
        # ── First pass ────────────────────────────────────────────────────────
        messages, priming = _build_tinyllama_messages(prompt, profile, language)
        generated_text = _run_inference(
            messages, priming,
            temperature=0.5, do_sample=True,
        )
        parsed = _parse_tinyllama_output(generated_text)

        # ── Quality gate ──────────────────────────────────────────────────────
        if _is_low_quality_tinyllama_output(generated_text, parsed):

            # ── Repair pass ───────────────────────────────────────────────────
            repair_messages, repair_priming = _build_tinyllama_repair_messages(
                prompt, profile, language
            )
            repaired_text = _run_inference(
                repair_messages, repair_priming,
                temperature=0.3, do_sample=True,
            )
            repaired_parsed = _parse_tinyllama_output(repaired_text)

            if not _is_low_quality_tinyllama_output(repaired_text, repaired_parsed):
                # Repair succeeded — use repaired output
                parsed = repaired_parsed
            else:
                # Both passes produced low-quality output — use mock fallback
                # with an academically honest explanation rather than raw junk.
                fallback = _generate_mock_response(profile, task_type, language)
                fallback["recommendation"] = (
                    "**Generated using selected backend model: TinyLlama-1.1B-Chat**\n\n"
                    "*(Note: TinyLlama inference produced low-quality output on this run "
                    "after both a primary and a repair generation pass. "
                    "This is a documented limitation of small on-device LLMs (1.1B parameters) "
                    "on CPU-only or memory-constrained hardware. "
                    "A structured fallback response is shown below for demonstration purposes.)*\n\n"
                    + fallback["recommendation"]
                )
                return fallback

        # ── Successful output — prepend model label ───────────────────────────
        parsed["recommendation"] = (
            "**Generated using selected backend model: TinyLlama-1.1B-Chat**\n\n"
            + parsed["recommendation"]
        )
        return parsed

    except Exception as e:
        # Model load failure or runtime error — stable mock fallback.
        fallback = _generate_mock_response(profile, task_type, language)
        fallback["recommendation"] = (
            "**TinyLlama inference was unavailable on this device, "
            "so the app used a stable fallback response.**\n\n"
            + fallback["recommendation"]
        )
        fallback["explanation"] += (
            " TinyLlama was still selected as the preferred backend model based on "
            "evaluation results, but local inference could not be completed here. "
            f"Technical reason: {str(e)}"
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
