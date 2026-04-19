"""
llm_backend.py — LLM response engine for the Smart Personal Finance Assistant
COMP8420 Assignment 2 — Large Language Models

Modes:
- Mock Mode: rule-based structured responses, stable for all demos.
- TinyLlama (Selected Model): real local inference (English generation always),
  then translated to Bangla via googletrans if the user selected Bangla.

Public API:
    generate_financial_response(profile, prompt, mode="Mock Mode", language="English")
    Returns: dict with keys recommendation, action_steps, explanation, disclaimer
             or dict with key error on failure.
"""

import re

from utils import (
    safe_float,
    format_currency,
    compute_surplus,
    compute_savings_rate,
    compute_debt_to_income_ratio,
)

# ---------------------------------------------------------------------------
# Optional heavy dependencies — fail gracefully if absent
# ---------------------------------------------------------------------------

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _TRANSFORMERS_OK = True
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    _TRANSFORMERS_OK = False

try:
    from googletrans import Translator as _GTranslator
    _GOOGLETRANS_OK = True
except Exception:
    _GTranslator = None
    _GOOGLETRANS_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELECTED_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}\n{% endif %}"
    "{% if message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n{% endif %}"
    "{% if message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}\n{% endif %}"
    "{% endfor %}<|assistant|>\n"
)

# Module-level lazy-load slots
_tok = None
_mdl = None
_load_err = None


# ---------------------------------------------------------------------------
# ── Small helper utilities ──
# ---------------------------------------------------------------------------

def _s(value, default: str = "") -> str:
    """Return a stripped string; never raises."""
    try:
        t = str(value).strip()
        return t if t else default
    except Exception:
        return default


def _amt(value) -> str:
    """Format a number as $X,XXX for prompt injection."""
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"


def _clean(text: str) -> str:
    """Strip special tokens and common instruction echoes from model output."""
    if not text:
        return ""
    t = text
    for tok in ("<|assistant|>", "<|user|>", "<|system|>", "<s>", "</s>"):
        t = t.replace(tok, " ")
    t = re.sub(r"\[[^\]]{1,80}\]", "", t)           # [placeholder]
    t = re.sub(r"\{[^\}]{1,80}\}", "", t)            # {placeholder}
    t = re.sub(r"(?i)^instructions?:\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"(?i)use (only|this|the) (profile|exact|actual).*", "", t)
    t = re.sub(r"(?i)do not (use|copy|repeat|invent|include).*", "", t)
    t = re.sub(r"(?i)write (the|a|your|all).*section.*", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _strip_bullet(line: str) -> str:
    return re.sub(r"^[\-\•\*\d\.\)\s]+", "", line).strip()


def _disclaimer_en() -> str:
    return (
        "This information is for educational purposes only and does not constitute "
        "personalised financial advice. Please consult a registered financial adviser "
        "before making any financial decisions."
    )


def _disclaimer_bn() -> str:
    return (
        "এই তথ্য কেবল শিক্ষামূলক উদ্দেশ্যে প্রদান করা হয়েছে এবং এটি ব্যক্তিগত আর্থিক পরামর্শ নয়। "
        "কোনো আর্থিক সিদ্ধান্ত নেওয়ার আগে একজন নিবন্ধিত আর্থিক উপদেষ্টার সঙ্গে পরামর্শ করুন।"
    )


# ---------------------------------------------------------------------------
# ── Googletrans integration ──
# ---------------------------------------------------------------------------

def _translate_text(text: str) -> str:
    """Translate a single text string from English to Bangla. Returns original on failure."""
    if not _GOOGLETRANS_OK or not text or not text.strip():
        return text
    try:
        tr = _GTranslator()
        result = tr.translate(text, src="en", dest="bn")
        return result.text if result and result.text else text
    except Exception:
        return text


def _translate_result_to_bangla(result: dict) -> dict:
    """
    Translate a full structured English result dict into Bangla section-by-section.
    Falls back gracefully on any translation error.
    """
    try:
        rec_translated = _translate_text(_s(result.get("recommendation", "")))
        exp_translated = _translate_text(_s(result.get("explanation", "")))

        steps = result.get("action_steps", [])
        if isinstance(steps, list):
            steps_translated = [_translate_text(_s(step)) for step in steps]
        else:
            steps_translated = [_translate_text(_s(steps))]

        # Always use canonical Bangla disclaimer rather than translating
        return {
            "recommendation": rec_translated,
            "action_steps": steps_translated,
            "explanation": exp_translated,
            "disclaimer": _disclaimer_bn(),
        }
    except Exception:
        # Last-resort Bangla wrapper if translation completely fails
        return {
            "recommendation": (
                "**বাংলা অনুবাদ পাওয়া যায়নি।** নিচে ইংরেজিতে পরামর্শ দেওয়া হলো:\n\n"
                + _s(result.get("recommendation", ""))
            ),
            "action_steps": result.get("action_steps", []),
            "explanation": _s(result.get("explanation", "")),
            "disclaimer": _disclaimer_bn(),
        }


# ---------------------------------------------------------------------------
# ── Mock Mode — task generators ──
# ---------------------------------------------------------------------------

def _mock_budget(profile: dict) -> dict:
    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    surplus = compute_surplus(income, expenses)
    rate = compute_savings_rate(income, expenses)
    goal = _s(profile.get("financial_goal", "your financial goals"))

    if rate >= 20:
        framework = "50/30/20 rule"
        note = f"Your savings rate of {rate:.1f}% is strong — maintain this discipline."
    elif rate >= 10:
        framework = "60/20/20 rule"
        note = f"Your savings rate of {rate:.1f}% is moderate. Cutting 10% from discretionary spend could accelerate your progress."
    else:
        framework = "zero-based budgeting"
        note = f"Your savings rate is only {rate:.1f}%. Zero-based budgeting will help you assign every dollar intentionally."

    needs = income * 0.50
    wants = income * 0.30
    save = income * 0.20

    return {
        "recommendation": (
            f"Based on your income of {format_currency(income)} and expenses of {format_currency(expenses)}, "
            f"your monthly surplus is {format_currency(surplus)}. We recommend the **{framework}** "
            f"to structure your spending and progress toward {goal}. {note}"
        ),
        "action_steps": [
            "Track every expense for 30 days using a free budgeting app.",
            f"Allocate {format_currency(needs)}/month (50%) to essential needs.",
            f"Limit wants and discretionary spending to {format_currency(wants)}/month (30%).",
            f"Transfer {format_currency(save)}/month (20%) to savings on payday.",
            "Cancel unused subscriptions to free up additional cash each month.",
            "Review your budget at the end of every month and adjust as needed.",
        ],
        "explanation": (
            f"The {framework} is recommended because your surplus of {format_currency(surplus)} "
            f"({rate:.1f}% of income) provides {'healthy' if rate >= 20 else 'moderate' if rate >= 10 else 'limited'} "
            f"financial headroom. A structured framework builds the habit of consistent saving while keeping day-to-day "
            f"spending sustainable."
        ),
    }


def _mock_savings(profile: dict) -> dict:
    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    savings = safe_float(profile.get("current_savings", 0))
    surplus = compute_surplus(income, expenses)
    goal = _s(profile.get("financial_goal", "your savings goal"))
    horizon = _s(profile.get("investment_horizon", "Medium (2-5 years)"))

    if "Long" in horizon:
        pct, vehicle = 0.25, "a high-interest savings account for emergencies, then low-cost index ETFs"
    elif "Medium" in horizon:
        pct, vehicle = 0.20, "a high-interest savings account and term deposits"
    else:
        pct, vehicle = 0.15, "a dedicated high-interest savings account or mortgage offset account"

    monthly = max(surplus * pct, 0)
    ef = expenses * 3

    return {
        "recommendation": (
            f"With a monthly surplus of {format_currency(surplus)}, we recommend saving "
            f"**{format_currency(monthly)}/month** ({pct * 100:.0f}% of surplus) into {vehicle}. "
            f"This positions you to reach {goal}. Over 12 months you would accumulate roughly "
            f"{format_currency(monthly * 12)} on top of your existing {format_currency(savings)}."
        ),
        "action_steps": [
            f"Build an emergency fund of {format_currency(ef)} (3 months of expenses) first.",
            "Open a dedicated savings account separate from your everyday account.",
            f"Set up an automatic transfer of {format_currency(monthly)} on payday.",
            f"Once the emergency fund is complete, direct surplus into {vehicle}.",
            "Do not withdraw savings for non-emergency spending.",
            "Reassess your savings rate every 6 months as income grows.",
        ],
        "explanation": (
            f"A {horizon.lower()} horizon calls for {vehicle.split(',')[0]}. "
            f"The {pct * 100:.0f}% target is calibrated to your current surplus of "
            f"{format_currency(surplus)}/month, ensuring the goal remains achievable without "
            f"straining your monthly budget."
        ),
    }


def _mock_debt(profile: dict) -> dict:
    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    debt = safe_float(profile.get("current_debt", 0))
    surplus = compute_surplus(income, expenses)
    dti = compute_debt_to_income_ratio(debt, income)
    goal = _s(profile.get("financial_goal", "become debt-free"))

    if debt == 0:
        return {
            "recommendation": (
                f"You have no reported debt — excellent position! "
                f"With a monthly surplus of {format_currency(surplus)}, "
                f"focus entirely on building savings and working toward {goal}."
            ),
            "action_steps": [
                "Pay credit card balances in full every month to stay debt-free.",
                "Build a 3–6 month emergency fund to avoid future debt.",
                "Redirect what would have been debt repayments into a savings or investment account.",
            ],
            "explanation": (
                "With zero debt your financial foundation is strong. The priority now is building "
                "wealth steadily while maintaining the zero-debt position."
            ),
        }

    if dti > 5:
        urgency, strategy, pct = "high", "Debt Avalanche (highest interest first)", 0.40
    elif dti > 2:
        urgency, strategy, pct = "moderate", "Debt Snowball (smallest balance first)", 0.30
    else:
        urgency, strategy, pct = "manageable", "balanced repayment plan", 0.20

    payment = max(surplus * pct, 0)
    months = (debt / payment) if payment > 0 else 999

    return {
        "recommendation": (
            f"Your debt-to-income ratio is **{dti:.1f}x** monthly income — {urgency} pressure. "
            f"We recommend the **{strategy}** to work toward {goal}. "
            f"Allocating {format_currency(payment)}/month as your extra repayment could clear "
            f"your debt in approximately **{months:.0f} months** (excluding interest)."
        ),
        "action_steps": [
            "List all debts, their interest rates, and minimum monthly repayments.",
            "Pay the minimum on every debt each month without exception.",
            f"Direct {format_currency(payment)}/month as an extra repayment using the {strategy}.",
            "Contact lenders to request lower interest rates or hardship assistance.",
            "Avoid taking on any new debt while repaying existing obligations.",
            "Once debt-free, redirect the repayment amount into savings or investments.",
        ],
        "explanation": (
            f"A debt-to-income ratio of {dti:.1f}x indicates {urgency} financial pressure. "
            f"The {strategy} is appropriate because "
            + ("it minimises total interest paid over time." if urgency == "high"
               else "it builds early momentum through quick wins." if urgency == "moderate"
               else "your debt level allows a balanced approach without sacrificing savings.")
        ),
    }


def _mock_invest(profile: dict) -> dict:
    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    debt = safe_float(profile.get("current_debt", 0))
    risk = _s(profile.get("risk_tolerance", "Medium"))
    horizon = _s(profile.get("investment_horizon", "Medium (2-5 years)"))
    goal = _s(profile.get("financial_goal", "grow wealth"))
    age = int(safe_float(profile.get("age", 25)))
    surplus = compute_surplus(income, expenses)

    profiles = {
        "Low":    ("high-interest savings accounts, government bonds, and term deposits",
                   "80% defensive / 20% growth", "Capital preservation is the priority."),
        "Medium": ("diversified index ETFs and balanced managed funds",
                   "50% growth / 50% defensive", "Balanced growth with moderate volatility."),
        "High":   ("growth-oriented ETFs and diversified equity funds",
                   "80% growth / 20% defensive", "Higher potential returns, higher short-term volatility."),
    }
    vehicles, allocation, note = profiles.get(risk, profiles["Medium"])
    monthly_inv = max(surplus * 0.20, 0)
    dti = compute_debt_to_income_ratio(debt, income)
    debt_warn = (
        " **Note:** With a high debt-to-income ratio, prioritise debt repayment before investing."
        if dti > 3 else ""
    )

    return {
        "recommendation": (
            f"As a beginner investor with **{risk.lower()} risk tolerance** and a "
            f"**{horizon.lower()} horizon**, start with **{vehicles}**. "
            f"A target allocation of {allocation} is appropriate. "
            f"Invest {format_currency(monthly_inv)}/month consistently to progress toward {goal}.{debt_warn}"
        ),
        "action_steps": [
            "Build a 3-month emergency fund before committing money to investments.",
            "Open a low-fee brokerage account suitable for beginners.",
            f"Start investing in {vehicles}.",
            f"Invest {format_currency(monthly_inv)}/month using dollar-cost averaging.",
            f"Maintain a {allocation} portfolio allocation.",
            "Reinvest dividends/distributions to compound returns over time.",
            "Review your portfolio annually — avoid reacting to short-term market moves.",
        ],
        "explanation": (
            f"The recommended approach suits your {risk.lower()} risk tolerance and {horizon.lower()} horizon. "
            f"At age {age}, you have {'substantial' if age < 40 else 'meaningful'} time to benefit from compounding. "
            f"{note} Beginning with diversified options reduces single-asset risk, which is appropriate for a new investor."
        ),
    }


def _generate_mock_response(profile: dict, task_type: str, language: str) -> dict:
    """Generate a mock structured response and optionally translate to Bangla."""
    _generators = {
        "Budget Planning": _mock_budget,
        "Savings Strategy": _mock_savings,
        "Debt Management": _mock_debt,
        "Beginner Investment Guidance": _mock_invest,
    }
    result = _generators.get(task_type, _mock_budget)(profile)
    result["disclaimer"] = _disclaimer_en()

    if language == "Bangla":
        result = _translate_result_to_bangla(result)

    return result


# ---------------------------------------------------------------------------
# ── Model loading ──
# ---------------------------------------------------------------------------

def _load_model():
    """Lazily load TinyLlama. Returns (tokenizer, model). Raises on failure."""
    global _tok, _mdl, _load_err

    if _tok is not None and _mdl is not None:
        return _tok, _mdl

    if _load_err is not None:
        raise RuntimeError(_load_err)

    if not _TRANSFORMERS_OK:
        _load_err = (
            "PyTorch / Transformers not installed. "
            "Run: pip install transformers torch accelerate sentencepiece safetensors"
        )
        raise RuntimeError(_load_err)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        _tok = AutoTokenizer.from_pretrained(_SELECTED_MODEL, use_fast=True)
        if _tok.pad_token is None:
            _tok.pad_token = _tok.eos_token
        _tok.padding_side = "left"
        if getattr(_tok, "chat_template", None) is None:
            _tok.chat_template = _FALLBACK_CHAT_TEMPLATE

        kwargs = {"torch_dtype": dtype}
        if device == "cuda":
            kwargs["device_map"] = "auto"
            kwargs["low_cpu_mem_usage"] = True

        _mdl = AutoModelForCausalLM.from_pretrained(_SELECTED_MODEL, **kwargs)
        if device != "cuda":
            _mdl = _mdl.to(device)
        _mdl.eval()

        return _tok, _mdl

    except Exception as exc:
        _load_err = f"Failed to load {_SELECTED_MODEL}: {exc}"
        raise RuntimeError(_load_err)


# ---------------------------------------------------------------------------
# ── TinyLlama prompting — English only, always ──
# ---------------------------------------------------------------------------

def _profile_summary(profile: dict) -> str:
    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    surplus = compute_surplus(income, expenses)
    return (
        f"age {int(safe_float(profile.get('age', 25)))}; "
        f"employment: {_s(profile.get('employment_status', 'unknown'))}; "
        f"income {_amt(income)}/month; "
        f"expenses {_amt(expenses)}/month; "
        f"surplus {_amt(surplus)}/month; "
        f"savings {_amt(profile.get('current_savings', 0))}; "
        f"debt {_amt(profile.get('current_debt', 0))}; "
        f"risk tolerance: {_s(profile.get('risk_tolerance', 'Medium'))}; "
        f"goal: {_s(profile.get('financial_goal', 'improve finances'))}; "
        f"horizon: {_s(profile.get('investment_horizon', 'Medium (2-5 years)'))}; "
        f"preferences: {_s(profile.get('extra_preferences', 'none'))}"
    )


def _task_focus(task_type: str) -> str:
    focuses = {
        "Budget Planning": (
            "Give a personalised monthly budget plan using the income and expense figures above. "
            "Recommend a specific budgeting framework. "
            "Do NOT recommend stocks, ETFs, or investment products unless the surplus is explicitly large."
        ),
        "Savings Strategy": (
            "Give a personalised savings strategy. Recommend a specific monthly savings target and vehicle. "
            "Include an emergency fund target based on the expenses figure."
        ),
        "Debt Management": (
            "Give a personalised debt repayment plan using the debt and surplus figures. "
            "Recommend a repayment strategy and estimate a realistic payoff timeline. "
            "Do NOT recommend investing while debt is the focus."
        ),
        "Beginner Investment Guidance": (
            "Give beginner-friendly investment guidance based on risk tolerance and investment horizon. "
            "Recommend specific investment vehicles suitable for a beginner. "
            "Use the surplus figure for a monthly investment amount."
        ),
    }
    return focuses.get(task_type, focuses["Budget Planning"])


def _build_messages(profile: dict, task_type: str) -> tuple:
    """
    Build English-only TinyLlama chat messages.
    Returns (messages_list, priming_prefix).
    """
    system = (
        "You are a practical personal finance assistant. "
        "Write complete, specific financial advice using only the profile facts given. "
        "Never invent numbers, interest rates, weekly values, or amounts not in the profile. "
        "Never use placeholders like [Name], [Amount], or [X%]. "
        "Never copy these instructions into your answer. "
        "Never repeat the profile line by line. "
        "Never leave a section heading empty. "
        "Always write all four sections with real content."
    )

    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    surplus = compute_surplus(income, expenses)

    user = (
        f"Task: {task_type}\n"
        f"Profile: {_profile_summary(profile)}\n\n"
        f"Focus: {_task_focus(task_type)}\n\n"
        "Write the answer in English with exactly these four sections:\n"
        "RECOMMENDATION:\n"
        "ACTION STEPS:\n"
        "1.\n2.\n3.\n4.\n"
        "EXPLANATION:\n"
        "DISCLAIMER:"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    priming = (
        f"RECOMMENDATION: Based on your income of {_amt(income)} and expenses of "
        f"{_amt(expenses)}, your surplus is {_amt(surplus)}. "
    )

    return messages, priming


def _build_repair_messages(profile: dict, task_type: str) -> tuple:
    """
    Build a stricter English-only repair prompt for second-pass generation.
    """
    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    surplus = compute_surplus(income, expenses)

    system = (
        "The previous answer was incomplete or low quality. "
        "Rewrite it fully using only the profile facts below. "
        "No placeholders. No copied instructions. No empty headings. "
        "Do not invent any number, rate, or value not given in the profile. "
        "Write all four sections completely."
    )

    user = (
        f"Rewrite {task_type} advice for:\n"
        f"{_profile_summary(profile)}\n\n"
        f"Focus: {_task_focus(task_type)}\n\n"
        "Return all four sections in English:\n"
        "RECOMMENDATION:\n"
        "ACTION STEPS:\n"
        "1.\n2.\n3.\n4.\n"
        "EXPLANATION:\n"
        "DISCLAIMER:"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    priming = (
        f"RECOMMENDATION: Given income of {_amt(income)}, expenses of {_amt(expenses)}, "
        f"and surplus of {_amt(surplus)}, "
    )

    return messages, priming


# ---------------------------------------------------------------------------
# ── TinyLlama output parser ──
# ---------------------------------------------------------------------------

def _parse(raw: str) -> dict:
    """
    Parse raw model output into recommendation / action_steps / explanation / disclaimer.
    Handles imperfect formatting; recovers useful content where possible.
    """
    text = _clean(raw)

    # Extract named sections via regex
    pattern = re.compile(
        r"(?is)(RECOMMENDATION|ACTION\s*STEPS|EXPLANATION|DISCLAIMER)\s*:?\s*"
        r"(.*?)(?=(?:\n\s*(?:RECOMMENDATION|ACTION\s*STEPS|EXPLANATION|DISCLAIMER)\s*:)|\Z)"
    )
    secs = {
        m.group(1).upper().replace(" ", "_"): m.group(2).strip()
        for m in pattern.finditer(text)
    }

    recommendation = secs.get("RECOMMENDATION", "")
    action_block   = secs.get("ACTION_STEPS", "")
    explanation    = secs.get("EXPLANATION", "")
    disclaimer     = secs.get("DISCLAIMER", "")

    # Parse action steps
    action_steps = []
    if action_block:
        for line in action_block.splitlines():
            step = _strip_bullet(line)
            if step and len(step.split()) >= 3 and "recommendation" not in step.lower():
                action_steps.append(step)

    # Fallback: numbered lines anywhere in text
    if not action_steps:
        for line in text.splitlines():
            if re.match(r"^\d+[\.\)]\s+", line.strip()):
                step = _strip_bullet(line.strip())
                if step and len(step.split()) >= 3:
                    action_steps.append(step)

    # Trim excessive steps
    if len(action_steps) > 6:
        action_steps = action_steps[:6]

    # Fallback recommendation from first long paragraph
    if not recommendation or len(recommendation.split()) < 12:
        for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
            upper = para.upper()
            if not any(upper.startswith(h) for h in ("ACTION", "EXPLANATION", "DISCLAIMER")):
                if len(para.split()) >= 12:
                    recommendation = para
                    break

    # Trim recommendation to only its own section
    if recommendation:
        recommendation = re.split(
            r"\n\s*(?:ACTION\s*STEPS|EXPLANATION|DISCLAIMER)\s*:?",
            recommendation
        )[0].strip()

    # Fallback explanation
    if not explanation or len(explanation.split()) < 8:
        for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
            if ("because" in para.lower() or "based on" in para.lower()) and len(para.split()) >= 8:
                upper = para.upper()
                if not upper.startswith("RECOMMENDATION") and not upper.startswith("ACTION"):
                    explanation = para
                    break

    # Fallback disclaimer
    if not disclaimer:
        for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
            if "educational" in para.lower() or "financial advice" in para.lower():
                disclaimer = para
                break

    if not explanation:
        explanation = (
            "This advice is based on the user's income, expenses, savings, debt, "
            "financial goal, investment horizon, and risk tolerance."
        )

    if not disclaimer:
        disclaimer = _disclaimer_en()

    return {
        "recommendation": recommendation.strip(),
        "action_steps":   action_steps,
        "explanation":    explanation.strip(),
        "disclaimer":     disclaimer.strip(),
    }


# ---------------------------------------------------------------------------
# ── Quality checker ──
# ---------------------------------------------------------------------------

_JUNK_PATTERNS = [
    r"\[[^\]]+\]",
    r"\bplaceholder\b",
    r"\byour name\b",
    r"\binsert\b",
    r"write 4 (action|step)",
    r"write the answer",
    r"use only the profile",
    r"return all four sections",
    r"do not (copy|repeat|invent|use placeholder)",
    # hallucination pattern: invented weekly amounts
    r"\$[\d,]+\s*per week",
    # invented percentages not matching any profile field
    r"\b[3-9][0-9]%\s*(interest|return|yield)",
]

_JUNK_RE = re.compile("|".join(_JUNK_PATTERNS), re.IGNORECASE)


def _is_poor(raw: str, parsed: dict) -> bool:
    """
    Return True if output quality is too low to show.
    Balanced — avoids rejecting imperfect but usable responses.
    """
    low = raw.lower() if raw else ""

    if _JUNK_RE.search(low):
        return True

    if len(low.split()) < 45:
        return True

    rec = _s(parsed.get("recommendation", ""))
    if len(rec.split()) < 8:
        return True

    steps = [
        st for st in parsed.get("action_steps", [])
        if isinstance(st, str) and len(st.split()) >= 4
        and "action steps" not in st.lower()
    ]
    if len(steps) < 3:
        return True

    exp = _s(parsed.get("explanation", ""))
    if len(exp.split()) < 6:
        return True

    return False


# ---------------------------------------------------------------------------
# ── TinyLlama inference helper ──
# ---------------------------------------------------------------------------

def _infer(messages: list, priming: str, temperature: float, top_p: float) -> str:
    """Run a single TinyLlama forward pass and return decoded text."""
    tok, mdl = _load_model()

    formatted = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ) + priming

    inputs = tok(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )

    new_ids = out[0][input_len:]
    raw = tok.decode(new_ids, skip_special_tokens=True).strip()
    return _clean(priming + raw)


# ---------------------------------------------------------------------------
# ── TinyLlama main generation flow ──
# ---------------------------------------------------------------------------

def _generate_tinyllama(profile: dict, prompt: dict, language: str) -> dict:
    """
    Full TinyLlama generation flow:
    1. English-only first pass
    2. Quality check
    3. English-only repair pass if needed
    4. Fallback to mock if repair also fails
    5. Translate to Bangla if language == "Bangla"
    """
    task_type = _s(prompt.get("task_type", "Budget Planning"))

    def _fallback(reason: str) -> dict:
        fb = _generate_mock_response(profile, task_type, "English")  # get English first
        if language == "Bangla":
            fb_bn = _translate_result_to_bangla(fb)
            fb_bn["recommendation"] = (
                f"**মডেল আউটপুট পাওয়া যায়নি ({reason}) — নির্ভরযোগ্য বিকল্প উত্তর দেওয়া হচ্ছে।**\n\n"
                + fb_bn["recommendation"]
            )
            return fb_bn
        fb["recommendation"] = (
            f"**TinyLlama output was unavailable ({reason}). A reliable fallback response is shown below.**\n\n"
            + fb["recommendation"]
        )
        return fb

    try:
        # ── First pass ──
        msgs1, prime1 = _build_messages(profile, task_type)
        raw1 = _infer(msgs1, prime1, temperature=0.55, top_p=0.90)
        parsed1 = _parse(raw1)

        if not _is_poor(raw1, parsed1):
            final = parsed1
        else:
            # ── Repair pass ──
            msgs2, prime2 = _build_repair_messages(profile, task_type)
            raw2 = _infer(msgs2, prime2, temperature=0.35, top_p=0.85)
            parsed2 = _parse(raw2)

            if not _is_poor(raw2, parsed2):
                final = parsed2
            else:
                return _fallback("low quality after repair")

        # ── Attach model credit ──
        final["recommendation"] = (
            "**Generated using selected backend model: TinyLlama-1.1B-Chat**\n\n"
            + final["recommendation"]
        )


        # ── Translate final structured result to Bangla if requested ──
        if language == "Bangla":
            credit_en = "**Generated using selected backend model: TinyLlama-1.1B-Chat**\n\n"
            recommendation_body = _s(final.get("recommendation", "")).replace(credit_en, "").strip()

            english_structured = {
                "recommendation": recommendation_body,
                "action_steps": final.get("action_steps", []),
                "explanation": final.get("explanation", ""),
                "disclaimer": final.get("disclaimer", ""),
            }

            translated = _translate_result_to_bangla(english_structured)

            final = {
                "recommendation": (
                    "**নির্বাচিত মডেল ব্যবহার করে তৈরি: TinyLlama-1.1B-Chat**\n\n"
                    + translated.get("recommendation", "")
                ),
                "action_steps": translated.get("action_steps", []),
                "explanation": translated.get("explanation", ""),
                "disclaimer": translated.get("disclaimer", ""),
            }


        return final

    except Exception as exc:
        return _fallback(str(exc))


# ---------------------------------------------------------------------------
# ── Public API ──
# ---------------------------------------------------------------------------

def generate_financial_response(
    profile: dict,
    prompt: dict,
    mode: str = "Mock Mode",
    language: str = "English",
) -> dict:
    """
    Main entry point for the Smart Personal Finance Assistant LLM backend.

    Args:
        profile:  User financial profile dict.
        prompt:   Structured prompt dict from prompt_templates.build_structured_prompt().
        mode:     "Mock Mode" or "TinyLlama (Selected Model)".
        language: "English" or "Bangla".

    Returns:
        dict with keys: recommendation, action_steps, explanation, disclaimer
        or dict with key: error
    """
    try:
        task_type = _s(prompt.get("task_type", "Budget Planning"))

        if mode == "Mock Mode":
            return _generate_mock_response(profile, task_type, language)

        if mode == "TinyLlama (Selected Model)":
            return _generate_tinyllama(profile, prompt, language)

        return {"error": f"Unknown mode '{mode}'. Use 'Mock Mode' or 'TinyLlama (Selected Model)'."}

    except Exception as exc:
        return {"error": f"Unexpected error in LLM backend: {exc}"}