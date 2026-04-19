"""
prompt_templates.py — Prompt construction for the Smart Personal Finance Assistant
COMP8420 Assignment 2 — Large Language Models

Provides:
- SUPPORTED_LANGUAGES
- SYSTEM_PROMPTS
- TASK_INSTRUCTIONS
- build_system_prompt(language) -> str
- build_user_prompt(profile, task_type, language) -> str
- build_structured_prompt(profile, task_type, language) -> dict
"""

from utils import safe_float, compute_surplus

# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = ["English", "Bangla"]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "English": (
        "You are a practical personal finance assistant. "
        "When given a user's financial profile and task, you write complete, specific financial advice. "
        "Rules you must follow:\n"
        "- Use the actual numbers from the profile. Never use placeholders like [Name], [Amount], or [Age].\n"
        "- Do not copy or repeat the prompt or instructions.\n"
        "- Do not repeat the profile line by line.\n"
        "- Never leave a section heading empty.\n"
        "- Always write all four sections: RECOMMENDATION, ACTION STEPS, EXPLANATION, DISCLAIMER.\n"
        "- Keep advice practical, specific, and financially sensible.\n"
        "- Write in plain English. Be direct and concise."
    ),
    "Bangla": (
        "You are a practical personal finance assistant. "
        "Write the advice mostly in simple Bangla (বাংলা). "
        "Keep section labels in English exactly as shown. "
        "You may use English for short financial terms like budget, ETF, savings, risk, or debt if needed. "
        "Rules you must follow:\n"
        "- Use the actual numbers from the profile. Never use placeholders.\n"
        "- Do not copy or repeat the prompt.\n"
        "- Never leave a section empty.\n"
        "- Always write all four sections: RECOMMENDATION, ACTION STEPS, EXPLANATION, DISCLAIMER.\n"
        "- Keep advice practical, specific, and easy to understand in Bangla."
    ),
}

# ---------------------------------------------------------------------------
# Task-specific instructions
# ---------------------------------------------------------------------------

TASK_INSTRUCTIONS = {
    "Budget Planning": (
        "Create a personalised monthly budget plan. "
        "Use the user's actual income and expenses. "
        "Recommend a specific budgeting framework (e.g. 50/30/20 or zero-based). "
        "Give concrete spending targets for needs, wants, and savings. "
        "Explain why this framework suits the user's situation."
    ),
    "Savings Strategy": (
        "Design a personalised savings strategy. "
        "Use the user's actual monthly surplus and savings goal. "
        "Recommend a specific monthly savings amount and a suitable savings vehicle. "
        "Include an emergency fund target. "
        "Explain the reasoning based on the user's horizon and current savings."
    ),
    "Debt Management": (
        "Create a personalised debt repayment plan. "
        "Use the user's actual debt amount and monthly surplus. "
        "Recommend a specific repayment strategy (e.g. Debt Avalanche or Debt Snowball). "
        "Give a concrete monthly repayment amount and estimated payoff timeline. "
        "Explain why this approach suits the user's debt-to-income situation."
    ),
    "Beginner Investment Guidance": (
        "Provide beginner-friendly personalised investment guidance. "
        "Use the user's actual risk tolerance, investment horizon, and monthly surplus. "
        "Recommend specific investment vehicles suitable for a beginner. "
        "Suggest a concrete monthly investment amount. "
        "Explain the reasoning based on the user's profile and goals."
    ),
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_amount(value) -> str:
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"


def _build_profile_block(profile: dict) -> str:
    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    surplus = compute_surplus(income, expenses)

    lines = [
        f"Age: {int(safe_float(profile.get('age', 25)))}",
        f"Employment: {profile.get('employment_status', 'Not stated')}",
        f"Monthly income: {_format_amount(income)}",
        f"Monthly expenses: {_format_amount(expenses)}",
        f"Monthly surplus: {_format_amount(surplus)}",
        f"Current savings: {_format_amount(profile.get('current_savings', 0))}",
        f"Current debt: {_format_amount(profile.get('current_debt', 0))}",
        f"Risk tolerance: {profile.get('risk_tolerance', 'Medium')}",
        f"Financial goal: {profile.get('financial_goal', 'Not stated')}",
        f"Investment horizon: {profile.get('investment_horizon', 'Medium (2-5 years)')}",
        f"Extra preferences: {profile.get('extra_preferences', 'None stated')}",
    ]
    return "\n".join(lines)


def _build_output_format_block(language: str) -> str:
    if language == "Bangla":
        return (
            "Write the answer with exactly these four sections. "
            "Keep the labels in English. Write the content mostly in Bangla.\n\n"
            "RECOMMENDATION:\n"
            "Write 2-4 complete sentences of specific advice using the actual numbers above.\n\n"
            "ACTION STEPS:\n"
            "1. Write a specific immediate action.\n"
            "2. Write a specific short-term action.\n"
            "3. Write a specific medium-term action.\n"
            "4. Write a specific long-term action.\n\n"
            "EXPLANATION:\n"
            "Write 2-3 sentences explaining why this advice suits this user.\n\n"
            "DISCLAIMER:\n"
            "Write a short educational disclaimer in Bangla."
        )
    return (
        "Write the answer with exactly these four sections:\n\n"
        "RECOMMENDATION:\n"
        "Write 2-4 complete sentences of specific advice using the actual dollar amounts and numbers above.\n\n"
        "ACTION STEPS:\n"
        "1. Write a specific immediate action.\n"
        "2. Write a specific short-term action.\n"
        "3. Write a specific medium-term action.\n"
        "4. Write a specific long-term action.\n\n"
        "EXPLANATION:\n"
        "Write 2-3 sentences explaining why this advice suits this user's specific profile.\n\n"
        "DISCLAIMER:\n"
        "Write a short disclaimer that this is educational information only."
    )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def build_system_prompt(language: str = "English") -> str:
    """
    Return the system prompt for the given language.
    Falls back to English if an unsupported language is passed.
    """
    return SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["English"])


def build_user_prompt(profile: dict, task_type: str, language: str = "English") -> str:
    """
    Build a complete, injected user prompt from the financial profile.

    Designed to:
    - clearly inject all real user values
    - enforce output structure
    - minimise TinyLlama instruction echoing and placeholder generation
    """
    task_instruction = TASK_INSTRUCTIONS.get(task_type, TASK_INSTRUCTIONS["Budget Planning"])
    profile_block = _build_profile_block(profile)
    format_block = _build_output_format_block(language)

    if language == "Bangla":
        lang_note = (
            "Answer mostly in Bangla. Keep section labels in English. "
            "You may keep short financial terms in English.\n"
        )
    else:
        lang_note = ""

    prompt = (
        f"Task: {task_type}\n\n"
        f"User Profile:\n{profile_block}\n\n"
        f"Instruction: {task_instruction}\n\n"
        f"{lang_note}"
        "Important rules:\n"
        "- Use the actual numbers from the profile above. Do not write placeholders.\n"
        "- Do not copy these instructions into your answer.\n"
        "- Do not repeat the user profile in your answer.\n"
        "- Do not leave any section empty.\n\n"
        f"{format_block}"
    )

    return prompt


def build_structured_prompt(profile: dict, task_type: str, language: str = "English") -> dict:
    """
    Build and return a structured prompt dict for use by llm_backend and app.py.

    Returns:
        dict with keys:
            "system"      — system prompt string
            "user"        — user prompt string
            "task_type"   — task type string
            "zero_shot"   — zero-shot version for prompt preview
            "structured"  — structured version for prompt preview
            "few_shot"    — few-shot example block for prompt preview
    """
    system = build_system_prompt(language)
    user = build_user_prompt(profile, task_type, language)

    income = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    surplus = compute_surplus(income, expenses)
    goal = profile.get("financial_goal", "improve my finances")
    risk = profile.get("risk_tolerance", "Medium")
    horizon = profile.get("investment_horizon", "Medium (2-5 years)")

    zero_shot = (
        f"You are a personal finance assistant. Provide {task_type} advice for a user "
        f"with income {_format_amount(income)}/month, expenses {_format_amount(expenses)}/month, "
        f"surplus {_format_amount(surplus)}/month, risk tolerance {risk}, "
        f"goal: {goal}."
    )

    structured = (
        f"Task: {task_type}\n"
        f"Income: {_format_amount(income)}/month | Expenses: {_format_amount(expenses)}/month | "
        f"Surplus: {_format_amount(surplus)}/month\n"
        f"Risk: {risk} | Horizon: {horizon} | Goal: {goal}\n"
        "Provide: RECOMMENDATION, ACTION STEPS (1-4), EXPLANATION, DISCLAIMER."
    )

    few_shot = (
        "Example — Budget Planning for a user with $4,000 income and $3,200 expenses:\n"
        "RECOMMENDATION: With a monthly surplus of $800 (20% of income), we recommend the 50/30/20 "
        "budgeting rule. Allocate $2,000 to needs, $1,200 to wants, and $800 to savings each month.\n"
        "ACTION STEPS:\n"
        "1. Track all spending this month using a free budgeting app.\n"
        "2. Open a dedicated savings account and transfer $800 on payday.\n"
        "3. Review and cancel unused subscriptions to free up extra cash.\n"
        "4. Increase savings allocation by 1% each time you receive a pay rise.\n"
        "EXPLANATION: The 50/30/20 framework suits this user because the 20% surplus is already "
        "strong and the rule maintains a sustainable lifestyle while building savings consistently.\n"
        "DISCLAIMER: This is educational information only. Please consult a registered "
        "financial adviser before making financial decisions."
    )

    return {
        "system": system,
        "user": user,
        "task_type": task_type,
        "zero_shot": zero_shot,
        "structured": structured,
        "few_shot": few_shot,
    }