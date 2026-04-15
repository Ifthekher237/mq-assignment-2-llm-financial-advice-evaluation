"""
prompt_templates.py — Reusable prompt-building functions for SmartFinance AI Assistant
Structured for LLM input and designed to support future model comparison experiments.
"""

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Bangla": "bn",
}

SYSTEM_PROMPTS = {
    "en": (
        "You are a knowledgeable, empathetic, and practical personal finance assistant. "
        "Your role is to analyse a user's financial situation and provide clear, "
        "actionable, and personalised recommendations. "
        "Always explain your reasoning in plain language. "
        "Avoid jargon unless you explain it. "
        "Never promote specific financial products or companies. "
        "Remind the user that your advice is educational and not a substitute "
        "for professional financial advice."
    ),
    "bn": (
        "আপনি একজন জ্ঞানী, সহানুভূতিশীল এবং ব্যবহারিক ব্যক্তিগত অর্থ সহকারী। "
        "আপনার ভূমিকা হলো ব্যবহারকারীর আর্থিক পরিস্থিতি বিশ্লেষণ করা এবং "
        "স্পষ্ট, কার্যকর এবং ব্যক্তিগতকৃত পরামর্শ প্রদান করা। "
        "সর্বদা সহজ ভাষায় আপনার যুক্তি ব্যাখ্যা করুন। "
        "মনে করিয়ে দিন যে আপনার পরামর্শ শিক্ষামূলক এবং পেশাদার আর্থিক পরামর্শের বিকল্প নয়।"
    ),
}

TASK_INSTRUCTIONS = {
    "Budget Planning": (
        "Focus on analysing the user's income, expenses, and monthly surplus. "
        "Suggest a practical budget framework (e.g., 50/30/20 rule or similar). "
        "Identify areas where spending can be reduced if needed."
    ),
    "Savings Strategy": (
        "Focus on building a savings plan tailored to the user's surplus and goals. "
        "Recommend how much to save monthly, suitable savings accounts or vehicles, "
        "and strategies to grow the savings over time."
    ),
    "Debt Management": (
        "Focus on the user's debt situation. "
        "Recommend a repayment strategy (e.g., avalanche or snowball method). "
        "Advise on balancing debt repayment with savings and living expenses."
    ),
    "Beginner Investment Guidance": (
        "Focus on introducing the user to investing concepts appropriate to their "
        "risk tolerance and investment horizon. "
        "Suggest beginner-friendly investment options (e.g., index funds, ETFs, "
        "high-interest savings accounts). "
        "Emphasise diversification and long-term thinking."
    ),
}


def build_system_prompt(language: str = "English") -> str:
    """
    Return the system-level instruction for the LLM.

    Args:
        language: User's selected language (e.g., "English", "Bangla").

    Returns:
        A string containing the system-level prompt.
    """
    lang_code = SUPPORTED_LANGUAGES.get(language, "en")
    return SYSTEM_PROMPTS.get(lang_code, SYSTEM_PROMPTS["en"])


def build_user_prompt(profile: dict, task_type: str) -> str:
    """
    Build a detailed user-facing prompt from the financial profile.

    Args:
        profile:   Dictionary containing the user's financial inputs.
        task_type: The selected financial task category.

    Returns:
        A formatted string prompt.
    """
    task_instruction = TASK_INSTRUCTIONS.get(task_type, "")
    surplus = profile.get("monthly_income", 0) - profile.get("monthly_expenses", 0)

    prompt = f"""
User Financial Profile:
- Age: {profile.get('age')}
- Employment Status: {profile.get('employment_status')}
- Monthly Income: ${profile.get('monthly_income', 0):,.2f}
- Monthly Expenses: ${profile.get('monthly_expenses', 0):,.2f}
- Monthly Surplus: ${surplus:,.2f}
- Current Savings: ${profile.get('current_savings', 0):,.2f}
- Current Debt: ${profile.get('current_debt', 0):,.2f}
- Risk Tolerance: {profile.get('risk_tolerance')}
- Financial Goal: {profile.get('financial_goal')}
- Investment Horizon: {profile.get('investment_horizon')}
- Extra Preferences/Constraints: {profile.get('extra_preferences') or 'None stated'}

Task: {task_type}
Instruction: {task_instruction}

Please provide:
1. A personalised recommendation for this user's situation.
2. A numbered list of concrete action steps they should take.
3. An explanation of why this advice is specifically suited to their profile.
4. A brief disclaimer about the limitations of this advice.
""".strip()

    return prompt


def build_structured_prompt(profile: dict, task_type: str, language: str = "English") -> dict:
    """
    Combine system and user prompts into a structured format ready for LLM input.
    Mirrors the standard chat-completion message format used by most LLMs.

    Returns:
        A dict with 'system' and 'user' keys, each containing prompt strings.
    """
    return {
        "system": build_system_prompt(language),
        "user": build_user_prompt(profile, task_type),
    }