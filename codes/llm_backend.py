"""
llm_backend.py — LLM response generation for SmartFinance AI Assistant

Supports two modes:
  - Mock Mode: structured, personalised demo response based on user profile data.
  - Local Model Placeholder: stub for future Hugging Face model integration.

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
# Mock Mode — Task-specific response generators
# ---------------------------------------------------------------------------

def _budget_planning_mock(profile: dict) -> dict:
    income   = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    surplus  = compute_surplus(income, expenses)
    savings_rate = compute_savings_rate(income, expenses)
    goal = profile.get("financial_goal", "your financial goals")

    if savings_rate >= 20:
        framework  = "50/30/20 rule (50% needs, 30% wants, 20% savings/investments)"
        budget_note = (
            f"Your current savings rate of {savings_rate:.1f}% is healthy. "
            "The 50/30/20 framework will help you maintain this discipline."
        )
    elif savings_rate >= 10:
        framework  = "60/20/20 rule (60% needs, 20% wants, 20% savings)"
        budget_note = (
            f"Your current savings rate of {savings_rate:.1f}% is moderate. "
            "Tightening discretionary spending by 10% could significantly "
            "accelerate progress toward your goal."
        )
    else:
        framework  = "Zero-based budgeting (every dollar assigned a purpose)"
        budget_note = (
            f"Your current savings rate is only {savings_rate:.1f}%, which leaves "
            "little room for saving or investing. Zero-based budgeting can help you "
            "identify and cut unnecessary expenses."
        )

    needs_target   = income * 0.50
    wants_target   = income * 0.30
    savings_target = income * 0.20

    recommendation = (
        f"Based on your monthly income of {format_currency(income)} and expenses of "
        f"{format_currency(expenses)}, your monthly surplus is {format_currency(surplus)}. "
        f"We recommend the **{framework}** to structure your budget and work towards "
        f"**{goal}**. {budget_note}"
    )

    action_steps = [
        "Track all spending for 30 days using a free app (e.g., YNAB, Pocketbook) "
        "to get a clear picture of where your money goes.",
        f"Allocate {format_currency(needs_target)}/mo (50%) to essential needs "
        "(rent, food, transport, utilities).",
        f"Limit discretionary spending to {format_currency(wants_target)}/mo (30%).",
        f"Direct at least {format_currency(savings_target)}/mo (20%) to a dedicated "
        "savings or investment account on payday.",
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

    return {"recommendation": recommendation, "action_steps": action_steps, "explanation": explanation}


def _savings_strategy_mock(profile: dict) -> dict:
    income   = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    savings  = safe_float(profile.get("current_savings", 0))
    surplus  = compute_surplus(income, expenses)
    goal     = profile.get("financial_goal", "your savings goal")
    horizon  = profile.get("investment_horizon", "Medium (2-5 years)")

    if "Long" in horizon:
        save_pct = 0.25
        vehicle  = "a High-Interest Savings Account (HISA) for your emergency fund, then low-cost index ETFs"
    elif "Medium" in horizon:
        save_pct = 0.20
        vehicle  = "a High-Interest Savings Account (HISA) and term deposits"
    else:
        save_pct = 0.15
        vehicle  = "a dedicated High-Interest Savings Account (HISA) or offset account"

    monthly_save         = surplus * save_pct if surplus > 0 else 0
    annual_save          = monthly_save * 12
    emergency_fund_target = expenses * 3

    recommendation = (
        f"With a monthly surplus of {format_currency(surplus)}, we recommend saving at least "
        f"**{format_currency(monthly_save)}/month** ({save_pct*100:.0f}% of surplus) into "
        f"{vehicle}. This will help you build toward {goal}. "
        f"Over 12 months, you could accumulate approximately {format_currency(annual_save)} "
        f"in new savings on top of your existing {format_currency(savings)}."
    )

    action_steps = [
        f"Build an emergency fund of {format_currency(emergency_fund_target)} "
        "(3 months of expenses) before investing.",
        "Open a dedicated savings account separate from your everyday account.",
        f"Set up an automatic transfer of {format_currency(monthly_save)} on payday "
        "('pay yourself first' principle).",
        f"Once the emergency fund is complete, direct additional savings to {vehicle}.",
        "Avoid dipping into savings for non-emergency purchases.",
        "Reassess your savings rate every 6 months as your income grows.",
    ]

    horizon_label = horizon.split("(")[0].strip().lower()
    explanation = (
        f"Your {horizon.lower()} investment horizon guides the choice of savings vehicle. "
        f"For a {horizon_label} horizon, liquidity and capital preservation are prioritised, "
        f"making {vehicle} appropriate. The {save_pct*100:.0f}% target is calibrated to your "
        f"current surplus of {format_currency(surplus)}/month."
    )

    return {"recommendation": recommendation, "action_steps": action_steps, "explanation": explanation}


def _debt_management_mock(profile: dict) -> dict:
    income   = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    debt     = safe_float(profile.get("current_debt", 0))
    surplus  = compute_surplus(income, expenses)
    dti      = compute_debt_to_income_ratio(debt, income)
    goal     = profile.get("financial_goal", "become debt-free")

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
            "Focus on building your emergency fund and then investing "
            f"toward your goal of {goal}."
        )
        action_steps = [
            "Maintain a zero-debt policy by paying credit card balances in full each month.",
            "Redirect funds that would go to debt repayment into a savings or investment account.",
            "Build an emergency fund (3-6 months of expenses) to avoid future debt.",
        ]
        explanation = (
            "With no current debt, your financial foundation is solid. "
            "The focus should be on maintaining this position and building wealth."
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
            "Pay the minimum on all debts every month to protect your credit score.",
            f"Allocate {format_currency(monthly_debt_payment)}/month as your extra repayment "
            f"using the {strategy} approach.",
            "Contact lenders to negotiate a lower interest rate — often possible for credit cards.",
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

    return {"recommendation": recommendation, "action_steps": action_steps, "explanation": explanation}


def _investment_guidance_mock(profile: dict) -> dict:
    income   = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    savings  = safe_float(profile.get("current_savings", 0))
    debt     = safe_float(profile.get("current_debt", 0))
    risk     = profile.get("risk_tolerance", "Medium")
    horizon  = profile.get("investment_horizon", "Medium (2-5 years)")
    goal     = profile.get("financial_goal", "grow your wealth")
    age      = safe_float(profile.get("age", 25))
    surplus  = compute_surplus(income, expenses)

    risk_map = {
        "Low":    {
            "vehicles":   "High-Interest Savings Accounts, government bonds, term deposits",
            "allocation": "80% defensive (bonds/cash), 20% growth (broad index ETFs)",
            "note":       "Capital preservation is the priority.",
        },
        "Medium": {
            "vehicles":   "Diversified ETFs (ASX 200 index, global index), balanced managed funds",
            "allocation": "50% growth (ETFs/shares), 50% defensive (bonds/cash)",
            "note":       "Balanced growth with moderate risk.",
        },
        "High":   {
            "vehicles":   "Growth ETFs, individual shares, property investment trusts (REITs)",
            "allocation": "80% growth (shares/ETFs), 20% defensive (bonds/cash)",
            "note":       "Higher potential returns with higher short-term volatility.",
        },
    }

    pd          = risk_map.get(risk, risk_map["Medium"])
    monthly_inv = surplus * 0.20 if surplus > 0 else 0
    dti         = compute_debt_to_income_ratio(debt, income)
    debt_warning = (
        " **Important:** Your current debt level suggests you should prioritise "
        "debt repayment before investing, as most debt interest rates exceed "
        "typical investment returns."
        if dti > 3 else ""
    )

    recommendation = (
        f"As a beginner investor with **{risk.lower()} risk tolerance** and a "
        f"**{horizon.lower()} horizon**, we recommend starting with "
        f"**{pd['vehicles']}**. "
        f"A suggested allocation is {pd['allocation']}. "
        f"Consider investing {format_currency(monthly_inv)}/month consistently "
        f"to work towards {goal}.{debt_warning}"
    )

    action_steps = [
        "Ensure you have 3 months of expenses saved as an emergency fund before investing.",
        "Open a brokerage account with a low-fee provider (e.g., CommSec Pocket, Stake, SelfWealth).",
        f"Start with {pd['vehicles']} — beginner-friendly and well-diversified.",
        f"Invest {format_currency(monthly_inv)}/month consistently (dollar-cost averaging).",
        f"Follow a {pd['allocation']} portfolio allocation.",
        "Reinvest dividends to compound your returns over time.",
        "Review your portfolio every 6-12 months — avoid checking it daily.",
        "Read: Barefoot Investor (AU), MoneySmart (ASIC), or r/personalfinance.",
    ]

    age_note      = "substantial" if age < 40 else "moderate"
    horizon_label = horizon.split("(")[0].strip().lower()
    explanation = (
        f"The recommended approach is tailored to your {risk.lower()} risk tolerance and "
        f"{horizon_label} investment horizon. At age {int(age)}, you have {age_note} time "
        f"to benefit from compound growth. {pd['note']} Starting with diversified ETFs "
        "reduces single-stock risk, ideal for a beginner investor."
    )

    return {"recommendation": recommendation, "action_steps": action_steps, "explanation": explanation}


def _generate_mock_response(profile: dict, task_type: str, language: str) -> dict:
    """Route to the correct task-specific mock response generator."""
    generators = {
        "Budget Planning":            _budget_planning_mock,
        "Savings Strategy":           _savings_strategy_mock,
        "Debt Management":            _debt_management_mock,
        "Beginner Investment Guidance": _investment_guidance_mock,
    }
    result = generators.get(task_type, _budget_planning_mock)(profile)

    result["disclaimer"] = (
        "This recommendation is generated for educational purposes as part of an "
        "academic project (COMP8420 Assignment 2). It is not personalised financial advice. "
        "Please consult a registered financial adviser before making any financial decisions."
    )
    if language == "Bangla":
        result["disclaimer"] += (
            " | বিঃদ্রঃ বর্তমান সংস্করণে ইংরেজিতে পরামর্শ প্রদান করা হচ্ছে। "
            "ভবিষ্যতে বাংলা ভাষায় সম্পূর্ণ সমর্থন যোগ করা হবে।"
        )
    return result


# ---------------------------------------------------------------------------
# Local Model Placeholder
# ---------------------------------------------------------------------------

def _generate_local_model_response(profile: dict, prompt: dict, language: str) -> dict:
    """Placeholder for a locally-run LLM (e.g., Mistral-7B via Hugging Face)."""
    return {
        "recommendation": (
            "**Local Model Integration — Coming Soon**\n\n"
            "This mode is reserved for a locally-run open-source language model "
            "(e.g., Mistral-7B, LLaMA-3, or Phi-3) loaded via Hugging Face Transformers. "
            "Integration will be added in the next stage of this assignment."
        ),
        "action_steps": [
            "Install: pip install transformers accelerate",
            "Load a model checkpoint (e.g., mistralai/Mistral-7B-Instruct-v0.2)",
            "Pass the structured prompt from prompt_templates.py to the model pipeline.",
            "Parse the output and populate this response dictionary.",
        ],
        "explanation": (
            "The local model mode is designed so that only this function needs updating "
            "when integrating a real model. All other modules remain unchanged."
        ),
        "disclaimer": (
            "This is a placeholder response. Local model integration will be completed "
            "in a later stage of COMP8420 Assignment 2."
        ),
    }


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
        mode:     "Mock Mode" or "Local Model Placeholder".
        language: User selected language.

    Returns:
        A dict with keys: recommendation, action_steps, explanation, disclaimer.
    """
    try:
        if mode == "Mock Mode":
            return _generate_mock_response(profile, prompt.get("task_type", "Budget Planning"), language)
        elif mode == "Local Model Placeholder":
            return _generate_local_model_response(profile, prompt, language)
        else:
            return {"error": f"Unknown mode: '{mode}'. Choose 'Mock Mode' or 'Local Model Placeholder'."}
    except Exception as e:
        return {"error": f"Unexpected error in LLM backend: {str(e)}"}