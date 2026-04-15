"""
utils.py — Helper utilities for SmartFinance AI Assistant
Provides input validation, formatting, and safe conversion functions.
"""


def safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float; return default on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default: int = 0) -> int:
    """Safely convert a value to int; return default on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def format_currency(amount, symbol: str = "$", decimals: int = 2) -> str:
    """Format a numeric value as a currency string."""
    try:
        return f"{symbol}{float(amount):,.{decimals}f}"
    except (ValueError, TypeError):
        return f"{symbol}0.00"


def compute_surplus(income: float, expenses: float) -> float:
    """Return the monthly surplus (income minus expenses)."""
    return income - expenses


def compute_savings_rate(income: float, expenses: float) -> float:
    """Return savings rate as a percentage of income. Returns 0.0 if income is 0."""
    if income <= 0:
        return 0.0
    surplus = income - expenses
    return round((surplus / income) * 100, 2)


def compute_debt_to_income_ratio(debt: float, income: float) -> float:
    """
    Return the debt-to-monthly-income ratio.
    A ratio > 5 (debt > 5x monthly income) is considered high.
    """
    if income <= 0:
        return 0.0
    return round(debt / income, 2)


def validate_profile(profile: dict) -> list:
    """
    Validate the user-submitted financial profile.
    Returns a list of error message strings. Empty list means all clear.
    """
    errors = []

    income   = safe_float(profile.get("monthly_income", 0))
    expenses = safe_float(profile.get("monthly_expenses", 0))
    savings  = safe_float(profile.get("current_savings", 0))
    debt     = safe_float(profile.get("current_debt", 0))
    age      = safe_int(profile.get("age", 0))
    goal     = str(profile.get("financial_goal", "")).strip()

    if income <= 0:
        errors.append("Monthly income must be greater than zero.")
    if expenses < 0:
        errors.append("Monthly expenses cannot be negative.")
    if savings < 0:
        errors.append("Current savings cannot be negative.")
    if debt < 0:
        errors.append("Current debt cannot be negative.")
    if age < 16 or age > 100:
        errors.append("Age must be between 16 and 100.")
    if not goal:
        errors.append("Please enter a financial goal (e.g., 'Save for a house deposit').")
    if income > 0 and expenses > income * 5:
        errors.append(
            "Monthly expenses appear unusually high relative to income. "
            "Please double-check your figures."
        )

    return errors


def summarise_profile(profile: dict) -> str:
    """Return a concise human-readable profile summary string."""
    income   = format_currency(profile.get("monthly_income", 0))
    expenses = format_currency(profile.get("monthly_expenses", 0))
    savings  = format_currency(profile.get("current_savings", 0))
    debt     = format_currency(profile.get("current_debt", 0))
    surplus  = compute_surplus(
        safe_float(profile.get("monthly_income", 0)),
        safe_float(profile.get("monthly_expenses", 0)),
    )
    savings_rate = compute_savings_rate(
        safe_float(profile.get("monthly_income", 0)),
        safe_float(profile.get("monthly_expenses", 0)),
    )

    return (
        f"Age {profile.get('age')}, {profile.get('employment_status')} | "
        f"Income: {income}/mo | Expenses: {expenses}/mo | "
        f"Surplus: {format_currency(surplus)}/mo ({savings_rate}%) | "
        f"Savings: {savings} | Debt: {debt} | "
        f"Risk Tolerance: {profile.get('risk_tolerance')} | "
        f"Goal: {profile.get('financial_goal')}"
    )