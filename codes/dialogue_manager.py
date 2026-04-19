"""
dialogue_manager.py — Orchestration layer for SmartFinance AI Assistant

Receives raw user inputs, structures them into a validated profile,
builds the appropriate prompt, calls the LLM backend, and returns
a clean structured response to the Streamlit frontend.
"""

from prompt_templates import (
    build_structured_prompt,
    build_system_prompt,
    build_user_prompt,
)
from llm_backend import generate_financial_response
from utils import safe_float, safe_int, summarise_profile


def build_profile(raw_inputs: dict) -> dict:
    """
    Normalise and structure raw form inputs into a clean profile dictionary.

    Args:
        raw_inputs: The raw dict from the Streamlit form.

    Returns:
        A clean, typed profile dictionary.
    """
    return {
        "age": safe_int(raw_inputs.get("age", 25)),
        "employment_status": str(raw_inputs.get("employment_status", "")).strip(),
        "monthly_income": safe_float(raw_inputs.get("monthly_income", 0.0)),
        "monthly_expenses": safe_float(raw_inputs.get("monthly_expenses", 0.0)),
        "current_savings": safe_float(raw_inputs.get("current_savings", 0.0)),
        "current_debt": safe_float(raw_inputs.get("current_debt", 0.0)),
        "risk_tolerance": str(raw_inputs.get("risk_tolerance", "Medium")).strip(),
        "financial_goal": str(raw_inputs.get("financial_goal", "")).strip(),
        "investment_horizon": str(
            raw_inputs.get("investment_horizon", "Medium (2-5 years)")
        ).strip(),
        "extra_preferences": str(raw_inputs.get("extra_preferences", "")).strip(),
    }


def run_financial_dialogue(
    profile: dict,
    task_type: str,
    mode: str = "Mock Mode",
    language: str = "English",
) -> dict:
    """
    Orchestrate the full dialogue pipeline:
      1. Normalise and structure the user profile.
      2. Build the LLM prompt using prompt_templates.
      3. Call the LLM backend to generate a response.
      4. Return the structured result with a profile summary attached.

    Args:
        profile:   User financial profile dict (already validated by app.py).
        task_type: The selected financial task (e.g., "Budget Planning").
        mode:      Model mode — "Mock Mode" or "TinyLlama (Selected Model)".
        language:  Selected language for the response.

    Returns:
        Result dict with keys: recommendation, action_steps, explanation,
        disclaimer, profile_summary. 'error' key only present on failure.
    """
    try:
        # Step 1: Ensure the profile is cleanly structured
        clean_profile = build_profile(profile)

        # Step 2: Build strong prompts for the backend
        system_prompt = build_system_prompt(language)
        user_prompt = build_user_prompt(clean_profile, task_type, language)

        # Keep structured prompt pieces for preview / assignment evidence
        preview_prompt = build_structured_prompt(clean_profile, task_type, language)

        prompt = {
            "system": system_prompt,
            "user": user_prompt,
            "task_type": task_type,
            "zero_shot": preview_prompt.get("zero_shot", ""),
            "structured": preview_prompt.get("structured", ""),
            "few_shot": preview_prompt.get("few_shot", ""),
        }

        # Step 3: Generate the response
        result = generate_financial_response(
            profile=clean_profile,
            prompt=prompt,
            mode=mode,
            language=language,
        )

        # Step 4: Attach metadata and prompt evidence
        result["profile_summary"] = summarise_profile(clean_profile)
        result["prompt_used"] = {
            "task_type": task_type,
            "system": system_prompt,
            "user": user_prompt,
            "zero_shot": preview_prompt.get("zero_shot", ""),
            "structured": preview_prompt.get("structured", ""),
            "few_shot": preview_prompt.get("few_shot", ""),
        }

        return result

    except Exception as e:
        return {
            "recommendation": "",
            "action_steps": [],
            "explanation": "",
            "disclaimer": "",
            "profile_summary": "",
            "error": f"Dialogue manager error: {str(e)}",
        }


class DialogueManager:
    """
    Small compatibility wrapper so the app can use:
        manager = DialogueManager()
        result = manager.generate_response(...)
    """

    def generate_response(
        self,
        profile: dict,
        task_type: str,
        mode: str = "Mock Mode",
        language: str = "English",
    ) -> dict:
        return run_financial_dialogue(
            profile=profile,
            task_type=task_type,
            mode=mode,
            language=language,
        )