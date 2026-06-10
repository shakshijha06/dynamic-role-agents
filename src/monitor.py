from agents import monitor_decide, monitor_finalize
import re


def _extract_numeric_value(text: str) -> str:
    """Extract a clean numeric string from text."""
    if not text:
        return "N/A"
    cleaned = text.replace("$", "").replace("€", "").replace("£", "").replace("%", "")
    cleaned = cleaned.replace(",", "")
    match = re.search(r'-?\d+(?:\.\d+)?', cleaned)
    if match:
        return match.group(0)
    return "N/A"


def _numeric_equal(predicted: str, actual: str) -> bool:
    """Returns True if numeric values are within 0.01 tolerance."""
    p_val = _extract_numeric_value(predicted)
    a_val = _extract_numeric_value(actual)
    if p_val == "N/A" or a_val == "N/A":
        return predicted.strip().replace(",", "") == actual.strip().replace(",", "")
    try:
        return abs(float(p_val) - float(a_val)) < 0.01
    except:
        return p_val == a_val


class Monitor:
    """Monitor agent wrapper — tracks metrics and delegates to LLM decisions"""

    def __init__(self, threshold: int = 7, max_attempts: int = 3):
        self.threshold = threshold
        self.max_attempts = max_attempts
        self.swap_count = 0
        self.swap_successes = 0
        self.total_questions = 0
        self.agreements = 0
        self.disagreements = 0
        self.interventions = 0

    def decide(self, question: str, solver_sol: str, solver_ans: str, critic_sol: str, critic_ans: str, attempt: int) -> tuple[str, str, str]:
        """
        Decides whether to accept or swap with a specialized role.
        Returns: (decision, role_name, context)
        """
        # if answers match numerically, accept immediately — no LLM call needed
        if _numeric_equal(solver_ans, critic_ans):
            return "accept", "", ""

        # call LLM monitor to analyze disagreement and pick best role
        decision, role, context = monitor_decide(
            question, solver_sol, solver_ans, critic_sol, critic_ans, attempt, self.max_attempts
        )

        if decision == "swap":
            self.swap_count += 1

        return decision, role, context

    def finalize(self, question: str, history: list) -> str:
        """Delegates final consensus decision to LLM monitor"""
        return monitor_finalize(question, history)

    def record_question_start(self):
        self.total_questions += 1

    def record_initial_comparison(self, matched: bool):
        if matched:
            self.agreements += 1
        else:
            self.disagreements += 1
            if self.max_attempts > 1:
                self.interventions += 1

    def get_stats(self) -> dict:
        """Returns swap and agreement statistics for paper metrics"""
        agreement_rate = (self.agreements / self.total_questions * 100) if self.total_questions > 0 else 0.0
        disagreement_rate = (self.disagreements / self.total_questions * 100) if self.total_questions > 0 else 0.0
        intervention_rate = (self.interventions / self.total_questions * 100) if self.total_questions > 0 else 0.0

        return {
            "total_questions": self.total_questions,
            "total_swaps": self.swap_count,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "interventions": self.interventions,
            "agreement_rate": round(agreement_rate, 1),
            "disagreement_rate": round(disagreement_rate, 1),
            "intervention_rate": round(intervention_rate, 1)
        }

    def reset(self):
        """Reset all metrics"""
        self.total_questions = 0
        self.swap_count = 0
        self.swap_successes = 0
        self.agreements = 0
        self.disagreements = 0
        self.interventions = 0