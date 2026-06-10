import os
import json
import pandas as pd
import re
from dotenv import load_dotenv
from agents import solve, critique, critic_solve
from monitor import Monitor

load_dotenv()

# ─── CHANGE THIS TO SWITCH MODES ───────────────────────────
MODE = "single_agent"  # options: single_agent | fixed_roles | random_swap | monitor_swap
NUM_QUESTIONS = 5      # 5 for quick test, 50 for MVP, 200 for paper
# ────────────────────────────────────────────────────────────

def load_questions(path: str, n: int) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    return data[:n]

def extract_numeric_value(text: str) -> str:
    """Extracts a clean numeric string from a raw answer string."""
    if not text:
        return "N/A"
    text = text.replace("$", "").replace("€", "").replace("£", "").replace("%", "")
    text = text.replace(",", "")
    match = re.search(r'-?\d+(?:\.\d+)?', text)
    if match:
        return match.group(0)
    return "N/A"

def extract_final_answer(solution: str) -> str:
    """Extract the number after FINAL ANSWER: from solver output"""
    if not solution:
        return "N/A"
    for line in solution.strip().split("\n"):
        if "FINAL ANSWER:" in line:
            raw = line.replace("FINAL ANSWER:", "").strip()
            if not raw:
                return "N/A"
            val = extract_numeric_value(raw)
            if val != "N/A":
                return val
            return raw
    # fallback — scan all lines for a number if FINAL ANSWER not found
    for line in reversed(solution.strip().split("\n")):
        val = extract_numeric_value(line)
        if val != "N/A":
            return val
    return "N/A"

def is_correct(predicted: str, actual: str) -> bool:
    p_val = extract_numeric_value(predicted)
    a_val = extract_numeric_value(actual)
    if p_val == "N/A" or a_val == "N/A":
        return predicted.strip().replace(",", "") == actual.strip().replace(",", "")
    try:
        return abs(float(p_val) - float(a_val)) < 0.01
    except:
        return p_val == a_val


# ─── BASELINE 1: Single Agent, no critic, no loop ───────────
def run_single_agent(question: dict) -> dict:
    """Weakest baseline — one shot, no feedback"""
    from agents import tracker
    tracker.reset()

    solution = solve(question["question"])
    predicted = extract_final_answer(solution)
    correct = is_correct(predicted, str(question["answer"]))

    return {
        "predicted": predicted,
        "correct": correct,
        "swaps": 0,
        "rounds": 1,
        "latency": tracker.total_latency,
        "prompt_tokens": tracker.total_prompt_tokens,
        "completion_tokens": tracker.total_completion_tokens,
        "cost": tracker.total_cost,
        "agreed": True,
        "critic_disagreed": False,
        "monitor_intervened": False
    }


# ─── BASELINE 2: Fixed Roles — Solver + Critic loop, no swap ─
def run_fixed_roles(question: dict, max_rounds: int = 3) -> dict:
    """Critic gives feedback but roles never change"""
    from agents import tracker
    tracker.reset()

    best_answer = None
    best_score = 0
    solution = ""
    round_num = 0

    for round_num in range(max_rounds):
        solution = solve(question["question"])
        feedback, score = critique(question["question"], solution)
        print(f"    Round {round_num+1} | Score: {score}")

        if score > best_score:
            best_score = score
            best_answer = solution

        if score >= 7:
            break

    predicted = extract_final_answer(best_answer)
    correct = is_correct(predicted, str(question["answer"]))

    return {
        "predicted": predicted,
        "correct": correct,
        "swaps": 0,
        "rounds": round_num + 1,
        "latency": tracker.total_latency,
        "prompt_tokens": tracker.total_prompt_tokens,
        "completion_tokens": tracker.total_completion_tokens,
        "cost": tracker.total_cost,
        "agreed": True if best_score >= 7 else False,
        "critic_disagreed": True if best_score < 7 else False,
        "monitor_intervened": False
    }


# ─── BASELINE 3: Random Swap — swaps randomly, not intelligently
def run_random_swap(question: dict, max_rounds: int = 3) -> dict:
    """Swaps happen randomly — proves monitor swap is smarter"""
    import random
    random.seed(42)
    from agents import tracker
    tracker.reset()

    best_answer = None
    best_score = 0
    solution = ""
    swaps = 0
    round_num = 0

    random_contexts = [
        "Try a different method than before.",
        "Use estimation to solve this problem.",
        "Break the problem into smaller parts."
    ]

    feedback_text = ""
    for round_num in range(max_rounds):
        solution = solve(question["question"], feedback=feedback_text)  # ← FIXED
        score_feedback, score = critique(question["question"], solution)
        print(f"    Round {round_num+1} | Score: {score}")

        if score > best_score:
            best_score = score
            best_answer = solution

        # random swap — 40% chance regardless of score
        if random.random() < 0.4:
            swaps += 1
            feedback_text = random.choice(random_contexts)
            print(f"  [RANDOM] Swap triggered randomly!")

        if score >= 7:
            break

    predicted = extract_final_answer(best_answer)
    correct = is_correct(predicted, str(question["answer"]))

    return {
        "predicted": predicted,
        "correct": correct,
        "swaps": swaps,
        "rounds": round_num + 1,
        "latency": tracker.total_latency,
        "prompt_tokens": tracker.total_prompt_tokens,
        "completion_tokens": tracker.total_completion_tokens,
        "cost": tracker.total_cost,
        "agreed": True if best_score >= 7 else False,
        "critic_disagreed": True if best_score < 7 else False,
        "monitor_intervened": swaps > 0
    }


# ─── YOUR NOVEL SYSTEM: Monitor Swap ─────────────────────────
def run_monitor_swap(question: dict) -> dict:
    """
    YOUR NOVEL CONTRIBUTION.
    Monitor compares independent Solver and Critic solutions.
    If they differ, Monitor uses LLM analysis to choose specialized solver roles.
    If all attempts are exhausted, Monitor reviews history for final consensus.
    """
    from agents import solve, critic_solve, tracker

    tracker.reset()

    monitor = Monitor(max_attempts=3)
    monitor.record_question_start()

    history = []

    # Attempt 1 — Solver and Critic both solve independently
    solver_sol = solve(question["question"])
    solver_ans = extract_final_answer(solver_sol)
    history.append({
        "agent": "Solver Attempt 1 (Default)",
        "solution": solver_sol,
        "answer": solver_ans
    })

    critic_sol = critic_solve(question["question"])
    critic_ans = extract_final_answer(critic_sol)
    history.append({
        "agent": "Critic Independent Solve",
        "solution": critic_sol,
        "answer": critic_ans
    })

    matched = is_correct(solver_ans, critic_ans)
    monitor.record_initial_comparison(matched)

    predicted = solver_ans
    rounds = 1

    if matched:
        print("    Attempt 1 | Solver and Critic AGREE immediately!")
    else:
        print(f"    Attempt 1 | DISAGREEMENT! Solver: {solver_ans} vs Critic: {critic_ans}")

        attempt = 1
        role_name = ""
        feedback = ""

        while attempt < 3:
            decision, role_name, feedback = monitor.decide(
                question["question"], solver_sol, solver_ans, critic_sol, critic_ans, attempt - 1
            )

            if decision == "accept":
                print(f"    [MONITOR] Accepted Solver answer '{solver_ans}' at attempt {attempt}")
                predicted = solver_ans
                break

            print(f"    [MONITOR] Intervened | Swap triggered! Role -> {role_name} | Feedback: {feedback}")

            solver_sol = solve(question["question"], role_name=role_name, feedback=feedback)
            solver_ans = extract_final_answer(solver_sol)

            history.append({
                "agent": f"Solver Attempt {attempt+1} ({role_name})",
                "solution": solver_sol,
                "answer": solver_ans
            })

            rounds = attempt + 1
            print(f"    Attempt {attempt+1} ({role_name}) | Solver: {solver_ans} vs Critic: {critic_ans}")

            if is_correct(solver_ans, critic_ans):
                print(f"    [MONITOR] Agreement reached with Critic!")
                predicted = solver_ans
                break

            attempt += 1
        else:
            # all attempts exhausted — run final consensus
            print("    [MONITOR] No consensus reached. Running Final Consensus Review...")
            from collections import Counter
            all_answers = [item["answer"] for item in history if item["answer"] != "N/A"]
            if all_answers:
                most_common = Counter(all_answers).most_common(1)[0][0]
                predicted = most_common
                print(f"    [MONITOR] Majority vote selected: {predicted}")

            else:
                final_sol = monitor.finalize(question["question"], history)
                predicted = extract_final_answer(final_sol)  # fallback to last solver answer
            print(f"    [MONITOR] Final consensus answer selected: {predicted}")

    correct = is_correct(predicted, str(question["answer"]))

    return {
        "predicted": predicted,
        "correct": correct,
        "swaps": monitor.swap_count,
        "rounds": rounds,
        "latency": tracker.total_latency,
        "prompt_tokens": tracker.total_prompt_tokens,
        "completion_tokens": tracker.total_completion_tokens,
        "cost": tracker.total_cost,
        "agreed": matched,
        "critic_disagreed": not matched,
        "monitor_intervened": monitor.swap_count > 0
    }


# ─── MAIN ────────────────────────────────────────────────────
def main():
    print(f"\n{'='*50}")
    print(f"  MODE: {MODE} | Questions: {NUM_QUESTIONS}")
    print(f"{'='*50}\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "questions.json")
    results_dir = os.path.join(base_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    questions = load_questions(data_path, NUM_QUESTIONS)

    results = []
    correct_count = 0
    total_swaps = 0

    for i, q in enumerate(questions):
        print(f"Q{i+1}: {q['question'][:60]}...")

        if MODE == "single_agent":
            result = run_single_agent(q)
        elif MODE == "fixed_roles":
            result = run_fixed_roles(q)
        elif MODE == "random_swap":
            result = run_random_swap(q)
        elif MODE == "monitor_swap":
            result = run_monitor_swap(q)
        else:
            raise ValueError(f"Unknown MODE: {MODE}")

        if result["correct"]:
            correct_count += 1
        total_swaps += result["swaps"]

        results.append({
            "question_id": i + 1,
            "question": q["question"][:80],
            "expected": q["answer"],
            "predicted": result["predicted"],
            "correct": result["correct"],
            "swaps": result["swaps"],
            "rounds": result["rounds"],
            "latency": result.get("latency", 0.0),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "cost": result.get("cost", 0.0),
            "agreed": result.get("agreed", False),
            "critic_disagreed": result.get("critic_disagreed", False),
            "monitor_intervened": result.get("monitor_intervened", False),
            "mode": MODE
        })

        status = "CORRECT" if result["correct"] else "INCORRECT"
        print(f"  -> Predicted: {result['predicted']} | Expected: {q['answer']} | {status}")
        print(f"    Rounds: {result['rounds']} | Swaps: {result['swaps']}")
        print(f"    Latency: {result.get('latency', 0.0):.2f}s | Cost: ${result.get('cost', 0.0):.5f} | Tokens: {result.get('prompt_tokens', 0) + result.get('completion_tokens', 0)}")
        print()

    # ─── FINAL SUMMARY ───────────────────────────────────────
    df = pd.DataFrame(results)
    accuracy = correct_count / NUM_QUESTIONS * 100
    avg_latency = df["latency"].mean()
    total_cost = df["cost"].sum()
    avg_tokens = (df["prompt_tokens"] + df["completion_tokens"]).mean()

    print(f"\n{'='*50}")
    print(f"  MODE:                      {MODE}")
    print(f"  ACCURACY:                  {accuracy:.1f}% ({correct_count}/{NUM_QUESTIONS})")
    print(f"  SWAPS:                     {total_swaps} total across all questions")
    print(f"  AVG SWAPS PER QUESTION:    {total_swaps/NUM_QUESTIONS:.2f}")
    print(f"  AVG LATENCY PER QUESTION:  {avg_latency:.2f}s")
    print(f"  AVG TOKENS PER QUESTION:   {avg_tokens:.1f}")
    print(f"  TOTAL COST:                ${total_cost:.5f}")
    print(f"{'='*50}\n")

    output_path = os.path.join(results_dir, f"{MODE}_results.csv")
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()