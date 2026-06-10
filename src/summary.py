import os
import sys
import pandas as pd

# Reconfigure stdout to use UTF-8 just in case there are unicode chars
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def show_summary():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "..", "results")

    modes = ["single_agent", "fixed_roles", "random_swap", "monitor_swap"]

    summary = []

    for mode in modes:
        path = os.path.join(results_dir, f"{mode}_results.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            total = len(df)
            correct = int(df["correct"].sum())
            accuracy = correct / total * 100
            
            # Optional new columns with defaults if not present
            avg_rounds = df["rounds"].mean() if "rounds" in df.columns else 0.0
            avg_swaps = df["swaps"].mean() if "swaps" in df.columns else 0.0
            
            avg_latency = df["latency"].mean() if "latency" in df.columns else 0.0
            avg_cost = df["cost"].mean() if "cost" in df.columns else 0.0
            
            # Avg tokens is prompt + completion
            if "prompt_tokens" in df.columns and "completion_tokens" in df.columns:
                avg_tokens = (df["prompt_tokens"] + df["completion_tokens"]).mean()
            else:
                avg_tokens = 0.0
                
            # Agreement rate (how often they agreed initially)
            if "agreed" in df.columns:
                agreed_pct = df["agreed"].mean() * 100
            else:
                agreed_pct = 0.0
                
            # Intervention rate
            if "monitor_intervened" in df.columns:
                intervened_pct = df["monitor_intervened"].mean() * 100
            else:
                intervened_pct = 0.0

            summary.append({
                "mode": mode,
                "questions": total,
                "correct": correct,
                "accuracy": f"{accuracy:.1f}%",
                "avg_latency": f"{avg_latency:.2f}s",
                "avg_tokens": f"{avg_tokens:.1f}",
                "avg_cost": f"${avg_cost:.5f}",
                "avg_swaps": f"{avg_swaps:.2f}",
                "agreement_rate": f"{agreed_pct:.1f}%" if "agreed" in df.columns else "-",
                "intervention_rate": f"{intervened_pct:.1f}%" if "monitor_intervened" in df.columns else "-"
            })
        else:
            summary.append({
                "mode": mode,
                "questions": "-",
                "correct": "-",
                "accuracy": "NOT RUN YET",
                "avg_latency": "-",
                "avg_tokens": "-",
                "avg_cost": "-",
                "avg_swaps": "-",
                "agreement_rate": "-",
                "intervention_rate": "-"
            })

    print("\n" + "="*110)
    print("  EXPERIMENT SUMMARY — Dynamic Role Specialization")
    print("="*110)
    headers = f"{'Mode':<18} {'Questions':<10} {'Correct':<8} {'Accuracy':<10} {'Avg Latency':<12} {'Avg Tokens':<11} {'Avg Cost':<11} {'Avg Swaps':<10} {'Agreement':<10} {'Intervent'}"
    print(headers)
    print("-"*110)
    for row in summary:
        tag = " <- OURS" if row["mode"] == "monitor_swap" else ""
        print(f"{row['mode']:<18} {str(row['questions']):<10} {str(row['correct']):<8} {row['accuracy']:<10} {row['avg_latency']:<12} {row['avg_tokens']:<11} {row['avg_cost']:<11} {row['avg_swaps']:<10} {row['agreement_rate']:<10} {row['intervention_rate']}{tag}")
    print("="*110)

    # print best mode if results exist
    ran = [r for r in summary if r["accuracy"] != "NOT RUN YET"]
    if ran:
        best = max(ran, key=lambda x: float(x["accuracy"].replace("%", "")))
        print(f"\n  Best performing mode: {best['mode']} at {best['accuracy']}")

    print()

if __name__ == "__main__":
    show_summary()