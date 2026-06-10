import os
import re
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_PRICING = {
    "llama-3.3-70b-versatile": {
        "input": 0.59 / 1_000_000,
        "output": 0.79 / 1_000_000
    },
    "llama-3.1-8b-instant": {
        "input": 0.05 / 1_000_000,
        "output": 0.08 / 1_000_000
    }
}


class UsageTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.total_latency = 0.0
        self.model_usage = {}
        self.details = []

    def add_call(self, model, prompt_tokens, completion_tokens, latency):
        self.total_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_latency += latency

        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (prompt_tokens * pricing["input"]) + (completion_tokens * pricing["output"])
        self.total_cost += cost

        if model not in self.model_usage:
            self.model_usage[model] = {"prompt": 0, "completion": 0, "cost": 0.0, "calls": 0}
        self.model_usage[model]["prompt"] += prompt_tokens
        self.model_usage[model]["completion"] += completion_tokens
        self.model_usage[model]["cost"] += cost
        self.model_usage[model]["calls"] += 1

        self.details.append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "latency": latency
        })


tracker = UsageTracker()


def call_with_retry(model, messages, max_tokens):
    """Calls Groq API and auto-waits if rate limited — never crashes, tracks metrics"""
    start_time = time.time()
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages
            )
            latency = time.time() - start_time
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            tracker.add_call(model, prompt_tokens, completion_tokens, latency)
            return response
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                wait = 60
                match = re.search(r'try again in (\d+)m(\d+)', error_msg)
                if match:
                    wait = int(match.group(1)) * 60 + int(match.group(2)) + 10
                print(f"\n  [RATE LIMIT] Waiting {wait} seconds automatically...\n")
                time.sleep(wait)
            else:
                raise


ROLE_PROMPTS = {
    "Arithmetic Specialist": """You are a mathematical specialist focusing on absolute numerical and arithmetic precision.
Your role is to double-check every single addition, subtraction, multiplication, and division.
STRICT RULES:
1. Write down each calculation on a new line.
2. Verify each calculation by recalculating it using a different method (e.g., if you multiply, verify by division or addition).
3. Do not skip any steps.
4. Output your final answer as: FINAL ANSWER: [integer only, no units, no decimals]""",

    "Equation Builder": """You are a mathematical specialist focusing on constructing clear, step-by-step algebraic equations.
Your role is to translate the word problem into precise mathematical equations before solving.
STRICT RULES:
1. Define variables for all unknown values.
2. Write down the algebraic equations representing the relationships in the problem.
3. Solve the equations step-by-step, showing all algebraic manipulations.
4. Output your final answer as: FINAL ANSWER: [integer only, no units, no decimals]""",

    "Backward Reasoner": """You are a mathematical specialist who solves problems by reasoning backwards from the goal.
Your role is to analyze what the final question is asking, assume a path to the answer, and work backwards to verify assumptions.
STRICT RULES:
1. Start with the final quantity to find.
2. Identify what intermediate values are needed to compute it.
3. Work backwards to the given starting values.
4. Solve the steps in reverse order, then verify forwards.
5. Output your final answer as: FINAL ANSWER: [integer only, no units, no decimals]""",

    "Verification Agent": """You are a mathematical verification specialist with zero tolerance for errors.
Your role is to check the problem for hidden constraints, verify all assumptions, and check the dimensions/units.
STRICT RULES:
1. List all constraints and numbers given in the question.
2. After solving, check if your answer makes physical and logical sense.
3. Verify that you have used all necessary information and have not introduced external assumptions.
4. Output your final answer as: FINAL ANSWER: [integer only, no units, no decimals]""",

    "Word-Problem Translator": """You are a linguistic and mathematical translator.
Your role is to translate complex, wordy problems into clean, structured facts and facts-to-variables lists.
STRICT RULES:
1. Read the problem word-by-word.
2. Write down a structured bulleted list of "Given Facts".
3. Write down a bulleted list of "Relations" between facts.
4. Solve the problem sequentially using these structured facts.
5. Output your final answer as: FINAL ANSWER: [integer only, no units, no decimals]"""
}


def solve(question: str, role_name: str = "", feedback: str = "") -> str:
    """Solver agent — answers GSM8K math question with dynamic role specialization"""

    # default prompt
    system_prompt = """You are a world-class mathematician specializing in precise arithmetic reasoning.

STRICT RULES:
1. Read the problem twice before starting
2. Identify all given values and what is being asked
3. Write Step 1, Step 2, Step 3... for every single operation
4. After every arithmetic operation, verify it immediately
5. Never skip steps even if they seem obvious
6. Before writing final answer, re-read the question to confirm you answered what was asked
7. Write your final answer as: FINAL ANSWER: [integer only, no units, no decimals]"""

    # if monitor assigned a specialized role, use that role's prompt
    if role_name in ROLE_PROMPTS:
        system_prompt = ROLE_PROMPTS[role_name]

    # if monitor provided feedback, append as critical instruction
    if feedback:
        system_prompt += f"\n\nCRITICAL INSTRUCTION FROM MONITOR: {feedback}"

    message = call_with_retry(
        model="llama-3.3-70b-versatile",
        max_tokens=768,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Solve this math problem:\n\n{question}"}
        ]
    )
    time.sleep(0.5)
    return message.choices[0].message.content


def critic_solve(question: str) -> str:
    """Critic agent — solves independently using llama-3.1-8b-instant"""
    system_prompt = """You are an independent, high-precision mathematician.
Solve the given math problem step-by-step. Double-check your arithmetic.
Always write your final answer as: FINAL ANSWER: [integer only, no units, no decimals]"""

    message = call_with_retry(
        model="llama-3.1-8b-instant",
        max_tokens=512,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Solve this math problem:\n\n{question}"}
        ]
    )
    time.sleep(0.5)
    return message.choices[0].message.content


def monitor_decide(question: str, solver_sol: str, solver_ans: str, critic_sol: str, critic_ans: str, attempt: int, max_attempts: int) -> tuple[str, str, str]:
    """
    Monitor agent — analyzes Solver and Critic solutions and decides
    whether to accept or swap with a specialized role.
    Uses llama-3.1-8b-instant.
    """
    system_prompt = """You are an expert math meta-cognitive Monitor. Your job is to analyze the reasoning paths of a Solver and an independent Critic, identify the source of any disagreement or error, and select the best strategy for the next attempt.

NOTE: The Solver uses a larger, more capable model than the Critic.
When there is disagreement, prefer the Solver's answer unless the Critic's
step-by-step reasoning is clearly more rigorous and catches a specific error.

Available specialized roles for Solver:
- "Arithmetic Specialist": Best for calculation slips or simple arithmetic mistakes.
- "Equation Builder": Best for translating word problems into algebraic equations.
- "Backward Reasoner": Best for checking if working backwards from the answer makes sense.
- "Verification Agent": Best for reviewing constraints, checking if all facts are used, and avoiding traps.
- "Word-Problem Translator": Best for parsing wordy text and listing given facts/variables.

Analyze the solver and critic solutions, and choose:
1. DECISION: "swap" to try again with a specialized role, or "accept" if you believe the Solver is correct.
2. ROLE: One of the 5 roles above.
3. CONTEXT: One sentence of direct feedback instructing the Solver what to correct.

You must end your response with these exact three lines, and nothing after them:
DECISION: [swap or accept]
ROLE: [one of the 5 roles above]
CONTEXT: [one sentence of feedback]"""

    user_content = f"""Question: {question}

Solver's Solution:
{solver_sol}
Solver's Final Answer: {solver_ans}

Critic's Solution:
{critic_sol}
Critic's Final Answer: {critic_ans}

Current Attempt: {attempt + 1} of {max_attempts}"""

    message = call_with_retry(
        model="llama-3.1-8b-instant",
        max_tokens=300,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )

    content = message.choices[0].message.content

    decision = "swap"
    role = "Arithmetic Specialist"
    context = "Please verify your calculations."

    for line in content.strip().split("\n"):
        if line.startswith("DECISION:"):
            decision = line.replace("DECISION:", "").strip().lower()
            decision = "accept" if "accept" in decision else "swap"
        elif line.startswith("ROLE:"):
            role_candidate = line.replace("ROLE:", "").strip()
            for possible_role in ROLE_PROMPTS.keys():
                if possible_role.lower() in role_candidate.lower():
                    role = possible_role
                    break
        elif line.startswith("CONTEXT:"):
            context = line.replace("CONTEXT:", "").strip()

    time.sleep(0.5)
    return decision, role, context


def monitor_finalize(question: str, history: list) -> str:
    """
    Monitor reviews entire attempt history and selects the best final answer.
    Uses llama-3.1-8b-instant.
    """
    system_prompt = """You are an expert math meta-cognitive Monitor.
You will be given a math question and a history of attempts by a Solver and Critic.
Your job is to evaluate all the solutions, determine which numeric answer is correct based on the strongest mathematical reasoning, and output that final answer.

Your output must end with this exact line, and nothing after it:
FINAL ANSWER: [integer only, no units, no decimals]"""

    history_str = ""
    for idx, item in enumerate(history):
        history_str += f"\n--- Solution {idx+1} ({item['agent']}) ---\nAnswer: {item['answer']}\nReasoning:\n{item['solution']}\n"

    message = call_with_retry(
        model="llama-3.1-8b-instant",
        max_tokens=256,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n{history_str}"}
        ]
    )

    time.sleep(0.5)
    return message.choices[0].message.content


def critique(question: str, solution: str) -> tuple[str, int]:
    """Critic agent — scores solution 1 to 10, used by baseline modes"""

    system_prompt = """You are an unforgiving math exam checker with zero tolerance for errors.

YOUR JOB:
You will be given a math problem and a student's solution.
You must independently solve the problem yourself first, then compare.

STEP 1 — SOLVE IT YOURSELF:
- Solve the problem completely on your own without looking at the student's answer
- Write YOUR OWN SOLUTION clearly

STEP 2 — COMPARE:
- Compare your answer to the student's FINAL ANSWER
- If the numbers are different, score is automatically 4 or below, no exceptions
- If the numbers match, check if their reasoning was correct
- Even one wrong arithmetic step = deduct 2 points

STEP 3 — SCORE:
- 9-10: Answer correct, all steps correct, clear reasoning
- 7-8: Answer correct, minor reasoning gaps
- 5-6: Answer correct but steps have errors or are missing
- 3-4: Answer wrong but approach was partially right
- 1-2: Answer wrong and reasoning is completely off

END your response with these exact two lines, nothing after them:
SCORE: [single integer 1-10]
REASON: [one sentence only]"""

    message = call_with_retry(
        model="llama-3.3-70b-versatile",
        max_tokens=256,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nSolution to evaluate:\n{solution}"}
        ]
    )

    time.sleep(0.5)
    feedback = message.choices[0].message.content

    score = 5
    for line in feedback.strip().split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = int(line.replace("SCORE:", "").strip())
                score = max(1, min(10, score))
            except ValueError:
                pass

    return feedback, score