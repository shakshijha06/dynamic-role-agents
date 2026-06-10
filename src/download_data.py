import json
import os
from datasets import load_dataset

def download_gsm8k(num_questions: int = 200, split: str = "test"):
    print("Downloading GSM8K dataset from HuggingFace...")
    
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    
    questions = []
    for i, item in enumerate(dataset):
        if i >= num_questions:
            break
        
        answer_text = item["answer"]
        final_answer = answer_text.split("####")[-1].strip()
        
        # Normalize commas: "2,125" -> "2125"
        final_answer = final_answer.replace(",", "")
        
        questions.append({ 
            "id": i,
            "question": item["question"],
            "answer": final_answer
        })
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "questions.json")
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
    
    print(f"Saved {len(questions)} questions to {output_path}")

if __name__ == "__main__":
    download_gsm8k(num_questions=200)
