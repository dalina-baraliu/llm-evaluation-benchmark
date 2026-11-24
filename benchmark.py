
import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests



# 1. CONFIGURATION


MODELS = [
    "llama3.2:1b",
    "qwen2.5:1.5b"
]

API_URL = "http://localhost:11434/api/generate"

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "questions.csv"
RESULTS_PATH = BASE_DIR / "results.csv"



# 2. LOAD EVALUATION DATASET


def load_questions():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {DATASET_PATH.name}. "
            "Create it in your project folder."
        )

    df = pd.read_csv(
        DATASET_PATH,
        dtype=str,
        keep_default_na=False
    )

    required = {"category", "question", "answer"}

    if not required.issubset(df.columns):
        raise ValueError(
            "questions.csv must contain: "
            "category, question, answer"
        )

    df = df.fillna("")

    for column in required:
        if df[column].str.strip().eq("").any():
            raise ValueError(
                f"Empty values found in column: {column}"
            )

    return df.to_dict(orient="records")



# 3. NORMALIZE MODEL ANSWERS


def normalize_answer(text):
    """
    Ignore capitalization, punctuation,
    and extra whitespace.
    """
    text = str(text).casefold().strip()

    text = re.sub(r"[^\w]+", " ", text)

    return " ".join(text.split())


# 4. EVALUATION / SCORING


def check_answer(prediction, expected, category):
    """
    Returns:
        correct
        instruction_compliant
        needs_review
        scoring_method
    """

    prediction = str(prediction).strip()
    expected = str(expected).strip()

    pred = normalize_answer(prediction)
    exp = normalize_answer(expected)

 
    if category.casefold() == "instructions":
        exact = prediction == expected

        return (
            int(exact),
            int(exact),
            0,
            "strict_format"
        )

    # Normalized exact match
    if pred == exp:
        return (
            1,
            1,
            0,
            "exact_match"
        )

    pattern = rf"(?<!\w){re.escape(exp)}(?!\w)"

    if re.search(pattern, pred):
        return (
            1,
            0,
            1,
            "provisional_match"
        )

    # Review manually.
    return (
        0,
        0,
        1,
        "unmatched_review"
    )



# 5. CALL OLLAMA MODEL

def generate_response(model, question):
    prompt = (
        question
        + "\n\nReply with only the answer. "
        + "Do not include an explanation."
    )

    start_time = time.perf_counter()

    response = requests.post(
        API_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 128
            }
        },
        timeout=300
    )

    response.raise_for_status()

    elapsed = time.perf_counter() - start_time

    data = response.json()

    prediction = data.get("response", "").strip()

    return (
        prediction,
        round(elapsed, 3),
        data.get("eval_count", 0)
    )



# 6. BENCHMARK EVALUATION


def run_benchmark(limit=None):

    questions = load_questions()

    if limit is not None:
        questions = questions[:limit]

    if not questions:
        print("No questions to evaluate.")
        return

    results = []

    total = len(questions) * len(MODELS)

    print("\n================================")
    print("LLM EVALUATION BENCHMARK")
    print("================================")
    print(f"Models: {len(MODELS)}")
    print(f"Questions per model: {len(questions)}")
    print(f"Total evaluations: {total}")

    for model in MODELS:

        print(f"\nEvaluating model: {model}")
        print("--------------------------------")

        for index, item in enumerate(questions, start=1):

            question = item["question"]
            expected = item["answer"]
            category = item["category"]

            try:
                prediction, latency, tokens = generate_response(
                    model,
                    question
                )

            except requests.RequestException as error:
                print(f"\nOllama request failed: {error}")
                print(
                    "Check that Ollama is running "
                    "and the models are installed."
                )
                raise SystemExit(1)

            (
                correct,
                instruction_compliant,
                needs_review,
                scoring_method
            ) = check_answer(
                prediction,
                expected,
                category
            )

            results.append({
                "model": model,
                "category": category,
                "question": question,
                "expected": expected,
                "prediction": prediction,
                "correct": correct,
                "instruction_compliant": instruction_compliant,
                "needs_review": needs_review,
                "scoring_method": scoring_method,
                "latency_seconds": latency,
                "output_tokens": tokens
            })

            print(f"\nQuestion {index}/{len(questions)}")
            print(f"Category: {category}")
            print(f"Expected: {expected}")
            print(f"Prediction: {prediction}")
            print(f"Correct: {bool(correct)}")
            print(f"Review needed: {bool(needs_review)}")
            print(f"Latency: {latency:.2f}s")


    # 7. SAVE RESULTS
  

    df = pd.DataFrame(results)

    df.to_csv(RESULTS_PATH, index=False)

    print("\n================================")
    print("FINAL MODEL COMPARISON")
    print("================================")

    summary = df.groupby("model").agg(
        accuracy=("correct", "mean"),
        avg_latency=("latency_seconds", "mean"),
        instruction_compliance=(
            "instruction_compliant", "mean"
        ),
        review_count=("needs_review", "sum"),
        total_questions=("correct", "count")
    )

    summary["accuracy"] *= 100
    summary["instruction_compliance"] *= 100

    print(summary.round(2).to_string())

   
    # 8. PERFORMANCE BY CATEGORY
   

    print("\n================================")
    print("PERFORMANCE BY CATEGORY")
    print("================================")

    category_summary = df.groupby(
        ["model", "category"]
    ).agg(
        accuracy=("correct", "mean"),
        avg_latency=("latency_seconds", "mean")
    )

    category_summary["accuracy"] *= 100

    print(category_summary.round(2).to_string())

    
    # 9. MANUAL REVIEW SUMMARY
 

    review = df[df["needs_review"] == 1]

    print("\n================================")
    print("MANUAL REVIEW")
    print("================================")

    print(f"Responses needing review: {len(review)}")

    if not review.empty:
        print(
            review[
                [
                    "model",
                    "question",
                    "expected",
                    "prediction",
                    "scoring_method"
                ]
            ].to_string(index=False)
        )

    print("\n================================")
    print("BENCHMARK COMPLETE")
    print("================================")
    print(f"Results saved to: {RESULTS_PATH}")
    print(
        "NOTE: Accuracy is provisional until "
        "flagged answers are reviewed."
    )



# 10. MAIN ENTRY POINT


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Local LLM Evaluation Benchmark"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of questions to test per model"
    )

    args = parser.parse_args()

    run_benchmark(limit=args.limit)

