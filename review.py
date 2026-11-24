
from pathlib import Path
import argparse
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results.csv"
QUEUE = BASE / "review_queue.csv"
FINAL = BASE / "reviewed_results.csv"

def export_reviews():
    if QUEUE.exists():
        print("Review queue already exists.")
        print("Rename or remove it to create a new queue.")
        return

    df = pd.read_csv(
        RESULTS, dtype=str, keep_default_na=False
    )

    df.insert(0, "row_id", range(len(df)))
    flagged = df[df["needs_review"] == "1"].copy()

    columns = [
        "row_id", "model", "category",
        "question", "expected", "prediction",
        "correct"
    ]

    queue = flagged[columns].copy()
    queue["human_correct"] = ""
    queue.to_csv(QUEUE, index=False)

    print(f"Exported {len(queue)} responses.")
    print("Open review_queue.csv and fill human_correct.")
    print("Use 1 for correct and 0 for incorrect.")


def apply_reviews():
    df = pd.read_csv(
        RESULTS, dtype=str, keep_default_na=False
    )
    queue = pd.read_csv(
        QUEUE, dtype=str, keep_default_na=False
    )

    flagged_ids = set(
        str(i) for i in df.index
        if df.at[i, "needs_review"] == "1"
    )

    if (
        queue["row_id"].duplicated().any()
        or set(queue["row_id"]) != flagged_ids
    ):
        raise ValueError(
            "Review queue does not match flagged results."
        )

    df["final_correct"] = df["correct"]
    df["review_status"] = "automatic"

    for _, row in queue.iterrows():
        index = int(row["row_id"])
        decision = row["human_correct"].strip()

        if decision not in ("0", "1"):
            raise ValueError(
                f"Row {index}: enter 0 or 1 in human_correct."
            )

        for field in [
            "model", "category", "question",
            "expected", "prediction", "correct"
        ]:
            if row[field] != df.at[index, field]:
                raise ValueError(
                    "Review queue differs from results.csv. "
                    "Regenerate the queue before reviewing."
                )

        df.at[index, "final_correct"] = decision
        df.at[index, "review_status"] = "human_reviewed"

    df["final_correct"] = df["final_correct"].astype(int)
    df.to_csv(FINAL, index=False)

    summary = (
        df.groupby("model")["final_correct"]
        .mean() * 100
    )

    print("\nReviewed benchmark accuracy (%):")
    print(summary.round(2))
    print(f"\nSaved: {FINAL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["export", "apply"]
    )
    args = parser.parse_args()

    if args.action == "export":
        export_reviews()
    else:
        apply_reviews()
