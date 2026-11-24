
from pathlib import Path

import pandas as pd
import streamlit as st



# 1. PAGE CONFIGURATION


st.set_page_config(
    page_title="LLM Benchmark Lab",
    layout="wide"
)

st.title("LLM Evaluation Benchmark")
st.caption(
    "A local benchmarking framework for comparing "
    "large language models."
)

RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"



# 2. LOAD RESULTS


if not RESULTS_PATH.exists():
    st.error(
        "results.csv was not found. "
        "Run benchmark.py first."
    )
    st.stop()

df = pd.read_csv(RESULTS_PATH, keep_default_na=False)

required = {
    "model",
    "category",
    "question",
    "expected",
    "prediction",
    "correct",
    "latency_seconds"
}

missing = required - set(df.columns)

if missing:
    st.error(f"Missing columns in results.csv: {missing}")
    st.info("Run the latest benchmark.py to regenerate results.")
    st.stop()

if df.empty:
    st.warning("The results file is empty.")
    st.stop()

# Handle numerical fields
df["correct"] = pd.to_numeric(
    df["correct"], errors="coerce"
).fillna(0).astype(int)

df["latency_seconds"] = pd.to_numeric(
    df["latency_seconds"], errors="coerce"
)

# Support older CSV files
if "needs_review" not in df.columns:
    df["needs_review"] = 0

df["needs_review"] = pd.to_numeric(
    df["needs_review"], errors="coerce"
).fillna(0).astype(int)

if "scoring_method" not in df.columns:
    df["scoring_method"] = "legacy"

if "instruction_compliant" not in df.columns:
    df["instruction_compliant"] = 0

df["instruction_compliant"] = pd.to_numeric(
    df["instruction_compliant"], errors="coerce"
).fillna(0).astype(int)


df["strict_correct"] = (
    (df["correct"] == 1) &
    (df["needs_review"] == 0)
).astype(int)



# 3. SIDEBAR FILTERS


st.sidebar.header("🔎 Filters")

all_models = sorted(df["model"].unique())
all_categories = sorted(df["category"].unique())

selected_models = st.sidebar.multiselect(
    "Select models",
    all_models,
    default=all_models
)

selected_categories = st.sidebar.multiselect(
    "Select categories",
    all_categories,
    default=all_categories
)

filtered = df[
    df["model"].isin(selected_models) &
    df["category"].isin(selected_categories)
].copy()

if filtered.empty:
    st.warning("No results match the selected filters.")
    st.stop()

st.sidebar.divider()

st.sidebar.info(
    "Accuracy is provisional until all flagged "
    "responses have been manually reviewed."
)


# 4. SUMMARY METRICS


st.subheader("Benchmark Overview")

col1, col2, col3, col4 = st.columns(4)

accuracy = filtered["correct"].mean() * 100
average_latency = filtered["latency_seconds"].mean()
review_count = int(filtered["needs_review"].sum())

col1.metric(
    "Models evaluated",
    filtered["model"].nunique()
)

col2.metric(
    "Provisional accuracy",
    f"{accuracy:.1f}%"
)

col3.metric(
    "Average latency",
    f"{average_latency:.2f}s"
)

col4.metric(
    "Responses to review",
    review_count
)

st.divider()


# 5. SUMMARY BY MODEL


summary = filtered.groupby("model").agg(
    accuracy=("correct", "mean"),
    strict_success=("strict_correct", "mean"),
    avg_latency=("latency_seconds", "mean"),
    total=("correct", "count"),
    review_count=("needs_review", "sum")
).reset_index()

summary["accuracy_pct"] = summary["accuracy"] * 100
summary["strict_success_pct"] = (
    summary["strict_success"] * 100
)



# 6. DASHBOARD TABS


tab1, tab2, tab3, tab4 = st.tabs([
    "Model Comparison",
    "Category Analysis",
    "Manual Review",
    "All Results"
])


# TAB 1 - MODEL COMPARISON


with tab1:

    st.header("Model Performance Comparison")

    st.subheader("Accuracy by Model (%)")

    st.bar_chart(
        summary.set_index("model")[
            ["accuracy_pct", "strict_success_pct"]
        ]
    )

    st.caption(
        "Provisional accuracy includes potential matches "
        "that require review. Strict success counts only "
        "unflagged correct responses."
    )

    st.subheader("Average Response Time (seconds)")

    st.bar_chart(
        summary.set_index("model")["avg_latency"]
    )

    st.subheader("Performance Summary")

    st.dataframe(
        summary[
            [
                "model",
                "accuracy_pct",
                "strict_success_pct",
                "avg_latency",
                "total",
                "review_count"
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )



# TAB 2 - CATEGORY ANALYSIS


with tab2:

    st.header("Performance Across Categories")

    category_summary = (
        filtered.groupby(["category", "model"])
        .agg(
            accuracy=("correct", "mean"),
            avg_latency=("latency_seconds", "mean"),
            total=("correct", "count")
        )
        .reset_index()
    )

    category_summary["accuracy_pct"] = (
        category_summary["accuracy"] * 100
    )

    st.subheader("Accuracy by Category (%)")

    st.bar_chart(
        category_summary,
        x="category",
        y="accuracy_pct",
        color="model"
    )

    st.subheader("Average Latency by Category")

    st.bar_chart(
        category_summary,
        x="category",
        y="avg_latency",
        color="model"
    )

    st.subheader("Detailed Category Results")

    st.dataframe(
        category_summary.round(2),
        use_container_width=True,
        hide_index=True
    )

    instructions = filtered[
        filtered["category"].str.casefold()
        == "instructions"
    ]

    if not instructions.empty:
        st.subheader("Instruction Following")

        compliance = (
            instructions.groupby("model")[
                "instruction_compliant"
            ].mean() * 100
        )

        st.bar_chart(compliance)

        st.caption(
            "Measures exact output-format compliance "
            "for instruction-following questions."
        )



# TAB 3 - MANUAL REVIEW


with tab3:

    st.header("Responses Requiring Manual Review")

    review_df = filtered[
        filtered["needs_review"] == 1
    ].copy()

    st.metric(
        "Flagged responses",
        len(review_df)
    )

    st.info(
        "These responses may be correct or incorrect. "
        "Inspect each answer before treating "
        "the benchmark accuracy as final."
    )

    if review_df.empty:
        st.success(
            "No responses are currently flagged for review."
        )

    else:
        st.dataframe(
            review_df[
                [
                    "model",
                    "category",
                    "question",
                    "expected",
                    "prediction",
                    "correct",
                    "scoring_method"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="Download responses for review",
            data=review_df.to_csv(index=False),
            file_name="manual_review.csv",
            mime="text/csv"
        )

    st.caption(
        "This tab displays review candidates. "
        "It does not automatically verify or "
        "change the benchmark scores."
    )



# TAB 4 - ALL RESULTS


with tab4:

    st.header("Complete Evaluation Results")

    status_filter = st.selectbox(
        "Filter answers",
        [
            "All responses",
            "Provisional correct",
            "Provisional incorrect",
            "Needs review"
        ]
    )

    displayed = filtered.copy()

    if status_filter == "Provisional correct":
        displayed = displayed[displayed["correct"] == 1]

    elif status_filter == "Provisional incorrect":
        displayed = displayed[displayed["correct"] == 0]

    elif status_filter == "Needs review":
        displayed = displayed[
            displayed["needs_review"] == 1
        ]

    st.dataframe(
        displayed,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Download filtered results CSV",
        data=displayed.to_csv(index=False),
        file_name="benchmark_results.csv",
        mime="text/csv"
    )



# 7. FOOTER


st.divider()

st.caption(
    "LLM Benchmark Lab | Python + Ollama + "
    "pandas + Streamlit"
)

st.caption(
    "Evaluation limitations: small dataset, "
    "heuristic scoring, and hardware-dependent latency."
)


