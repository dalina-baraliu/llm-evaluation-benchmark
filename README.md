# LLM Evaluation Benchmark

A Python-based framework for evaluating and comparing locally hosted large language models through automated scoring, performance analysis, and human review.

## Overview

This project evaluates two language models running locally through Ollama:

- `llama3.2:1b`
- `qwen2.5:1.5b`

The benchmark measures answer accuracy, instruction-following compliance, and response latency across four categories: Knowledge, Math, Reasoning, and Instructions.

It includes a human-review workflow for ambiguous answers and an interactive Streamlit dashboard for exploring evaluation results.

## Technologies

Python · Ollama · pandas · requests · Streamlit · unittest

## Features

- Automated evaluation of multiple language models
- CSV-based evaluation dataset
- Normalized answer matching
- Strict output-format checks for instruction-following tasks
- Human-review flags for ambiguous responses
- Response latency and output token tracking
- Model-level and category-level performance summaries
- Interactive Streamlit dashboard
- CSV export of evaluation results
- Seven unit tests covering core scoring behavior

## Evaluation Methodology

The benchmark sends each question to both models through Ollama's local API, using temperature 0 and a maximum generation setting of 128 tokens.

Responses are scored using three approaches:

1. **Strict formatting:** Instruction-following tasks require an exact response match.
2. **Normalized matching:** Other answers are compared after removing capitalization, punctuation, and extra whitespace differences.
3. **Provisional matching:** Longer answers containing the expected answer are provisionally accepted and flagged for review.

Other unmatched answers are also flagged for human review.

Automated accuracy is provisional until flagged responses have been reviewed.

## Benchmark Results

| Model | Reviewed Accuracy |
|---|---:|
| llama3.2:1b | 84.38% |
| qwen2.5:1.5b | 100.00% |

- Total evaluations: **64**
- Questions per model: **32**
- Categories: Knowledge, Math, Reasoning, Instructions
- Responses reviewed by a human: **8**

These results reflect performance on this specific dataset and should not be interpreted as general model rankings.

## Project Files

- `benchmark.py` — Runs model evaluation and generates provisional results.
- `review.py` — Exports flagged answers and applies human-review decisions.
- `dashboard.py` — Displays evaluation metrics, comparisons, filters, and review candidates.
- `questions.csv` — Evaluation dataset.
- `results.csv` — Automated evaluation results.
- `review_queue.csv` — Human-review queue.
- `reviewed_results.csv` — Final results after review.
- `tests/` — Unit tests for scoring behavior.

## Installation

Install Python, Git, and Ollama before starting.

Clone the repository:

```bash
git clone https://github.com/dalina-baraliu/llm-evaluation-benchmark.git
cd llm-evaluation-benchmark
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the required models:

```bash
ollama pull llama3.2:1b
ollama pull qwen2.5:1.5b
```

Ensure Ollama is running and its API is accessible at `http://localhost:11434`.

## Running the Benchmark

Run the full evaluation:

```bash
python benchmark.py
```

Run a smaller evaluation:

```bash
python benchmark.py --limit 5
```

Automated evaluation results are saved to `results.csv`.

## Human Review

Export flagged responses:

```bash
python review.py export
```

Open `review_queue.csv` and enter a value in the `human_correct` column for each flagged response:

- `1` — Correct
- `0` — Incorrect

Apply completed reviews:

```bash
python review.py apply
```

Final reviewed scores are saved to `reviewed_results.csv`.

If rerunning the benchmark, preserve or remove the old review queue before exporting a new one. The script intentionally avoids overwriting an existing review queue.

## Dashboard

Launch the Streamlit dashboard:

```bash
python -m streamlit run dashboard.py
```

The dashboard provides:

- Model comparison
- Category analysis
- Manual-review candidates
- Complete results and CSV downloads

**Note:** The current dashboard reads provisional scores from `results.csv`. Final reviewed accuracy is available in `reviewed_results.csv`.

## Running Tests

```bash
python -m unittest discover -s tests -v
```

Latest verified result: **7 tests passed**.

## Limitations

- Small evaluation dataset of 32 questions per model
- Heuristic answer matching can produce false positives or false negatives
- Human review involves subjective correctness decisions
- Latency depends on local hardware and environment
- The benchmark evaluates two specific model versions
- The dashboard currently displays provisional rather than final reviewed accuracy

## Future Improvements

- Expand evaluation datasets
- Add more language models
- Improve semantic correctness evaluation
- Add regression testing
- Integrate reviewed accuracy into the dashboard

## Author

Dalina Baraliu

[LinkedIn](https://www.linkedin.com/in/dalina-baraliu/)