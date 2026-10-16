# APMOA — Autonomous Process Mining & Optimization Agent

APMOA is a self-adaptive multi-agent framework for business process discovery and optimization. It automatically analyzes event logs, evaluates multiple discovery algorithms (Inductive, Alpha, Alpha+, Heuristics, Split Miner), and uses a reasoning agent with a RAG knowledge base to select the best model or dynamically adapt if quality thresholds are unmet.

## Architecture
The system follows a multi-agent architecture:
- **Dataset Manager**: Ingests, normalizes, and validates event logs (CSV, XLSX, XES, XES.GZ).
- **Process Discovery Agent**: Applies various discovery algorithms.
- **Model Evaluation Agent**: Computes Fitness, Precision, Generalization, and Simplicity.
- **Model Selection Agent**: Scores models based on configurable weights.
- **RAG Knowledge Base**: Uses semantic search to provide domain context to the reasoning agent.
- **Reasoning Agent**: Integrates LLMs (via Groq) to diagnose issues and recommend adaptations.
- **Adaptation Agent**: Executes autonomous retry strategies.
- **Process Intelligence / Optimization Agents**: Analyzes bottlenecks, rework, deviations, and suggests optimizations.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables
Create a `.env` file in the root directory (see `.env.example`):
```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

## Dataset Format
APMOA dynamically normalizes event logs into a canonical schema:
- `case_id`: The process instance identifier.
- `activity`: The executed task.
- `timestamp`: Execution time.
- `resource`: (Optional) The user or system performing the task.

Supported formats: `.csv`, `.xlsx`, `.xls`, `.xes`, `.xes.gz`

## How to Run

### Dashboard
Start the Streamlit interactive control center:
```bash
streamlit run dashboard.py
```

### Experiments
To run the full multi-dataset experiment pipeline:
```bash
export PYTHONPATH=.
python tests/test_multiple_datasets.py
```
Results are saved to `results/experiments/multi_dataset/experiment_summary.json`.

## Dashboard Usage
The dashboard provides several views:
- **Overview**: High-level execution metrics.
- **Datasets**: Upload and map dataset schemas dynamically.
- **Experiments**: View persisted experiment runs and parameters.
- **Process Discovery**: Explore generated process models.
- **Model Evaluation**: Compare algorithms against quality metrics.
- **RAG Knowledge**: Manage semantic documents (TXT, MD, PDF, DOCX) and toggle RAG.
- **Autonomous Adaptation**: Timeline of algorithmic adjustments.
- **Process Intelligence & Optimization**: Bottlenecks, rework metrics, and actionable recommendations.
- **What-If Analysis**: Simulate hypothetical metric threshold changes.

## RAG Usage
The RAG Knowledge Base loads files from `data/knowledge/`.
Supported formats: `.txt`, `.md`, `.pdf`, `.docx`.
You can toggle RAG ON/OFF through the dashboard's Settings page or pass configuration via the experiment manager.

## Testing
Run lightweight unit tests for components before full dataset execution:
```bash
export PYTHONPATH=.
python tests/test_dataset_adapter.py
python tests/test_rag.py
```

## Results & Logs
- **Models**: `results/discovery/`
- **Experiments**: `results/experiments/`
- **Dashboard Logs**: Displayed directly in the Execution Log tab.

## Troubleshooting
- **Missing Required Columns**: Ensure your event log has a discernible case ID, activity name, and timestamp.
- **PM4Py Errors**: Ensure you have graphviz installed on your system if visualization fails.
- **RAG Embedding Issues**: Ensure `sentence-transformers` can access the huggingface hub to download the default embedding model (`all-MiniLM-L6-v2`), or configure offline models.
