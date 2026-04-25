# Getting Started

## Requirements

- Python 3.11 or newer
- An OpenAI-compatible API key
- A telemetry repository containing logs, traces, and metrics to analyze

## Install

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the project in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

## Configure Environment

Create a local `.env` file:

```bash
cp .env.example .env
```

Then set at least:

```bash
OPENAI_API_KEY=your_api_key
```

Optional values can be added later for model selection, database storage, parser attempts, and runtime toggles. See [Configuration](configuration.md).

## First Run

Run the full parser + executor flow:

```bash
aware assess \
  --query "On 2021-03-04 between 18:30:00 and 19:00:00 checkout timeout" \
  --repo /path/to/repository
```

The command produces a validated BuildSpec, launches executor sub-agents, and saves run artifacts under `output/`.
