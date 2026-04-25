# AWARE Assess ADK MVP

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)
![Google ADK](https://img.shields.io/badge/Google%20ADK-enabled-4285F4?style=flat&logo=google&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-enabled-412991?style=flat&logo=openai&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-supported-009688?style=flat&logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-2.x-E92063?style=flat&logo=pydantic&logoColor=white)
![Typer](https://img.shields.io/badge/Typer-CLI-111827?style=flat)
![SQLite](https://img.shields.io/badge/SQLite-supported-003B57?style=flat&logo=sqlite&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-ready-0A9EDC?style=flat&logo=pytest&logoColor=white)

Minimal **Assess** prototype for AWARE (Parser + Executor), with explicit separation:

- `tools/` for file/data handling
- `templates/` for agent templates to instantiate
- minimal ADK agents (`ParserAgent`, `ExecutorAgent`, dynamic sub-agents)
- LLM used as the decision engine (Parser and Executor sub-agents)
- strict analysis filtering on `failure_time_range_ts`

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Create your local `.env`:

```bash
cp .env.example .env
# or use the existing .env and fill in OPENAI_API_KEY
```

## Run

```bash
aware parse \
  --query "On 2021-03-04 between 18:30:00 and 19:00:00 checkout timeout" \
  --repo /agentfactory/data/Bank/telemetry/2021_03_04
```

Run the Executor step from an existing BuildSpec:

```bash
aware execute \
  --buildspec-json /path/to/buildspec.json \
  --repo /path/to/repository
```

Parser + Executor in a single command:

```bash
aware assess \
  --query "On 2021-03-04 between 18:30:00 and 19:00:00 checkout timeout" \
  --repo /path/to/repository
```

## UI Conversation Live

Start the UI:

```bash
aware ui --host 127.0.0.1 --port 8787
```

Then open:

```text
http://127.0.0.1:8787
```

In the UI:
- enter the user query
- enter the repository path
- start `Run Assess` (Parser + Executor)
- watch the live conversation:
  - parser initialization
  - reasoning / LLM attempt
  - validation
  - correction if invalid
  - Executor run (LogsAgent / TraceAgent / MetricsAgent)
  - final success + full JSON output

## Architecture Runtime

### 1) Agent Templates (Dynamic Instantiation)

Executor sub-agents are defined in:

- `templates/assess_templates.py`

Each template defines:

- `agent_name`
- `role`
- `objective`
- BuildSpec `target_field`
- `tools` to use
- `domain`

### 2) Tools (Data Handling)

Read/window/parse operations are in:

- `tools/telemetry_tools.py`

Examples:

- `load_csv_window`
- `build_llm_observation_context`
- `count_matches`
- `sample_matching_lines`
- `max_numeric_column`

### 3) Minimal Agents + LLM Decision Engine

`ExecutorAgent` instantiates sub-agents from the templates.

Each sub-agent:

- runs tools on its target file
- builds a compact observation context
- asks the LLM to decide the findings
- falls back to a heuristic path if the LLM response is invalid

Implementation:

- `agents/executor_agent.py`

Each run is automatically saved in:
- `output/json/<timestamp>_<run_id>.json`
- `output/txt/<timestamp>_<run_id>.txt`

## Knowledge DB (SQLite)

Executor sub-agent information is stored in a SQLite database:
- table `runs`
- table `events`
- table `task_results`
- table `findings`

Default DB:
- `sqlite:///output/assess.db`

You can override it with:
- UI: `DB URL` field
- CLI: `--db-url sqlite:///...`

## Provider LLM

- OpenAI uniquement (`openai-compatible`).

Useful `.env` variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional, default: `https://api.openai.com/v1`)
- `OPENAI_MODEL` (optional, default: `gpt-5-mini`)
- `PARSER_MAX_ATTEMPTS`
- `AWARE_PARSER_KB_FILE` (optional, default: `knowledge/parser_buildspec_kb.md`)
- `AWARE_EXECUTOR_KB_FILE` (optional, default: `knowledge/executor_rca_kb.md`)
- `EXECUTOR_MAX_AGENTS` (optional: global limit for instantiated sub-agents)
  - recommended V1 default: `5`
- `AWARE_ENABLE_REASONING` (`true|false`, default `true`)
  - `false`: the LLM is still used, but in fast reasoning mode (more direct prompts)
- `AWARE_ENABLE_MEMORY` (`true|false`, default `true`)
  - `false`: disables shared inter-agent memory and SQLite writes for agent results

### BuildSpec Note (Multiple Files)

BuildSpec now accepts multiple files per domain:

- `absolute_log_file`: list of absolute paths
- `absolute_trace_file`: list of absolute paths
- `absolute_metrics_file`: list of absolute paths

## Knowledge Base (Separate File)

`ParserAgent` loads its BuildSpec instructions from an external file:

- `knowledge/parser_buildspec_kb.md`

This file contains:
- the `task_1..task_7` mapping
- the BuildSpec contract
- normalization rules
- file selection rules (log/trace/metrics)

`ExecutorAgent` also loads its knowledge base:

- `knowledge/executor_rca_kb.md`

This file contains:
- possible components/reasons
- telemetry file structure (header + timestamp units)
- analysis rules to avoid false components (for example, file name)

You can point to another file with:

```bash
export AWARE_PARSER_KB_FILE=/path/to/my_kb.md
```

## Tests

```bash
pytest -q
```

## Support

Questions, ideas, or collaborations are welcome. Feel free to reach out: brell.sanwouo@inria.fr | sanwouobrell@gmail.com
