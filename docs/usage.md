# Usage

Aware exposes a CLI through the `aware` command.

## Parse a Request

Use `parse` when you only want to generate and validate a BuildSpec.

```bash
aware parse \
  --query "On 2021-03-04 between 18:30:00 and 19:00:00 checkout timeout" \
  --repo /path/to/repository
```

## Execute an Existing BuildSpec

Use `execute` when a BuildSpec already exists and you want to run the executor stage.

```bash
aware execute \
  --buildspec-json /path/to/buildspec.json \
  --repo /path/to/repository
```

## Run the Full Assessment

Use `assess` for the full Parser + Executor workflow.

```bash
aware assess \
  --query "On 2021-03-04 between 18:30:00 and 19:00:00 checkout timeout" \
  --repo /path/to/repository
```

## Useful Options

| Option | Scope | Description |
| --- | --- | --- |
| `--repo` | CLI | Path to the telemetry repository. |
| `--query` | Parser / Assess | Natural-language incident request. |
| `--buildspec-json` | Execute | Path to an existing BuildSpec JSON file. |
| `--db-url` | Execute / Assess | SQLite database URL override. |
| `--max-agents` | Execute / Assess | Maximum number of executor sub-agents. |
| `--llm-model` | Parser / Assess | OpenAI model override. |

## BuildSpec Multiple Files

The BuildSpec supports multiple files per telemetry domain:

- `absolute_log_file`: list of absolute log paths
- `absolute_trace_file`: list of absolute trace paths
- `absolute_metrics_file`: list of absolute metrics paths
