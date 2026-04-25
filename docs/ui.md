# Web UI

The web UI provides a live view of the Parser and Executor runtime. It is useful when you want to inspect agent events, generated BuildSpecs, findings, anomalies, and root-cause synthesis without reading raw artifacts.

## Start the UI

```bash
aware ui --host 127.0.0.1 --port 8787
```

Open:

```text
http://127.0.0.1:8787
```

## Main Inputs

| Field | Description |
| --- | --- |
| Source Path | Path to the telemetry repository or source directory. |
| DB URL | Optional SQLite database URL. |
| Max Agents | Optional limit for executor sub-agent instantiation. |
| User Query | Incident request sent to the Parser agent. |

## Runtime Panels

- **Live Agent Conversation** shows streamed events from the user, system, Parser agent, Executor agent, and dynamic sub-agents.
- **Current Agent Tasks** summarizes each active agent and its latest status.
- **BuildSpec** displays the Parser output.
- **Root Cause Synthesis**, **Findings**, **Anomalies**, and **Final Reporting** display the structured assessment output.

## Typical Flow

1. Enter the telemetry repository path.
2. Enter the incident query.
3. Click `Run Assess`.
4. Watch Parser validation and Executor analysis events.
5. Review the final JSON sections in the result panels.
