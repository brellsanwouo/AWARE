## POSSIBLE ROOT CAUSE REASONS:
- high CPU usage
- high memory usage
- network latency
- network packet loss
- high disk I/O read usage
- high disk space usage
- high JVM CPU load
- JVM Out of Memory (OOM) Heap

## POSSIBLE ROOT CAUSE COMPONENTS:
- apache01
- apache02
- Tomcat01
- Tomcat02
- Tomcat03
- Tomcat04
- MG01
- MG02
- IG01
- IG02
- Mysql01
- Mysql02
- Redis01
- Redis02

## DATA SCHEMA
### metric_app.csv
- header: `timestamp,rr,sr,cnt,mrt,tc`
- timestamp unit: seconds
- component/service field: `tc`
- key metrics: `rr`, `sr`, `cnt`, `mrt`

### metric_container.csv
- header: `timestamp,cmdb_id,kpi_name,value`
- timestamp unit: seconds
- component field: `cmdb_id`
- KPI name field: `kpi_name`
- KPI numeric field: `value`

### trace_span.csv
- header: `timestamp,cmdb_id,parent_id,span_id,trace_id,duration`
- timestamp unit: milliseconds
- component field: `cmdb_id`
- trace graph fields: `parent_id`, `span_id`, `trace_id`
- timing field: `duration`

### log_service.csv
- header: `log_id,timestamp,cmdb_id,log_name,value`
- timestamp unit: seconds
- component field: `cmdb_id`
- reason/message fields: `log_name`, `value`

## RCA EXECUTION RULES FOR EXECUTOR AGENTS
- Analyze rows strictly within `failure_time_range_ts`.
- Use header semantics first: infer component/reason candidates from column roles.
- Root cause component must come from telemetry values (typically `cmdb_id` or `tc`) and not from file names.
- Prefer trace evidence for downstream faulty component selection when multiple component candidates exist.
- Cross-check log evidence to confirm reason and timestamp when possible.
- Keep uncertainty as `known|unknown` based on evidence completeness.
