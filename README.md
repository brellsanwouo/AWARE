# AWARE Assess ADK MVP

Prototype minimal **Assess** pour AWARE (Parser + Executor), avec séparation explicite:

- `tools/` pour la manipulation fichiers/données
- `templates/` pour les templates d'agents à instancier
- agents ADK minimaux (`ParserAgent`, `ExecutorAgent`, sous-agents dynamiques)
- LLM utilisé comme cerveau décisionnel (Parser et sous-agents Executor)
- filtrage strict des analyses sur `failure_time_range_ts`

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Créer ton `.env` local:

```bash
cp .env.example .env
# ou utilise le .env déjà présent et remplis OPENAI_API_KEY
```

## Lancement

```bash
aware parse \
  --query "On 2021-03-04 between 18:30:00 and 19:00:00 checkout timeout" \
  --repo /agentfactory/data/Bank/telemetry/2021_03_04
```

Exécuter l'étape Executor à partir d'un BuildSpec existant:

```bash
aware execute \
  --buildspec-json /path/to/buildspec.json \
  --repo /path/to/repository
```

Parser + Executor en une seule commande:

```bash
aware assess \
  --query "On 2021-03-04 between 18:30:00 and 19:00:00 checkout timeout" \
  --repo /path/to/repository
```

## UI Conversation Live

Lance l'UI:

```bash
aware ui --host 127.0.0.1 --port 8787
```

Puis ouvre:

```text
http://127.0.0.1:8787
```

Dans l'UI:
- tu entres la requête utilisateur
- tu entres le chemin repository
- tu lances `Run Assess` (Parser + Executor)
- tu vois la conversation live:
  - init parser
  - réflexion / tentative LLM
  - validation
  - correction si invalide
  - exécution Executor (LogsAgent / TraceAgent / MetricsAgent)
  - succès final + sortie JSON complète

## Architecture Runtime

### 1) Templates d'agents (instanciation dynamique)

Les sous-agents Executor sont définis dans:

- `templates/assess_templates.py`

Chaque template définit:

- `agent_name`
- `role`
- `objective`
- `target_field` du BuildSpec
- `tools` à utiliser
- `domain`

### 2) Tools (manipulation de données)

Les opérations de lecture/fenêtrage/parse sont dans:

- `tools/telemetry_tools.py`

Exemples:

- `load_csv_window`
- `build_llm_observation_context`
- `count_matches`
- `sample_matching_lines`
- `max_numeric_column`

### 3) Agents minimaux + LLM décideur

`ExecutorAgent` instancie les sous-agents à partir des templates.

Chaque sous-agent:

- exécute les tools sur son fichier cible
- construit un contexte compact des observations
- demande au LLM de décider des findings
- retombe sur un fallback heuristique si la réponse LLM est invalide

Implémentation:

- `agents/executor_agent.py`

Chaque run est sauvegardé automatiquement dans:
- `output/json/<timestamp>_<run_id>.json`
- `output/txt/<timestamp>_<run_id>.txt`

## Knowledge DB (SQLite)

Les informations des sous-agents (Executor) sont stockées dans une base SQLite:
- table `runs`
- table `events`
- table `task_results`
- table `findings`

DB par défaut:
- `sqlite:///output/assess.db`

Tu peux surcharger:
- UI: champ `DB URL`
- CLI: `--db-url sqlite:///...`

## Provider LLM

- OpenAI uniquement (`openai-compatible`).

Variables `.env` utiles:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optionnel, défaut: `https://api.openai.com/v1`)
- `OPENAI_MODEL` (optionnel, défaut: `gpt-5-mini`)
- `PARSER_MAX_ATTEMPTS`
- `AWARE_PARSER_KB_FILE` (optionnel, défaut: `knowledge/parser_buildspec_kb.md`)
- `AWARE_EXECUTOR_KB_FILE` (optionnel, défaut: `knowledge/executor_rca_kb.md`)
- `EXECUTOR_MAX_AGENTS` (optionnel: limite globale de sous-agents instanciés)
  - défaut recommandé V1: `5`
- `AWARE_ENABLE_REASONING` (`true|false`, défaut `true`)
  - `false`: le LLM reste utilisé, mais en mode raisonnement rapide (prompts plus directs)
- `AWARE_ENABLE_MEMORY` (`true|false`, défaut `true`)
  - `false`: désactive la mémoire partagée inter-agents et l'écriture des résultats d'agents en SQLite

### Note BuildSpec (fichiers multiples)

Le BuildSpec accepte désormais plusieurs fichiers par domaine:

- `absolute_log_file`: liste de chemins absolus
- `absolute_trace_file`: liste de chemins absolus
- `absolute_metrics_file`: liste de chemins absolus

## Base de Connaissance (fichier séparé)

Le ParserAgent charge ses instructions BuildSpec depuis un fichier externe:

- `knowledge/parser_buildspec_kb.md`

Ce fichier contient:
- le mapping `task_1..task_7`
- le contrat BuildSpec
- les règles de normalisation
- les règles de sélection de fichiers (log/trace/metrics)

L'ExecutorAgent charge aussi sa base de connaissance:

- `knowledge/executor_rca_kb.md`

Ce fichier contient:
- les composants/réasons possibles
- la structure des fichiers telemetry (header + unités timestamp)
- les règles d'analyse pour éviter les faux composants (ex: nom de fichier)

Tu peux pointer un autre fichier via:

```bash
export AWARE_PARSER_KB_FILE=/chemin/vers/mon_kb.md
```

## Tests

```bash
pytest -q
```
