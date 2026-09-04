"""
MetricGuard - Ingestion Adapters
==================================
Four ways for real teams to plug their metrics into MetricGuard.
Every adapter normalises its input into the standard MetricDefinition
schema and writes / appends to data/metric_definitions.json.

ADAPTER 1 — CSV
  Teams export their metric catalogue as a spreadsheet.
  Expected columns (extra columns are silently ignored):
    team, metric_name, sql, description
  Optional columns:
    filters, includes_refunds, time_grain, id

ADAPTER 2 — dbt manifest.json
  Point MetricGuard at your dbt project's manifest.json.
  All nodes of type 'metric' or models tagged with 'metric' are
  extracted automatically — no manual work needed.

ADAPTER 3 — Database (SQLAlchemy)
  Connect to any SQL database (Postgres, MySQL, Snowflake, BigQuery,
  Redshift…). MetricGuard queries information_schema to pull view
  definitions and optionally a metadata table you define.

ADAPTER 4 — Manual JSON
  Append one or more metrics defined as plain dicts. Useful for
  scripting or one-off additions from a CI pipeline.

Usage:
    python src/ingest.py --help
    python src/ingest.py csv  --file metrics.csv  --team "Finance"
    python src/ingest.py dbt  --manifest target/manifest.json
    python src/ingest.py db   --url "postgresql://user:pass@host/db"
    python src/ingest.py json --file my_metrics.json
"""

import csv
import json
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data"
METRICS_PATH = DATA_DIR / "metric_definitions.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_existing() -> list[dict]:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return []


def _save(metrics: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved %d metrics to %s", len(metrics), METRICS_PATH)


def _make_id(existing: list[dict]) -> str:
    """Generate a unique metric ID (m01, m02 … or UUID if slots exhausted)."""
    used = {m.get("id", "") for m in existing}
    for i in range(1, 10000):
        candidate = f"m{i:02d}"
        if candidate not in used:
            return candidate
    return f"m_{uuid.uuid4().hex[:8]}"


def _normalise(raw: dict, existing: list[dict], default_team: str = "") -> dict:
    """
    Coerce an arbitrary dict into a valid MetricDefinition-compatible dict.
    Missing optional fields get safe defaults.
    """
    metric_id = str(raw.get("id") or _make_id(existing))
    team      = str(raw.get("team") or default_team or "Unknown").strip()
    name      = str(raw.get("metric_name") or raw.get("name") or "").strip()
    sql       = str(raw.get("sql") or raw.get("compiled_sql") or raw.get("raw_sql") or "").strip()
    desc      = str(raw.get("description") or raw.get("label") or name).strip()

    # filters: accept list[str] or pipe-separated string
    filters_raw = raw.get("filters", [])
    if isinstance(filters_raw, str):
        filters = [f.strip() for f in filters_raw.split("|") if f.strip()]
    elif isinstance(filters_raw, list):
        filters = [str(f) for f in filters_raw]
    else:
        filters = []

    includes_refunds = raw.get("includes_refunds")
    if isinstance(includes_refunds, str):
        includes_refunds = includes_refunds.lower() in ("true", "yes", "1")

    time_grain = str(raw.get("time_grain") or "unknown").strip()

    return {
        "id":               metric_id,
        "team":             team,
        "metric_name":      name,
        "sql":              sql,
        "description":      desc,
        "filters":          filters,
        "includes_refunds": includes_refunds,
        "time_grain":       time_grain,
    }


def _merge(existing: list[dict], new_metrics: list[dict]) -> tuple[list[dict], int, int]:
    """
    Merge new_metrics into existing, deduplicating by (team, metric_name).
    Returns (merged_list, added_count, updated_count).
    """
    index = {(m["team"], m["metric_name"]): i for i, m in enumerate(existing)}
    added = updated = 0
    result = existing[:]

    for nm in new_metrics:
        key = (nm["team"], nm["metric_name"])
        if key in index:
            result[index[key]] = nm
            updated += 1
        else:
            index[key] = len(result)
            result.append(nm)
            added += 1

    return result, added, updated


# ---------------------------------------------------------------------------
# ADAPTER 1 — CSV
# ---------------------------------------------------------------------------

def ingest_csv(
    file_path: str | Path,
    default_team: str = "",
    replace: bool = False,
) -> tuple[int, int]:
    """
    Load metrics from a CSV file.

    Required columns : team, metric_name, sql, description
    Optional columns : id, filters, includes_refunds, time_grain

    Returns (added, updated) counts.

    Example CSV row:
        Finance,monthly_revenue,"SELECT SUM(amount) FROM orders WHERE status='completed'",
        "Total completed-order revenue",status = 'completed',false,month
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    existing = [] if replace else _load_existing()
    new_metrics = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Normalise column names (strip whitespace, lowercase)
        reader.fieldnames = [c.strip().lower() for c in (reader.fieldnames or [])]

        required = {"team", "metric_name", "sql", "description"}
        missing  = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}\n"
                f"Found columns: {reader.fieldnames}\n"
                "See data/sample_metrics.csv for the expected format."
            )

        for row in reader:
            if not row.get("metric_name", "").strip():
                continue   # skip blank rows
            nm = _normalise(row, existing + new_metrics, default_team)
            new_metrics.append(nm)

    merged, added, updated = _merge(existing, new_metrics)
    _save(merged)
    logger.info("CSV ingest: %d added, %d updated from %s", added, updated, path.name)
    return added, updated


# ---------------------------------------------------------------------------
# ADAPTER 2 — dbt manifest.json
# ---------------------------------------------------------------------------

def ingest_dbt_manifest(
    manifest_path: str | Path,
    team: str = "",
    replace: bool = False,
) -> tuple[int, int]:
    """
    Parse a dbt project's manifest.json and extract metric/model definitions.

    What gets extracted:
      - Nodes of node_type 'metric'    (dbt Semantic Layer metrics)
      - Nodes of node_type 'model'     (SQL models — tagged 'metric' or all)

    The team name is read from the dbt project name if not supplied.

    Usage:
        # In your dbt project root:
        dbt compile
        python src/ingest.py dbt --manifest target/manifest.json --team "Analytics"
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found: {path}")

    with open(path) as f:
        manifest = json.load(f)

    project_name = manifest.get("metadata", {}).get("project_name", team or "dbt")
    team_name    = team or project_name

    existing    = [] if replace else _load_existing()
    new_metrics = []

    nodes   = manifest.get("nodes",   {})
    metrics = manifest.get("metrics", {})

    # --- dbt Semantic Layer metrics (dbt >= 1.6) ---
    for key, node in metrics.items():
        label       = node.get("label") or node.get("name", "")
        description = node.get("description") or label
        # Build a representative SQL from the measure definition
        measures    = node.get("type_params", {}).get("measure", {})
        agg         = node.get("type_params", {}).get("agg", "SUM")
        expr        = measures.get("expr") or node.get("name", "value")
        model_ref   = node.get("model") or ""
        sql         = f"SELECT {agg}({expr}) FROM {model_ref}  -- dbt metric"

        raw = {
            "team":        team_name,
            "metric_name": node.get("name", label).lower().replace(" ", "_"),
            "sql":         sql,
            "description": description,
            "time_grain":  node.get("time_granularity", "unknown"),
            "filters":     [f.get("where", "") for f in node.get("filter", {}).get("where_filters", [])],
        }
        new_metrics.append(_normalise(raw, existing + new_metrics))

    # --- dbt model nodes (SQL models) ---
    for key, node in nodes.items():
        if node.get("resource_type") != "model":
            continue
        tags = node.get("tags", [])
        # Include all models, or only those tagged 'metric'
        # (remove the 'metric' condition below to ingest ALL models)
        if "metric" not in tags and not node.get("config", {}).get("meta", {}).get("is_metric"):
            continue

        raw_sql      = node.get("raw_code") or node.get("raw_sql") or ""
        compiled_sql = node.get("compiled_code") or node.get("compiled_sql") or raw_sql

        raw = {
            "team":        team_name,
            "metric_name": node.get("name", "").lower(),
            "sql":         compiled_sql or raw_sql,
            "description": node.get("description") or node.get("name", ""),
            "filters":     [],
        }
        new_metrics.append(_normalise(raw, existing + new_metrics))

    if not new_metrics:
        logger.warning(
            "No metrics found in manifest.json. "
            "Tag your dbt models with 'metric' or use dbt Semantic Layer metrics."
        )

    merged, added, updated = _merge(existing, new_metrics)
    _save(merged)
    logger.info("dbt manifest ingest: %d added, %d updated", added, updated)
    return added, updated


# ---------------------------------------------------------------------------
# ADAPTER 3 — Database (SQLAlchemy)
# ---------------------------------------------------------------------------

def ingest_from_db(
    connection_url: str,
    team: str,
    query: str | None = None,
    replace: bool = False,
) -> tuple[int, int]:
    """
    Connect to any SQLAlchemy-compatible database and pull metric definitions.

    Supported databases (install the matching driver):
      PostgreSQL : pip install psycopg2-binary
      MySQL      : pip install pymysql
      Snowflake  : pip install snowflake-sqlalchemy
      BigQuery   : pip install sqlalchemy-bigquery
      Redshift   : pip install sqlalchemy-redshift
      SQLite     : built-in (for testing)

    OPTION A — Metadata table
      If your team maintains a metrics catalogue table, pass a custom query:
        query = "SELECT metric_name, sql_definition as sql, description,
                        filters, time_grain FROM analytics.metric_registry"

    OPTION B — View definitions (auto-discovery)
      Without a query, MetricGuard scans information_schema.views and
      extracts all views whose names contain 'metric', 'kpi', or 'measure'.

    Example:
        python src/ingest.py db \\
          --url "postgresql://analyst:pass@dwh.company.com:5432/prod" \\
          --team "Analytics" \\
          --query "SELECT * FROM analytics.metric_registry"
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        raise ImportError(
            "sqlalchemy is required for DB ingestion.\n"
            "Install it: pip install sqlalchemy\n"
            "Then install your database driver (e.g. pip install psycopg2-binary)"
        )

    engine = create_engine(connection_url)
    existing    = [] if replace else _load_existing()
    new_metrics = []

    with engine.connect() as conn:
        if query:
            # Custom query — caller controls the schema
            rows = conn.execute(text(query)).mappings().all()
            for row in rows:
                raw = {
                    "team":        row.get("team", team),
                    "metric_name": row.get("metric_name") or row.get("name", ""),
                    "sql":         row.get("sql") or row.get("sql_definition") or row.get("query", ""),
                    "description": row.get("description") or row.get("label", ""),
                    "filters":     row.get("filters", []),
                    "time_grain":  row.get("time_grain", "unknown"),
                    "includes_refunds": row.get("includes_refunds"),
                }
                new_metrics.append(_normalise(raw, existing + new_metrics, team))
        else:
            # Auto-discovery: scan information_schema.views
            logger.info("No query supplied — scanning information_schema.views")
            view_query = """
                SELECT table_name, view_definition
                FROM   information_schema.views
                WHERE  LOWER(table_name) ~ 'metric|kpi|measure'
            """
            try:
                rows = conn.execute(text(view_query)).mappings().all()
            except Exception:
                # Some DBs (Snowflake, BigQuery) use different system tables
                logger.warning("information_schema.views query failed — try passing --query explicitly")
                rows = []

            for row in rows:
                name = str(row.get("table_name", "")).lower()
                raw  = {
                    "team":        team,
                    "metric_name": name,
                    "sql":         row.get("view_definition", ""),
                    "description": f"Auto-discovered view: {name}",
                }
                new_metrics.append(_normalise(raw, existing + new_metrics, team))

    merged, added, updated = _merge(existing, new_metrics)
    _save(merged)
    logger.info("DB ingest: %d added, %d updated from %s", added, updated, connection_url.split("@")[-1])
    return added, updated


# ---------------------------------------------------------------------------
# ADAPTER 4 — Manual JSON
# ---------------------------------------------------------------------------

def ingest_json(
    file_path: str | Path | None = None,
    metrics: list[dict] | None = None,
    replace: bool = False,
) -> tuple[int, int]:
    """
    Append metrics from a JSON file or a Python list of dicts.

    JSON file format — array of objects:
    [
      {
        "team": "Finance",
        "metric_name": "monthly_revenue",
        "sql": "SELECT SUM(amount) FROM orders WHERE status='completed'",
        "description": "Total revenue from completed orders.",
        "filters": ["status = 'completed'"],
        "includes_refunds": false,
        "time_grain": "month"
      }
    ]

    Example:
        python src/ingest.py json --file my_metrics.json
    """
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")
        with open(path) as f:
            metrics = json.load(f)

    if not metrics:
        raise ValueError("No metrics provided — pass file_path or metrics list")

    existing    = [] if replace else _load_existing()
    new_metrics = [_normalise(m, existing) for m in metrics]

    merged, added, updated = _merge(existing, new_metrics)
    _save(merged)
    logger.info("JSON ingest: %d added, %d updated", added, updated)
    return added, updated


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(
        prog="python src/ingest.py",
        description="MetricGuard ingestion CLI — add your team's metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/ingest.py csv  --file metrics.csv --team "Finance"
  python src/ingest.py dbt  --manifest target/manifest.json --team "Analytics"
  python src/ingest.py db   --url "postgresql://user:pass@host/db" --team "Data"
                            --query "SELECT * FROM analytics.metric_registry"
  python src/ingest.py json --file my_metrics.json

After ingestion, run the pipeline:
  python src/report.py
  # or click "Re-run pipeline" in the dashboard
        """,
    )
    sub = p.add_subparsers(dest="adapter", required=True)

    # CSV
    csv_p = sub.add_parser("csv", help="Ingest from a CSV file")
    csv_p.add_argument("--file",    required=True, help="Path to CSV file")
    csv_p.add_argument("--team",    default="",    help="Default team name (overrides CSV column)")
    csv_p.add_argument("--replace", action="store_true", help="Replace all existing metrics")

    # dbt
    dbt_p = sub.add_parser("dbt", help="Ingest from dbt manifest.json")
    dbt_p.add_argument("--manifest", required=True, help="Path to dbt target/manifest.json")
    dbt_p.add_argument("--team",     default="",    help="Override team name (default: dbt project name)")
    dbt_p.add_argument("--replace",  action="store_true")

    # DB
    db_p = sub.add_parser("db", help="Ingest from a SQL database")
    db_p.add_argument("--url",     required=True, help="SQLAlchemy connection URL")
    db_p.add_argument("--team",    required=True, help="Team name for ingested metrics")
    db_p.add_argument("--query",   default=None,  help="Custom SQL query to fetch metrics")
    db_p.add_argument("--replace", action="store_true")

    # JSON
    json_p = sub.add_parser("json", help="Ingest from a JSON file")
    json_p.add_argument("--file",    required=True)
    json_p.add_argument("--replace", action="store_true")

    args = p.parse_args()

    try:
        if args.adapter == "csv":
            added, updated = ingest_csv(args.file, args.team, args.replace)
        elif args.adapter == "dbt":
            added, updated = ingest_dbt_manifest(args.manifest, args.team, args.replace)
        elif args.adapter == "db":
            added, updated = ingest_from_db(args.url, args.team, args.query, args.replace)
        elif args.adapter == "json":
            added, updated = ingest_json(args.file, replace=args.replace)
        else:
            p.print_help(); sys.exit(1)

        print(f"\n✅  Ingestion complete: {added} added, {updated} updated")
        print(f"    Metrics file: {METRICS_PATH}")
        print("\nNext step — run the analysis:")
        print("    python src/report.py\n")

    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"\n❌  {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
