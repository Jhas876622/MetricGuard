"""
MetricGuard - Interactive Setup Wizard
=======================================
Guides a user through adding their team's metrics without touching any code.

Run:
    python src/cli.py

What it does:
  1. Asks for team name
  2. Loops: collect metric name, SQL, description, filters, time grain
  3. Writes everything to data/metric_definitions.json
  4. Offers to run the pipeline immediately

No programming knowledge required.
"""

import json
import sys
from pathlib import Path

ROOT_DIR     = Path(__file__).parent.parent
DATA_DIR     = ROOT_DIR / "data"
METRICS_PATH = DATA_DIR / "metric_definitions.json"


# ── terminal colours ─────────────────────────────────────────────────────────
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

BOLD  = lambda t: _c("1",     t)
DIM   = lambda t: _c("2",     t)
CYAN  = lambda t: _c("96",    t)
GREEN = lambda t: _c("92",    t)
WARN  = lambda t: _c("93",    t)
ERR   = lambda t: _c("91",    t)


def _ask(prompt: str, default: str = "", required: bool = False) -> str:
    """Prompt the user, show default in brackets, enforce required fields."""
    suffix = f" [{default}]" if default else ""
    while True:
        val = input(f"  {CYAN('→')} {prompt}{DIM(suffix)}: ").strip()
        if not val:
            if default:
                return default
            if required:
                print(ERR("    This field is required. Please enter a value."))
                continue
        return val or ""


def _ask_yn(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    val = input(f"  {CYAN('→')} {prompt}{DIM(suffix)}: ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes")


def _load_existing() -> list[dict]:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return []


def _next_id(existing: list[dict]) -> str:
    used = {m.get("id", "") for m in existing}
    for i in range(1, 10000):
        cid = f"m{i:02d}"
        if cid not in used:
            return cid
    return f"m_{len(existing)+1:04d}"


def _save(metrics: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def _collect_metric(team: str, existing: list[dict]) -> dict | None:
    """Interactively collect one metric definition. Returns None to stop."""
    print()
    print(BOLD("  ── New metric ─────────────────────────────────────────────"))

    name = _ask("Metric name (snake_case, e.g. monthly_revenue)", required=True)
    if not name:
        return None

    print(DIM("  Enter the SQL definition. You can paste multi-line SQL;"))
    print(DIM("  press Enter twice when done."))
    lines = []
    while True:
        line = input("  " + DIM("SQL> "))
        if not line and lines and not lines[-1]:
            break
        lines.append(line)
    sql = "\n".join(lines).strip()
    if not sql:
        print(WARN("  No SQL entered — skipping this metric."))
        return None

    description = _ask(
        "Plain-English description",
        default=f"Computes {name.replace('_', ' ')}",
    )

    print(DIM("  Filters (optional) — comma-separated SQL conditions"))
    print(DIM("  e.g.  status = 'completed', event_date >= CURRENT_DATE - 30"))
    filters_raw = _ask("Filters", default="")
    filters = [f.strip() for f in filters_raw.split(",") if f.strip()]

    time_grain = _ask(
        "Time grain",
        default="month",
    )
    print(DIM("  Common values: month, 30d_rolling, calendar_month, 7d_rolling, all_time"))

    includes_refunds_str = _ask("Includes refunds? (yes / no / skip)", default="skip")
    includes_refunds: bool | None = None
    if includes_refunds_str.lower() in ("yes", "y", "true"):
        includes_refunds = True
    elif includes_refunds_str.lower() in ("no", "n", "false"):
        includes_refunds = False

    metric = {
        "id":               _next_id(existing),
        "team":             team,
        "metric_name":      name,
        "sql":              sql,
        "description":      description,
        "filters":          filters,
        "includes_refunds": includes_refunds,
        "time_grain":       time_grain,
    }

    print()
    print(GREEN("  Preview:"))
    print(DIM(f"  {json.dumps(metric, indent=4)}"))
    if not _ask_yn("  Save this metric?", default=True):
        print(WARN("  Metric discarded."))
        return None

    return metric


def run_wizard() -> None:
    """Main entry point for the interactive wizard."""
    print()
    print(BOLD("  ╔══════════════════════════════════════════════════╗"))
    print(BOLD("  ║        MetricGuard  —  Setup Wizard             ║"))
    print(BOLD("  ╚══════════════════════════════════════════════════╝"))
    print()
    print("  This wizard adds your team's metric definitions to MetricGuard.")
    print("  No coding required — just paste your SQL and descriptions.")
    print()

    existing  = _load_existing()
    new_count = 0

    if existing:
        print(DIM(f"  Found {len(existing)} existing metric(s) in {METRICS_PATH.name}"))
        if _ask_yn("  Add to existing metrics? (No = replace all)", default=True):
            metrics = existing[:]
        else:
            metrics = []
            print(WARN("  All existing metrics will be replaced."))
    else:
        metrics = []
        print(DIM("  Starting fresh — no existing metrics found."))

    print()
    team = _ask("Your team name (e.g. Finance, Analytics, Growth)", required=True)

    print()
    print(f"  Adding metrics for team {BOLD(team)}.")
    print(DIM("  Type a metric name to start, or press Enter to finish."))

    while True:
        metric = _collect_metric(team, metrics)
        if metric is None:
            # User pressed Enter without a name → stop
            if not _ask_yn("\n  Add another metric?", default=True):
                break
            continue

        metrics.append(metric)
        new_count += 1
        print(GREEN(f"\n  ✓ Metric '{metric['metric_name']}' saved."))

        if not _ask_yn("  Add another metric?", default=True):
            break

    if new_count == 0:
        print(WARN("\n  No metrics were added. Exiting."))
        return

    _save(metrics)
    print()
    print(GREEN(f"  ✅  Saved {new_count} new metric(s) to {METRICS_PATH}"))
    print(f"  Total metrics in file: {len(metrics)}")
    print()

    if _ask_yn("  Run the MetricGuard pipeline now?", default=True):
        print()
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "report.py")],
            cwd=str(ROOT_DIR),
        )
        if result.returncode == 0:
            print()
            print(GREEN("  ✅  Pipeline complete! Open output/dashboard.html in your browser."))
        else:
            print(ERR("  Pipeline failed — check the output above for errors."))
    else:
        print()
        print("  When ready, run:")
        print(CYAN("    python src/report.py"))
        print()
        print("  Or start the web server:")
        print(CYAN("    uvicorn src.app:app --reload --port 8000"))
        print()


if __name__ == "__main__":
    try:
        run_wizard()
    except KeyboardInterrupt:
        print("\n\n  Wizard cancelled.")
        sys.exit(0)
