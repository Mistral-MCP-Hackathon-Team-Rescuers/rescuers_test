# mcp_aequitas_server.py
# Works with DB (SQLAlchemy) and large CSV/Parquet; Pydantic-safe type hints.

from typing import Optional, Literal, Any
import os
import sys
import json
import uuid
import pandas as pd

from sqlalchemy import create_engine, text as _sql_text
from mcp.server.fastmcp import FastMCP

from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness

mcp = FastMCP("Aequitas MCP Server", port=3000, stateless_http=True, debug=True)

# ------------------------------
# In-memory store for generated reports
# ------------------------------
_REPORTS: dict[str, dict[str, Any]] = {}
_LAST_ID: Optional[str] = None

# ------------------------------
# Data loading helpers
# ------------------------------
def _conn_from_alias_or_url(conn_alias: Optional[str], conn_url: Optional[str]) -> str:
    """Resolve SQLAlchemy URL from env alias or direct URL."""
    if conn_url:
        return conn_url
    if conn_alias:
        v = os.getenv(conn_alias)
        if not v:
            raise ValueError(f"Env var {conn_alias} is not set")
        return v
    raise ValueError("Provide either conn_alias or conn_url")

def _load_df(
    data: list[dict[str, Any]] | str | dict[str, Any],
    file_format: Literal["csv", "parquet"] = "csv",
) -> pd.DataFrame:
    """
    Accepts one of:
      1) list of dict rows
      2) path/URL to CSV or Parquet
      3) SQL dict:
         {
           "kind": "sql",
           "conn": "<sqlalchemy_url>" | None,
           "conn_alias": "<ENV_VAR_WITH_URL>" | None,
           "query": "SELECT ... WHERE batch_id=:bid" | None,
           "table": "table_name" | None,
           "schema": Optional[str],
           "params": Optional[dict],
           "limit": Optional[int]  # best-effort LIMIT for Postgres/MySQL/SQLite
         }
      4) CSV dict (large-file friendly):
         {
           "kind": "csv",
           "path": "/path/to/file.csv",
           "usecols": Optional[list[str]],
           "dtype": Optional[dict[str,str]],
           "nrows": Optional[int],
           "chunksize": Optional[int],
           "storage_options": Optional[dict]
         }
      5) Parquet dict:
         {
           "kind": "parquet",
           "path": "/path/to/file.parquet",
           "columns": Optional[list[str]]
         }
    """
    # (1) JSON rows
    if isinstance(data, list):
        return pd.DataFrame(data)

    # (3) SQL source
    if isinstance(data, dict) and data.get("kind") == "sql":
        conn_str = _conn_from_alias_or_url(data.get("conn_alias"), data.get("conn"))
        query = data.get("query")
        table = data.get("table")
        schema = data.get("schema")
        params = data.get("params") or {}
        limit = data.get("limit")

        engine = create_engine(conn_str)
        with engine.connect() as con:
            if query:
                qtext = query.strip().rstrip(";")
                # naive LIMIT appender for non-MSSQL
                if limit and "limit" not in qtext.lower() and not conn_str.lower().startswith("mssql+"):
                    qtext = f"{qtext} LIMIT {int(limit)}"
                return pd.read_sql_query(_sql_text(qtext), con, params=params)
            if table:
                return pd.read_sql_table(table, con, schema=schema)
            raise ValueError("SQL source requires 'query' or 'table'")

    # (4) CSV source (large-file friendly)
    if isinstance(data, dict) and data.get("kind") == "csv":
        path = data["path"]
        usecols = data.get("usecols")
        dtype = data.get("dtype")
        nrows = data.get("nrows")
        chunksize = data.get("chunksize")
        storage_options = data.get("storage_options")

        if chunksize and chunksize > 0:
            # Stream chunks; concatenate up to nrows
            chunks: list[pd.DataFrame] = []
            rows_read = 0
            for chunk in pd.read_csv(
                path,
                usecols=usecols,
                dtype=dtype,
                chunksize=chunksize,
                storage_options=storage_options,
            ):
                if nrows:
                    remaining = nrows - rows_read
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        chunk = chunk.iloc[:remaining]
                chunks.append(chunk)
                rows_read += len(chunk)
            if not chunks:
                return pd.DataFrame(columns=usecols or [])
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.read_csv(
                path,
                usecols=usecols,
                dtype=dtype,
                nrows=nrows,
                storage_options=storage_options,
            )

    # (5) Parquet source
    if isinstance(data, dict) and data.get("kind") == "parquet":
        path = data["path"]
        columns = data.get("columns")
        return pd.read_parquet(path, columns=columns)

    # (2) File path/URL string
    if isinstance(data, str):
        if file_format == "csv":
            return pd.read_csv(data)
        elif file_format == "parquet":
            return pd.read_parquet(data)
        raise ValueError("Unsupported file_format")

    raise ValueError("Unsupported data type for 'data'")

def _format_markdown_summary(report: dict[str, Any]) -> str:
    """Compact, human-readable markdown summary from an Aequitas report dict."""
    counts = report.get("counts", [])
    bias = report.get("bias", [])
    fairness = report.get("fairness", [])

    def _short_rows(rows: list[dict[str, Any]], keys: list[str], limit: int = 12) -> str:
        if not rows or not keys:
            return "_(no rows)_"
        header = "| " + " | ".join(keys) + " |\n|" + "|".join(["---"] * len(keys)) + "|\n"
        body: list[str] = []
        for r in rows[:limit]:
            body.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
        extra = f"\n_… {len(rows)-limit} more rows not shown_" if len(rows) > limit else ""
        return header + "\n".join(body) + extra

    md = [
        "# Aequitas Fairness Report",
        "",
        "## Crosstabs (counts)",
        _short_rows(counts, keys=[k for k in counts[0].keys()] if counts else []),
        "",
        "## Disparities (selected)",
        _short_rows(bias, keys=[k for k in bias[0].keys()][:8] if bias else []),
        "",
        "## Fairness (metrics)",
        _short_rows(fairness, keys=[k for k in fairness[0].keys()][:8] if fairness else []),
        "",
        "> Tip: Use the `aequitas.explain` prompt to generate a plain-language analysis.",
    ]
    return "\n".join(md)

# ------------------------------
# Core computation
# ------------------------------
def _run_aequitas(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    protected_attrs: list[str],
    decision_threshold: Optional[float],
    ref_groups: Optional[dict[str, Any]],
    fairness_criteria: Optional[list[str]],
) -> dict[str, Any]:
    # Threshold scores into decisions when provided
    col_for_decision = score_col
    if decision_threshold is not None:
        df = df.copy()
        df["score_thresholded"] = (df[score_col].astype(float) >= float(decision_threshold)).astype(int)
        col_for_decision = "score_thresholded"

    # Normalize column names for Aequitas
    work = pd.DataFrame()
    work["label_value"] = df[label_col].astype(int)
    work["score"] = df[col_for_decision].astype(float)
    for a in protected_attrs:
        work[a] = df[a]

    g = Group()
    xtab, _ = g.get_crosstabs(
        work,
        score_col="score",
        label_col="label_value",
        attr_cols=protected_attrs,
    )

    b = Bias()
    bias_df = b.get_disparity_predefined_groups(
        xtab,
        original_df=work,
        ref_groups_dict=ref_groups,
        alpha=0.05,
        mask_significance=True,
    )

    f = Fairness()
    fair_df = f.get_group_value_fairness(
        bias_df,
        fair_metrics=fairness_criteria,
    )

    return {
        "counts": xtab.reset_index().to_dict(orient="records"),
        "bias": bias_df.reset_index().to_dict(orient="records"),
        "fairness": fair_df.reset_index().to_dict(orient="records"),
    }

def _store_report(payload: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    report = {**payload, "meta": meta}
    rid = str(uuid.uuid4())
    _REPORTS[rid] = report
    global _LAST_ID
    _LAST_ID = rid
    json_uri = f"aequitas://report/{rid}.json"
    md_uri = f"aequitas://report/{rid}.md"
    return {
        "report_id": rid,
        "resources": {"json": json_uri, "markdown": md_uri},
        **report,
    }

# ------------------------------
# Tools
# ------------------------------
@mcp.tool()
def compute_bias_report(
    data: list[dict[str, Any]] | str | dict[str, Any],
    label_col: str,
    score_col: str,
    protected_attrs: list[str],
    decision_threshold: Optional[float] = None,
    ref_groups: Optional[dict[str, Any]] = None,
    fairness_criteria: Optional[list[Literal[
        "demographic_parity", "impact_parity", "fpr_parity", "fnr_parity"
    ]]] = None,
    file_format: Literal["csv", "parquet"] = "csv",
) -> dict[str, Any]:
    """Generic entry point: JSON rows, file path/URL, SQL dict, CSV dict, Parquet dict."""
    df = _load_df(data, file_format=file_format)
    payload = _run_aequitas(
        df=df,
        label_col=label_col,
        score_col=score_col,
        protected_attrs=protected_attrs,
        decision_threshold=decision_threshold,
        ref_groups=ref_groups,
        fairness_criteria=fairness_criteria,
    )
    meta = {
        "protected_attrs": protected_attrs,
        "label_col": label_col,
        "score_col": score_col,
        "threshold": decision_threshold,
        "ref_groups": ref_groups,
        "fairness_criteria": fairness_criteria,
        "source": "generic",
    }
    return _store_report(payload, meta)

@mcp.tool()
def compute_bias_report_sql(
    conn_alias: Optional[str] = None,
    conn_url: Optional[str] = None,
    sql: Optional[str] = None,
    table: Optional[str] = None,
    schema: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
    limit: Optional[int] = None,
    label_col: str = "label",
    score_col: str = "score",
    protected_attrs: list[str] = ["sex", "race"],
    decision_threshold: Optional[float] = None,
    ref_groups: Optional[dict[str, Any]] = None,
    fairness_criteria: Optional[list[Literal[
        "demographic_parity", "impact_parity", "fpr_parity", "fnr_parity"
    ]]] = None,
) -> dict[str, Any]:
    """Convenience wrapper for SQL sources. Prefer conn_alias (env var) for secrets."""
    df = _load_df({
        "kind": "sql",
        "conn": conn_url,
        "conn_alias": conn_alias,
        "query": sql,
        "table": table,
        "schema": schema,
        "params": params or {},
        "limit": limit,
    })
    payload = _run_aequitas(
        df=df,
        label_col=label_col,
        score_col=score_col,
        protected_attrs=protected_attrs,
        decision_threshold=decision_threshold,
        ref_groups=ref_groups,
        fairness_criteria=fairness_criteria,
    )
    meta = {
        "protected_attrs": protected_attrs,
        "label_col": label_col,
        "score_col": score_col,
        "threshold": decision_threshold,
        "ref_groups": ref_groups,
        "fairness_criteria": fairness_criteria,
        "source": "sql",
        "sql_meta": {
            "conn_alias": conn_alias,
            "table": table,
            "schema": schema,
            "limit": limit,
            "query_present": bool(sql),
        }
    }
    return _store_report(payload, meta)

@mcp.tool()
def compute_bias_report_csv(
    path: str,
    label_col: str,
    score_col: str,
    protected_attrs: list[str],
    usecols: Optional[list[str]] = None,
    dtype: Optional[dict[str, str]] = None,
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None,
    decision_threshold: Optional[float] = None,
    ref_groups: Optional[dict[str, Any]] = None,
    fairness_criteria: Optional[list[Literal[
        "demographic_parity", "impact_parity", "fpr_parity", "fnr_parity"
    ]]] = None,
) -> dict[str, Any]:
    """CSV-focused convenience tool with large-file friendly options."""
    df = _load_df({
        "kind": "csv",
        "path": path,
        "usecols": usecols,
        "dtype": dtype,
        "nrows": nrows,
        "chunksize": chunksize,
    })
    payload = _run_aequitas(
        df=df,
        label_col=label_col,
        score_col=score_col,
        protected_attrs=protected_attrs,
        decision_threshold=decision_threshold,
        ref_groups=ref_groups,
        fairness_criteria=fairness_criteria,
    )
    meta = {
        "protected_attrs": protected_attrs,
        "label_col": label_col,
        "score_col": score_col,
        "threshold": decision_threshold,
        "ref_groups": ref_groups,
        "fairness_criteria": fairness_criteria,
        "source": "csv",
        "csv_meta": {
            "path": path,
            "usecols": usecols,
            "nrows": nrows,
            "chunksize": chunksize,
        }
    }
    return _store_report(payload, meta)

@mcp.tool()
def summarize_fairness_profile(
    fairness_profile: Literal["punitive", "assistive"],
    metrics: Optional[list[Literal[
        "fpr_parity", "fnr_parity", "impact_parity", "demographic_parity"
    ]]] = None,
) -> dict[str, Any]:
    """Pick fairness metrics aligned to a rough policy profile."""
    if metrics:
        selected = metrics
    elif fairness_profile == "punitive":
        selected = ["fpr_parity", "impact_parity"]
    else:
        selected = ["fnr_parity", "impact_parity"]
    return {"selected_metrics": selected}

@mcp.tool()
def list_reports() -> list[dict[str, Any]]:
    """List IDs of reports stored in this server process, with minimal metadata."""
    out: list[dict[str, Any]] = []
    for rid, rep in _REPORTS.items():
        out.append({
            "report_id": rid,
            "resources": {
                "json": f"aequitas://report/{rid}.json",
                "markdown": f"aequitas://report/{rid}.md",
            },
            "meta": rep.get("meta", {}),
        })
    return out

@mcp.tool()
def get_report(report_id: str) -> dict[str, Any]:
    """Return a stored report by ID, or raise if not found."""
    if report_id not in _REPORTS:
        raise ValueError(f"Unknown report_id: {report_id}")
    rep = _REPORTS[report_id]
    return {
        "report_id": report_id,
        "resources": {
            "json": f"aequitas://report/{report_id}.json",
            "markdown": f"aequitas://report/{report_id}.md",
        },
        **rep,
    }

@mcp.tool()
def clear_reports() -> dict[str, Any]:
    """Clear all stored reports in memory (useful during iteration)."""
    _REPORTS.clear()
    global _LAST_ID
    _LAST_ID = None
    return {"cleared": True}

# ------------------------------
# Resources
# ------------------------------
@mcp.resource("aequitas://help", name="Aequitas MCP Help")
def read_help() -> str:
    return (
        "# Aequitas MCP Help\n\n"
        "**Tools**: `compute_bias_report`, `compute_bias_report_sql`, `compute_bias_report_csv`, "
        "`summarize_fairness_profile`, `list_reports`, `get_report`, `clear_reports`.\n\n"
        "**Resources**: `aequitas://report/<id>.json`, `aequitas://report/<id>.md`, `aequitas://help`.\n\n"
        "**Workflow**:\n"
        "1) Run `compute_bias_report_sql` (DB) or `compute_bias_report_csv` (large CSV), or the generic `compute_bias_report`.\n"
        "2) Open the returned Markdown resource.\n"
        "3) (Optional) Use prompts to generate a plain-language narrative.\n\n"
        "**Notes**:\n"
        "- Prefer `conn_alias` pointing to an env var with a read-only DB user.\n"
        "- For large CSVs, use `usecols`, `dtype`, `nrows`, or `chunksize` to keep memory in check.\n"
        "- Reports are in-memory only; restarting clears state."
    )

@mcp.resource("aequitas://report/{rid}.json", name="Aequitas Report JSON")
def read_report_json(rid: str) -> str:
    rep = _REPORTS.get(rid)
    if rep is None:
        raise FileNotFoundError(f"No report for {rid}")
    return json.dumps(rep, indent=2)

@mcp.resource("aequitas://report/{rid}.md", name="Aequitas Report (Markdown)")
def read_report_md(rid: str) -> str:
    rep = _REPORTS.get(rid)
    if rep is None:
        raise FileNotFoundError(f"No report for {rid}")
    return _format_markdown_summary(rep)

# ------------------------------
# Prompts (Pydantic-safe annotations)
# ------------------------------
@mcp.prompt(
    name="aequitas.explain",
    description="Explain an Aequitas fairness report in plain language."
)
def prompt_explain(
    bias_rows: list[dict[str, Any]],
    fairness_rows: list[dict[str, Any]],
    context: str = ""
) -> str:
    return (
        "You are a fairness analyst. Given Aequitas outputs, write a concise, action-focused summary.\n\n"
        "Guidelines:\n"
        "- Define each reported metric the first time you use it (1 short sentence).\n"
        "- Call out the worst-affected subgroup(s) and whether disparities are statistically significant.\n"
        "- Suggest 1–2 next steps: threshold tuning, data collection, or mitigation.\n"
        "- Keep it under 200 words.\n\n"
        f"Context (optional): {context}\n\n"
        f"Bias rows JSON:\n{json.dumps(bias_rows, indent=2)}\n\n"
        f"Fairness rows JSON:\n{json.dumps(fairness_rows, indent=2)}\n"
    )

@mcp.prompt(
    name="aequitas.choose_metrics",
    description="Recommend fairness metrics for a task profile."
)
def prompt_choose_metrics(profile: Literal["punitive", "assistive"], notes: str = "") -> str:
    return (
        "Recommend 2–3 fairness metrics to prioritize, given the profile.\n"
        "- punitive: avoid false positives (suggest fpr_parity, impact_parity, or equalized odds variants).\n"
        "- assistive: avoid false negatives (suggest fnr_parity, impact_parity, or equalized odds variants).\n"
        f"Notes: {notes}\n"
    )

# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    # Usage: python mcp_aequitas_server.py [stdio|http|streamable-http|sse] [host] [port]
    transport = sys.argv[1] if len(sys.argv) > 1 else "streamable-http"
    host = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8080
    mcp.run(transport="streamable-http")
