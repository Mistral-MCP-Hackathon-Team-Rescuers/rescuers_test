"""
Fairness Proxy MCP Server (no local Fairlearn deps)
- Calls your FastAPI fairness endpoint and returns metrics.
"""

import os
import requests
import numpy as np
from pydantic import Field
from mcp.server.fastmcp import FastMCP

# ---- Config -----------------------------------------------------------------
FAIRNESS_API_URL = os.getenv(
    "FAIRNESS_API_URL",
    "https://python-server-r8dp.onrender.com/fairness-check"
)
PORT = int(os.getenv("PORT", "3000"))

mcp = FastMCP("Fairness Proxy MCP", port=PORT, stateless_http=True, debug=True)

# ---- Tools ------------------------------------------------------------------
@mcp.tool(
    title="Fairness Assessment (via FastAPI)",
    description="Proxies to the external fairness API and returns selection rates & TPR by group, "
                "plus derived disparate impact and TPR gap."
)
def fairness_assessment(
    y_true: list[int] = Field(description="True labels (0/1)"),
    y_pred: list[int] = Field(description="Predicted labels (0/1)"),
    sensitive_features: list[str] = Field(description="Group identifiers (e.g., 'male', 'female')")
) -> dict:
    # Basic validation
    n, n2, n3 = len(y_true), len(y_pred), len(sensitive_features)
    if not (n and n == n2 == n3):
        return {
            "error": "Length mismatch: y_true, y_pred, and sensitive_features must have the same non-zero length.",
            "lengths": {"y_true": n, "y_pred": n2, "sensitive_features": n3}
        }

    payload = {
        "y_true": y_true,
        "y_pred": y_pred,
        "sensitive_features": sensitive_features,
    }

    try:
        resp = requests.post(FAIRNESS_API_URL, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()  # expects: { "selection_rates": {...}, "tpr_by_group": {...} }
    except Exception as e:
        return {
            "error": f"API call failed: {e}",
            "api_url": FAIRNESS_API_URL
        }

    # Derive handy extras for downstream use
    overall_selection_rate = float(np.mean(np.array(y_pred, dtype=float)))

    sr = data.get("selection_rates", {}) or {}
    tpr = data.get("tpr_by_group", {}) or {}

    # Disparate impact = min(selection_rate) / max(selection_rate)
    di = None
    try:
        sr_vals = [float(v) for v in sr.values() if v is not None]
        if sr_vals and max(sr_vals) > 0:
            di = float(min(sr_vals) / max(sr_vals))
    except Exception:
        pass

    # TPR gap = max(TPR) - min(TPR)
    tpr_gap = None
    try:
        tpr_vals = [float(v) for v in tpr.values() if v is not None]
        if tpr_vals:
            tpr_gap = float(max(tpr_vals) - min(tpr_vals))
    except Exception:
        pass

    return {
        "api_url": FAIRNESS_API_URL,
        "overall_selection_rate": overall_selection_rate,
        "selection_rates": sr,
        "tpr_by_group": tpr,
        "disparate_impact": di,
        "tpr_gap": tpr_gap
    }


@mcp.tool(
    title="Fairness API Health",
    description="Checks reachability of the external fairness API (GET /)."
)
def fairness_api_health() -> dict:
    base = FAIRNESS_API_URL.rsplit("/", 1)[0] or FAIRNESS_API_URL
    try:
        r = requests.get(base + "/", timeout=10)
        ok = r.ok
        body = r.json() if "application/json" in r.headers.get("content-type", "") else r.text
        return {"ok": ok, "status_code": r.status_code, "body": body, "base_url": base}
    except Exception as e:
        return {"ok": False, "error": str(e), "base_url": base}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
