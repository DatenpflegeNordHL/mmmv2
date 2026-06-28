"""Small standalone HTML report writer."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def write_html_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write a readable static HTML report."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = report.get("after") or report.get("metrics") or report
    readiness = metrics.get("release_readiness", {})
    body = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Audio Quality Report</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:920px;margin:2rem auto;padding:0 1rem;background:#111827;color:#e5e7eb}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{border:1px solid #374151;padding:.45rem;text-align:left}"
        ".score{font-size:2rem;font-weight:800;color:#67e8f9}code{color:#fbbf24}</style></head><body>",
        "<h1>Audio Quality Report</h1>",
        f"<p class='score'>Readiness: {html.escape(str(readiness.get('score', 'n/a')))} / 100 "
        f"({html.escape(str(readiness.get('grade', 'n/a')))})</p>",
        "<h2>Key Metrics</h2><table><tbody>",
    ]
    for key in (
        "duration_seconds",
        "sample_rate",
        "channels",
        "integrated_lufs",
        "estimated_true_peak_dbtp",
        "sample_peak_dbfs",
        "crest_factor",
        "peak_to_loudness_ratio",
        "stereo_correlation",
        "low_end_width",
        "spectral_centroid",
        "spectral_rolloff",
        "clipping_sample_count",
    ):
        if key in metrics:
            body.append(f"<tr><th>{html.escape(key)}</th><td><code>{html.escape(str(metrics[key]))}</code></td></tr>")
    body.extend(["</tbody></table>", "<h2>Recommendations</h2><ul>"])
    for item in readiness.get("recommendations", []):
        body.append(f"<li>{html.escape(str(item))}</li>")
    body.extend(["</ul>", "</body></html>"])
    output_path.write_text("".join(body), encoding="utf-8")
    return output_path
