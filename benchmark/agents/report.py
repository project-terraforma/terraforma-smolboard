"""HTML report generation for benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Any


def _format_metric(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
        return f"{float(value) * 100:.1f}%"
    return str(value)


def write_report(rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Write a small HTML report containing embedded benchmark data."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    serialized_rows = [dict(row) for row in rows]
    html = f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>Benchmark Report</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 2rem; color: #123; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border: 1px solid #ccc; padding: 0.75rem; text-align: left; }}
      th {{ background: #f2f2f2; }}
    </style>
  </head>
  <body>
    <h1>Benchmark Report</h1>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Dataset</th>
          <th>Accuracy</th>
          <th>F1</th>
          <th>Total Runtime</th>
        </tr>
      </thead>
      <tbody>
        {''.join(
            f"<tr><td>{row.get('model_name', 'n/a')}</td><td>{row.get('run_inputs', {}).get('dataset', 'n/a')}</td><td>{_format_metric(row.get('metrics', {}).get('accuracy'))}</td><td>{_format_metric(row.get('metrics', {}).get('f1'))}</td><td>{_format_metric(row.get('metrics', {}).get('total_runtime_seconds'))}</td></tr>"
            for row in serialized_rows
        )}
      </tbody>
    </table>
    <script>
      window.benchmarkData = {json.dumps(serialized_rows)};
    </script>
  </body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    return output
