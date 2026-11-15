
## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push:

1. **quality**: `black --check` and `flake8`.
2. **test**: installs dependencies, pulls model files with DVC from `ci-remote`, registers the model in MLflow, runs `pytest`.
3. **build**: only on `main` and only if tests pass. Builds the Docker image and pushes it to `ghcr.io` (tags `latest` and the commit SHA).

A failing step stops the pipeline, so a broken test never produces an image.

## DVC remotes

| Remote | Where | Used by |
|---|---|---|
| `myremote` (default) | Google Drive | local development |
| `ci-remote` | `dvc-store/` folder in this repo | CI, so no cloud credentials are needed |

## Monitoring

- `GET /metrics`: Prometheus request count and latency.
- `logs/predictions.jsonl`: every prediction, kept in the `api_logs` volume.
- `python scripts/drift_report.py`: PSI drift check, exits 1 on alert.
- Alert rules and thresholds: [docs/monitoring.md](docs/monitoring.md).
