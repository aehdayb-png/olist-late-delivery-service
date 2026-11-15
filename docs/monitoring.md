# Monitoring and alerts

## What is collected

| Signal | Where |
|---|---|
| Request count, latency, status codes | `GET /metrics` (Prometheus format) |
| Every prediction: features, probability, class, model version | `logs/predictions.jsonl` (Docker volume `api_logs`) |
| Service events and errors | `logs/service.log` and console |
| Prediction drift | `python scripts/drift_report.py` (PSI on probability and key features) |

Inside Docker: `docker compose exec api python scripts/drift_report.py`.

## What I would alert on

Thresholds are starting points. Tune them after real traffic.

| Alert | Rule | Action |
|---|---|---|
| Service down | `/health` fails for 1 minute | Page, check API and MLflow containers |
| Error rate | 5xx responses above 1% for 5 minutes | Page, read `service.log` |
| Latency | p95 above 500 ms for 5 minutes | Warn, check load |
| Bad input | 422 responses above 10% for 15 minutes | Warn, the upstream sender changed |
| Data drift | PSI 0.10 to 0.25 on a feature: warn. Above 0.25: alert | Investigate, consider retraining |
| Prediction drift | Predicted-late rate moves more than 15 points from baseline | Warn, compare with real late rate |
| Logging broken | Requests keep arriving but `predictions.jsonl` stops growing | Alert, evaluation data is being lost |
| Model quality | When real delivery dates arrive, join them to the logs. ROC-AUC more than 0.05 below the notebook value (0.709) | Retrain |

## Limits

The drift script compares the first N logged predictions with the latest N.
It does not compare against the training data, so it detects change after launch, not a bad start.
