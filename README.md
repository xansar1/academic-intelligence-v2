# Academic Intelligence AI V2.1 — Research Demonstration

A portfolio-grade Python + Streamlit + scikit-learn app, fully separated from the legacy Academic Dashboard codebase. No real student records, credentials, or private data are shipped. Generates **960 synthetic students** deterministically.

## Features

- Command Center: KPIs, score distribution, program mix
- Student Explorer: filters, search, individual subject chart
- Prediction Lab: logistic regression inference and model sensitivity
- Model Insights: 240 held-out fictional test records, PR-AUC, ROC-AUC, precision, recall, confusion matrix
- Dark Streamlit theme and improved Safari/mobile UI
- CI tests, local launcher, reproducible synthetic data

**Not an educational risk prediction service.** Labels are generated from simulated processes; all scores represent behavior on synthetic labels and must never be used for decisions about real students.

## Run locally on Mac

```bash
bash scripts/start_mac.sh
```

Then open http://localhost:8501 . To stop: Control+C in Terminal.

## Tests

```bash
python3 -m pip install -r requirements.txt -r requirements-dev.txt
python3 -m pytest -q tests/
```

## Publish to GitHub (no legacy secrets)

1. Create an empty public repository `xansar1/academic-intelligence-v2` at https://github.com/new . Do **not** initialize with README, LICENSE or gitignore.
2. From the extracted release root run `bash scripts/push_to_github.sh`.
3. Confirm the new clean repository on GitHub. Only synthetic-data demo files should be present.

## Deploy to Streamlit Community Cloud

1. Open https://share.streamlit.io/ and choose **Create app / Deploy an app**.
2. Repository `xansar1/academic-intelligence-v2`, branch `main`, entrypoint `academic_v2_demo.py`.
3. Use Python 3.12 if your settings offer a choice. No environment variables or secrets are required.
4. Deploy and confirm all four sections load. Test Prediction Lab, Student Explorer, and metrics.
5. Only after a real live URL works, replace the Academic Dashboard demo URL in the 3D portfolio.

## Legacy project security issue (action required)

The separate legacy public repository `xansar1/ai-academic-analytics-dashboard` contains a tracked `.streamlit/secrets.toml`, predictable demo passwords and `data/Private_data.csv` with student names. If its email app password is real, **revoke/rotate it immediately**, remove inappropriate data from public access, and review Git history exposure. Removing the file from the latest commit does not remove past commits. Do not copy any legacy dataset/secrets or deploy its historical `app.py` publicly until reviewed.

### Release checklist

- [x] Synthetic demo isolated from legacy app
- [x] 10 unit tests pass locally
- [x] Code compiles
- [ ] New clean GitHub repository published
- [ ] Streamlit Cloud public demo deployed and URL verified
- [ ] Portfolio project #2 live demo linked
