"""Reproducible *fictional* academic data. Never touches real student files."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["Mathematics", "Science", "English", "Attendance", "Assignments", "Previous Score"]
UNIVERSITIES = ["Northbridge Academy", "Cedar Grove College", "Horizon Institute", "Riverbend Campus"]
PROGRAMS = ["Science", "Commerce", "Technology", "Humanities"]


def generate_demo_data(n: int = 960, seed: int = 2031) -> pd.DataFrame:
    """Create plausible-looking but entirely synthetic scores and future needs labels."""
    if not 100 <= n <= 12000:
        raise ValueError("Demo dataset size must be between 100 and 12000")
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n)
    engagement = rng.normal(0, 1, n)
    attendance = np.clip(79 + 11 * engagement + 3 * ability + rng.normal(0, 7, n), 25, 100)
    assignments = np.clip(74 + 11 * engagement + 6 * ability + rng.normal(0, 10, n), 0, 100)
    previous = np.clip(67 + 15 * ability + rng.normal(0, 10, n), 0, 100)
    maths = np.clip(68 + 14 * ability + 2 * engagement + rng.normal(0, 10, n), 0, 100)
    science = np.clip(71 + 13 * ability + rng.normal(0, 10, n), 0, 100)
    english = np.clip(72 + 11 * ability + 2 * engagement + rng.normal(0, 11, n), 0, 100)
    current = np.mean(np.array([maths, science, english]).T, axis=1)
    # Synthetic future labels are generated stochastically; not inferred from real records.
    logits = (-2.0 + 0.058 * (65 - current) + 0.043 * (78 - attendance)
              + 0.020 * (70 - assignments) + 0.016 * (65 - previous))
    probability = 1 / (1 + np.exp(-logits))
    support = (rng.random(n) < probability).astype(int)
    df = pd.DataFrame({
        "Student ID": [f"DEMO-{i:04d}" for i in range(1, n + 1)],
        "Institution": rng.choice(UNIVERSITIES, size=n),
        "Program": rng.choice(PROGRAMS, size=n),
        "Semester": rng.integers(1, 7, n),
        "Mathematics": np.rint(maths).astype(int),
        "Science": np.rint(science).astype(int),
        "English": np.rint(english).astype(int),
        "Attendance": np.rint(attendance).astype(int),
        "Assignments": np.rint(assignments).astype(int),
        "Previous Score": np.rint(previous).astype(int),
        "Needs Support Next Term": support,
    })
    df["Average Score"] = df[["Mathematics", "Science", "English"]].mean(axis=1).round(1)
    return df


def validate_uploaded_csv(raw: bytes, max_bytes: int = 2_500_000) -> pd.DataFrame:
    """Read classroom mark CSV in local mode; validate *only* numeric score columns."""
    from io import BytesIO
    if not raw or len(raw) > max_bytes:
        raise ValueError("Choose a nonempty CSV below 2.5 MB")
    try:
        df = pd.read_csv(BytesIO(raw), dtype=str, nrows=10001)
    except Exception as exc:
        raise ValueError("CSV could not be read") from exc
    if df.empty or len(df) > 10000:
        raise ValueError("CSV must contain 1–10,000 student records")
    if not df.columns.is_unique:
        raise ValueError("CSV contains duplicate column names")
    df.columns = [str(c).strip() for c in df.columns]
    if "Student ID" not in df.columns:
        df.insert(0, "Student ID", [f"LOCAL-{i:04d}" for i in range(1, len(df) + 1)])
    marks = [col for col in FEATURES if col in df.columns]
    if not marks:
        raise ValueError("Include one or more columns: " + ", ".join(FEATURES))
    for col in marks:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().any() or not numeric.between(0, 100).all():
            raise ValueError(f"{col} must contain numbers between 0 and 100")
        df[col] = numeric
    df["Average Score"] = df[[c for c in ("Mathematics", "Science", "English") if c in marks]].mean(axis=1).round(1) if any(c in marks for c in ("Mathematics", "Science", "English")) else df[marks].mean(axis=1).round(1)
    return df
