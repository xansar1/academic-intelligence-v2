"""Session-local descriptive analytics (does not transmit or persist uploaded data)."""
import pandas as pd
from .data import FEATURES


def dataset_kpis(df: pd.DataFrame) -> dict:
    if df.empty: return {"students":0,"average":0,"attendance":None,"needs_attention":0}
    score=pd.to_numeric(df["Average Score"],errors="coerce")
    attend=pd.to_numeric(df["Attendance"],errors="coerce") if "Attendance" in df else None
    return {"students":int(len(df)),"average":round(float(score.mean()),1),
            "attendance":round(float(attend.mean()),1) if attend is not None else None,
            "needs_attention":int((score<60).sum())}


def filter_records(df: pd.DataFrame, institution: list[str], program: list[str]) -> pd.DataFrame:
    filtered=df.copy()
    if institution and "Institution" in df: filtered=filtered[filtered["Institution"].isin(institution)]
    if program and "Program" in df: filtered=filtered[filtered["Program"].isin(program)]
    return filtered
