import io
import pandas as pd
import pytest
from academic_v2.data import generate_demo_data, validate_uploaded_csv, FEATURES
from academic_v2.analytics import dataset_kpis, filter_records
from academic_v2.model import train_demo_model, score_record

@pytest.fixture(scope="module")
def data(): return generate_demo_data()

@pytest.fixture(scope="module")
def model(data): return train_demo_model(data)

def test_deterministic(data):
    assert len(data)==960 and data.equals(generate_demo_data())
    assert all(data[k].between(0,100).all() for k in FEATURES)
    assert 0 < data["Needs Support Next Term"].sum() < 960

def test_no_real_identifiers(data):
    assert data["Student ID"].str.startswith("DEMO-").all()
    assert "STUDENT_NAME" not in data.columns

def test_metrics(model):
    assert model.test_size == 240
    assert 0 <= model.scores["pr_auc"] <= 1
    assert 0 <= model.scores["precision"] <= 1
    assert sum(model.scores[k] for k in ("true_negative","false_positive","false_negative","true_positive")) == model.test_size

def test_prediction(data,model):
    r=score_record(model,data.iloc[0][FEATURES].to_dict())
    assert 0 <= r["probability"] <=1 and len(r["contributions"]) == len(FEATURES)

def test_invalid_inputs(model):
    with pytest.raises(ValueError): score_record(model,{key: -1 for key in FEATURES})
    with pytest.raises(ValueError): score_record(model,{key: float("nan") for key in FEATURES})

def test_kpis(data):
    metrics=dataset_kpis(data)
    assert metrics["students"] == 960 and 0 <= metrics["average"] <=100

def test_filters(data):
    f=filter_records(data,["Northbridge Academy"],["Science"])
    assert f["Institution"].eq("Northbridge Academy").all()
    assert f["Program"].eq("Science").all()

def test_valid_csv():
    raw=b"Student ID,Mathematics,Science,English,Attendance\nX1,55,78,81,80\n"
    df=validate_uploaded_csv(raw)
    assert df["Average Score"].iloc[0] == round((55+78+81)/3,1)

def test_invalid_csv():
    with pytest.raises(ValueError):validate_uploaded_csv(b"Student ID,Mathematics\nX1,600\n")
    with pytest.raises(ValueError):validate_uploaded_csv(b"")

def test_metadata_not_a_score():
    raw=b"Student ID,Mathematics,Phone\nX1,52,9876543210\n"
    df=validate_uploaded_csv(raw)
    assert df["Average Score"].iloc[0]==52
