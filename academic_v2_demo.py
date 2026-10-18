"""Academic Intelligence V2 — public-safe fictional data portfolio dashboard.

Launch with: streamlit run academic_v2_demo.py
This is NOT the old authenticated multi-tenant application.
"""
from __future__ import annotations
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from academic_v2.data import FEATURES, generate_demo_data, validate_uploaded_csv
from academic_v2.analytics import dataset_kpis, filter_records
from academic_v2.model import score_record, train_demo_model

st.set_page_config(page_title="Academic Intelligence · V2", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root{--mint:#87f9d9;--blue:#88bafb;--pink:#ed8eb4;--panel:#111d30;--line:#273850}
[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 100% 0%,#222047 0,#080e1a 39%,#080d19 100%);color:#e9f1fd}
[data-testid="stHeader"]{background:#080f1bd4}
[data-testid="stSidebar"]{background:linear-gradient(170deg,#0d1425,#0a1020);border-right:1px solid #26354b}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#9eb0c6}
[data-testid="stMainBlockContainer"]{max-width:1460px;padding-top:2.2rem;padding-bottom:5rem}
h1,h2,h3,p{letter-spacing:-.016em}h1{letter-spacing:-.055em!important;font-weight:800!important}
.ai-brand{display:flex;align-items:center;gap:12px;margin:9px 0 35px}.ai-logo{background:linear-gradient(125deg,#83fbd7,#74cde9);color:#0e2331;display:grid;place-items:center;height:46px;width:46px;border-radius:15px;font-size:23px;font-weight:800}.ai-brand strong{display:block;font-size:21px;color:#f3f8ff;line-height:1.1}.ai-brand small{color:#7690aa;letter-spacing:.20em;font-size:9px}
.hero{padding:32px 34px;border:1px solid #34455f;border-radius:18px;background:radial-gradient(circle at 100% 40%,#49408360 0,transparent 44%),linear-gradient(105deg,#13293a,#10172b);margin:22px 0 23px;position:relative;overflow:hidden}.hero small{letter-spacing:.18em;font-size:11px;color:#8af5db;font-weight:800}.hero h1{font-size:clamp(38px,4.2vw,70px);line-height:1.08;margin:17px 0 10px}.hero h1 em{color:#83f5d4;font-style:normal}.hero p{color:#a9c0d0;font-size:15px;max-width:670px;margin:0 0 18px}
.eyebrow{font-size:11px;letter-spacing:.16em;font-weight:800;color:#82f4d5}.eyebrow .dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#82f4d5;margin-right:8px;box-shadow:0 0 12px #82f4d5}
.stMetric{background:#101a2b;border:1px solid #2a394f;padding:16px 20px;border-radius:13px}.stMetric [data-testid="stMetricValue"]{color:#e9f5ff;letter-spacing:-.04em}.stMetric [data-testid="stMetricLabel"]{color:#9eaec1}
[data-testid="stPlotlyChart"]{background:#111a2b;border:1px solid #293952;border-radius:15px;padding:7px}
[data-testid="stDataFrame"]{border:1px solid #34455e;border-radius:12px}
.stButton button[kind="primary"],.stFormSubmitButton button[kind="primary"]{background:#86f8d6;color:#082223;border-color:#86f8d6;font-weight:800}
.safe-note{background:#1b2132;color:#d3bba1;border:1px solid #5a4a3d;border-radius:10px;padding:13px 18px;font-size:12px}
.info-block{background:#111b2d;color:#b6c5d7;padding:19px;border:1px solid #34455f;border-radius:14px;line-height:1.65;font-size:14px}
.metric-foot{font-size:11px;color:#86a1b3;letter-spacing:.08em;padding-top:5px}
[data-testid="stSidebarNav"]{display:none}
@media(max-width:680px){.hero{padding:22px 20px}.hero h1{font-size:38px}.ai-brand{margin-bottom:18px}}
</style>
""", unsafe_allow_html=True)

# academic_v2/ui_polish.css • V2.1
from pathlib import Path as _V2Path
_v2_css = _V2Path(__file__).resolve().parent / 'academic_v2' / 'ui_polish.css'
if _v2_css.is_file():
    st.markdown('<style>' + _v2_css.read_text(encoding='utf-8') + '</style>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def demo_data() -> pd.DataFrame:
    return generate_demo_data()

@st.cache_resource(show_spinner=False)
def model_bundle():
    return train_demo_model(generate_demo_data())

records=demo_data()
bundle=model_bundle()
st.sidebar.markdown('<div class="ai-brand"><span class="ai-logo">✦</span><span><strong>academic<span style="color:#87f9d9">.ai</span></strong><small>INTELLIGENCE STUDIO</small></span></div>',unsafe_allow_html=True)
st.sidebar.caption("WORKSPACE / DEMO MODE")
page=st.sidebar.radio("Explore",["Command Center","Student Explorer","Prediction Lab","Model Insights"],label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.success("● Synthetic dataset online")
st.sidebar.caption("Research demonstration · no real student records · no user accounts")
st.sidebar.markdown("### Data source")
source=st.sidebar.selectbox("Dataset",["Fictional demo students"] + (["Local CSV (descriptive only)"] if os.environ.get("ACADEMIC_V2_LOCAL_CSV")=="1" else []),label_visibility="collapsed")
custom=False
if source.startswith("Local"):
    st.sidebar.warning("LOCAL ONLY. Never upload real student records to a public hosting service.")
    uploaded=st.sidebar.file_uploader("CSV of fictional records only",type=["csv"])
    if uploaded:
        try:
            records=validate_uploaded_csv(uploaded.getvalue())
            custom=True
            st.sidebar.info("Scores only. ML model remains trained solely on fictional demo data.")
        except ValueError as exc:
            st.sidebar.error(str(exc))

institution_options=sorted(records["Institution"].dropna().unique().tolist()) if "Institution" in records else []
program_options=sorted(records["Program"].dropna().unique().tolist()) if "Program" in records else []
st.sidebar.markdown("### Workspace filters")
institutions=st.sidebar.multiselect("Institution",institution_options,default=[])
programs=st.sidebar.multiselect("Program",program_options,default=[])
filtered=filter_records(records,institutions,programs)

st.markdown('<div class="eyebrow"><span class="dot"></span> RESEARCH ENVIRONMENT &nbsp; / &nbsp; 2026</div>',unsafe_allow_html=True)
st.caption(f"WORKSPACE  /  {page.upper()}   ·   FICTIONAL DATA ONLY")

COMMON_LAYOUT=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#c4d5e6",family="Arial"),margin=dict(l=35,r=25,t=45,b=25),xaxis=dict(gridcolor="#26364b"),yaxis=dict(gridcolor="#26364b"))
def plot(fig):
    fig.update_layout(**COMMON_LAYOUT)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

if page=="Command Center":
    st.markdown('<div class="hero"><small>THE FUTURE OF ACADEMIC INSIGHT</small><h1>Better signals.<br><em>Better support.</em></h1><p>Explore academic performance patterns and responsibly evaluated support-priority scores using exclusively fictional student records.</p><small>PYTHON &nbsp; / &nbsp; STREAMLIT &nbsp; / &nbsp; SCIKIT-LEARN</small></div>',unsafe_allow_html=True)
    st.markdown('<div class="safe-note"><b>RESEARCH DEMO:</b> The students, institutions, outcomes and model training labels are simulated. Results are not validated educational risk predictions.</div>',unsafe_allow_html=True)
    st.write("")
    data_kpis=dataset_kpis(filtered)
    m=st.columns(4)
    m[0].metric("Sample students",f"{data_kpis['students']:,}")
    m[1].metric("Average score",f"{data_kpis['average']:.1f} / 100")
    m[2].metric("Mean attendance",f"{data_kpis['attendance']:.1f}%" if data_kpis['attendance'] is not None else "N/A")
    m[3].metric("Below 60 average",f"{data_kpis['needs_attention']:,}")
    st.caption("Descriptive counts, not a diagnosis or a predicted dropout rate.")
    c1,c2=st.columns([1.4,1])
    with c1:
        st.subheader("Score distribution")
        if not filtered.empty:
            fig=px.histogram(filtered,x="Average Score",nbins=18,color_discrete_sequence=["#88f7da"])
            fig.update_traces(marker_line_width=0,opacity=.82)
            fig.update_layout(xaxis_title="Average score",yaxis_title="Students")
            plot(fig)
        else: st.info("No records match current filters.")
    with c2:
        st.subheader("Program mix")
        if "Program" in filtered and not filtered.empty:
            d=filtered["Program"].value_counts().reset_index()
            fig=px.pie(d,values="count",names="Program",hole=.69,color_discrete_sequence=["#84f4d6","#9b9af5","#6bafd8","#f29bc2"])
            fig.update_traces(textinfo="percent",textfont_size=12)
            plot(fig)
        else: st.info("Program details unavailable.")
    st.subheader("Recent demo records")
    visible=[c for c in ["Student ID","Institution","Program","Mathematics","Science","English","Average Score","Attendance"] if c in filtered]
    st.dataframe(filtered[visible].head(16),hide_index=True,use_container_width=True)
    st.download_button("Download synthetic sample CSV",data=demo_data().to_csv(index=False),file_name="academic_ai_synthetic_demo.csv",mime="text/csv")

elif page=="Student Explorer":
    st.title("Explore the student signals.")
    st.caption("Only fictional student IDs are shown. Filtering, sorting and comparisons are real operations on demo records.")
    if filtered.empty:
        st.warning("No records match the filters.")
    else:
        c1,c2=st.columns([1.2,1])
        with c1:
            st.subheader("Subject overview")
            subjects=[col for col in ["Mathematics","Science","English"] if col in filtered]
            if subjects:
                group=filtered[subjects].mean().reset_index()
                group.columns=["Subject","Average"]
                fig=px.bar(group,x="Average",y="Subject",orientation="h",color_discrete_sequence=["#8af2d6"],range_x=[0,100])
                plot(fig)
            else:st.info("No subject mark columns.")
        with c2:
            st.subheader("Performance by program")
            if "Program" in filtered:
                df=filtered.groupby("Program",as_index=False)["Average Score"].mean()
                fig=px.bar(df,x="Program",y="Average Score",color_discrete_sequence=["#9c9bff"],range_y=[0,100])
                plot(fig)
            else: st.info("Program grouping unavailable.")
        query=st.text_input("Filter Student ID",placeholder="DEMO-0001")
        subset=filtered[filtered["Student ID"].astype(str).str.contains(query,case=False,regex=False)] if query else filtered
        columns=[c for c in ["Student ID","Institution","Program","Semester"]+FEATURES+["Average Score"] if c in subset]
        st.dataframe(subset[columns],height=385,hide_index=True,use_container_width=True)
        st.caption(f"Showing {len(subset):,} of {len(filtered):,} filtered fictional records.")
        selected=st.selectbox("Inspect one fictional student",filtered["Student ID"].astype(str).tolist())
        chosen=filtered[filtered["Student ID"].astype(str)==selected].iloc[0]
        scores=[k for k in ["Mathematics","Science","English"] if k in filtered]
        if scores:
            fig=go.Figure(go.Scatterpolar(r=[float(chosen[c]) for c in scores]+[float(chosen[scores[0]])],theta=scores+[scores[0]],fill="toself",line_color="#8cf9d9"))
            fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",radialaxis=dict(visible=True,range=[0,100],gridcolor="#38445d")),showlegend=False)
            plot(fig)

elif page=="Prediction Lab":
    st.title("Meet the Prediction Lab.")
    st.caption("Simulated probability that a fictional student may need extra support next term. Not a real-world educational prediction.")
    st.markdown('<div class="safe-note"><b>ML LIMITATION:</b> This classifier learned from generated academic records and generated outcomes. Do not use it for decisions about actual students.</div>',unsafe_allow_html=True)
    st.write("")
    left,right=st.columns([1.25,1])
    with left:
        st.subheader("01 / Student signals")
        preset=st.selectbox("Load example",["Typical example","Higher support need","Strong academic profile"])
        examples={"Typical example":[70,74,72,79,75,68],"Higher support need":[32,38,40,45,50,39],"Strong academic profile":[83,81,88,96,91,82]}
        with st.form("predict_form"):
            values={}
            cs=st.columns(2)
            for i,feature in enumerate(FEATURES):
                values[feature]=cs[i%2].number_input(feature,0,100,value=examples[preset][i],step=1)
            submitted=st.form_submit_button("Analyze fictional profile  →",type="primary",use_container_width=True)
        if submitted:
            st.session_state["academic_v2_risk"] = score_record(bundle,values)
            st.session_state["academic_v2_profile"] = preset
    with right:
        st.subheader("02 / Model output")
        result=st.session_state.get("academic_v2_risk")
        if result:
            prob=result["probability"]
            st.metric("Synthetic support-priority score",f"{prob*100:.1f}%")
            st.progress(min(100,max(0,round(prob*100))))
            if result["review_suggested"]:st.warning("Above demonstration cutoff: a faculty review would be suggested in this fictional example.")
            else:st.success("Below demonstration cutoff. No review suggestion in this fictional example.")
            contributions=pd.DataFrame(result["contributions"])
            contributions["direction"]=np.where(contributions["log_odds_shift"]>0,"Raises score","Lowers score")
            fig=px.bar(contributions,x="log_odds_shift",y="feature",orientation="h",color="direction",color_discrete_map={"Raises score":"#ef8aaf","Lowers score":"#81f5d7"})
            plot(fig)
            st.caption("Model log-odds shift relative to reference medians. This is one-feature-at-a-time sensitivity, not a causal explanation or contribution in percentage points.")
        else:
            st.markdown('<div class="info-block">Choose fictional marks and select <b>Analyze fictional profile</b> to see an actual trained model inference. Detailed feature sensitivities will appear here.</div>',unsafe_allow_html=True)

elif page=="Model Insights":
    st.title("Beyond the black box.")
    st.caption("Logistic Regression · 75% train / 25% independent test · entirely synthetic outcomes")
    st.markdown('<div class="safe-note"><b>NOT VALIDATED ON REAL STUDENTS:</b> These metrics describe only a simulated-label dataset. The model must not be deployed to make educational interventions or admissions decisions.</div>',unsafe_allow_html=True)
    st.write("")
    m=st.columns(4)
    m[0].metric("Test PR-AUC",f"{bundle.scores['pr_auc']:.3f}")
    m[1].metric("Test precision",f"{bundle.scores['precision']:.1%}")
    m[2].metric("Test recall",f"{bundle.scores['recall']:.1%}")
    m[3].metric("Test ROC-AUC",f"{bundle.scores['roc_auc']:.3f}")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Confusion matrix · held-out test")
        cm=np.array([[bundle.scores["true_negative"],bundle.scores["false_positive"]],[bundle.scores["false_negative"],bundle.scores["true_positive"]]])
        fig=px.imshow(cm,text_auto=True,labels=dict(x="Model flag",y="Simulated truth"),x=["No review","Review"],y=["No need","Need"],color_continuous_scale=[[0,"#1c2843"],[1,"#85f7d6"]])
        plot(fig)
    with c2:
        st.subheader("Transparent methodology")
        st.markdown(f"""<div class="info-block"><b>Training dataset:</b> 960 fictional students<br><b>Test set:</b> {bundle.test_size} untouched records<br><b>Baseline PR-AUC:</b> {bundle.scores['baseline_pr_auc']:.3f}<br><b>Review cutoff:</b> {bundle.scores['threshold']:.0%} (fixed before test evaluation)<br><b>Brier score:</b> {bundle.scores['brier']:.3f}<br><b>Classifier:</b> StandardScaler + Logistic Regression</div>""",unsafe_allow_html=True)
        st.write("")
        st.info("Generated outcome labels make this a methodology demonstration, not evidence of real student-risk predictive validity.")
    st.subheader("Model coefficients (standardized feature space)")
    coef=bundle.pipeline.named_steps["logisticregression"].coef_[0]
    feature_effect=pd.DataFrame({"feature":FEATURES,"coefficient":coef,"direction":["Raises score" if x>0 else "Lowers score" for x in coef]})
    fig=px.bar(feature_effect,x="coefficient",y="feature",orientation="h",color="direction",color_discrete_map={"Raises score":"#eb93ae","Lowers score":"#8af4db"})
    plot(fig)
    st.caption("Positive coefficient increases the simulated support-need log-odds given other normalized features; not an explanation for an individual student.")

st.markdown("---")
st.caption("© 2026 Muhammed Ansar · Academic Intelligence V2 · Synthetic portfolio demonstration · Not suitable for real student decisions")
