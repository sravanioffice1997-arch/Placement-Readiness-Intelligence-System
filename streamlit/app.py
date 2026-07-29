import streamlit as st
import PyPDF2
import json
import re
import joblib
import numpy as np
from groq import Groq
from fpdf import FPDF
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
GROQ_API_KEY = "PASTE_YOUR_GROQ_API_KEY_HERE"
client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Placement Readiness Intelligence System", layout="wide")

# ============================================================
# LOAD ML MODEL (cached so it loads once, not on every click)
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load("readiness_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, label_encoder, feature_cols

model, label_encoder, feature_cols = load_model()

# ============================================================
# CORE FUNCTIONS
# ============================================================
def extract_resume_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def call_groq(prompt, temperature=0.2):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)

def analyze_resume(resume_text):
    prompt = f"""
You are an expert technical recruiter. Read the resume text below and extract structured information.
Do NOT rely on any fixed skill list — infer skills contextually from projects, experience, and tools mentioned.

Return ONLY valid JSON, no markdown, no extra text, in this exact structure:
{{
  "skills": ["list", "of", "all", "technical", "and", "soft", "skills", "found"],
  "projects": ["short description of each project"],
  "certifications": ["list of certifications, empty list if none"],
  "internships": ["list of internships/work experience, empty list if none"],
  "education": "highest degree and field",
  "tools_and_technologies": ["list of tools/frameworks/languages"],
  "resume_completeness_score": <integer 0-100, based on how complete/detailed the resume is>
}}

Resume text:
\"\"\"{resume_text}\"\"\"
"""
    return call_groq(prompt, temperature=0)

def analyze_jd(jd_text):
    prompt = f"""
You are an expert technical recruiter. Read the job description below and extract structured requirements.
Do NOT rely on any fixed skill list — infer required skills contextually.

IMPORTANT — how to classify skills:
- If the JD has explicit sections like "Required Skills" / "Must Have" / "Essential" —
  put those skills in "required_skills".
- If the JD has explicit sections like "Good to Have" / "Nice to Have" / "Preferred" / "Bonus" —
  put those skills in "optional_skills", NOT in "required_skills".
- Within "required_skills", mark a skill as "critical" only if the JD phrasing signals it's
  non-negotiable (e.g. "must have", "strong experience in", "proficiency in", listed first,
  core to the role). Mark supporting/secondary required skills as required but not critical.
- If the JD has no explicit Required/Good-to-Have structure, use your judgment based on
  emphasis and phrasing.

Return ONLY valid JSON, no markdown, no extra text, in this exact structure:
{{
  "job_title": "string",
  "role_category": "string (e.g. Data Science, Web Development, DevOps)",
  "required_skills": ["all required/must-have skills, from explicit Required section if present"],
  "critical_skills": ["subset of required_skills that are absolutely essential/non-negotiable"],
  "optional_skills": ["all Good-to-Have / Preferred / Bonus skills, kept separate from required_skills"],
  "tools_and_frameworks": ["list"],
  "soft_skills": ["list"],
  "responsibilities": ["list of key responsibilities"]
}}

Job description text:
\"\"\"{jd_text}\"\"\"
"""
    return call_groq(prompt, temperature=0)

def analyze_skill_gap(resume_data, jd_data):
    prompt = f"""
Compare the candidate's resume profile with the job requirements below.
Match skills semantically and generously, for BOTH required and optional skills:
- "Statistics" satisfies "statistical modeling" or "statistics and probability"
- Training/evaluating ML models (e.g. "trained XGBoost and Random Forest models") satisfies "model evaluation" even if not explicitly named as a skill
- "React.js" matches "React", "ML" matches "Machine Learning", etc.
- If a skill is demonstrated through a PROJECT even if not listed as a standalone skill, count it as present — this applies equally to optional skills. For example, if a project mentions "Deep Learning & NLP" or "sentiment analysis model", that counts as evidence for deep learning frameworks and/or NLP even if not listed under resume skills.
- Read project descriptions carefully for technology and domain mentions, not just the skills list.

Resume skills: {resume_data['skills']}
Resume tools: {resume_data['tools_and_technologies']}
Resume projects: {resume_data['projects']}
JD required skills: {jd_data['required_skills']}
JD critical skills: {jd_data['critical_skills']}
JD optional/good-to-have skills: {jd_data.get('optional_skills', [])}

Return ONLY valid JSON, no markdown, in this exact structure:
{{
  "matched_skills": ["required skills present, including those demonstrated via projects"],
  "missing_skills": ["required skills genuinely not found anywhere in resume/skills/projects"],
  "critical_missing_skills": ["critical skills not found"],
  "matched_optional_skills": ["optional/good-to-have skills the candidate does have"],
  "missing_optional_skills": ["optional/good-to-have skills the candidate does NOT have"],
  "skills_to_learn_first": ["top 3-5 priority skills to learn, required skills first"]
}}
"""
    return call_groq(prompt, temperature=0)

def build_features(resume_data, jd_data, gap_data):
    """
    Converts LLM outputs into the 10 numeric features.
    Trusts gap_data's semantic matching (from the LLM comparison) for all
    skill-related features instead of doing brittle literal string matching.
    """
    required = jd_data.get("required_skills", []) or []
    critical = jd_data.get("critical_skills", []) or []
    optional = jd_data.get("optional_skills", []) or []
    matched = gap_data.get("matched_skills", []) or []
    missing = gap_data.get("missing_skills", []) or []
    critical_missing = gap_data.get("critical_missing_skills", []) or []
    matched_optional = gap_data.get("matched_optional_skills", []) or []

    # skill_match_percentage: matched / JD-required (JD-anchored)
    skill_match_percentage = round((len(matched) / len(required)) * 100) if required else 0

    # critical_skill_match_percentage
    matched_lower = [m.lower() for m in matched]
    critical_matched = [s for s in critical if s.lower() in matched_lower]
    critical_skill_match_percentage = round((len(critical_matched) / len(critical)) * 100) if critical else 100

    missing_skills_count = len(missing)
    critical_missing_skills_count = len(critical_missing)

    # optional_skill_match_percentage: its OWN feature now (11th feature),
    # so the trained model learns the right weight for it instead of us
    # forcing it into another feature.
    optional_skill_match_percentage = round((len(matched_optional) / len(optional)) * 100) if optional else 100

    # keyword_match_score: back to representing required-skill match only
    keyword_match_score = skill_match_percentage

    # project_relevance_score: fraction of MATCHED skills backed by evidence
    # in projects/tools, not literal phrase-matching
    project_text = " ".join(resume_data.get("projects", [])).lower()
    tools_text = " ".join(resume_data.get("tools_and_technologies", [])).lower()
    evidence_text = project_text + " " + tools_text
    if matched:
        supported = sum(1 for s in matched if any(word in evidence_text for word in s.lower().split()))
        project_relevance_score = round((supported / len(matched)) * 100)
    else:
        project_relevance_score = 40
    num_projects = len(resume_data.get("projects", []))
    project_relevance_score = min(100, project_relevance_score + min(num_projects * 3, 15))

    certs = resume_data.get("certifications", []) or []
    certification_relevance_score = 70 if certs else 20

    internships = resume_data.get("internships", []) or []
    internship_relevance_score = 70 if internships else 20

    resume_completeness_score = resume_data.get("resume_completeness_score", 50)

    # role_category_match_score: based on overall skill match strength
    if skill_match_percentage >= 80:
        role_category_match_score = 90
    elif skill_match_percentage >= 60:
        role_category_match_score = 70
    elif skill_match_percentage >= 40:
        role_category_match_score = 50
    else:
        role_category_match_score = 25

    return {
        "skill_match_percentage": skill_match_percentage,
        "critical_skill_match_percentage": critical_skill_match_percentage,
        "missing_skills_count": missing_skills_count,
        "critical_missing_skills_count": critical_missing_skills_count,
        "optional_skill_match_percentage": optional_skill_match_percentage,
        "project_relevance_score": project_relevance_score,
        "certification_relevance_score": certification_relevance_score,
        "internship_relevance_score": internship_relevance_score,
        "resume_completeness_score": resume_completeness_score,
        "keyword_match_score": keyword_match_score,
        "role_category_match_score": role_category_match_score,
    }

def predict_readiness(resume_data, jd_data, gap_data):
    features = build_features(resume_data, jd_data, gap_data)
    X = np.array([[features[col] for col in feature_cols]])
    pred_encoded = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    pred_proba = model.predict_proba(X)[0]

    class_order = ["Not Ready Yet", "Needs Improvement", "Moderately Ready", "Highly Ready"]
    label_to_rank = {l: i for i, l in enumerate(class_order)}
    proba_dict = dict(zip(label_encoder.inverse_transform(range(len(pred_proba))), pred_proba))
    weighted_rank = sum(label_to_rank[label] * prob for label, prob in proba_dict.items())
    readiness_score = round((weighted_rank / (len(class_order) - 1)) * 100)
    # Cap at 90: no resume is truly "perfect" against a JD, so we reserve
    # 91-100 to avoid implying flawlessness even at maximum model confidence.
    readiness_score = min(readiness_score, 90)

    return {
        "features": features,
        "readiness_label": pred_label,
        "readiness_score": readiness_score,
        "class_probabilities": proba_dict,
    }

def generate_feedback(resume_data, jd_data, gap_data, prediction_result):
    prompt = f"""
You are a career coach helping a student prepare for a job application.

Job Title: {jd_data.get('job_title')}
Role Category: {jd_data.get('role_category')}

Placement Readiness Score: {prediction_result['readiness_score']}/100
Readiness Level: {prediction_result['readiness_label']}

Matched Skills: {gap_data.get('matched_skills')}
Missing Skills: {gap_data.get('missing_skills')}
Critical Missing Skills: {gap_data.get('critical_missing_skills')}

Candidate's Projects: {resume_data.get('projects')}
Candidate's Certifications: {resume_data.get('certifications')}
Candidate's Education: {resume_data.get('education')}

Based on this, generate a coaching report. Return ONLY valid JSON, no markdown, in this exact structure:
{{
  "summary": "10-20 line easy-to-read summary explaining whether this student is suitable for this job role, written directly to the student in second person",
  "strengths": ["list of 3-5 specific strengths based on their actual resume"],
  "gaps": ["list of specific gaps, referencing missing_skills if any, otherwise note minor improvement areas"],
  "seven_day_plan": ["list of 4-6 concrete, specific daily/short-term action items"],
  "thirty_day_plan": ["list of 4-6 concrete, longer-term action items"],
  "resume_improvement_suggestions": ["list of 3-5 specific suggestions to improve the resume itself"],
  "interview_tips": ["list of 3-5 tips specific to this role and this candidate's background"]
}}
"""
    return call_groq(prompt, temperature=0.4)

# ============================================================
# PDF REPORT GENERATION
# ============================================================
def generate_pdf_report(resume_data, jd_data, gap_data, prediction, feedback):
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def h1(text):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(20, 20, 20)
        pdf.multi_cell(0, 10, text)
        pdf.ln(2)

    def h2(text):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 8, text)
        pdf.ln(1)

    def body(text):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        clean = str(text).encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, clean)
        pdf.ln(1)

    def bullet_list(items):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        for item in items:
            pdf.set_x(pdf.l_margin)
            clean_item = str(item).encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 6, f"- {clean_item}")
        pdf.ln(2)

    # Header
    h1("Placement Readiness Report")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 6, f"Job Title: {jd_data.get('job_title', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Score
    h2(f"Placement Readiness Score: {prediction['readiness_score']} / 100")
    body(f"Readiness Level: {prediction['readiness_label']}")
    pdf.ln(2)

    # Matched / Missing Skills
    h2("Matched Skills")
    bullet_list(gap_data.get("matched_skills", []) or ["None"])

    h2("Missing Skills")
    bullet_list(gap_data.get("missing_skills", []) or ["None"])

    if gap_data.get("critical_missing_skills"):
        h2("Critical Missing Skills")
        bullet_list(gap_data["critical_missing_skills"])

    # Summary
    h2("Summary")
    body(feedback.get("summary", ""))

    # Strengths / Gaps
    h2("Strengths")
    bullet_list(feedback.get("strengths", []))

    h2("Gaps to Address")
    bullet_list(feedback.get("gaps", []))

    # Plans
    h2("7-Day Improvement Plan")
    bullet_list(feedback.get("seven_day_plan", []))

    h2("30-Day Improvement Plan")
    bullet_list(feedback.get("thirty_day_plan", []))

    # Resume suggestions
    h2("Resume Improvement Suggestions")
    bullet_list(feedback.get("resume_improvement_suggestions", []))

    # Interview tips
    h2("Interview Tips")
    bullet_list(feedback.get("interview_tips", []))

    return bytes(pdf.output())

# ============================================================
# STREAMLIT UI
# ============================================================
st.title("🎯 Placement Readiness Intelligence System")
st.caption("Upload your resume and paste a job description to see how ready you are.")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
with col2:
    jd_text = st.text_area("Paste the job description here", height=250)

analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

if analyze_clicked:
    if not resume_file:
        st.error("Please upload a resume PDF first.")
    elif not jd_text.strip():
        st.error("Please paste a job description first.")
    else:
        with st.spinner("Reading resume..."):
            resume_text = extract_resume_text(resume_file)

        with st.spinner("Analyzing resume with AI..."):
            resume_data = analyze_resume(resume_text)

        with st.spinner("Analyzing job description with AI..."):
            jd_data = analyze_jd(jd_text)

        with st.spinner("Comparing resume with job requirements..."):
            gap_data = analyze_skill_gap(resume_data, jd_data)

        with st.spinner("Calculating placement readiness score..."):
            prediction = predict_readiness(resume_data, jd_data, gap_data)

        with st.spinner("Generating personalized feedback..."):
            feedback = generate_feedback(resume_data, jd_data, gap_data, prediction)

        # Store in session state
        st.session_state["resume_data"] = resume_data
        st.session_state["jd_data"] = jd_data
        st.session_state["gap_data"] = gap_data
        st.session_state["prediction"] = prediction
        st.session_state["feedback"] = feedback

        st.success("Analysis complete!")

# ---- Display results if available ----
if "prediction" in st.session_state:
    prediction = st.session_state["prediction"]
    gap_data = st.session_state["gap_data"]
    feedback = st.session_state["feedback"]
    jd_data = st.session_state["jd_data"]
    resume_data = st.session_state["resume_data"]

    st.divider()
    score = prediction["readiness_score"]
    label = prediction["readiness_label"]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Placement Readiness Score", f"{score} / 100")
        st.subheader(label)
    with c2:
        st.write("**Class Probabilities**")
        for cls, prob in prediction["class_probabilities"].items():
            st.progress(prob, text=f"{cls}: {prob*100:.1f}%")

    with st.expander("🔧 Debug: Raw Features (for troubleshooting)"):
        st.json(prediction["features"])

    st.divider()
    sc1, sc2 = st.columns(2)
    with sc1:
        st.write("### ✅ Matched Skills")
        st.write(", ".join(gap_data["matched_skills"]) or "None")
    with sc2:
        st.write("### ❌ Missing Skills")
        st.write(", ".join(gap_data["missing_skills"]) or "None")
        if gap_data["critical_missing_skills"]:
            st.write("**⚠️ Critical Missing:**", ", ".join(gap_data["critical_missing_skills"]))

    if gap_data.get("missing_optional_skills"):
        st.write("### 🌟 Nice-to-Have Skills You're Missing")
        st.write(", ".join(gap_data["missing_optional_skills"]))
        st.caption("These aren't required, but having them strengthens your profile slightly.")

    st.divider()
    st.write("### 📋 Summary")
    st.write(feedback["summary"])

    fc1, fc2 = st.columns(2)
    with fc1:
        st.write("### 💪 Strengths")
        for s in feedback["strengths"]:
            st.write("- ", s)
    with fc2:
        st.write("### 🎯 Gaps to Address")
        for g in feedback["gaps"]:
            st.write("- ", g)

    st.divider()
    pc1, pc2 = st.columns(2)
    with pc1:
        st.write("### 📅 7-Day Plan")
        for d in feedback["seven_day_plan"]:
            st.write("- ", d)
    with pc2:
        st.write("### 📆 30-Day Plan")
        for d in feedback["thirty_day_plan"]:
            st.write("- ", d)

    st.divider()
    st.write("### 📝 Resume Improvement Suggestions")
    for r in feedback["resume_improvement_suggestions"]:
        st.write("- ", r)

    st.write("### 🗣️ Interview Tips")
    for t in feedback["interview_tips"]:
        st.write("- ", t)

    st.divider()
    pdf_bytes = generate_pdf_report(resume_data, jd_data, gap_data, prediction, feedback)
    st.download_button(
        label="📄 Download Full Report (PDF)",
        data=pdf_bytes,
        file_name=f"placement_readiness_report_{jd_data.get('job_title', 'report').replace(' ', '_')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )