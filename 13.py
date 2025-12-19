# student_stress_analyzer_v3.py
"""
Student Stress Analyzer — Full integrated Streamlit app (v3)

Features:
- Attractive UI with readable cards and remedy styles.
- Student assessment, PDF export, ML predictor.
- Companion (Ollama) support.
- Emails via Google API (if configured) or SMTP fallback.
- Consultation system broadcast to ALL doctors:
    - Students request consultations (preferred datetime).
    - All registered doctors receive email notifications.
    - Each doctor sees pending requests with Accept / Deny buttons.
    - Option A behavior: first doctor to Accept locks the consultation (status -> 'accepted').
    - Deny by a doctor marks as denied_by_<email> but does not block others from accepting.
    - When accepted: Google Calendar event attempted (creates Google Meet link), otherwise Jitsi fallback.
    - Student receives email upon accept/deny.
- Doctors' dashboard shows notifications, pending requests, scheduled consultations, and student records.
- Safe fallbacks and robust CSV-based persistence.
- Run with: streamlit run student_stress_analyzer_v3.py

Configuration:
- To use Google APIs (Calendar + Gmail) place credentials.json in project root and install google-auth-oauthlib & google-api-python-client.
- To use SMTP fallback, set environment variables: SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS
- Optional: streamlit-webrtc for webcam preview (doctors) and Ollama server for local LLM companion.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import hashlib
import json
import requests
import time
import random
import joblib
import base64
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from textblob import TextBlob
import plotly.express as px

# Optional google libs (used when available)
try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    GOOGLE_AVAILABLE = True
except Exception:
    GOOGLE_AVAILABLE = False

# Optional webcam support (doctors)
try:
    from streamlit_webrtc import webrtc_streamer
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

# ----------------------------
# File & directory setup
# ----------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

USERS_CSV = os.path.join(DATA_DIR, "users.csv")
PROGRESS_CSV = os.path.join(DATA_DIR, "progress.csv")
CONSULT_CSV = os.path.join(DATA_DIR, "consultations.csv")
MESSAGES_CSV = os.path.join(DATA_DIR, "messages.csv")
CHATBOT_CSV = os.path.join(DATA_DIR, "chatbot_responses.csv")
MODEL_PKL = os.path.join(DATA_DIR, "stress_predictor.pkl")
SYNTH_CSV = os.path.join(DATA_DIR, "academic_stress_dataset.csv")
LAST_USER = os.path.join(DATA_DIR, "last_user.txt")
GOOGLE_TOKEN = os.path.join(DATA_DIR, "google_token.json")
CREDENTIALS_FILE = "credentials.json"  # put your Google OAuth client credentials here

def ensure_csv(path, cols):
    if not os.path.exists(path):
        pd.DataFrame(columns=cols).to_csv(path, index=False)
    else:
        # if file exists but headers don't match, recreate
        try:
            df = pd.read_csv(path, nrows=0)
            existing = list(df.columns)
            if not all(c in existing for c in cols):
                pd.DataFrame(columns=cols).to_csv(path, index=False)
        except Exception:
            pd.DataFrame(columns=cols).to_csv(path, index=False)

ensure_csv(USERS_CSV, ["name","email","password_hash","role","assigned_patients"])
ensure_csv(PROGRESS_CSV, ["email","date","score_pct","sleep_hours","bp_systolic","bp_diastolic","notes"])
ensure_csv(CONSULT_CSV, ["consult_id","student","doctor","date_requested","preferred_time","scheduled_time","meeting_link","status","notes"])
ensure_csv(MESSAGES_CSV, ["msg_id","from_user","to_user","date","content","via_ollama"])
ensure_csv(CHATBOT_CSV, ["id","email","role","user_msg","bot_reply","timestamp"])

# ----------------------------
# Questionnaire, remedies, media
# ----------------------------
QUESTIONS = [
    "1. I feel overwhelmed by assignments or deadlines.",
    "2. I struggle to maintain focus during lectures or self-study.",
    "3. I procrastinate academic work due to stress.",
    "4. I feel anxious before exams or presentations.",
    "5. I often sacrifice sleep for studying or projects.",
    "6. I find it hard to balance academics with extracurriculars.",
    "7. I feel unsupported by professors or peers.",
    "8. I feel my performance does not reflect my effort.",
    "9. I skip meals or neglect health due to workload.",
    "10. I feel burnout towards the end of a semester.",
    "11. I am confident in managing multiple academic tasks. (reverse)",
    "12. I compare my performance with others frequently.",
    "13. I feel uncertain about my academic future or career.",
    "14. I experience headaches or fatigue during study sessions.",
    "15. I struggle with time management (planning/prioritizing).",
    "16. I rely on caffeine and late nights before exams.",
    "17. I feel pressure from family expectations regarding grades.",
    "18. I feel unmotivated despite having goals.",
    "19. I worry about financial or scholarship concerns.",
    "20. I feel like I study just for grades rather than understanding."
]
REVERSE_INDEXES = [10]
LIKERT_LABELS = {0:"Never",1:"Almost never",2:"Sometimes",3:"Fairly often",4:"Very often"}
MAX_SCORE = len(QUESTIONS) * 4

HAPPY_GIFS = [
    "https://media.giphy.com/media/111ebonMs90YLu/giphy.gif",
    "https://media.giphy.com/media/3oriO0OEd9QIDdllqo/giphy.gif",
    "https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif"
]
SAD_GIFS = [
    "https://media.giphy.com/media/9Y5BbDSkSTiY8/giphy.gif",
    "https://media.giphy.com/media/OPU6wzx8JrHna/giphy.gif",
    "https://media.giphy.com/media/3orieSZy6Eke4B9y5K/giphy.gif"
]

VIDEOS = {
    "Low": ["https://www.youtube.com/watch?v=2OEL4P1Rz04","https://www.youtube.com/watch?v=GPGY8mF8WQw"],
    "Moderate": ["https://www.youtube.com/watch?v=inpok4MKVLM","https://www.youtube.com/watch?v=ZToicYcHIOU"],
    "High": ["https://www.youtube.com/watch?v=MIr3RsUWrdo","https://www.youtube.com/watch?v=inpok4MKVLM"],
    "Very High": ["https://www.youtube.com/watch?v=ZToicYcHIOU","https://www.youtube.com/watch?v=MIr3RsUWrdo"]
}

BASE_REMEDIES = {
    "Low": [
        "Keep a consistent sleep schedule (7-9 hours)",
        "Take a 20-minute walk daily",
        "Keep in touch with friends weekly",
        "Practice gratitude journaling (3 items/day)",
        "Continue calming hobbies (music, art, reading)"
    ],
    "Moderate": [
        "Try 5–10 minutes of daily mindfulness",
        "Use Pomodoro (25/5) for study blocks",
        "Limit caffeine after 3 PM",
        "Take nature walks during breaks",
        "Break large tasks into 3–4 small steps"
    ],
    "High": [
        "Schedule sessions with campus counselling",
        "Introduce 30 min structured exercise 3x/week",
        "Use a physical daily planner to prioritize tasks",
        "Practice CBT journaling for anxiety patterns",
        "Consider reducing commitments temporarily"
    ],
    "Very High": [
        "Contact campus health or mental health services immediately",
        "If having thoughts of self-harm, contact emergency services or a crisis hotline",
        "Talk to family or a trusted adult for immediate support",
        "Create a safety plan and remove access to means of self-harm",
        "Consider urgent psychiatric evaluation if recommended"
    ]
}

def expand_remedies(base):
    pool = {}
    for cat, items in base.items():
        out = []
        for it in items:
            out.append(it)
            out.append(it + " — try this consistently for 2 weeks")
            out.append("Set a daily reminder to: " + it.lower())
        extras = [
            "Deep breathing: 4-4-8 pattern for 5 minutes",
            "Progressive muscle relaxation for 10 minutes before bed",
            "Digital detox: 1 evening per week without screens",
            "Organize your tasks using time-blocking",
            "Talk to a peer support group on campus",
            "Practice a calming bedtime routine: warm shower + reading"
        ]
        out.extend(extras)
        pool[cat] = list(dict.fromkeys(out))
    return pool

REMEDY_POOL = expand_remedies(BASE_REMEDIES)

# ----------------------------
# helpers (csv, hashing, persist)
# ----------------------------
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def save_csv(df, path):
    df.to_csv(path, index=False)

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def remember_user(email: str):
    try:
        with open(LAST_USER, "w") as f:
            f.write(email)
    except Exception:
        pass

def get_remembered_user():
    try:
        if os.path.exists(LAST_USER):
            with open(LAST_USER, "r") as f:
                return f.read().strip()
    except Exception:
        pass
    return None

# restore remembered
if 'user' not in st.session_state:
    st.session_state['user'] = None

remembered = get_remembered_user()
if remembered and st.session_state['user'] is None:
    users = load_csv(USERS_CSV)
    if not users.empty and remembered in users.get('email', []):
        st.session_state['user'] = users[users['email'] == remembered].iloc[0].to_dict()

# ----------------------------
# ML generate+train
# ----------------------------
def generate_synthetic_dataset(n=1000, outcsv=SYNTH_CSV):
    rows=[]
    rng=np.random.RandomState(42)
    for _ in range(n):
        sleep=max(2,min(10,rng.normal(7,1.5)))
        study=max(0,min(14,rng.normal(4,3)))
        bp_sys=int(max(90,min(160,rng.normal(120,12))))
        bp_dia=int(max(60,min(100,rng.normal(78,8))))
        hr=int(max(50,min(120,rng.normal(75,10))))
        mood=int(max(1,min(10,rng.normal(6,2))))
        gpa=round(max(0.0,min(4.0,rng.normal(3.0,0.5))),2)
        procrast=int(max(0,min(10,rng.normal(4,3))))
        support=int(max(0,min(10,rng.normal(6,3))))
        anxiety=int(max(0,min(10,rng.normal(5,3))))
        base=(10-sleep)*6 + max(0,study-4)*3 + procrast*4 + (10-mood)*3 + (10-support)*2 + anxiety*4
        pct=int(min(100,max(0,(base/(6*10+3*10+4*10+3*10+2*10+4*10))*100*1.1)))
        if pct<=30:
            label="Low"
        elif pct<=60:
            label="Moderate"
        elif pct<=80:
            label="High"
        else:
            label="Very High"
        rows.append({
            "sleep_hours":round(sleep,2),"study_hours":round(study,2),"bp_sys":bp_sys,"bp_dia":bp_dia,
            "heart_rate":hr,"mood_score":mood,"gpa":gpa,"procrastination":procrast,
            "support_level":support,"academic_anxiety":anxiety,"stress_pct":pct,"stress_level":label
        })
    df=pd.DataFrame(rows)
    df.to_csv(outcsv,index=False)
    return df

def train_model_from_csv(csv_path=SYNTH_CSV, model_out=MODEL_PKL):
    df=pd.read_csv(csv_path)
    features=["sleep_hours","study_hours","bp_sys","bp_dia","heart_rate","mood_score","gpa","procrastination","support_level","academic_anxiety"]
    X=df[features]
    mapping={"Low":0,"Moderate":1,"High":2,"Very High":3}
    y=df["stress_level"].map(mapping)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    clf=RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train,y_train)
    preds=clf.predict(X_test)
    acc=accuracy_score(y_test,preds)
    jr=classification_report(y_test,preds)
    joblib.dump(clf, model_out)
    return clf, acc, jr

MODEL=None
if os.path.exists(MODEL_PKL):
    try:
        MODEL=joblib.load(MODEL_PKL)
    except Exception:
        MODEL=None

# ----------------------------
# Ollama streaming companion
# ----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

def stream_ollama_companion(prompt, model=OLLAMA_MODEL):
    personality = (
        "You are a compassionate academic support companion. "
        "Listen empathetically, validate emotions, and offer practical coping steps concisely.\n\n"
        f"Student: {prompt}\nCompanion:"
    )
    payload={"model":model,"prompt":personality,"stream":True,"max_tokens":400,"temperature":0.7}
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as r:
            if r.status_code!=200:
                yield f"[Ollama error {r.status_code}: {r.text}]"
                return
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj=json.loads(line)
                    if isinstance(obj, dict):
                        if "response" in obj and obj["response"]:
                            yield obj["response"]
                        elif "output" in obj and obj["output"]:
                            yield obj["output"]
                        else:
                            for v in obj.values():
                                if isinstance(v,str) and v.strip():
                                    yield v
                                    break
                    else:
                        yield str(obj)
                except Exception:
                    yield line
    except requests.exceptions.ConnectionError:
        yield "⚠️ Ollama server not reachable. Start it with `ollama serve`."
    except Exception as e:
        yield f"[Ollama exception: {e}]"

# ----------------------------
# Sentiment & PDF export
# ----------------------------
def sentiment_of_text(text):
    try:
        s=TextBlob(text).sentiment.polarity
        if s>0.1: return "positive"
        if s<-0.1: return "negative"
        return "neutral"
    except Exception:
        return "neutral"

def export_progress_pdf(email):
    df=load_csv(PROGRESS_CSV)
    my=df[df['email']==email] if not df.empty else pd.DataFrame()
    fname=os.path.join(DATA_DIR, f"{email.replace('@','_')}_stress_report.pdf")
    c=canvas.Canvas(fname,pagesize=A4)
    c.setFont("Helvetica-Bold",16)
    c.drawString(40,820,"Student Stress Report")
    c.setFont("Helvetica",11)
    c.drawString(40,800,f"Email: {email}")
    c.drawString(40,785,f"Generated: {datetime.now().isoformat()}")
    y=760
    if my.empty:
        c.drawString(40,y,"No progress records available.")
    else:
        my=my.copy()
        try:
            my['date']=pd.to_datetime(my['date'])
        except Exception:
            pass
        my=my.sort_values('date', ascending=False).head(30)
        for _,row in my.iterrows():
            try:
                date_str = row['date'].strftime('%Y-%m-%d %H:%M') if hasattr(row['date'],'strftime') else str(row['date'])
            except Exception:
                date_str = str(row.get('date',''))
            c.drawString(40,y,f"{date_str} — Stress%: {row.get('score_pct', '')} Sleep: {row.get('sleep_hours','')}")
            y-=14
            if y<80:
                c.showPage()
                y=800
    c.save()
    return fname

# ----------------------------
# Google OAuth helpers (Calendar + Gmail)
# ----------------------------
SCOPES = ['https://www.googleapis.com/auth/calendar.events','https://www.googleapis.com/auth/gmail.send','openid','https://www.googleapis.com/auth/userinfo.email']

def get_google_creds():
    """Load saved token if exists, otherwise run installed app flow to obtain credentials.
    Requires credentials.json (Google OAuth client) in project root."""
    if not GOOGLE_AVAILABLE:
        return None
    # load saved token
    if os.path.exists(GOOGLE_TOKEN):
        try:
            creds = Credentials.from_authorized_user_file(GOOGLE_TOKEN, SCOPES)
            return creds
        except Exception:
            pass
    # if no credentials file
    if not os.path.exists(CREDENTIALS_FILE):
        return None
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    # save token
    with open(GOOGLE_TOKEN, 'w') as f:
        f.write(creds.to_json())
    return creds

def create_calendar_event(creds, summary, start_dt, duration_minutes=30, attendees=None):
    try:
        service = build('calendar', 'v3', credentials=creds)
        end_dt = start_dt + timedelta(minutes=duration_minutes)
        event = {
            'summary': summary,
            'start': {'dateTime': start_dt.isoformat(), 'timeZone': 'Asia/Kolkata'},
            'end': {'dateTime': end_dt.isoformat(), 'timeZone': 'Asia/Kolkata'},
            'attendees': [{'email': e} for e in (attendees or [])],
            # request conferencing
            'conferenceData': {'createRequest': {'requestId': hashlib.sha1(f"{summary}{time.time()}".encode()).hexdigest()}}
        }
        created = service.events().insert(calendarId='primary', body=event, conferenceDataVersion=1).execute()
        # get joining link if available
        link = created.get('hangoutLink') or (created.get('conferenceData') or {}).get('entryPoints',[{}])[0].get('uri','')
        return created, link
    except Exception:
        return None, None

def send_gmail_message(creds, to_email, subject, body_text):
    try:
        service = build('gmail', 'v1', credentials=creds)
        message_bytes = f"To: {to_email}\r\nSubject: {subject}\r\n\r\n{body_text}".encode('utf-8')
        raw = base64.urlsafe_b64encode(message_bytes).decode('utf-8')
        message = {'raw': raw}
        service.users().messages().send(userId='me', body=message).execute()
        return True
    except Exception:
        return False

# ----------------------------
# SMTP helper for sending emails with attachments (fallback)
# ----------------------------
def send_email_via_smtp(to_email, subject, body_text, attachments=None):
    server = os.environ.get("SMTP_SERVER")
    port = os.environ.get("SMTP_PORT")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    if not (server and port and user and password):
        return False, "SMTP not configured"
    try:
        msg = EmailMessage()
        msg["From"] = user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body_text)
        if attachments:
            for p in attachments:
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                        maintype = "application"
                        subtype = "octet-stream"
                        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=os.path.basename(p))
                except Exception:
                    pass
        smtp = smtplib.SMTP(server, int(port))
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
        smtp.quit()
        return True, "Sent via SMTP"
    except Exception as e:
        return False, str(e)

def send_email_auto(to_email, subject, body_text, attachments=None):
    # try Google API first
    if GOOGLE_AVAILABLE:
        creds = get_google_creds()
        if creds:
            ok = send_gmail_message(creds, to_email, subject, body_text)
            if ok:
                return True, "Sent via Google API"
    # fallback to SMTP
    ok,msg = send_email_via_smtp(to_email, subject, body_text, attachments=attachments)
    return ok, msg

# ----------------------------
# UI styling (improved & readable)
# ----------------------------
st.set_page_config(page_title="Student Stress Analyzer", layout="wide", page_icon="🎓")
st.markdown(
    """
<style>
:root {
  --bg-dark: #0b0f12;
  --card-text: #0f1724;
  --muted: #94a3b8;
  --accent-1: #7c3aed;
  --accent-2: #06b6d4;
  --glass: rgba(255,255,255,0.03);
}
[data-testid="stAppViewContainer"] {animation: fadeIn 0.5s ease-in-out; background: linear-gradient(180deg,#0b0f12 0%, #0f1724 100%); color: #e6eef8;}
@keyframes fadeIn {from {opacity:0; transform: translateY(6px);} to {opacity:1; transform: translateY(0);}}
.header {
  background: linear-gradient(90deg,var(--accent-1),var(--accent-2));
  padding: 18px; border-radius: 12px; color: white; margin-bottom: 16px; box-shadow: 0 8px 30px rgba(2,6,23,0.6);
  transition: transform 0.18s ease;
}
.header:hover { transform: translateY(-2px); }
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(250,250,250,0.86));
  border-radius: 14px; padding: 16px; box-shadow: 0 12px 30px rgba(2,6,23,0.35);
  margin-bottom: 14px; color: var(--card-text);
  transition: box-shadow 0.18s ease, transform 0.18s ease;
}
.card:hover { box-shadow: 0 18px 40px rgba(2,6,23,0.5); transform: translateY(-4px); }
.card h2, .card h3, .card h4 { color: #0f1724 !important; font-weight:700 !important; }
.small-muted {color: var(--muted); font-size:13px; margin-top:6px;}
.remedy {margin:8px 0; padding:12px; border-radius:10px; font-weight:500; font-size:15px; line-height:1.4em;}
.low {background: linear-gradient(90deg,#d8f3dc,#b7e4c7); color:#081c15}
.mod {background: linear-gradient(90deg,#fff8e1,#fff3cd); color:#3a2b00}
.high {background: linear-gradient(90deg,#ffdede,#ffc6c6); color:#58151c}
.very {background: linear-gradient(90deg,#f5b7b1,#f1948a); color:#4a0b0b}
.sidebar-card {background: rgba(255,255,255,0.03); padding:10px; border-radius:10px; color:#e6eef8}
button.stButton > button { background: transparent; border: 1px solid rgba(255,255,255,0.08); padding: 8px 12px; border-radius: 8px; color: #e6eef8; }
.notice {background: rgba(255,255,255,0.02); padding:10px; border-radius:8px; color:#cbd5e1}
</style>
""", unsafe_allow_html=True
)

def landing_banner():
    st.markdown('<div class="header"><h1>🎓 Student Stress Analyzer</h1><div class="small-muted">Understand your stress, track progress, get remedies, and talk to a companion.</div></div>', unsafe_allow_html=True)

def page_header(title, subtitle=None):
    st.markdown(f'<div class="card"><h2 style="margin:0;">{title}</h2><div class="small-muted">{subtitle or ""}</div></div>', unsafe_allow_html=True)

# ----------------------------
# Authentication helpers + routing (with email on login)
# ----------------------------
def register_user(name, email, password, role):
    users = load_csv(USERS_CSV)
    if not users.empty and email in users.get('email', []):
        return False, "Email already registered."
    pw_hash = hash_password(password)
    new = {"name": name, "email": email, "password_hash": pw_hash, "role": role, "assigned_patients": ""}
    users = pd.concat([users, pd.DataFrame([new])], ignore_index=True) if not users.empty else pd.DataFrame([new])
    save_csv(users, USERS_CSV)
    remember_user(email)
    return True, "Registered and logged in."

def login_user(email, password, role):
    users = load_csv(USERS_CSV)
    if users.empty:
        return None
    matched = users[(users["email"]==email) & (users["password_hash"]==hash_password(password)) & (users["role"]==role)]
    if not matched.empty:
        remember_user(email)
        return matched.iloc[0].to_dict()
    return None

def auto_login_if_exists(email):
    users = load_csv(USERS_CSV)
    if users.empty: return None
    matched = users[users['email']==email]
    if not matched.empty:
        return matched.iloc[0].to_dict()
    return None

def send_login_notification(user_record):
    try:
        name = user_record.get("name","User")
        email = user_record.get("email")
        subject = f"Login notification — Student Stress Analyzer"
        body = f"Hello {name},\n\nYou have successfully logged into Student Stress Analyzer on {datetime.now().isoformat()}.\nIf this wasn't you, please change your password immediately.\n\nRegards,\nStudent Stress Analyzer"
        ok,msg = send_email_auto(email, subject, body)
        try:
            if ok:
                st.sidebar.success("Login notification sent to your email.")
            else:
                st.sidebar.info("Login notification not sent (email not configured).")
        except Exception:
            pass
    except Exception:
        pass

def send_assessment_report(email, pct, category, remedies, pdf_path=None):
    try:
        subject = f"Your Assessment Report — {pct}% — {category}"
        body_lines = [
            f"Hello,",
            "",
            f"Your recent stress assessment result: {pct}% — {category}",
            "",
            "Suggested remedies:",
        ]
        for r in remedies:
            body_lines.append(f" - {r}")
        body_lines.append("")
        body_lines.append("We attached your progress report (if available). Please reach out if you need support.")
        body = "\n".join(body_lines)
        attachments = [pdf_path] if pdf_path and os.path.exists(pdf_path) else None
        ok,msg = send_email_auto(email, subject, body, attachments=attachments)
        return ok,msg
    except Exception as e:
        return False,str(e)

# ----------------------------
# Sidebar navigation and auth UI
# ----------------------------
if 'page' not in st.session_state:
    st.session_state['page']='dashboard'

st.sidebar.title("Navigate")
if st.sidebar.button("🏠 Dashboard"): st.session_state['page']='dashboard'
if st.sidebar.button("🧾 Assessment"): st.session_state['page']='assessment'
if st.sidebar.button("📈 Progress"): st.session_state['page']='progress'
if st.sidebar.button("📅 Consultations"): st.session_state['page']='consultations'
if st.sidebar.button("💬 Messages"): st.session_state['page']='chat'
if st.sidebar.button("🤖 Companion"): st.session_state['page']='ollama'
if st.sidebar.button("📊 ML Predictor"): st.session_state['page']='ml'
if st.sidebar.button("⚙️ Admin"): st.session_state['page']='admin'

st.sidebar.markdown("---")
st.sidebar.title("Account")

if st.session_state.get('user') is None:
    remembered = get_remembered_user()
    auth_mode = st.sidebar.radio("Action", ["Login","Register"])
    role_choice = st.sidebar.selectbox("I am a", ["Student","Doctor","Admin"])
    if auth_mode=="Register":
        name = st.sidebar.text_input("Full name")
        email = st.sidebar.text_input("Email")
        pw = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Register"):
            ok,msg = register_user(name or email, email, pw, role_choice)
            if ok:
                st.sidebar.success(msg)
                st.session_state['user']=auto_login_if_exists(email)
                try:
                    send_login_notification(st.session_state['user'])
                except Exception:
                    pass
                st.rerun()
            else:
                st.sidebar.error(msg)
    else:
        email = st.sidebar.text_input("Email", value=remembered or "")
        pw = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            user = login_user(email, pw, role_choice)
            if user:
                try:
                    send_login_notification(user)
                except Exception:
                    pass
                st.session_state['user']=user
                st.sidebar.success(f"Welcome, {user['name']}")
                st.rerun()
            else:
                users = load_csv(USERS_CSV)
                if email and (not users.empty) and (email in users.get('email', [])):
                    u = users[users['email']==email].iloc[0].to_dict()
                    if u['role']==role_choice:
                        st.sidebar.warning("Auto-logging in for convenience (remembered).")
                        try:
                            send_login_notification(u)
                        except Exception:
                            pass
                        st.session_state['user']=u
                        remember_user(email)
                        st.rerun()
                st.sidebar.error("Invalid credentials or role mismatch")
else:
    st.sidebar.markdown(f'<div class="sidebar-card">Signed in: **{st.session_state["user"]["name"]}**  \n({st.session_state["user"]["role"]})</div>', unsafe_allow_html=True)
    if st.sidebar.button("Logout"):
        st.session_state['user']=None
        try:
            if os.path.exists(LAST_USER):
                os.remove(LAST_USER)
        except Exception:
            pass
        st.rerun()

# enforce login
if st.session_state.get('user') is None:
    landing_banner()
    st.info("Please register or login from the sidebar. We remember the last user on this machine.")
    st.stop()

USER = st.session_state['user']
USER_EMAIL = USER['email']
USER_ROLE = USER['role']

# ----------------------------
# Pages implementations
# ----------------------------
def page_dashboard():
    page_header("Dashboard — Student Progress & Improvements", f"Welcome, {USER['name']}. Overview of your recent records.")
    df = load_csv(PROGRESS_CSV)
    my = df[df['email']==USER_EMAIL] if not df.empty else pd.DataFrame()
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("<div class='card'><h3>Recent history</h3></div>", unsafe_allow_html=True)
        if my.empty:
            st.info("No records yet. Try taking the assessment.")
        else:
            myc=my.copy()
            try:
                myc['date']=pd.to_datetime(myc['date'])
            except Exception:
                pass
            st.dataframe(myc.sort_values('date',ascending=False).head(20))
    with col2:
        st.markdown("<div class='card'><h3>Quick actions</h3></div>", unsafe_allow_html=True)
        if st.button("Take Assessment"):
            st.session_state['page']='assessment'
            st.rerun()
        if st.button("Open Companion"):
            st.session_state['page']='ollama'
            st.rerun()

    if not my.empty and 'score_pct' in my.columns and my['score_pct'].notna().any():
        try:
            fig=px.line(my.sort_values('date'), x='date', y='score_pct', title='Stress % over time')
            st.plotly_chart(fig, use_container_width=True)
            sorted_my=my.sort_values('date')
            first=sorted_my['score_pct'].iloc[0]
            last=sorted_my['score_pct'].iloc[-1]
            change=last-first
            st.metric("Change since first record", f"{change} pct points", delta=f"{(last-first):+.1f}")
        except Exception:
            pass

    cb = load_csv(CHATBOT_CSV)
    recent = cb[cb['email']==USER_EMAIL] if not cb.empty else pd.DataFrame()
    if not recent.empty:
        st.markdown("<div class='card'><h3>Last Companion Conversation</h3></div>", unsafe_allow_html=True)
        latest=recent.sort_values('timestamp', ascending=False).iloc[0]
        st.write("You:", latest['user_msg'])
        st.write("Companion:", latest['bot_reply'][:500] + ("..." if len(latest['bot_reply'])>500 else ""))

def page_assessment():
    if USER_ROLE.lower() == 'doctor':
        st.warning("Doctors do not take stress assessments. Please use your dashboard to view student data and consultations.")
        return

    page_header("Assessment — Student Stress Analyzer", "Answer the questions and optionally give sleep/BP context.")
    st.markdown("<div class='card'><h4>Perceived Academic Stress Questionnaire (20 items)</h4></div>", unsafe_allow_html=True)
    cols=st.columns(2)
    responses=[None]*len(QUESTIONS)
    for i,q in enumerate(QUESTIONS):
        with cols[i%2]:
            responses[i]=st.selectbox(q, options=list(LIKERT_LABELS.keys()), format_func=lambda x: LIKERT_LABELS[x], key=f"q_{i}_{USER_EMAIL}")
    st.markdown("---")
    st.subheader("Daily context (optional)")
    c1,c2,c3=st.columns([1,1,1])
    with c1:
        sleep_hours=st.number_input("Sleep hours last night", 0.0, 24.0, 7.0, step=0.25, key=f"sleep_{USER_EMAIL}")
    with c2:
        bp_sys=st.number_input("BP systolic (optional)", 0, 300, 0, key=f"bps_{USER_EMAIL}")
    with c3:
        bp_dia=st.number_input("BP diastolic (optional)", 0, 200, 0, key=f"bpd_{USER_EMAIL}")
    notes=st.text_area("Notes (optional)", key=f"notes_{USER_EMAIL}")
    if st.button("Submit Assessment"):
        scored=[]
        for i,val in enumerate(responses):
            v=int(val)
            if i in REVERSE_INDEXES:
                v=4-v
            scored.append(v)
        total=sum(scored)
        pct=int(round((total/MAX_SCORE)*100))
        category=("Low" if pct<=30 else ("Moderate" if pct<=60 else ("High" if pct<=80 else "Very High")))
        dfp=load_csv(PROGRESS_CSV)
        row={"email":USER_EMAIL,"date":datetime.now().isoformat(),"score_pct":pct,
             "sleep_hours":float(sleep_hours) if sleep_hours else np.nan,
             "bp_systolic":int(bp_sys) if bp_sys>0 else np.nan,
             "bp_diastolic":int(bp_dia) if bp_dia>0 else np.nan,"notes":notes}
        dfp=pd.concat([dfp,pd.DataFrame([row])], ignore_index=True)
        save_csv(dfp, PROGRESS_CSV)
        st.success(f"Recorded: {pct}% — {category}")
        gif=random.choice(HAPPY_GIFS if pct<=30 else SAD_GIFS)
        st.image(gif, use_column_width=True)
        pool=REMEDY_POOL.get(category, [])
        sample=min(len(pool), 12)
        selected=random.sample(pool, sample) if pool else []
        css="low" if category=="Low" else ("mod" if category=="Moderate" else ("high" if category=="High" else "very"))
        st.markdown(f"<div class='card {css}'>", unsafe_allow_html=True)
        st.markdown(f"### Suggested remedies — {category}")
        for r in selected:
            st.markdown(f"<div class='remedy'>• {r}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        vids=VIDEOS.get(category, [])
        if vids:
            st.markdown("### Short videos you may find helpful")
            for v in vids:
                st.video(v)
        if MODEL is not None:
            study_hours=4.0
            try:
                words=notes.split()
                nums=[float(w) for w in words if w.replace('.','',1).isdigit()]
                if nums: study_hours=float(nums[0])
            except Exception:
                study_hours=4.0
            feats=np.array([[sleep_hours, study_hours, bp_sys if bp_sys>0 else 120, bp_dia if bp_dia>0 else 80, 75, 6, 3.0, 4, 6, 5]])
            try:
                pred=MODEL.predict(feats)[0]
                maplab={0:"Low",1:"Moderate",2:"High",3:"Very High"}
                st.info(f"ML Model suggests: {maplab.get(pred,'Unknown')}")
            except Exception:
                st.info("ML prediction currently unavailable.")
        st.balloons()

        # generate PDF and attempt to email user
        pdf_path = export_progress_pdf(USER_EMAIL)
        ok,msg = send_assessment_report(USER_EMAIL, pct, category, selected[:10], pdf_path)
        if ok:
            st.success("Assessment report and remedies emailed to you.")
        else:
            st.warning(f"Couldn't send email automatically: {msg}. You can download the PDF below.")
        try:
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF report", f.read(), file_name=os.path.basename(pdf_path), mime="application/pdf")
        except Exception:
            pass

def page_progress():
    page_header("Progress — Historical Data & Trends", "Track your stress percentage and context over time")
    df=load_csv(PROGRESS_CSV)
    my=df[df['email']==USER_EMAIL] if not df.empty else pd.DataFrame()
    if my.empty:
        st.info("No progress data yet.")
        return
    my=my.copy()
    try:
        my['date']=pd.to_datetime(my['date'])
    except Exception:
        pass
    my_sorted=my.sort_values('date')
    c1,c2=st.columns(2)
    with c1:
        fig=px.line(my_sorted, x='date', y='score_pct', title='Stress % over time', markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Latest Stress %", f"{my_sorted['score_pct'].iloc[-1]}%")
    with c2:
        if 'sleep_hours' in my_sorted.columns and my_sorted['sleep_hours'].notna().any():
            fig2=px.line(my_sorted, x='date', y='sleep_hours', title='Sleep hours', markers=True)
            st.plotly_chart(fig2, use_container_width=True)
        if 'bp_systolic' in my_sorted.columns and my_sorted['bp_systolic'].notna().any():
            fig3=px.line(my_sorted, x='date', y='bp_systolic', title='BP Systolic', markers=True)
            st.plotly_chart(fig3, use_container_width=True)
    st.markdown("### Recent entries")
    st.dataframe(my_sorted.sort_values('date', ascending=False).head(50))
    if st.button("Download progress PDF"):
        fname=export_progress_pdf(USER_EMAIL)
        with open(fname,"rb") as f:
            st.download_button("Download PDF", f.read(), file_name=os.path.basename(fname), mime="application/pdf")

def page_consultations():
    page_header("Consultations — Request or Accept", "Schedule a consultation with a doctor")
    # reload consults fresh
    cdf = load_csv(CONSULT_CSV)
    users_df = load_csv(USERS_CSV)

    # Student flow: request consultation (broadcast to all doctors)
    if USER_ROLE.lower()=="student":
        st.subheader("Request a consultation with any available doctor")
        preferred_date = st.date_input("Preferred date", datetime.now().date() + timedelta(days=1))
        preferred_time = st.time_input("Preferred time", (datetime.now() + timedelta(hours=1)).time())
        preferred_dt = datetime.combine(preferred_date, preferred_time)
        notes = st.text_area("Notes for the doctor (optional)")

        if st.button("Request consultation (notify all doctors)"):
            docs = users_df[users_df['role'].str.lower()=='doctor'] if not users_df.empty else pd.DataFrame()
            if docs.empty:
                st.warning("No doctors registered yet.")
            else:
                cid = hashlib.sha1(f"{USER_EMAIL}{datetime.now().isoformat()}{random.random()}".encode()).hexdigest()[:12]
                meeting_link = ""  # not scheduled yet; created on Accept
                new = {
                    "consult_id": cid,
                    "student": USER_EMAIL,
                    "doctor": "",  # not assigned yet
                    "date_requested": datetime.now().isoformat(),
                    "preferred_time": preferred_dt.isoformat(),
                    "scheduled_time": "",
                    "meeting_link": meeting_link,
                    "status": "requested",
                    "notes": notes or ""
                }
                cdf = pd.concat([cdf, pd.DataFrame([new])], ignore_index=True)
                save_csv(cdf, CONSULT_CSV)
                st.success("Consultation request broadcast to all registered doctors.")

                # notify every doctor by email
                for _, d in docs.iterrows():
                    try:
                        doc_email = d['email']
                        doc_name = d.get('name', 'Doctor')
                        subj = f"New Consultation Request from {USER.get('name',USER_EMAIL)}"
                        body = (
                            f"Dear Dr. {doc_name},\n\n"
                            f"A new consultation request has been made by {USER.get('name',USER_EMAIL)} ({USER_EMAIL}).\n"
                            f"Preferred time: {preferred_dt.strftime('%Y-%m-%d %H:%M')}\n"
                            f"Notes: {notes}\n\n"
                            f"Please log in to Student Stress Analyzer and accept or deny this request. First doctor to accept will be assigned.\n\n"
                            "Regards,\nStudent Stress Analyzer System"
                        )
                        send_email_auto(doc_email, subj, body)
                    except Exception:
                        pass

        st.markdown("### Your consultation requests")
        st.dataframe(cdf[cdf['student']==USER_EMAIL].sort_values('date_requested', ascending=False))
        return

    # Doctor flow: view all pending requests (they were broadcast)
    if USER_ROLE.lower()=="doctor":
        st.subheader("Pending consultation requests (first-accept locks)")
        # fetch latest
        cdf = load_csv(CONSULT_CSV)
        pending = cdf[(cdf['status']=='requested')]
        if pending.empty:
            st.info("No pending consultation requests at the moment.")
        else:
            # show each pending request with Accept / Deny buttons
            for idx,row in pending.sort_values('date_requested', ascending=True).iterrows():
                consult_id = row['consult_id']
                student_email = row['student']
                student_name = ""
                try:
                    users = users_df
                    student_name = users.loc[users['email']==student_email, 'name'].values[0] if not users.empty and student_email in users['email'].values else student_email
                except Exception:
                    student_name = student_email
                preferred_time = row.get('preferred_time','')
                notes = row.get('notes','')
                st.markdown(f"**Consult ID:** {consult_id}  \n**Student:** {student_name} — {student_email}  \n**Preferred:** {preferred_time}  \n**Notes:** {notes}")
                cols = st.columns([1,1,1])
                if cols[0].button(f"Accept {consult_id}", key=f"accept_{consult_id}"):
                    # reload consults to avoid race conditions
                    latest = load_csv(CONSULT_CSV)
                    cur = latest[latest['consult_id']==consult_id]
                    if cur.empty:
                        st.warning("Consultation record not found (may have been processed).")
                    else:
                        status_now = cur.iloc[0]['status']
                        if status_now != 'requested':
                            st.warning(f"Cannot accept. Current status: {status_now}")
                        else:
                            # assign this doctor and set accepted
                            scheduled_dt = datetime.now() + timedelta(minutes=30)
                            meeting_link = ""
                            # try Google calendar if available
                            creds = None
                            if GOOGLE_AVAILABLE:
                                creds = get_google_creds()
                            if creds:
                                try:
                                    created, link = create_calendar_event(creds, f"Consultation {consult_id}", scheduled_dt, duration_minutes=30, attendees=[student_email, USER_EMAIL])
                                    if link:
                                        meeting_link = link
                                except Exception:
                                    meeting_link = ""
                            if not meeting_link:
                                # fallback to jitsi
                                meeting_link = f"https://meet.jit.si/{hashlib.sha1(f'{consult_id}{time.time()}'.encode()).hexdigest()[:10]}"

                            # update record atomically by reloading, modifying, saving
                            latest.loc[latest['consult_id']==consult_id, 'doctor'] = USER_EMAIL
                            latest.loc[latest['consult_id']==consult_id, 'status'] = 'accepted'
                            latest.loc[latest['consult_id']==consult_id, 'scheduled_time'] = scheduled_dt.isoformat()
                            latest.loc[latest['consult_id']==consult_id, 'meeting_link'] = meeting_link
                            save_csv(latest, CONSULT_CSV)

                            # notify student
                            try:
                                subj = f"Your consultation request has been accepted"
                                body = (
                                    f"Hello {student_name},\n\n"
                                    f"Dr. {USER.get('name',USER_EMAIL)} has accepted your consultation.\n"
                                    f"Scheduled: {scheduled_dt.strftime('%Y-%m-%d %H:%M')}\n"
                                    f"Meeting link: {meeting_link}\n\n"
                                    "Please join at the scheduled time.\n\nRegards,\nStudent Stress Analyzer"
                                )
                                send_email_auto(student_email, subj, body)
                            except Exception:
                                pass
                            st.success("Accepted and student notified. Consultation locked to you.")
                if cols[1].button(f"Deny {consult_id}", key=f"deny_{consult_id}"):
                    # reload and check status
                    latest = load_csv(CONSULT_CSV)
                    cur = latest[latest['consult_id']==consult_id]
                    if cur.empty:
                        st.warning("Consultation record not found.")
                    else:
                        status_now = cur.iloc[0]['status']
                        if status_now != 'requested':
                            st.warning(f"Cannot deny. Current status: {status_now}")
                        else:
                            latest.loc[latest['consult_id']==consult_id, 'status'] = f"denied_by_{USER_EMAIL}"
                            save_csv(latest, CONSULT_CSV)
                            # notify student about denial
                            try:
                                subj = f"Your consultation request was denied"
                                body = (
                                    f"Hello {student_name},\n\n"
                                    f"Dr. {USER.get('name',USER_EMAIL)} has declined this consultation request.\n"
                                    f"The request is still open and other doctors may accept it.\n\nRegards,\nStudent Stress Analyzer"
                                )
                                send_email_auto(student_email, subj, body)
                            except Exception:
                                pass
                            st.info("You denied the request. It remains available to other doctors.")

                if cols[2].button(f"Open meeting link {consult_id}", key=f"open_{consult_id}"):
                    st.write(row.get('meeting_link','No link yet'))

        # show scheduled & historical
        st.markdown("### Scheduled consultations (assigned to you)")
        latest = load_csv(CONSULT_CSV)
        scheduled = latest[(latest['doctor']==USER_EMAIL) & (latest['status']=='accepted')]
        if scheduled.empty:
            st.info("No scheduled consultations assigned to you.")
        else:
            st.dataframe(scheduled[['consult_id','student','scheduled_time','meeting_link','status']].sort_values('scheduled_time', ascending=False))

        st.markdown("### Recent student records")
        progress = load_csv(PROGRESS_CSV)
        if progress.empty:
            st.info("No student records yet.")
        else:
            merged = progress.merge(users_df[['email','name']], left_on='email', right_on='email', how='left')
            display = merged[['name','email','score_pct','sleep_hours','bp_systolic','date']].sort_values(by='date', ascending=False)
            st.dataframe(display.head(200))
        return

    # other roles
    st.info("Consultation features are available to students and doctors.")

def page_chat():
    page_header("Messages — Share Notes & Companion logs")
    mdf=load_csv(MESSAGES_CSV)
    msg=st.text_area("Write a message (visible to you and doctors)")
    if st.button("Post message"):
        mid=hashlib.sha1(f"{USER_EMAIL}{datetime.now().isoformat()}{msg}".encode()).hexdigest()[:12]
        row={"msg_id":mid,"from_user":USER_EMAIL,"to_user":"all","date":datetime.now().isoformat(),"content":msg,"via_ollama":False}
        mdf=pd.concat([mdf,pd.DataFrame([row])], ignore_index=True)
        save_csv(mdf, MESSAGES_CSV)
        st.success("Posted")
    st.markdown("---")
    my_msgs=mdf[(mdf['from_user']==USER_EMAIL)|(mdf['to_user']==USER_EMAIL)|(mdf['to_user']=='all')] if not mdf.empty else pd.DataFrame()
    if my_msgs.empty:
        st.info("No messages yet.")
    else:
        for _,r in my_msgs.sort_values('date',ascending=False).head(50).iterrows():
            tag="🤖" if r.get('via_ollama') else "💬"
            st.markdown(f"**{tag} {r['from_user']}** — {r['date']}  \n> {r['content']}")

def page_ollama():
    page_header("Companion — Student Support (Ollama streaming)", "Students only — compassionate companion")
    if USER_ROLE.lower()!="student":
        st.info("Ollama companion available to students only.")
        return
    prompt=st.text_area("Share how you're feeling or ask for help", placeholder="I'm feeling anxious about exams...")
    if st.button("Send to Companion"):
        if not prompt.strip():
            st.warning("Write something first.")
            return
        msgs=load_csv(MESSAGES_CSV)
        mid=hashlib.sha1(f"{USER_EMAIL}{datetime.now().isoformat()}{prompt}".encode()).hexdigest()[:12]
        msgs=pd.concat([msgs,pd.DataFrame([{"msg_id":mid+"_u","from_user":USER_EMAIL,"to_user":"ollama","date":datetime.now().isoformat(),"content":prompt,"via_ollama":False}])], ignore_index=True)
        save_csv(msgs, MESSAGES_CSV)
        st.markdown("**Companion:**")
        box=st.empty()
        full=""
        with st.spinner("Companion is replying..."):
            for chunk in stream_ollama_companion(prompt):
                if isinstance(chunk, bytes):
                    chunk=chunk.decode(errors="ignore")
                full+=str(chunk)
                box.markdown(full)
                time.sleep(0.02)
        cb=load_csv(CHATBOT_CSV)
        bid=hashlib.sha1(f"{USER_EMAIL}{datetime.now().isoformat()}{full}".encode()).hexdigest()[:12]
        cb=pd.concat([cb,pd.DataFrame([{"id":bid,"email":USER_EMAIL,"role":USER_ROLE,"user_msg":prompt,"bot_reply":full,"timestamp":datetime.now().isoformat()}])], ignore_index=True)
        save_csv(cb, CHATBOT_CSV)
        msgs=load_csv(MESSAGES_CSV)
        msgs=pd.concat([msgs,pd.DataFrame([{"msg_id":bid+"_bot","from_user":"ollama","to_user":USER_EMAIL,"date":datetime.now().isoformat(),"content":full,"via_ollama":True}])], ignore_index=True)
        save_csv(msgs, MESSAGES_CSV)
        sentiment=sentiment_of_text(full)
        st.success("Companion reply saved.")
        gif=random.choice(HAPPY_GIFS if sentiment=="positive" else SAD_GIFS)
        st.image(gif)
        st.markdown(random.choice(["🌻 Keep going — small steps count.", "🌈 You deserve rest and care.", "💪 You're doing your best — that's enough for today."]))

def page_ml():
    page_header("ML Predictor — Train & Predict", "Train on synthetic dataset or load model")
    global MODEL
    if MODEL is None:
        st.warning("ML model not available.")
        if st.button("Generate synthetic dataset & train model now"):
            with st.spinner("Generating dataset and training..."):
                df=generate_synthetic_dataset(1200,SYNTH_CSV)
                clf,acc,rep=train_model_from_csv(SYNTH_CSV, MODEL_PKL)
                st.success(f"Trained model with accuracy {acc:.3f}")
                st.text(rep)
                MODEL=clf
        return
    st.markdown("Input features for a prediction")
    sleep=st.slider("Sleep hours",2.0,10.0,7.0)
    study=st.number_input("Study hours/day",0.0,16.0,4.0)
    bp_sys=st.number_input("BP systolic",90,160,120)
    bp_dia=st.number_input("BP diastolic",60,100,80)
    hr=st.number_input("Heart rate",50,120,75)
    mood=st.slider("Mood (1-10)",1,10,6)
    gpa=st.number_input("GPA",0.0,4.0,3.0)
    procrast=st.slider("Procrastination (0-10)",0,10,4)
    support=st.slider("Support level (0-10)",0,10,6)
    anxiety=st.slider("Academic anxiety (0-10)",0,10,5)
    if st.button("Predict"):
        X=np.array([[sleep,study,bp_sys,bp_dia,hr,mood,gpa,procrast,support,anxiety]])
        pred=MODEL.predict(X)[0]
        mapping={0:"Low",1:"Moderate",2:"High",3:"Very High"}
        label=mapping.get(pred,"Unknown")
        st.success(f"Predicted stress level: {label}")
        pool=REMEDY_POOL.get(label,[])
        sel=random.sample(pool,min(len(pool),12)) if pool else []
        st.markdown("### Suggested remedies")
        for r in sel:
            st.markdown(f"- {r}")
        for v in VIDEOS.get(label,[]):
            st.video(v)
        if st.button("Open Companion"):
            st.session_state['page']='ollama'
            st.rerun()

def page_admin():
    if USER_ROLE.lower()!='admin':
        st.warning("Admin page restricted.")
        return
    page_header("Admin — Data & Model")
    users=load_csv(USERS_CSV)
    progress=load_csv(PROGRESS_CSV)
    consult=load_csv(CONSULT_CSV)
    chatbot=load_csv(CHATBOT_CSV)
    st.subheader("Registered Users")
    st.dataframe(users)
    st.subheader("Progress Records")
    st.dataframe(progress)
    st.subheader("Consultations")
    st.dataframe(consult)
    st.subheader("Chatbot Logs")
    st.dataframe(chatbot)
    col1,col2=st.columns(2)
    with col1:
        if st.button("Reset all CSVs (careful)"):
            ensure_csv(USERS_CSV, ["name","email","password_hash","role","assigned_patients"])
            ensure_csv(PROGRESS_CSV, ["email","date","score_pct","sleep_hours","bp_systolic","bp_diastolic","notes"])
            ensure_csv(CONSULT_CSV, ["consult_id","student","doctor","date_requested","preferred_time","scheduled_time","meeting_link","status","notes"])
            ensure_csv(MESSAGES_CSV, ["msg_id","from_user","to_user","date","content","via_ollama"])
            ensure_csv(CHATBOT_CSV, ["id","email","role","user_msg","bot_reply","timestamp"])
            st.warning("CSV files reset. Restart app.")
    with col2:
        if st.button("Generate synthetic dataset & train model"):
            with st.spinner("Training..."):
                generate_synthetic_dataset(1200,SYNTH_CSV)
                clf,acc,rep=train_model_from_csv(SYNTH_CSV, MODEL_PKL)
                st.success(f"Model trained, accuracy {acc:.3f}")

# ----------------------------
# Main routing
# ----------------------------
page=st.session_state.get('page','dashboard')

if page=='dashboard':
    if USER_ROLE.lower() == 'doctor':
        st.markdown(f"<div class='card'><h3>Doctor Dashboard</h3><div class='small-muted'>Welcome, Dr. {USER['name']}. Notifications and student data are below.</div></div>", unsafe_allow_html=True)
        consult_df = load_csv(CONSULT_CSV)
        pending = consult_df[(consult_df['status']=='requested')]
        st.info(f"You have {len(pending)} pending consultation request(s). Go to Consultations to accept or deny.")
        page_consultations()
    else:
        page_dashboard()
elif page=='assessment':
    page_assessment()
elif page=='progress':
    page_progress()
elif page=='consultations':
    page_consultations()
elif page=='chat':
    page_chat()
elif page=='ollama':
    page_ollama()
elif page=='ml':
    page_ml()
elif page=='admin':
    page_admin()
else:
    page_dashboard()

st.markdown("---")
st.caption("Demo: local CSV storage. For production, use secure authentication, encrypted DB, and server-side LLM/Google integrations.")

