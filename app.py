
# app.py (updated to use config.py values; preserves all features)
import streamlit as st
from datetime import datetime, timedelta
from typing import List, Optional
import os
import io
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import openpyxl
from services.serp_search import search_academic_resources

# Optional Groq client (for richer feedback). If not installed or API key missing, fallback to local feedback.
try:
    from groq import Groq
except Exception:
    Groq = None

# Import app configuration (load env in config.py)
from config import (
    GROQ_API_KEY,
    SERP_API_KEY,
    SENDER_EMAIL,
    EMAIL_PASSWORD,
    APP_TITLE,
    PAGE_LAYOUT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_START_HOUR,
    DEFAULT_END_HOUR,
    DEFAULT_MODEL_SCOUT,
    DEFAULT_MODEL_MAVERICK,
)

from email.mime.base import MIMEBase
from email import encoders
import tempfile

# -------------------------
# Mock Classes for Standalone Run (preserve existing features)
# -------------------------
class Lesson:
    def __init__(self, title: str, duration_minutes: int = 30):
        self.title = title
        self.duration_minutes = duration_minutes
        self.start_time = None
        self.end_time = None
    
    def __repr__(self):
        return self.title


class Scheduler:
    def __init__(self, start_hour=DEFAULT_START_HOUR, end_hour=DEFAULT_END_HOUR):
        self.start_hour = start_hour
        self.end_hour = end_hour

    def create_schedule(self, lessons: List[Lesson], start_datetime: datetime):
        schedule = []
        current_time = start_datetime
        for lesson in lessons:
            end_time = current_time + timedelta(minutes=lesson.duration_minutes)
            lesson_copy = Lesson(lesson.title, lesson.duration_minutes)
            lesson_copy.start_time = current_time
            lesson_copy.end_time = end_time
            schedule.append(lesson_copy)
            current_time = end_time
        return schedule

class LessonPlanner:
    def generate_plan(self, prompt: str, grade: str, subject: str):
        return [
            {"title": f"Intro to {prompt}", "duration_minutes": 30},
            {"title": f"{prompt} Core Concepts", "duration_minutes": 45},
            {"title": f"{prompt} Practice", "duration_minutes": 60},
        ]

PLANNER_AVAILABLE = True

def send_grade_report_email(receiver_email: str, df):
    """
    Send Excel grade report to teacher (uses SENDER_EMAIL and EMAIL_PASSWORD from config).
    """
    try:
        # use config values
        if not SENDER_EMAIL or not EMAIL_PASSWORD:
            st.error("Email credentials not configured in config/.env.")
            return

        # Create Excel file in temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            excel_path = tmp.name
            df.to_excel(excel_path, index=False)

        # Build email
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = receiver_email
        msg["Subject"] = "Student Grade Report (Excel)"

        body = "Please find attached the student grade report."
        msg.attach(MIMEText(body, "plain"))

        # Attach file
        with open(excel_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="grade_report.xlsx"')
        msg.attach(part)

        # Send email
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        st.success("✅ Grade report sent successfully as Excel!")

    except Exception as e:
        st.error(f"Failed to send email: {e}")

def make_lesson(obj):
    """
    Accept Lesson instance, dict or raw and produce Lesson.
    """
    if isinstance(obj, Lesson):
        return obj
    if isinstance(obj, dict):
        title = obj.get("title", "Untitled Lesson")
        duration = obj.get("duration_minutes", 30)
    else:
        title = str(obj)
        duration = 30
    try:
        duration = int(duration)
    except:
        duration = 30
    return Lesson(title=title, duration_minutes=duration)

# -------------------------
# App config (use constants from config.py)
# -------------------------
st.set_page_config(page_title=APP_TITLE, layout=PAGE_LAYOUT)
st.title(APP_TITLE)

# -------------------------
# Session state init (preserve storage)
# -------------------------
for key in ["raw_lessons", "lessons", "schedule", "messages", "grade_df", "grade_report"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------
# Sidebar (preserve & improve)
# -------------------------
with st.sidebar:
    st.image("https://www.cais.usc.edu/wp-content/uploads/2022/10/Meta-logo-e1667246992948.png", width=140)
    st.header("⚙️ Control Panel")

    st.subheader("🧭 Navigation")
    # Navigation checkboxes — features hidden until toggled
    show_search = st.checkbox("🔍 Academic Search", value=False)
    show_exam = st.checkbox("📥 Exam & Grades Manager", value=False)
    show_scheduler = st.checkbox("🗓 Scheduler", value=False)
    show_analytics = st.checkbox("📊 Performance Analytics", value=False)

    model_option = st.radio("Choose Model:", ["Llama 4 Scout", "Llama 4 Maverick"], index=1)
    selected_model = DEFAULT_MODEL_SCOUT if "Scout" in model_option else DEFAULT_MODEL_MAVERICK

    memory_enabled = st.checkbox("Enable Chat Memory", value=True)
    system_prompt = st.text_area("System Persona", value=DEFAULT_SYSTEM_PROMPT, height=100)
    st.markdown("---")

    st.subheader("Scheduler Window")
    # use defaults from config for the slider defaults
    start_hour = st.slider("Start Hour", 0, 23, DEFAULT_START_HOUR)
    end_hour = st.slider("End Hour", 0, 23, DEFAULT_END_HOUR)

    if st.button("Clear Chat & Grades"):
        st.session_state["messages"] = []
        st.session_state["grade_df"] = None
        st.session_state["grade_report"] = None
        st.success("Cleared chat history and grade data.")

# -------------------------
# Layout: left = chat & grades, right = planner + scheduler
# -------------------------
left_col, right_col = st.columns([1.2, 1.8])

# -------------------------
# Chat / Llama (left)
# -------------------------
with left_col:
    st.subheader("💬 Llama Chat")
    client = None
    if GROQ_API_KEY and Groq:
        try:
            client = Groq(api_key=GROQ_API_KEY)
        except Exception as e:
            st.warning(f"Groq init failed: {e}")
            client = None
    else:
        st.info("Groq not configured — chat will use local fallback responses.")

    if "messages" not in st.session_state or st.session_state["messages"] is None:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        role = msg.get("role", "user")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))

    if client:
        prompt = st.chat_input("Ask a question to the assistant...")
        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                placeholder = st.empty()
                output = ""
                try:
                    messages_payload = [{"role": "system", "content": system_prompt}]
                    if memory_enabled:
                        messages_payload.extend(st.session_state["messages"][-10:])
                    else:
                        messages_payload.append({"role": "user", "content": prompt})
                    stream = client.chat.completions.create(
                        model=selected_model,
                        messages=messages_payload,
                        temperature=0.7,
                        max_tokens=600,
                        stream=True
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            output += chunk.choices[0].delta.content
                            placeholder.markdown(output + "▌")
                    placeholder.markdown(output)
                    st.session_state["messages"].append({"role": "assistant", "content": output})
                except Exception as e:
                    st.error(f"Chat error: {e}")
    else:
        prompt = st.chat_input("Ask (local fallback)...")
        if prompt:
            reply = f"I can't call Groq here. Local assistant echo: {prompt}"
            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.session_state["messages"].append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

    st.markdown("---")

    # -------------------------
    # Academic Search (hidden until toggled)
    # -------------------------

if show_search:
    st.markdown("## 🔍 Search Academic Materials")

    query = st.text_input(
        "Enter topic or subject (e.g. Primary Math Fractions):",
        key="search_query"
    )

    if st.button("Search Resources", key="search_btn"):
        if not query:
            st.warning("Please enter a search topic.")
        else:
            with st.spinner("Searching for relevant materials..."):
                results = search_academic_resources(query)

                if isinstance(results, dict) and results.get("error"):
                    st.error(results["error"])
                elif len(results) == 0:
                    st.info("No resources found.")
                else:
                    for res in results:
                        st.markdown(f"### {res['title']}")
                        st.write(res["snippet"])
                        st.markdown(f"[Open Resource]({res['link']})")
                        st.markdown("---")




    # if show_search:
    #     st.markdown("## 🔍 Search Academic Materials")
    #     query = st.text_input("Enter topic or subject (e.g. Primary Math Fractions):")

    #     if st.button("Search Resources"):
    #         if not query:
    #             st.warning("Please enter a search topic.")
    #         else:
    #             with st.spinner("Searching for relevant materials..."):
    #                 results = search_academic_resources(query)

    #                 if isinstance(results, dict) and results.get("error"):
    #                     st.error(results["error"])
    #                 elif len(results) == 0:
    #                     st.info("No resources found.")
    #                 else:
    #                     for res in results:
    #                         st.markdown(f"### {res['title']}")
    #                         st.write(res["snippet"])
    #                         st.markdown(f"[Open Resource]({res['link']})")
    #                         st.markdown("---")

    # -------------------------
    # Exam & Grades Manager (hidden until toggled)
    # -------------------------
    if show_exam:
        st.subheader("📥 Exam & Grades Manager")

        st.markdown("**Upload student scores (CSV) or enter manually.**")
        st.markdown("CSV columns: student_id, student_name, score (numeric). You may include multiple courses by uploading multiple files or editing.")
        uploaded = st.file_uploader("Upload CSV file", type=["csv"], accept_multiple_files=False)

        if uploaded:
            try:
                _df = pd.read_csv(uploaded)
                required = {"student_id", "student_name", "score"}
                if not required.issubset(set(_df.columns)):
                    st.error(f"CSV must contain columns: {required}. Found: {list(_df.columns)}")
                else:
                    _df['score'] = pd.to_numeric(_df['score'], errors='coerce').fillna(0)
                    st.session_state["grade_df"] = _df[['student_id','student_name','score']].copy()
                    st.success(f"Loaded {len(_df)} students.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

        st.markdown("Or add/edit students below (use the table editor).")
        if st.session_state.get("grade_df") is None:
            seed_df = pd.DataFrame([
                {"student_id": "S001", "student_name": "Alice", "score": 78},
                {"student_id": "S002", "student_name": "Bob", "score": 65},
                {"student_id": "S003", "student_name": "Charlie", "score": 52},
            ])
            st.session_state["grade_df"] = seed_df

        edited = st.data_editor(st.session_state["grade_df"], num_rows="dynamic", use_container_width=True)
        st.session_state["grade_df"] = edited

        st.markdown("### Compute Grades & Positions")
        col1, col2 = st.columns([1,1])
        with col1:
            grade_btn = st.button("Compute Grades")
        with col2:
            export_btn = st.button("Download Grade Report (CSV)")

        if grade_btn:
            df_calc = st.session_state["grade_df"].copy()
            df_calc['score'] = pd.to_numeric(df_calc['score'], errors='coerce').fillna(0)
            df_calc['percent'] = df_calc['score'].clip(0, 100)
            def assign_letter(p: float) -> str:
                if p >= 70: return 'A'
                if p >= 60: return 'B'
                if p >= 50: return 'C'
                if p >= 45: return 'D'
                return 'F'
            df_calc['grade'] = df_calc['percent'].apply(assign_letter)
            df_calc['position'] = df_calc['percent'].rank(method='dense', ascending=False).astype(int)
            df_calc = df_calc.sort_values(by=['position','student_name'])
            st.session_state['grade_df'] = df_calc.reset_index(drop=True)
            st.session_state['grade_report'] = df_calc
            st.success("Grades computed and positions assigned.")

        if export_btn:
            if st.session_state.get('grade_report') is None:
                st.warning("No grade report yet — compute grades first.")
            else:
                csv = st.session_state['grade_report'].to_csv(index=False)
                st.download_button("Download CSV", data=csv, file_name="grade_report.csv", mime="text/csv")

        # Feedback generation
        st.markdown("### Generate Feedback for Students")
        st.markdown("### 📧 Email Grade Report")
        teacher_email = st.text_input("Teacher's Email Address")

        if st.button("Send Excel Report via Email"):
            if st.session_state.get("grade_report") is None:
                st.warning("Please compute grades before sending the email.")
            elif not teacher_email:
                st.warning("Please enter the teacher's email.")
            else:
                send_grade_report_email(teacher_email, st.session_state["grade_report"])

        fb_col1, fb_col2 = st.columns([2,1])
        with fb_col1:
            fb_scope = st.selectbox("Feedback mode", ["All students", "Single student"], index=0)
        with fb_col2:
            fb_btn = st.button("Generate Feedback")

        def local_feedback(row) -> str:
            p = row['percent']
            grade = row['grade']
            tips = []
            if p >= 70:
                tone = "Excellent work — keep it up!"
                tips.append("Continue practicing advanced problems and extend with project tasks.")
            elif p >= 60:
                tone = "Good performance — you're on the right track."
                tips.append("Focus on strengthening weaker subtopics and timed practice.")
            elif p >= 50:
                tone = "Fair — there's room to improve."
                tips.append("Revise fundamental concepts and take short quizzes frequently.")
            elif p >= 45:
                tone = "Below average — needs attention."
                tips.append("Target key problem areas and seek remedial sessions.")
            else:
                tone = "Poor performance — immediate intervention recommended."
                tips.append("Start with fundamentals, use guided practice, and attend extra coaching.")
            rec = "Recommended resources: review class notes, sample exercises, and short video tutorials."
            return f"{tone}\nGrade: {grade}\nScore: {p}%\nSuggestions: {' '.join(tips)}\n{rec}"

        def groq_feedback_for_student(student_name: str, score: float, percent: float, grade_letter: str, subject: Optional[str]=None):
            if not Groq or not GROQ_API_KEY:
                return None
            try:
                client = Groq(api_key=GROQ_API_KEY)
                prompt = f"""You are an experienced primary/secondary teacher coach. Produce a concise, constructive feedback paragraph for student named {student_name}.
Subject: {subject or 'General'}.
Score: {score}. Percent: {percent}%. Grade: {grade_letter}.
Give: 1-sentence summary, 2 short targeted suggestions, 1 recommended resource (title + short reason). Keep it <= 80 words."""
                resp = client.chat.completions.create(
                    model=DEFAULT_MODEL_MAVERICK,
                    messages=[{"role":"system","content":"You are a helpful teacher assistant."},{"role":"user","content":prompt}],
                    max_tokens=200,
                    temperature=0.2
                )
                reply = resp.choices[0].message.content
                return reply
            except Exception as e:
                st.warning(f"Groq feedback failed: {e}")
                return None

        if fb_btn:
            if st.session_state.get('grade_report') is None:
                st.warning("Compute grades before generating feedback.")
            else:
                df_fb = st.session_state['grade_report'].copy()
                feedbacks = []
                with st.spinner("Generating feedback..."):
                    for idx, row in df_fb.iterrows():
                        groq_fb = None
                        if Groq and GROQ_API_KEY:
                            groq_fb = groq_feedback_for_student(row['student_name'], row['score'], row['percent'], row['grade'], subject=None)
                        fb = groq_fb if groq_fb else local_feedback(row)
                        feedbacks.append(fb)
                df_fb['feedback'] = feedbacks
                st.session_state['grade_report'] = df_fb
                st.success("Feedback generated and saved to grade report.")

        # display table and allow per-row feedback review
        if st.session_state.get('grade_report') is not None:
            st.markdown("### Grade Report Preview")
            st.dataframe(st.session_state['grade_report'], use_container_width=True)

# -------------------------
# Right column: Lesson Planner + Scheduler + Analytics
# -------------------------
with right_col:
    st.subheader("📘 Lesson Planner")
    topic = st.text_input("Topic (Planner)", placeholder="e.g. Python Lists")
    grade_label = st.text_input("Grade (Planner)", placeholder="e.g. Primary 5")
    subject_label = st.text_input("Subject (Planner)", placeholder="e.g. Computer Science")
    if st.button("Generate Lesson Plan (Planner)"):
        if not topic or not grade_label or not subject_label:
            st.warning("Fill Topic, Grade and Subject for planner.")
        else:
            with st.spinner("Generating lesson plan..."):
                try:
                    planner = LessonPlanner()
                    raw = planner.generate_plan(topic, grade_label, subject_label)
                    lessons = [make_lesson(x) for x in raw]
                    st.session_state['lessons'] = lessons
                    st.success(f"Generated {len(lessons)} lessons")
                except Exception as e:
                    st.error(f"Planner Error: {e}")
                    st.session_state['lessons'] = []

    if st.session_state.get('lessons'):
        st.write("### Proposed Lessons")
        for i, l in enumerate(st.session_state['lessons'], 1):
            st.info(f"{i}. {l.title} — ⏱ {l.duration_minutes} mins")

    st.markdown("---")

    
    if show_scheduler:

        st.subheader("🗓 Scheduler")
        if st.session_state.get('lessons'):
            start_date = st.date_input("Start Date", datetime.now())
            start_time = st.time_input("Start Time", datetime.now().time())
            if st.button("Create Schedule"):
                try:
                    dt = datetime.combine(start_date, start_time)
                    scheduler = Scheduler(start_hour=start_hour, end_hour=end_hour)
                    clean_lessons = [make_lesson(x) for x in st.session_state['lessons']]
                    st.session_state['schedule'] = scheduler.create_schedule(clean_lessons, dt)
                    st.success("Schedule created")
                except Exception as e:
                    st.error(f"Scheduling error: {e}")

        if st.session_state.get('schedule'):
            st.write("### 📅 Itinerary")

            for s in st.session_state['schedule']:
                try:
                    # Try common field names safely
                    lesson_title = ""

                    if hasattr(s, "lesson"):
                        lesson_title = str(s.lesson)
                    elif hasattr(s, "title"):
                        lesson_title = str(s.title)
                    elif hasattr(s, "name"):
                        lesson_title = str(s.name)

                    st.markdown(f"""
                    <div style='padding:10px;border-left:5px solid #4CAF50;background:#f9f9f9;margin:6px 0;color:black'>
                        <b>{s.start_time.strftime('%b %d, %H:%M')} - {s.end_time.strftime('%H:%M')}</b><br>
                        📘 {lesson_title}
                    </div>
                    """, unsafe_allow_html=True)

                except Exception:
                    pass




    # -------------------------
    # Analytics: bar chart + summary
    # -------------------------
    st.markdown("---")

    # Ensure df is defined even if analytics is hidden
    df = None

    if show_analytics:
        st.subheader("📊 Performance Analytics")
        if st.session_state.get('grade_report') is None:
            st.info("Compute grades to view analytics.")
        else:
            df = st.session_state['grade_report'].copy()
            # summary metrics
            avg = df['percent'].mean()
            median = df['percent'].median()
            highest = df['percent'].max()
            lowest = df['percent'].min()
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Class Average", f"{avg:.1f}%")
            colB.metric("Median", f"{median:.1f}%")
            colC.metric("Highest", f"{highest:.1f}%")
            colD.metric("Lowest", f"{lowest:.1f}%")

        # -------------------------
        # Send Report to Teacher Email
        # -------------------------
        st.markdown("### 📧 Send Report to Teacher")
        teacher_email = st.text_input("Enter teacher's email address")

        if st.button("Send Report to Email"):
            if not teacher_email:
                st.warning("Please enter an email.")
            elif df is None:
                st.warning("Generate the report first.")
            else:
                send_grade_report_email(teacher_email, df)

        # Charts (only if df exists)
        if df is not None:
            st.markdown("#### Score Distribution")
            chart_df = df[['student_name','percent']].set_index('student_name')
            st.bar_chart(chart_df)

            st.markdown("#### Grade Distribution")
            grade_counts = df['grade'].value_counts().rename_axis('grade').reset_index(name='count')
            st.dataframe(grade_counts)

            # Show strengths/weaknesses inference (simple)
            st.markdown("#### Quick Insights")
            low_perf = df[df['percent'] < 50]
            high_perf = df[df['percent'] >= 70]
            st.write(f"Students needing remediation: {len(low_perf)}")
            st.write(f"Students excelling: {len(high_perf)}")
            if len(low_perf) > 0:
                with st.expander("List of students needing remediation"):
                    for _, r in low_perf.iterrows():
                        st.markdown(f"- {r['student_name']} — {r['percent']}% — Grade {r['grade']}")

# -------------------------
# Footer / Help
# -------------------------
st.markdown("---")
st.caption("Lesson Agent + Exam Manager — lesson planning, scheduling, grading, analytics, and feedback generation.")

