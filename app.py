import streamlit as st
from datetime import datetime, timedelta
from typing import List
import os

try:
    from groq import Groq
except ImportError:
    st.warning("Groq module not installed, chat will be disabled.")
    Groq = None
from dotenv import load_dotenv

# Load the .env file immediately
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# -------------------------
# Mock Classes for Standalone Run
# -------------------------
class Lesson:
    def __init__(self, title: str, duration_minutes: int = 30):
        self.title = title
        self.duration_minutes = duration_minutes
        self.start_time = None
        self.end_time = None

class Scheduler:
    def __init__(self, start_hour=9, end_hour=17):
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

# -------------------------
# Helpers
# -------------------------
def make_lesson(obj):
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
# App Config
# -------------------------
st.set_page_config(page_title="Lesson Agent + Llama Chat", layout="wide")
st.title("🤖 Lesson Agent Dashboard + Llama 4 Chat")

# -------------------------
# Session State
# -------------------------
for key in ["raw_lessons", "lessons", "schedule", "messages"]:
    if key not in st.session_state:
        st.session_state[key] = []

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.image("https://www.cais.usc.edu/wp-content/uploads/2022/10/Meta-logo-e1667246992948.png", width=140)
    st.header("⚙️ Control Panel")

    #api_key = st.text_input("Groq API Key (optional)", type="password")

    model_option = st.radio("Choose Model:", ["Llama 4 Scout", "Llama 4 Maverick"], index=1)
    selected_model = "meta-llama/llama-4-scout-17b-16e-instruct" if "Scout" in model_option else "meta-llama/llama-4-maverick-17b-128e-instruct"

    memory_enabled = st.toggle("Enable Chat Memory", True)
    system_prompt = st.text_area("System Persona", value="You are a helpful assistant for teachers.", height=120)

    st.divider()
    start_hour = st.slider("Start Hour", 0, 23, 9)
    end_hour = st.slider("End Hour", 0, 23, 17)

    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()

    st.success("LessonPlanner Connected ✅" if PLANNER_AVAILABLE else "Planner Mock Mode ⚠️")

# -------------------------
# Main Layout
# -------------------------
chat_col, main_col = st.columns([1, 2])

# -------------------------
# Chatbot (Left)
# -------------------------
with chat_col:
    st.subheader("💬 Llama Chat")
    if api_key and Groq:
        client = Groq(api_key=api_key)
    elif not api_key or not Groq:
        st.warning("Groq chat is disabled.")
        client = None

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if client and (prompt := st.chat_input("Ask something...")):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        messages_payload = [{"role": "system", "content": system_prompt}]
        if memory_enabled:
            messages_payload.extend(st.session_state["messages"])
        else:
            messages_payload.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            output = ""
            try:
                stream = client.chat.completions.create(
                    model=selected_model,
                    messages=messages_payload,
                    temperature=0.7,
                    max_tokens=800,
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

# -------------------------
# Lesson Planner + Scheduler (Right)
# -------------------------
with main_col:
    tab1, tab2 = st.tabs(["📘 Lesson Planner", "🗓 Scheduler"])

    # Lesson Planner
    with tab1:
        st.subheader("Create Lesson Plan")
        topic = st.text_input("Topic", placeholder="e.g. Python Lists")
        grade = st.text_input("Grade", placeholder="e.g. Primary 5")
        subject = st.text_input("Subject", placeholder="e.g. Computer Science")

        if st.button("Generate Plan", type="primary"):
            if not topic or not grade or not subject:
                st.warning("Fill Topic, Grade and Subject.")
            else:
                with st.spinner("Generating..."):
                    try:
                        planner = LessonPlanner()
                        raw = planner.generate_plan(topic, grade, subject)
                        st.session_state["lessons"] = [make_lesson(x) for x in raw]
                        st.success(f"Generated {len(st.session_state['lessons'])} lessons")
                    except Exception as e:
                        st.error(f"Planner Error: {e}")
                        st.session_state["lessons"] = []

        if st.session_state["lessons"]:
            st.write("### Lessons")
            for i, l in enumerate(st.session_state["lessons"], 1):
                st.info(f"{i}. {l.title} — ⏱ {l.duration_minutes} mins")

    # Scheduler
    with tab2:
        st.subheader("Schedule Lessons")
        if st.session_state["lessons"]:
            start_date = st.date_input("Start Date", datetime.now())
            start_time = st.time_input("Start Time", datetime.now().time())

            if st.button("Create Schedule"):
                try:
                    dt = datetime.combine(start_date, start_time)
                    scheduler = Scheduler(start_hour=start_hour, end_hour=end_hour)
                    clean_lessons = [make_lesson(x) for x in st.session_state["lessons"]]
                    st.session_state["schedule"] = scheduler.create_schedule(clean_lessons, dt)
                    st.success("Schedule created")
                except Exception as e:
                    st.error(f"Scheduling error: {e}")

        if st.session_state["schedule"]:
            st.write("### Itinerary")
            for s in st.session_state["schedule"]:
                st.markdown(f"""
                <div style='padding:8px;border-left:5px solid #4CAF50;background:#f5f5f5;margin: 5px 0;color:black'>
                    <b>{s.start_time.strftime('%H:%M')} - {s.end_time.strftime('%H:%M')}</b><br>
                    {s.title}
                </div>
                """, unsafe_allow_html=True)
