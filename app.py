"""
DIP Diet RAG Agent — Streamlit Dashboard
Powered by IBM Granite (watsonx.ai) + ChromaDB Vector Store
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Load .env if present (fallback to .env.example)
try:
    from dotenv import load_dotenv
    if (BASE_DIR / ".env").exists():
        load_dotenv(BASE_DIR / ".env")
    elif (BASE_DIR / ".env.example").exists():
        load_dotenv(BASE_DIR / ".env.example")
except ImportError:
    pass

from rag_agent import DIPDietRAGAgent, UserProfile, generate_meal_plan, get_nutrition_profile
from ingest import ingest as run_ingest

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DIP Diet Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    /* ── Global ── */
    body { font-family: 'Segoe UI', sans-serif; }
    .main { background: #f0f7f0; }

    /* ── Header ── */
    .dip-header {
        background: linear-gradient(135deg, #1b5e20 0%, #388e3c 50%, #66bb6a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(27,94,32,0.3);
    }
    .dip-header h1 { font-size: 2.2rem; margin: 0; letter-spacing: 1px; }
    .dip-header p  { margin: 0.3rem 0 0; font-size: 1rem; opacity: 0.9; }

    /* ── Cards ── */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #388e3c;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 0.8rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #555; text-transform: uppercase; }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #1b5e20; }
    .metric-card .sub   { font-size: 0.8rem; color: #888; }

    /* ── Chat ── */
    .chat-user {
        background: #e8f5e9;
        border-radius: 12px 12px 2px 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        float: right;
        clear: both;
        color: #1b3a1e;
    }
    .chat-bot {
        background: white;
        border-radius: 12px 12px 12px 2px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        max-width: 85%;
        float: left;
        clear: both;
        border-left: 3px solid #388e3c;
        color: #222;
    }
    .chat-container { min-height: 300px; }

    /* ── Status badge ── */
    .badge-ok   { background:#c8e6c9; color:#1b5e20; padding:2px 10px; border-radius:20px; font-size:0.8rem; }
    .badge-warn { background:#fff3e0; color:#e65100; padding:2px 10px; border-radius:20px; font-size:0.8rem; }
    .badge-err  { background:#ffebee; color:#b71c1c; padding:2px 10px; border-radius:20px; font-size:0.8rem; }

    /* ── Meal plan table ── */
    .meal-row { background:white; border-radius:8px; padding:0.6rem 1rem; margin:0.4rem 0; border-left:3px solid #66bb6a; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background: #1b5e20 !important; }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stMultiSelect label { color: #c8e6c9 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: #388e3c; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="dip-header">
        <h1>🥗 DIP Diet RAG Assistant</h1>
        <p>Disciplined &amp; Intelligent Plan · Powered by IBM Granite + watsonx.ai</p>
    </div>
    """, unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "agent_status" not in st.session_state:
    st.session_state.agent_status = None
if "profile_saved" not in st.session_state:
    st.session_state.profile_saved = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Your Health Profile")
    st.markdown("---")

    name   = st.text_input("Name",           value="Health Seeker")
    age    = st.slider("Age",                10, 100, 35)
    weight = st.slider("Weight (kg)",        30, 200, 70)
    gender = st.selectbox("Gender",          ["Male", "Female", "Other"])
    height = st.slider("Height (cm)",        100, 220, 165)
    activity = st.selectbox("Activity Level", [
        "Sedentary", "Light", "Moderate", "Active", "Very Active"
    ], index=2)
    goals = st.text_input("Health Goals",    value="General wellness")
    conditions = st.text_input("Medical Conditions", value="None")
    restrictions = st.text_input("Dietary Restrictions", value="None")

    if st.button("💾 Save Profile"):
        st.session_state.profile_saved = True
        st.success("Profile saved!")

    # BMI widget
    user_profile = UserProfile(
        name=name, age=age, gender=gender,
        weight_kg=weight, height_cm=height,
        activity_level=activity, health_goals=goals,
        medical_conditions=conditions, dietary_restrictions=restrictions,
    )
    bmi_color = {"Underweight": "#66bb6a", "Normal weight": "#66bb6a",
                 "Overweight": "#ff9800", "Obese": "#f44336"}.get(user_profile.bmi_category, "#66bb6a")
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.15); border-radius:8px; padding:0.6rem; "
        f"text-align:center; margin-top:0.8rem;'>"
        f"<b>BMI:</b> {user_profile.bmi} &nbsp;|&nbsp; <b>{user_profile.bmi_category}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ System")

    if st.button("🔄 Build / Rebuild Vector Store"):
        with st.spinner("Ingesting DIP diet documents into vector store..."):
            try:
                run_ingest()
                st.session_state.ingestion_done = True
                st.session_state.agent = None  # force re-init
                st.success("✅ Vector store built!")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []

# ── Agent initialization ──────────────────────────────────────────────────────
if st.session_state.agent is None:
    with st.spinner("🔧 Initializing DIP Diet AI Agent..."):
        try:
            agent = DIPDietRAGAgent(top_k=5)
            status = agent.initialize()
            st.session_state.agent = agent
            st.session_state.agent_status = status
        except Exception as e:
            st.session_state.agent_status = {"errors": [str(e)]}
            st.error(f"Agent initialization error: {e}")

# ── Status bar ────────────────────────────────────────────────────────────────
agent   = st.session_state.agent
a_status = st.session_state.agent_status or {}

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    retriever_ok = a_status.get("retriever") is not None
    badge = "badge-ok" if retriever_ok else "badge-err"
    label = "✅ Vector Store" if retriever_ok else "❌ Vector Store"
    st.markdown(f'<span class="{badge}">{label}</span>', unsafe_allow_html=True)
with col_s2:
    llm_ok = a_status.get("llm") is not None
    badge = "badge-ok" if llm_ok else "badge-warn"
    label = "✅ IBM Granite LLM" if llm_ok else "⚠️ LLM (Fallback mode)"
    st.markdown(f'<span class="{badge}">{label}</span>', unsafe_allow_html=True)
with col_s3:
    chunks = a_status.get("retriever_chunks", 0)
    st.markdown(f'<span class="badge-ok">📚 {chunks} chunks retrieved</span>', unsafe_allow_html=True)

if a_status.get("errors"):
    for err in a_status["errors"]:
        st.warning(f"⚠️ {err}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_plan, tab_nutrition, tab_lifestyle = st.tabs([
    "💬 AI Chat", "📅 Meal Plan", "📊 Nutrition", "🌿 Lifestyle"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – AI Chat
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    # Quick questions
    st.markdown("**Quick Questions:**")
    quick_qs = [
        "What should I eat for breakfast on DIP Diet?",
        "How does DIP Diet help with diabetes?",
        "What are the core principles of DIP Diet?",
        "Can I eat rice on DIP Diet?",
        "What is the DIP Diet meal timing?",
    ]
    q_cols = st.columns(len(quick_qs))
    for i, q in enumerate(quick_qs):
        if q_cols[i].button(q[:30] + "…", key=f"quick_{i}"):
            st.session_state.messages.append({"role": "user", "content": q})

    # Chat history
    chat_html = '<div class="chat-container">'
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f'<div class="chat-user">🧑 <b>You:</b> {msg["content"]}</div>'
        else:
            chat_html += f'<div class="chat-bot">🥗 <b>DIP Assistant:</b><br>{msg["content"]}</div>'
    chat_html += '<div style="clear:both;"></div></div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask your DIP Diet question:",
            placeholder="e.g. What should I eat for dinner if I have high blood pressure?",
            key="chat_input",
        )
        submitted = st.form_submit_button("🚀 Ask")

    if submitted and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        if agent:
            with st.spinner("🔍 Searching DIP knowledge base & generating answer..."):
                result = agent.query(user_input, user_profile=user_profile)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            error   = result.get("error")

            if sources and "Sources:" not in answer:
                answer += f"\n\n*Sources: {', '.join(sources)}*"
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⚠️ Agent not initialized. Please click 'Build Vector Store' in the sidebar first.",
            })
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Meal Plan
# ═══════════════════════════════════════════════════════════════════════════════
with tab_plan:
    st.markdown(
        f"### 📅 Your Personalized DIP Diet Weekly Meal Plan\n"
        f"Customized for: **{name}** | Goals: {goals}"
    )

    meal_plan = generate_meal_plan(user_profile)
    days = list(meal_plan.keys())
    selected_day = st.selectbox("Select Day", days)

    day_meals = meal_plan[selected_day]

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### 🌅 Breakfast (7–9 AM)")
        st.markdown(f'<div class="meal-row">🍌 <b>Fruits:</b> {day_meals["breakfast"]}</div>', unsafe_allow_html=True)

        st.markdown("#### ☀️ Lunch (12–2 PM)")
        st.markdown(f'<div class="meal-row">🥗 <b>Plate 1 (Raw Salad):</b> {day_meals["lunch_salad"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meal-row">🍚 <b>Plate 2 (Cooked):</b> {day_meals["lunch_cooked"]}</div>', unsafe_allow_html=True)

    with col_m2:
        st.markdown("#### 🌙 Dinner (6–8 PM)")
        st.markdown(f'<div class="meal-row">🥗 <b>Plate 1 (Raw Salad):</b> {day_meals["dinner_salad"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meal-row">🍲 <b>Plate 2 (Cooked):</b> {day_meals["dinner_cooked"]}</div>', unsafe_allow_html=True)

        st.markdown("#### 🥜 Snack")
        st.markdown(f'<div class="meal-row">✨ {day_meals["snack"]}</div>', unsafe_allow_html=True)

    # Calories delta
    avg_cal = sum(d.get("calories", 1600) for d in meal_plan.values()) / len(meal_plan)
    delta = day_meals.get("calories", 1600) - avg_cal
    st.metric("Estimated Calories", f'{day_meals.get("calories", 1600)} kcal', delta=f"{delta:+.0f} vs avg")

    if "notes" in day_meals:
        st.info(f"💡 **Personalized Note:** {day_meals['notes']}")

    # Full week overview
    with st.expander("#### 📊 Full Week Overview"):
        rows = []
        for d, m in meal_plan.items():
            rows.append({"Day": d, "Breakfast": m["breakfast"], "Calories": m.get("calories", "...")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Nutrition Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab_nutrition:
    st.markdown("### 📊 DIP Diet Nutrition Dashboard")

    col_n1, col_n2, col_n3, col_n4 = st.columns(4)
    col_n1.markdown('<div class="metric-card"><h3>Daily Calories</h3><div class="value">~1,620</div><div class="sub">kcal/day average</div></div>', unsafe_allow_html=True)
    col_n2.markdown('<div class="metric-card"><h3>Daily Fiber</h3><div class="value">~52g</div><div class="sub">vs 25g recommended</div></div>', unsafe_allow_html=True)
    col_n3.markdown('<div class="metric-card"><h3>Antioxidant Score</h3><div class="value">95/100</div><div class="sub">Exceptional</div></div>', unsafe_allow_html=True)
    col_n4.markdown('<div class="metric-card"><h3>Plant Diversity</h3><div class="value">30+</div><div class="sub">unique foods/week</div></div>', unsafe_allow_html=True)

    nutrition = get_nutrition_profile()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Macros pie
        macros = nutrition["macros"]
        fig_macros = go.Figure(go.Pie(
            labels=list(macros.keys()),
            values=list(macros.values()),
            hole=0.4,
            textinfo="percent+label",
        ))
        fig_macros.update_layout(title="Macronutrient Distribution (%)", height=320)
        st.plotly_chart(fig_macros, use_container_width=True)

    with chart_col2:
        # Meal calories bar
        meal_cal = nutrition["meal_calories"]
        fig_meals = go.Figure(go.Bar(
            x=list(meal_cal.keys()),
            y=list(meal_cal.values()),
            marker_color=["#c8e6c9", "#1b5e20", "#388e3c", "#1b5e20", "#388e3c", "#c8e6c9"],
            text=list(meal_cal.values()),
            textposition="outside",
        ))
        fig_meals.update_layout(title="Calories by Meal", height=320)
        st.plotly_chart(fig_meals, use_container_width=True)

    # Weekly calorie line chart
    days_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = nutrition["weekly_avg_calories"]
    fig_weekly = go.Figure()
    fig_weekly.add_trace(go.Scatter(
        x=days_labels, y=weekly,
        mode="lines+markers",
        line=dict(color="#2e7d32", width=2),
        fill="tozeroy",
        fillcolor="rgba(102,187,106,0.15)",
        name="Calories",
    ))
    fig_weekly.update_layout(title="Weekly Calorie Intake", xaxis_title="Day", yaxis_title="Calories (kcal)", hovermode="x unified", height=300)
    st.plotly_chart(fig_weekly, use_container_width=True)

    # Radar chart
    nutrient_scores = nutrition["nutrient_scores"]
    categories = list(nutrient_scores.keys())
    values = list(nutrient_scores.values())
    fig_radar = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(46,125,50,0.25)",
        line=dict(color="#2e7d32"),
        name="DIP Diet Score",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="DIP Diet Nutrient Score Radar",
        height=350,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Nutrient targets
    st.markdown("#### 🔬 Key Nutrient Targets")
    targets = {
        "Fiber": ("~52g/day", "2× RDA — gut health, blood sugar"),
        "Vitamin C": ("~200mg/day", "Immune boost from raw fruits/veg"),
        "Potassium": ("~4700mg/day", "Heart & BP regulation"),
        "Magnesium": ("~420mg/day", "Muscle, nerve, bone health"),
        "Antioxidants": ("Very High", "Anti-inflammatory, anti-aging"),
    }
    for nutrient, (amount, note) in targets.items():
        st.markdown(
            f"<div class='meal-row'><b>{nutrient}:</b> {amount}"
            f"<br><small style='color:#555;'>{note}</small></div>",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Lifestyle
# ═══════════════════════════════════════════════════════════════════════════════
with tab_lifestyle:
    st.markdown("### 💡 DIP Diet Lifestyle Integration")

    life_col1, life_col2 = st.columns(2)

    with life_col1:
        st.markdown("""
            #### 🌅 Morning Routine
            <div class="meal-row">⏰ <b>5:30 AM</b> — Wake up</div>
            <div class="meal-row">💧 <b>6:00 AM</b> — 2-3 glasses warm water</div>
            <div class="meal-row">🧘 <b>6:15 AM</b> — 20 min Pranayama / Yoga</div>
            <div class="meal-row">🏃 <b>6:30 AM</b> — 30-45 min brisk walk</div>
            <div class="meal-row">☀️ <b>7:00 AM</b> — 20 min sunlight exposure</div>
            <div class="meal-row">🍌 <b>7:30 AM</b> — Fruit breakfast</div>
            """, unsafe_allow_html=True)

        st.markdown("""
            #### 🌙 Evening Routine
            <div class="meal-row">🥗 <b>6:00–7:30 PM</b> — Light dinner</div>
            <div class="meal-row">🚶 <b>7:45 PM</b> — 15-min evening walk</div>
            <div class="meal-row">📵 <b>9:00 PM</b> — No screens</div>
            <div class="meal-row">😴 <b>10:00 PM</b> — Sleep</div>
            """, unsafe_allow_html=True)

    with life_col2:
        st.markdown("""
            #### 💧 Hydration Rules
            <div class="meal-row">🌅 2-3 glasses warm water on waking</div>
            <div class="meal-row">🥤 1 glass water 30 min before each meal</div>
            <div class="meal-row">⏳ Wait 45 min after eating before drinking</div>
            <div class="meal-row">🥥 Coconut water as mid-morning snack</div>
            <div class="meal-row">💦 Total: 8-10 glasses/day</div>
            <div class="meal-row">🌡️ Prefer warm or room-temperature water</div>
            """, unsafe_allow_html=True)

        st.markdown("""
            #### 🚫 DIP Diet Discipline Rules
            <div class="meal-row">🍽️ No eating after 8 PM</div>
            <div class="meal-row">📦 Zero packaged / processed food</div>
            <div class="meal-row">🧈 No dairy in any form</div>
            <div class="meal-row">🍭 No sugar, honey, or artificial sweeteners</div>
            <div class="meal-row">☕ No tea, coffee, or alcohol</div>
            <div class="meal-row">🛢️ Maximum 1 tsp oil/day in cooking</div>
            """, unsafe_allow_html=True)

    st.markdown("#### 🔄 Smart Food Substitutions")
    subs = {
        "White rice → Brown rice / Millet": "Higher fiber, lower GI",
        "Refined flour (maida) → Whole wheat / Ragi flour": "Micronutrient rich",
        "Packaged juice → Whole fresh fruit": "Retains fiber & enzymes",
        "Dairy milk → Coconut water / Almond milk": "Dairy-free DIP-approved",
        "Sugar → None (allow natural fruit sugars)": "Stabilizes blood sugar",
        "Deep-fried → Steamed / Raw": "Preserves nutrients",
    }
    for sub, reason in subs.items():
        st.markdown(
            f"<div class='meal-row'><b>{sub}</b><br><small style='color:#555;'>{reason}</small></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### 🌿 DIP Diet Core Principles")
    principles = {
        "Plate 1 Rule": "50% raw uncooked salad at every meal — activates digestive enzymes",
        "Fruit Breakfast": "Only fruits until 10 AM — detox window, hormone reset",
        "Meal Timing": "Breakfast 7–9 AM, Lunch 12–2 PM, Dinner 6–8 PM",
        "Food Combining": "Avoid mixing fruits with cooked food — different digestion times",
        "Sunlight Therapy": "20–30 min daily — Vitamin D, circadian rhythm, immunity",
        "Earthing": "Barefoot on grass — reduces inflammation, improves sleep",
    }
    for principle, desc in principles.items():
        st.markdown(
            f"<div class='meal-row'><b>{principle}</b><br><small style='color:#555;'>{desc}</small></div>",
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; color:#666; font-size:0.8rem; padding:0.5rem;'>"
    "🥗 DIP Diet RAG Assistant · Powered by <b>IBM Granite</b> on <b>watsonx.ai</b> · "
    "For informational purposes only. Consult a healthcare professional for medical advice."
    "</div>",
    unsafe_allow_html=True,
)
