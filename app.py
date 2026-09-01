"""
ScreenSense — Smartphone Addiction Risk Predictor
----------------------------------------------------
Loads the trained model saved from the notebook (best_addiction_model.joblib)
and serves it through a fully bilingual (EN/AR), custom-styled UI.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="ScreenSense",
    page_icon="🌘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# LANGUAGE STATE
# ----------------------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# ----------------------------------------------------------------------------
# TRANSLATIONS
# ----------------------------------------------------------------------------
TEXT = {
    "en": {
        "hero_title": "🌘 ScreenSense",
        "hero_sub": "A model trained on real usage data — sleep, screen time, study balance "
                     "and notification habits — estimates your smartphone addiction risk and "
                     "shows exactly which habits are driving it.",
        "tab_predict": "🔮 Predict",
        "tab_cope": "🧭 Understand & Cope",
        "tab_performance": "📈 Model performance",
        "tab_about": "ℹ️ About this model",
        "performance_title": "📈 Model performance",
        "performance_sub": "A quick summary of the experiments from the training notebook.",
        "best_model": "Best model",
        "test_accuracy": "Test accuracy",
        "cv_accuracy": "5-fold CV accuracy",
        "cv_stability": "CV standard deviation",
        "comparison_title": "Model comparison",
        "performance_note": "The Decision Tree achieved the highest test accuracy, while Random Forest had the strongest mean cross-validation accuracy. The project selected Decision Tree as the final model based on the test-set ranking used in the notebook.",
        "sb_profile": "⚙️ Your profile",
        "sb_age": "Age",
        "sb_gender": "Gender",
        "sb_sleep": "🛌 Sleep & recovery",
        "sb_sleep_hours": "Sleep hours / night",
        "sb_screen": "📱 Screen habits",
        "sb_screen_time": "Daily screen time (hrs)",
        "sb_gaming": "Gaming (hrs/day)",
        "sb_social": "Social media (hrs/day)",
        "sb_weekend": "Weekend screen time (hrs)",
        "sb_notif": "Notifications / day",
        "sb_opens": "App opens / day",
        "sb_study": "🎓 Study & stress",
        "sb_study_hours": "Work / study hours (day)",
        "sb_stress": "Stress level",
        "sb_academic": "Is screen time hurting academics/work?",
        "run_btn": "🔮 Predict my risk",
        "gender_opts": ["Male", "Female"],
        "stress_opts": ["Low", "Medium", "High"],
        "yesno_opts": ["No", "Yes"],
        "result_label": "Predicted outcome",
        "addicted": "Addiction signals detected",
        "not_addicted": "No addiction signals detected",
        "risk_score_title": "🎯 Estimated risk score",
        "fingerprint_title": "📊 Habit fingerprint",
        "confidence_note": "Model confidence",
        "confidence_caveat": "⚠️ The selected model (Decision Tree) tends to be very confident "
                              "— close to 0% or 100% — because it fits training data almost "
                              "perfectly. Treat this score as a directional signal, not a "
                              "precise probability.",
        "insights_title": "💡 What's driving this",
        "tip_sleep": "Your sleep is falling short of 8 hours — sleep debt is one of the strongest signals in this model.",
        "tip_dependency": "Combined screen + gaming + social time is high — try setting app time limits.",
        "tip_balance": "Study/work time is small relative to screen time — the balance is skewed toward entertainment.",
        "tip_notif": "A high notification count keeps re-triggering phone checks — consider muting non-essential apps.",
        "tip_none": "Your habits look balanced across the factors this model weighs most heavily.",
        "empty_title": "👈 Set your habits in the sidebar",
        "empty_sub": "Then hit **Predict my risk** to see your result, a habit fingerprint, "
                     "and the model's confidence.",
        "footer": "ScreenSense · powered by the model trained in your notebook",
        "cope_title": "🧭 Understand & cope with screen dependency",
        "cope_sub": "Practical, evidence-informed habits — not a diagnosis, just a starting point.",
        "cope_cards": [
            ("🌙", "Fix your sleep window",
             "Set a hard cut-off time for screens, 30–60 minutes before bed. Sleep debt is "
             "one of the biggest drivers of compulsive phone use the next day."),
            ("🔕", "Tame your notifications",
             "Turn off non-essential push notifications. Every buzz is a tiny trigger that "
             "pulls you back into the app — fewer triggers, fewer unplanned sessions."),
            ("⏱️", "Use app timers, not willpower",
             "Set daily time limits on social media and games at the OS level. Willpower "
             "runs out; a hard limit doesn't."),
            ("🎯", "Replace, don't just remove",
             "Swap one scrolling session a day for a specific alternative — a short walk, "
             "a call, reading — so the habit loop has somewhere else to go."),
            ("📵", "Create phone-free zones",
             "Keep the phone out of the bedroom and off the dinner table. Environment "
             "design beats self-control most days."),
            ("🧑‍⚕️", "Know when to ask for help",
             "If screen use is seriously affecting sleep, work, relationships, or mood, "
             "that's worth talking to a counselor or doctor about — this app is not a "
             "diagnosis."),
        ],
        "about_title": "ℹ️ About this model",
        "about_body": (
            "**Data:** 7,500 user records with screen-time, sleep, study, stress and "
            "notification behavior.\n\n"
            "**Target:** `addicted_label` — a binary label (addicted / not addicted).\n\n"
            "**Best model:** Decision Tree — ~93.5% test accuracy, ~93.3% cross-validation "
            "accuracy.\n\n"
            "**Input design:** predictions are based on usage and behavior features such as "
            "screen time, sleep, notifications, stress, study/work balance and related "
            "engineered features. The app does not ask the user to label their own addiction level."
        ),
    },
    "ar": {
        "hero_title": "🌘 سكرين سينس",
        "hero_sub": "نموذج تعلّم آلي مُدرّب على بيانات استخدام حقيقية — النوم، وقت الشاشة، "
                     "توازن الدراسة، وعادات الإشعارات — بيقيّم مستوى خطر إدمانك للموبايل "
                     "ويوريك بالظبط أي عادة هي اللي بتأثر أكتر.",
        "tab_predict": "🔮 التقييم",
        "tab_cope": "🧭 افهم وتعامل مع المشكلة",
        "tab_performance": "📈 أداء النموذج",
        "tab_about": "ℹ️ عن النموذج",
        "performance_title": "📈 أداء النموذج",
        "performance_sub": "ملخص سريع لتجارب النماذج الموجودة في نوت بوك التدريب.",
        "best_model": "أفضل نموذج",
        "test_accuracy": "دقة الاختبار",
        "cv_accuracy": "دقة Cross-Validation",
        "cv_stability": "الانحراف المعياري للـ CV",
        "comparison_title": "مقارنة النماذج",
        "performance_note": "حقق Decision Tree أعلى دقة على بيانات الاختبار، بينما حقق Random Forest أعلى متوسط في الـ Cross-Validation. تم اختيار Decision Tree كنموذج نهائي بناءً على ترتيب دقة الاختبار المستخدم في النوت بوك.",
        "sb_profile": "⚙️ بياناتك",
        "sb_age": "العمر",
        "sb_gender": "النوع",
        "sb_sleep": "🛌 النوم والراحة",
        "sb_sleep_hours": "ساعات النوم / اليوم",
        "sb_screen": "📱 عادات استخدام الشاشة",
        "sb_screen_time": "وقت الشاشة اليومي (ساعات)",
        "sb_gaming": "الألعاب (ساعات/يوم)",
        "sb_social": "السوشيال ميديا (ساعات/يوم)",
        "sb_weekend": "وقت الشاشة في الويكند (ساعات)",
        "sb_notif": "الإشعارات / اليوم",
        "sb_opens": "عدد فتح التطبيقات / اليوم",
        "sb_study": "🎓 الدراسة والتوتر",
        "sb_study_hours": "ساعات الدراسة / الشغل (اليوم)",
        "sb_stress": "مستوى التوتر",
        "sb_academic": "وقت الشاشة بيأثر على دراستك/شغلك؟",
        "run_btn": "🔮 قيّم مستوى خطري",
        "gender_opts": ["ذكر", "أنثى"],
        "stress_opts": ["منخفض", "متوسط", "مرتفع"],
        "yesno_opts": ["لا", "نعم"],
        "result_label": "النتيجة المتوقعة",
        "addicted": "فيه مؤشرات إدمان",
        "not_addicted": "مفيش مؤشرات إدمان واضحة",
        "risk_score_title": "🎯 نسبة الخطورة المقدّرة",
        "fingerprint_title": "📊 بصمة عاداتك",
        "confidence_note": "ثقة النموذج",
        "confidence_caveat": "⚠️ النموذج المُختار (Decision Tree) بيميل يكون واثق جدًا "
                              "— قريب من 0% أو 100% — لأنه بيتوافق مع بيانات التدريب شبه "
                              "بشكل كامل. اعتبر الرقم ده مؤشر اتجاه مش احتمال دقيق.",
        "insights_title": "💡 أهم العوامل المؤثرة",
        "tip_sleep": "نومك أقل من 8 ساعات — قلة النوم من أقوى المؤشرات في النموذج ده.",
        "tip_dependency": "مجموع وقت الشاشة + الألعاب + السوشيال ميديا عالي — جرّب تحدد وقت لكل تطبيق.",
        "tip_balance": "وقت الدراسة/الشغل قليل مقارنة بوقت الشاشة — التوازن مايل ناحية الترفيه.",
        "tip_notif": "عدد الإشعارات عالي وده بيرجّعك للموبايل باستمرار — جرّب تكتم التطبيقات مش الضرورية.",
        "tip_none": "عاداتك متوازنة نسبيًا في أهم العوامل اللي النموذج بيعتمد عليها.",
        "empty_title": "👈 حدد عاداتك من القائمة الجانبية",
        "empty_sub": "بعدين دوس **قيّم مستوى خطري** عشان تشوف النتيجة، وبصمة عاداتك، وثقة النموذج.",
        "footer": "سكرين سينس · بيشتغل بالنموذج المُدرّب في المفكرة بتاعتك",
        "cope_title": "🧭 افهم وتعامل مع الاعتماد على الشاشة",
        "cope_sub": "عادات عملية — مش تشخيص طبي، بس نقطة بداية كويسة.",
        "cope_cards": [
            ("🌙", "ثبّت ميعاد نوم واضح",
             "حدد وقت تقفل فيه الشاشات قبل النوم بـ 30-60 دقيقة. قلة النوم من أكبر أسباب "
             "استخدام الموبايل بشكل قهري في اليوم التالي."),
            ("🔕", "قلل الإشعارات",
             "قفل الإشعارات مش الضرورية. كل إشعار هو trigger صغير بيرجّعك للتطبيق — كل ما "
             "الـ triggers تقل، الجلسات الغير مخططة تقل."),
            ("⏱️", "استخدم مؤقتات التطبيقات مش قوة الإرادة",
             "حدد وقت يومي للسوشيال ميديا والألعاب من إعدادات الجهاز. قوة الإرادة بتخلص، "
             "بس الحد اليومي مبيخلصش."),
            ("🎯", "استبدل العادة، مش بس تمنعها",
             "بدّل جلسة سكرول واحدة في اليوم بحاجة تانية محددة — مشي، مكالمة، قراءة — "
             "عشان العادة تلاقي مكان تاني تروحله."),
            ("📵", "اعمل مناطق من غير موبايل",
             "خلي الموبايل بره الأوضة وقت النوم وبعيد عن السفرة وقت الأكل. تصميم البيئة "
             "غالبًا بيغلب قوة الإرادة."),
            ("🧑‍⚕️", "اعرف إمتى تطلب مساعدة",
             "لو استخدام الموبايل بقى بيأثر بشكل جدي على نومك أو شغلك أو علاقاتك أو مزاجك، "
             "يستاهل تتكلم مع مختص — الأداة دي مش تشخيص."),
        ],
        "about_title": "ℹ️ عن النموذج",
        "about_body": (
            "**البيانات:** 7,500 سجل مستخدم فيهم وقت الشاشة، النوم، الدراسة، التوتر، "
            "وعادات الإشعارات.\n\n"
            "**الهدف (Target):** `addicted_label` — تصنيف ثنائي (مدمن / مش مدمن).\n\n"
            "**أفضل موديل:** Decision Tree — دقة على بيانات الاختبار ~93.5%، ودقة "
            "cross-validation ~93.3%.\n\n"
            "**تصميم المدخلات:** التوقع بيعتمد على عادات الاستخدام والسلوك زي وقت الشاشة، "
            "النوم، الإشعارات، التوتر، توازن الدراسة/الشغل، وبعض الخصائص المشتقة. التطبيق "
            "مش بيطلب منك تحدد بنفسك إذا كنت مدمن ولا لأ."
        ),
    },
}


def t(key):
    return TEXT[st.session_state.lang][key]


# ----------------------------------------------------------------------------
# LANGUAGE TOGGLE
# Kept before the theme/RTL CSS so language and layout update in the same rerun.
# ----------------------------------------------------------------------------
with st.sidebar:
    lang_choice = st.radio(
        "🌐 Language / اللغة",
        ["English", "العربية"],
        index=0 if st.session_state.lang == "en" else 1,
        horizontal=True,
        key="language_selector",
    )
    st.session_state.lang = "en" if lang_choice == "English" else "ar"

# ----------------------------------------------------------------------------
# THEME  — dark "twilight" base, ONE interpolated risk gradient instead of
# repeated traffic-light colors, colored accents used sparingly.
# ----------------------------------------------------------------------------
is_rtl = st.session_state.lang == "ar"
rtl_css = """
    .stApp, section[data-testid="stSidebar"] { direction: rtl; text-align: right; }
    .stSlider label, .stSelectbox label, .stRadio label { text-align: right; }
""" if is_rtl else ""

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=Cairo:wght@400;600;700&display=swap');

    :root{{
        --bg-0:#0a0e17;
        --indigo:#6366f1;
        --teal:#14b8a6;
        --amber:#f59e0b;
        --rose:#e11d48;
        --text-soft:#c7cbe0;
    }}

    html, body, [class*="css"]  {{
        font-family: {'Cairo' if is_rtl else 'Inter'}, sans-serif;
    }}

    .stApp {{
        background: radial-gradient(circle at 10% 0%, #14172e 0%, var(--bg-0) 45%),
                    radial-gradient(circle at 90% 100%, #0c1f2b 0%, var(--bg-0) 55%);
        color: var(--text-soft);
    }}

    h1, h2, h3, h4 {{
        font-family: {'Cairo' if is_rtl else "'Space Grotesk'"}, sans-serif !important;
        color: #f2f3ff !important;
    }}

    .hero {{
        padding: 2.2rem 2.4rem;
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(99,102,241,0.20), rgba(20,184,166,0.12));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 40px rgba(99,102,241,0.12);
        margin-bottom: 1.6rem;
    }}
    .hero h1 {{
        font-size: 2.05rem;
        margin-bottom: 0.35rem;
        background: linear-gradient(90deg, var(--teal), var(--indigo));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero p {{ color: var(--text-soft); font-size: 0.98rem; max-width: 680px; }}

    .glass {{
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.3rem 1.5rem;
        backdrop-filter: blur(6px);
        margin-bottom: 1rem;
    }}
    .glass h4 {{
        font-size: 0.95rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--teal) !important;
        margin-bottom: 0.8rem;
    }}

    /* Verdict: same dark glass background for every risk level — only a thin
       accent stripe + badge color shift with risk, so the page doesn't turn
       into a wall of red. */
    .verdict {{
        border-radius: 20px;
        padding: 1.6rem 1.9rem;
        text-align: center;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-{'right' if is_rtl else 'left'}: 6px solid var(--risk-color, var(--indigo));
    }}
    .verdict h2 {{ font-size: 1.7rem; margin: 0.2rem 0; color: var(--risk-color, var(--indigo)) !important; }}
    .verdict p {{ color: var(--text-soft); margin: 0; }}

    .tip-card {{
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
        height: 100%;
    }}
    .tip-card .tip-icon {{ font-size: 1.5rem; margin-{'left' if is_rtl else 'right'}: 0.4rem; }}
    .tip-card h5 {{ color: #f2f3ff; margin: 0.3rem 0 0.4rem 0; font-size: 1.02rem; }}
    .tip-card p {{ color: var(--text-soft); font-size: 0.9rem; margin: 0; }}

    .metric-card {{
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        text-align: center;
        height: 100%;
    }}
    .metric-card .metric-value {{
        color: var(--teal);
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}
    .metric-card .metric-label {{
        color: var(--text-soft);
        font-size: 0.82rem;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f1226 0%, #0a0e17 100%);
        border-{'left' if is_rtl else 'right'}: 1px solid rgba(255,255,255,0.06);
    }}

    .stButton>button {{
        background: linear-gradient(90deg, var(--indigo), var(--teal));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.3rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        width: 100%;
    }}
    .stButton>button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 22px rgba(99,102,241,0.35);
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{
        background: rgba(255,255,255,0.03);
        border-radius: 10px 10px 0 0;
        padding: 8px 16px;
        color: var(--text-soft);
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(99,102,241,0.18) !important;
        color: #f2f3ff !important;
    }}

    {rtl_css}

    footer, #MainMenu {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# RISK COLOR — single continuous gradient (teal -> amber -> rose), used
# sparingly, instead of four repeated block colors.
# ----------------------------------------------------------------------------


def risk_color(score_pct: float) -> str:
    stops = [(0, (20, 184, 166)), (50, (245, 158, 11)), (100, (225, 29, 72))]
    score_pct = max(0, min(100, score_pct))
    for (s0, c0), (s1, c1) in zip(stops, stops[1:]):
        if s0 <= score_pct <= s1:
            ratio = (score_pct - s0) / (s1 - s0) if s1 != s0 else 0
            r = int(c0[0] + (c1[0] - c0[0]) * ratio)
            g = int(c0[1] + (c1[1] - c0[1]) * ratio)
            b = int(c0[2] + (c1[2] - c0[2]) * ratio)
            return f"rgb({r},{g},{b})"
    return "rgb(99,102,241)"


# ----------------------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "best_addiction_model.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


bundle = load_model()

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>{t('hero_title')}</h1>
        <p>{t('hero_sub')}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if bundle is None:
    missing_msg = (
        "⚠️ Couldn't find **best_addiction_model.joblib** next to this script. "
        "Run the training notebook first (it saves this file automatically), "
        "then drop it in the same folder as `app.py`."
        if st.session_state.lang == "en" else
        "⚠️ ملقيتش ملف **best_addiction_model.joblib** جنب السكريبت. "
        "شغّل نوت بوك التدريب الأول (بيحفظ الملف ده تلقائي)، بعدين حطه في نفس فولدر `app.py`."
    )
    st.error(missing_msg)
    st.stop()

model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]

# ----------------------------------------------------------------------------
# SIDEBAR — inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### {t('sb_profile')}")
    age = st.slider(t("sb_age"), 13, 60, 21)
    gender_label = st.radio(t("sb_gender"), t("gender_opts"), horizontal=True)
    gender = t("gender_opts").index(gender_label)

    st.markdown(f"### {t('sb_sleep')}")
    sleep_hours = st.slider(t("sb_sleep_hours"), 0.0, 12.0, 6.5, 0.5)

    st.markdown(f"### {t('sb_screen')}")
    daily_screen_time_hours = st.slider(t("sb_screen_time"), 0.0, 16.0, 5.0, 0.5)
    gaming_hours = st.slider(t("sb_gaming"), 0.0, 10.0, 1.0, 0.5)
    social_media_hours = st.slider(t("sb_social"), 0.0, 10.0, 2.0, 0.5)
    weekend_screen_time = st.slider(t("sb_weekend"), 0.0, 16.0, 6.0, 0.5)
    notifications_per_day = st.slider(t("sb_notif"), 0, 300, 40, 5)
    app_opens_per_day = st.slider(t("sb_opens"), 0, 300, 35, 5)

    st.markdown(f"### {t('sb_study')}")
    work_study_hours = st.slider(t("sb_study_hours"), 0.0, 16.0, 4.0, 0.5)
    stress_label = st.select_slider(t("sb_stress"), options=t("stress_opts"), value=t("stress_opts")[1])
    stress_level = t("stress_opts").index(stress_label)
    academic_label = st.radio(t("sb_academic"), t("yesno_opts"), horizontal=True)
    academic_work_impact = t("yesno_opts").index(academic_label)

    st.markdown("---")
    run = st.button(t("run_btn"), use_container_width=True)

# ----------------------------------------------------------------------------
# BUILD FEATURE ROW (matches the exact features the notebook trained on,
# including the self-reported addiction_level column)
# ----------------------------------------------------------------------------


def build_features():
    raw = {
        "age": age,
        "gender": gender,
        "daily_screen_time_hours": daily_screen_time_hours,
        "social_media_hours": social_media_hours,
        "gaming_hours": gaming_hours,
        "work_study_hours": work_study_hours,
        "sleep_hours": sleep_hours,
        "notifications_per_day": notifications_per_day,
        "app_opens_per_day": app_opens_per_day,
        "weekend_screen_time": weekend_screen_time,
        "stress_level": stress_level,
        "academic_work_impact": academic_work_impact,
    }
    raw["sleep_deficit"] = 8 - raw["sleep_hours"]
    raw["screen_dependency"] = (
        raw["daily_screen_time_hours"] + raw["gaming_hours"] + raw["social_media_hours"]
    )
    raw["study_balance"] = (
        raw["work_study_hours"] / raw["daily_screen_time_hours"]
        if raw["daily_screen_time_hours"] != 0 else 0
    )
    return raw


def gauge_chart(score_pct: float, color: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score_pct,
            number={"suffix": "%", "font": {"size": 38, "color": "#f2f3ff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b8fb0"},
                "bar": {"color": color, "thickness": 0.32},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [{"range": [0, 100], "color": "rgba(255,255,255,0.05)"}],
            },
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(t=15, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#c7cbe0"},
    )
    return fig


def radar_chart(raw):
    if st.session_state.lang == "ar":
        categories = ["وقت الشاشة", "الألعاب", "السوشيال ميديا", "قلة النوم", "الإشعارات", "التوتر"]
    else:
        categories = ["Screen time", "Gaming", "Social media", "Sleep deficit", "Notifications", "Stress"]
    values = [
        min(raw["daily_screen_time_hours"] / 12, 1) * 10,
        min(raw["gaming_hours"] / 8, 1) * 10,
        min(raw["social_media_hours"] / 8, 1) * 10,
        min(max(raw["sleep_deficit"], 0) / 6, 1) * 10,
        min(raw["notifications_per_day"] / 150, 1) * 10,
        raw["stress_level"] / 2 * 10,
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + values[:1],
            theta=categories + categories[:1],
            fill="toself",
            fillcolor="rgba(99,102,241,0.32)",
            line=dict(color="#14b8a6", width=2),
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], showticklabels=False, gridcolor="rgba(255,255,255,0.15)"),
            angularaxis=dict(color="#c7cbe0"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        height=330,
        margin=dict(t=30, b=10, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#c7cbe0"},
    )
    return fig


# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_predict, tab_cope, tab_performance, tab_about = st.tabs(
    [t("tab_predict"), t("tab_cope"), t("tab_performance"), t("tab_about")]
)


with tab_predict:
    if run:
        raw = build_features()
        input_df = pd.DataFrame([raw]).reindex(columns=feature_names, fill_value=0)
        input_scaled = scaler.transform(input_df)

        pred = int(model.predict(input_scaled)[0])

        proba = None
        risk_score = 50.0
        if hasattr(model, "predict_proba"):
            proba_raw = model.predict_proba(input_scaled)[0]
            classes = list(model.classes_)
            idx_1 = classes.index(1) if 1 in classes else len(classes) - 1
            risk_score = float(proba_raw[idx_1] * 100)
            proba = proba_raw

        color = risk_color(risk_score)
        outcome_text = t("addicted") if pred == 1 else t("not_addicted")

        col_left, col_right = st.columns([1.1, 1])

        with col_left:
            st.markdown(
                f"""
                <div class="verdict" style="--risk-color:{color};">
                    <p>{t('result_label')}</p>
                    <h2>{outcome_text}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown(f"#### {t('fingerprint_title')}")
            st.plotly_chart(radar_chart(raw), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown(f"#### {t('risk_score_title')}")
            st.plotly_chart(gauge_chart(risk_score, color), use_container_width=True)
            st.caption(t("confidence_caveat"))
            st.markdown("</div>", unsafe_allow_html=True)

            if proba is not None:
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.markdown(f"#### 🧠 {t('confidence_note')}")
                labels = [t("not_addicted"), t("addicted")]
                for i, cls in enumerate(model.classes_):
                    name = labels[1] if cls == 1 else labels[0]
                    p = proba[i]
                    st.markdown(f"**{name}** — {p*100:.1f}%")
                    st.progress(float(p))
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown(f"#### {t('insights_title')}")
        tips = []
        if raw["sleep_deficit"] > 2:
            tips.append(t("tip_sleep"))
        if raw["screen_dependency"] > 8:
            tips.append(t("tip_dependency"))
        if raw["study_balance"] < 0.5 and raw["daily_screen_time_hours"] > 0:
            tips.append(t("tip_balance"))
        if raw["notifications_per_day"] > 80:
            tips.append(t("tip_notif"))
        if not tips:
            tips.append(t("tip_none"))
        for tip in tips:
            st.markdown(f"- {tip}")
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown(
            f"""
            <div class="glass" style="text-align:center; padding:3rem 1.5rem;">
                <h3>{t('empty_title')}</h3>
                <p>{t('empty_sub')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_cope:
    st.markdown(f"## {t('cope_title')}")
    st.caption(t("cope_sub"))
    cards = t("cope_cards")
    cols = st.columns(3)
    for i, (icon, title, body) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="tip-card">
                    <span class="tip-icon">{icon}</span><h5 style="display:inline;">{title}</h5>
                    <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# MODEL PERFORMANCE
# Values below come directly from the experiments in the training notebook.
# ----------------------------------------------------------------------------
with tab_performance:
    st.markdown(f"## {t('performance_title')}")
    st.caption(t("performance_sub"))

    performance = pd.DataFrame([
        {"Model": "Decision Tree", "Test Accuracy": 0.934667, "CV Mean Accuracy": 0.932500, "CV Std Dev": 0.006604},
        {"Model": "Random Forest", "Test Accuracy": 0.927333, "CV Mean Accuracy": 0.935833, "CV Std Dev": 0.003249},
        {"Model": "XGBoost", "Test Accuracy": 0.927333, "CV Mean Accuracy": 0.930833, "CV Std Dev": 0.005426},
        {"Model": "SVM", "Test Accuracy": 0.918000, "CV Mean Accuracy": 0.914833, "CV Std Dev": 0.007367},
        {"Model": "Logistic Regression", "Test Accuracy": 0.892667, "CV Mean Accuracy": 0.897500, "CV Std Dev": 0.005083},
        {"Model": "KNN", "Test Accuracy": 0.886000, "CV Mean Accuracy": 0.890667, "CV Std Dev": 0.008239},
        {"Model": "Naive Bayes", "Test Accuracy": 0.848000, "CV Mean Accuracy": 0.863500, "CV Std Dev": 0.008052},
    ])

    best_row = performance.loc[performance["Model"] == "Decision Tree"].iloc[0]
    metrics = [
        (t("best_model"), "Decision Tree"),
        (t("test_accuracy"), f"{best_row['Test Accuracy'] * 100:.2f}%"),
        (t("cv_accuracy"), f"{best_row['CV Mean Accuracy'] * 100:.2f}%"),
        (t("cv_stability"), f"± {best_row['CV Std Dev'] * 100:.2f}%"),
    ]

    cols = st.columns(4)
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown(f"#### {t('comparison_title')}")

    chart_df = performance.melt(
        id_vars="Model",
        value_vars=["Test Accuracy", "CV Mean Accuracy"],
        var_name="Metric",
        value_name="Accuracy",
    )

    fig = go.Figure()
    for metric in ["Test Accuracy", "CV Mean Accuracy"]:
        subset = chart_df[chart_df["Metric"] == metric]
        fig.add_trace(
            go.Bar(
                x=subset["Model"],
                y=subset["Accuracy"] * 100,
                name=metric,
            )
        )

    fig.update_layout(
        barmode="group",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#c7cbe0"},
        yaxis=dict(title="Accuracy (%)", range=[80, 100], gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=30, l=20, r=20),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)

    display_df = performance.copy()
    for column in ["Test Accuracy", "CV Mean Accuracy", "CV Std Dev"]:
        display_df[column] = display_df[column].map(lambda value: f"{value * 100:.2f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.info(t("performance_note"))

with tab_about:
    st.markdown(f"## {t('about_title')}")
    st.markdown(f'<div class="glass">{t("about_body").replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

st.markdown(
    f"<p style='text-align:center; color:#5c6084; font-size:0.8rem; margin-top:2rem;'>{t('footer')}</p>",
    unsafe_allow_html=True,
)