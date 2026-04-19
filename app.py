import streamlit as st
import pandas as pd
import joblib
from openai import OpenAI

st.set_page_config(page_title="Прогноз следующего пробного ЕНТ", layout="centered")

model = joblib.load("best_model.pkl")
model_columns = joblib.load("model_columns.pkl")


def get_openai_client():
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        if not api_key:
            return None
        return OpenAI(api_key=api_key.strip())
    except Exception:
        return None


def risk_level(score):
    if score < 70:
        return "Высокий риск"
    elif score < 100:
        return "Средний риск"
    else:
        return "Низкий риск"


def detect_issues(student_row):
    issues = []

    sleep_value = student_row["10. Среднее количество часов сна в сутки"]
    if sleep_value <= 4:
        issues.append("Критически низкий уровень сна")
    elif sleep_value <= 6:
        issues.append("Недостаток сна")

    interest_value = student_row["17. Насколько вам интересна учеба? 1 - совсем не интересна 5 - очень интересна"]
    if interest_value <= 2:
        issues.append("Низкий интерес к учебе")

    trial_value = student_row["41. Результат предыдущего пробного ЕНТ (ЕНТ который вы сдавали до последнего)"]
    if trial_value < 70:
        issues.append("Слабая академическая база")

    return issues


def generate_recommendations(issues, prediction):
    recommendations = []

    if "Критически низкий уровень сна" in issues:
        recommendations.append("Необходимо срочно нормализовать режим сна: желательно спать не менее 7–8 часов в сутки.")
    elif "Недостаток сна" in issues:
        recommendations.append("Рекомендуется увеличить продолжительность сна, так как недосып снижает концентрацию и продуктивность.")

    if "Низкий интерес к учебе" in issues:
        recommendations.append("Стоит повысить учебную мотивацию: поставить конкретные цели по баллам ЕНТ и разбить подготовку на небольшие этапы.")

    if "Слабая академическая база" in issues:
        recommendations.append("Рекомендуется начать с укрепления базовых тем и регулярно решать типовые задания по формату ЕНТ.")

    if prediction < 70:
        recommendations.append("Прогноз находится в зоне высокого риска, поэтому важно выстроить регулярный график подготовки и уделять внимание слабым темам.")
    elif prediction < 100:
        recommendations.append("Результат находится на среднем уровне, поэтому важно усилить практику и системность подготовки.")
    else:
        recommendations.append("Прогноз хороший, рекомендуется сохранить текущий темп подготовки и регулярно повторять изученный материал.")

    if not recommendations:
        recommendations.append("Выраженных проблем не обнаружено. Рекомендуется продолжать подготовку в стабильном режиме.")

    return recommendations


def get_ai_recommendation(prediction, risk, issues, recommendations):
    client = get_openai_client()
    if client is None:
        return "ИИ-рекомендации временно недоступны: API ключ не задан в настройках приложения."

    issues_text = "\n".join([f"- {item}" for item in issues]) if issues else "- Явных проблемных зон не обнаружено"
    recs_text = "\n".join([f"- {item}" for item in recommendations])

    prompt = f"""
Ты — образовательный AI-консультант.
Нужно написать персонализированные рекомендации ученику на русском языке.

Данные ученика:
- Прогнозируемый балл следующего пробного ЕНТ: {prediction}
- Уровень риска: {risk}

Выявленные проблемные зоны:
{issues_text}

Базовые рекомендации системы:
{recs_text}

Сформируй итоговый текст рекомендаций:
1. Напиши поддерживающе и понятно.
2. Дай конкретные действия на ближайшие 1–2 недели.
3. Не придумывай фактов, опирайся только на данные выше.
4. Ответ сделай в 1–2 абзацах без лишней воды.
"""

    try:
        response = client.responses.create(
            model="gpt-5.2",
            input=prompt
        )
        return response.output_text
    except Exception:
        return "ИИ-рекомендации временно недоступны из-за ограничений API или технической ошибки. Используйте базовые рекомендации системы."


st.title("Прогноз следующего пробного ЕНТ")
st.write(
    "Система предназначена для предварительной оценки результата следующего пробного ЕНТ "
    "на основе социально-академических факторов."
)

age = st.number_input("Возраст", min_value=10, max_value=25, value=17)
sleep = st.number_input("Часы сна", min_value=0.0, max_value=12.0, value=6.0, step=0.5)
interest = st.slider("Интерес к учебе", min_value=1, max_value=5, value=3)
trial = st.number_input("Текущий пробный ЕНТ", min_value=0.0, max_value=140.0, value=60.0, step=1.0)

if st.button("Рассчитать прогноз"):
    row = {col: 0 for col in model_columns}

    row["2. Возраст"] = age
    row["10. Среднее количество часов сна в сутки"] = sleep
    row["17. Насколько вам интересна учеба? 1 - совсем не интересна 5 - очень интересна"] = interest
    row["41. Результат предыдущего пробного ЕНТ (ЕНТ который вы сдавали до последнего)"] = trial

    df = pd.DataFrame([row])[model_columns]

    prediction = round(float(model.predict(df)[0]), 2)
    risk = risk_level(prediction)
    issues = detect_issues(df.iloc[0])
    recommendations = generate_recommendations(issues, prediction)

    st.subheader("Результат")
    st.write(f"**Прогнозируемый балл:** {prediction}")

    if risk == "Высокий риск":
        st.error(f"Уровень риска: {risk}")
    elif risk == "Средний риск":
        st.warning(f"Уровень риска: {risk}")
    else:
        st.success(f"Уровень риска: {risk}")

    st.subheader("Проблемные зоны")
    if issues:
        for issue in issues:
            st.write(f"- {issue}")
    else:
        st.write("Явных проблемных зон не обнаружено.")

    st.subheader("Рекомендации")
    for rec in recommendations:
        st.write(f"- {rec}")

    st.subheader("ИИ-рекомендации")
    with st.spinner("Генерируется персонализированная рекомендация..."):
        ai_text = get_ai_recommendation(prediction, risk, issues, recommendations)

    st.write(ai_text)
