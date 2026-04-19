import streamlit as st
import pandas as pd
import joblib
from openai import OpenAI

st.set_page_config(page_title="Прогноз следующего пробного ЕНТ", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns


def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", "")

    if not isinstance(api_key, str):
        return None

    api_key = api_key.strip()

    if api_key == "":
        return None

    try:
        OpenAI(api_key=api_key)
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

    sleep_value = student_row.get("10. Среднее количество часов сна в сутки", 0)
    if sleep_value <= 4:
        issues.append("Критически низкий уровень сна")
    elif sleep_value <= 6:
        issues.append("Недостаток сна")

    interest_value = student_row.get(
        "17. Насколько вам интересна учеба? 1 - совсем не интересна 5 - очень интересна", 0
    )
    if interest_value <= 2:
        issues.append("Низкий интерес к учебе")

    trial_value = student_row.get(
        "41. Результат предыдущего пробного ЕНТ (ЕНТ который вы сдавали до последнего)", 0
    )
    if trial_value < 70:
        issues.append("Слабая академическая база")

    return issues


def generate_recommendations(issues, prediction):
    recommendations = []

    if "Критически низкий уровень сна" in issues:
        recommendations.append(
            "Необходимо срочно нормализовать режим сна: желательно спать не менее 7–8 часов в сутки."
        )
    elif "Недостаток сна" in issues:
        recommendations.append(
            "Рекомендуется увеличить продолжительность сна, так как недосып снижает концентрацию и продуктивность."
        )

    if "Низкий интерес к учебе" in issues:
        recommendations.append(
            "Стоит повысить учебную мотивацию: поставить конкретные цели по баллам ЕНТ и разбить подготовку на небольшие этапы."
        )

    if "Слабая академическая база" in issues:
        recommendations.append(
            "Рекомендуется начать с укрепления базовых тем и регулярно решать типовые задания по формату ЕНТ."
        )

    if prediction < 70:
        recommendations.append(
            "Прогноз находится в зоне высокого риска, поэтому важно выстроить регулярный график подготовки и уделять внимание слабым темам."
        )
    elif prediction < 100:
        recommendations.append(
            "Результат находится на среднем уровне, поэтому важно усилить практику и системность подготовки."
        )
    else:
        recommendations.append(
            "Прогноз хороший, рекомендуется сохранить текущий темп подготовки и регулярно повторять изученный материал."
        )

    if not recommendations:
        recommendations.append(
            "Выраженных проблем не обнаружено. Рекомендуется продолжать подготовку в стабильном режиме."
        )

    return recommendations


def get_ai_recommendation(prediction, risk, issues, recommendations):
    client = get_openai_client()

    if client is None:
        return "ИИ-рекомендации временно недоступны: API ключ не найден или не загрузился."

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
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Ты образовательный AI-консультант. Пиши кратко, ясно и доброжелательно на русском языке."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка OpenAI: {str(e)}"
