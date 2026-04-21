import streamlit as st
import pandas as pd
import numpy as np
import joblib
from openai import OpenAI

st.set_page_config(page_title="Прогноз ЕНТ", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns


def get_openai_client():
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def risk_level(score):
    if score < 70:
        return "Высокий риск"
    elif score < 100:
        return "Средний риск"
    else:
        return "Низкий риск"


def set_if_exists(row, col_name, value):
    if col_name in row:
        row[col_name] = value


def build_student_df(user_input, model_columns):
    row = {col: np.nan for col in model_columns}

    for key, value in user_input.items():
        set_if_exists(row, key, value)

    student_df = pd.DataFrame([row])
    student_df = student_df.reindex(columns=model_columns)
    return student_df


def detect_issues_full(student_row):
    issues = []

    # Academic
    trial = student_row.get("41. Результат предыдущего пробного ЕНТ (ЕНТ который вы сдавали  до последнего)", np.nan)
    q1 = student_row.get("38. Средний балл за первую четверть (по 5-балльной шкале) (укажите число)", np.nan)
    q2 = student_row.get("39. Средний балл за вторую четверть (по 5-балльной шкале) (укажите число)", np.nan)
    study_hours = student_row.get("11, Общее количество часов самостоятельной учебной работы в день (включая выполнение домашних заданий и подготовку к ЕНТ)", np.nan)
    absences = student_row.get("12. Количество пропусков занятий за последний месяц", np.nan)
    hw = student_row.get("21. Как часто вы выполняете домашние задания полностью?", np.nan)
    repeat_days = student_row.get("22. Сколько дней в неделю вы дополнительно повторяете учебный материал?", np.nan)

    if pd.notna(trial) and trial < 70:
        issues.append("Низкий текущий пробный результат")
    if pd.notna(q1) and q1 < 4:
        issues.append("Слабая успеваемость за 1 четверть")
    if pd.notna(q2) and q2 < 4:
        issues.append("Слабая успеваемость за 2 четверть")
    if pd.notna(study_hours) and study_hours < 2:
        issues.append("Недостаточная самостоятельная подготовка")
    if pd.notna(absences) and absences > 4:
        issues.append("Высокое количество пропусков")
    if pd.notna(hw) and hw <= 2:
        issues.append("Домашние задания выполняются нерегулярно")
    if pd.notna(repeat_days) and repeat_days < 3:
        issues.append("Слишком мало повторения учебного материала")

    # Motivation
    interest = student_row.get("17. Насколько вам интересна учеба? 1 - совсем не интересна 5 - очень интересна", np.nan)
    goals = student_row.get("18. Насколько ясно вы понимаете свои учебные цели? 1 - совсем не понимаю 5 - полностью понимаю", np.nan)
    effort = student_row.get("19. Насколько вы готовы прилагать дополнительные усилия для учебы? 1 - совсем не готов 5 - полностью готов", np.nan)
    motivation_loss = student_row.get("20. Как часто вы теряете мотивацию к учебе?", np.nan)
    independence = student_row.get("36. Насколько вы считаете себя самостоятельным в учебе? 1 - совсем не самостоятельный 5 - полностью самостоятельный", np.nan)
    planning = student_row.get("37. Как часто вы планируете свою учебную деятельность (расписание, подготовку к экзаменам)?", np.nan)

    if pd.notna(interest) and interest <= 2:
        issues.append("Низкий интерес к учебе")
    if pd.notna(goals) and goals <= 2:
        issues.append("Неясные учебные цели")
    if pd.notna(effort) and effort <= 2:
        issues.append("Низкая готовность к дополнительным усилиям")
    if pd.notna(motivation_loss) and motivation_loss >= 4:
        issues.append("Частая потеря мотивации")
    if pd.notna(independence) and independence <= 2:
        issues.append("Низкая самостоятельность в учебе")
    if pd.notna(planning) and planning <= 2:
        issues.append("Слабое планирование учебной деятельности")

    # Wellbeing
    sleep = student_row.get("10. Среднее количество часов сна в сутки", np.nan)
    stress = student_row.get("23. Уровень учебного стресса", np.nan)
    anxiety = student_row.get("24. Тревожность перед экзаменами", np.nan)
    fatigue = student_row.get("25. Уровень усталости от учебной нагрузки", np.nan)
    confidence = student_row.get("26. Уверенность в своих учебных способностях", np.nan)
    concentration = student_row.get("27. Концентрация во время занятий", np.nan)
    overload = student_row.get("28. Как часто вы ощущаете учебную перегрузку?", np.nan)

    if pd.notna(sleep) and sleep <= 4:
        issues.append("Критически низкий уровень сна")
    elif pd.notna(sleep) and sleep <= 6:
        issues.append("Недостаток сна")
    if pd.notna(stress) and stress >= 4:
        issues.append("Высокий учебный стресс")
    if pd.notna(anxiety) and anxiety >= 4:
        issues.append("Высокая тревожность перед экзаменами")
    if pd.notna(fatigue) and fatigue >= 4:
        issues.append("Высокая усталость от учебной нагрузки")
    if pd.notna(confidence) and confidence <= 2:
        issues.append("Низкая уверенность в своих учебных способностях")
    if pd.notna(concentration) and concentration <= 2:
        issues.append("Слабая концентрация")
    if pd.notna(overload) and overload >= 4:
        issues.append("Частое ощущение перегрузки")

    # Digital
    online_use = student_row.get("29. Используете ли вы онлайн-ресурсы для подготовки к ЕНТ (онлайн-курсы, тестовые платформы, образовательные видео)?", np.nan)
    online_hours = student_row.get("30. Сколько часов в неделю вы используете онлайн-ресурсы для учебы?", np.nan)
    search_materials = student_row.get("31. Как часто вы самостоятельно ищете дополнительные материалы для изучения темы  (видео, статьи, тесты, конспекты)?", np.nan)
    extra_tests = student_row.get("32. Как часто вы решаете дополнительные тесты или задания сверх школьной программы?", np.nan)
    weekly_tests = student_row.get("33. Сколько раз в неделю вы проходите онлайн-тесты или тренировочные задания? (Укажите число)", np.nan)
    videos = student_row.get("34. Используете ли вы образовательные видео или онлайн-лекции для подготовки?", np.nan)
    online_help = student_row.get("35. Насколько онлайн-ресурсы помогают вам лучше понимать учебный материал? 1 - совсем не помогают 5 - значительно помогают", np.nan)

    if pd.notna(online_use) and online_use == 1:
        issues.append("Онлайн-ресурсы почти не используются")
    if pd.notna(online_hours) and online_hours < 2:
        issues.append("Слишком мало времени на онлайн-обучение")
    if pd.notna(search_materials) and search_materials <= 2:
        issues.append("Редкий поиск дополнительных материалов")
    if pd.notna(extra_tests) and extra_tests <= 2:
        issues.append("Редкое решение дополнительных заданий")
    if pd.notna(weekly_tests) and weekly_tests < 2:
        issues.append("Мало тренировочных тестов в неделю")
    if pd.notna(videos) and videos == 1:
        issues.append("Не используются образовательные видео")
    if pd.notna(online_help) and online_help <= 2:
        issues.append("Онлайн-ресурсы мало помогают в понимании материала")

    # Family
    parent_interest = student_row.get("13. Интерес родителей к вашей учебе", np.nan)
    family_support = student_row.get("14. Эмоциональная поддержка со стороны семьи", np.nan)
    parent_control = student_row.get("15. Контроль выполнения домашних заданий со стороны родителей", np.nan)
    parent_pressure = student_row.get("16. Давление со стороны родителей относительно результатов ЕНТ", np.nan)

    if pd.notna(parent_interest) and parent_interest <= 2:
        issues.append("Низкий интерес родителей к учебе")
    if pd.notna(family_support) and family_support <= 2:
        issues.append("Недостаточная эмоциональная поддержка семьи")
    if pd.notna(parent_control) and parent_control <= 2:
        issues.append("Слабый родительский контроль")
    if pd.notna(parent_pressure) and parent_pressure >= 4:
        issues.append("Высокое давление со стороны родителей")

    # Context
    workplace = student_row.get("4. Есть ли у вас отдельное рабочее место для учебы?", np.nan)
    tutor = student_row.get("5. Есть ли у вас репетитор по учебным предметам?", np.nan)
    extra_activity = student_row.get("7, Количество часов дополнительной занятости в неделю (спорт, кружки, работа)", np.nan)

    if pd.notna(workplace) and workplace == 0:
        issues.append("Нет отдельного рабочего места для учебы")
    if pd.notna(tutor) and tutor == 0:
        issues.append("Нет репетитора")
    if pd.notna(extra_activity) and extra_activity > 15:
        issues.append("Высокая дополнительная занятость")

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

    if "Неясные учебные цели" in issues:
        recommendations.append(
            "Полезно определить конкретную цель по баллам ЕНТ и разбить подготовку на недельные задачи."
        )

    if "Низкая готовность к дополнительным усилиям" in issues:
        recommendations.append(
            "Рекомендуется начать с небольших, но регулярных дополнительных занятий, чтобы постепенно сформировать устойчивый учебный ритм."
        )

    if "Частая потеря мотивации" in issues:
        recommendations.append(
            "Важно чередовать подготовку, тесты и отдых, а также отслеживать маленькие достижения, чтобы не терять мотивацию."
        )

    if "Слабая академическая база" in issues or "Низкий текущий пробный результат" in issues:
        recommendations.append(
            "Рекомендуется начать с укрепления базовых тем и регулярно решать типовые задания по формату ЕНТ."
        )

    if "Недостаточная самостоятельная подготовка" in issues:
        recommendations.append(
            "Нужно увеличить время самостоятельной подготовки и закрепить ежедневный график занятий."
        )

    if "Высокое количество пропусков" in issues:
        recommendations.append(
            "Стоит сократить пропуски занятий, так как систематическое присутствие влияет на качество подготовки."
        )

    if "Высокий учебный стресс" in issues or "Высокая тревожность перед экзаменами" in issues:
        recommendations.append(
            "Рекомендуется снизить перегрузку, добавить короткие перерывы и использовать более структурированный план подготовки."
        )

    if "Высокая усталость от учебной нагрузки" in issues or "Частое ощущение перегрузки" in issues:
        recommendations.append(
            "Важно перераспределить нагрузку, чтобы избежать переутомления, и соблюдать баланс между учебой и восстановлением."
        )

    if "Слабая концентрация" in issues:
        recommendations.append(
            "Полезно заниматься короткими блоками по 25–40 минут с перерывами, чтобы улучшить концентрацию."
        )

    if "Низкая уверенность в своих учебных способностях" in issues:
        recommendations.append(
            "Следует чаще фиксировать прогресс и начинать подготовку с посильных заданий, чтобы укрепить уверенность."
        )

    if "Онлайн-ресурсы почти не используются" in issues or "Не используются образовательные видео" in issues:
        recommendations.append(
            "Стоит подключить онлайн-ресурсы, тестовые платформы и образовательные видео для дополнительной практики."
        )

    if "Мало тренировочных тестов в неделю" in issues or "Редкое решение дополнительных заданий" in issues:
        recommendations.append(
            "Рекомендуется чаще проходить тренировочные тесты, чтобы лучше привыкнуть к формату ЕНТ."
        )

    if "Низкий интерес родителей к учебе" in issues or "Недостаточная эмоциональная поддержка семьи" in issues:
        recommendations.append(
            "Полезно обсудить с семьей учебные цели и договориться о более поддерживающей атмосфере дома."
        )

    if "Высокое давление со стороны родителей" in issues:
        recommendations.append(
            "Важно снизить избыточное давление и сделать акцент не только на результате, но и на стабильном прогрессе."
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

    user_prompt = f"""
Нужно написать подробные персонализированные рекомендации ученику на русском языке.

Данные ученика:
- Прогнозируемый балл ЕНТ: {prediction}
- Уровень риска: {risk}

Выявленные проблемные зоны:
{issues_text}

Базовые рекомендации системы:
{recs_text}

Сформируй развернутые рекомендации:
1. Пиши доброжелательно, понятно и мотивирующе.
2. Объясни, какие факторы сейчас сильнее всего влияют на прогнозируемый результат.
3. Для каждой ключевой проблемной зоны предложи конкретные шаги улучшения.
4. Дай практический план действий на ближайшие 1–2 недели.
5. Добавь рекомендации по режиму сна, самостоятельной подготовке, решению тестов, повторению тем и психологической устойчивости.
6. Если есть семейные или мотивационные трудности, мягко укажи, как с ними можно работать.
7. Не придумывай фактов, используй только данные выше.
8. Ответ сделай подробным: 4–6 абзацев.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Ты образовательный AI-консультант. Пиши кратко, ясно и доброжелательно на русском языке."
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ИИ-рекомендации временно недоступны. Ошибка OpenAI: {str(e)}"


try:
    model, model_columns = load_artifacts()
except Exception as e:
    st.error("Не удалось загрузить модель.")
    st.code(str(e))
    st.stop()


st.title("Прогноз ЕНТ")
st.write("Заполните анкету. Система использует все содержательные факторы анкеты и текущий пробный результат.")

with st.form("ent_form"):
    st.subheader("1. Общий контекст")
    col1, col2 = st.columns(2)

    with col1:
        class_num = st.selectbox("Класс обучения", [10, 11], index=1)
        age = st.number_input("Возраст", min_value=14, max_value=20, value=17)
        gender = st.selectbox(
            "Пол",
            [0, 1],
            format_func=lambda x: "Женский" if x == 0 else "Мужской"
        )
        workplace = st.selectbox(
            "Есть ли у вас отдельное рабочее место для учебы?",
            [1, 0],
            format_func=lambda x: "Да" if x == 1 else "Нет"
        )
        tutor = st.selectbox(
            "Есть ли у вас репетитор по учебным предметам?",
            [1, 0],
            format_func=lambda x: "Да" if x == 1 else "Нет"
        )
        income = st.selectbox(
            "Материальное положение семьи (доход)",
            options=[125, 350, 550, 825, 1100],
            format_func=lambda x: {
                125: "До 250 тыс.",
                350: "250–450 тыс.",
                550: "450–650 тыс.",
                825: "650–1000 тыс.",
                1100: "Более 1000 тыс."
            }[x]
        )
        extra_activity = st.number_input(
            "Количество часов дополнительной занятости в неделю",
            min_value=0.0, max_value=40.0, value=0.0, step=0.5
        )

    with col2:
        profile_subjects = st.selectbox(
            "Профильные предметы на ЕНТ",
            [
                "Биология - география",
                "Биология - химия",
                "Всемирная история - английский",
                "Всемирная история - география",
                "Всемирная история - основы права",
                "География - английский",
                "Казахский язык - Казахская литература",
                "Математика - география",
                "Математика - информатика",
                "Математика - физика",
                "Русский язык - Русская литература",
                "Химия - физика",
            ]
        )
        language = st.selectbox(
            "Язык обучения",
            ["Русский", "Казахский", "Английский", "Русский и казахский"]
        )
        sleep = st.number_input("Среднее количество часов сна в сутки", min_value=0.0, max_value=15.0, value=6.0, step=0.5)
        study_hours = st.number_input("Общее количество часов самостоятельной учебной работы в день", min_value=0.0, max_value=15.0, value=3.0, step=0.5)
        absences = st.number_input("Количество пропусков занятий за последний месяц", min_value=0, max_value=30, value=0, step=1)
        current_trial = st.number_input("Текущий результат пробного ЕНТ", min_value=0.0, max_value=140.0, value=70.0, step=1.0)

    st.subheader("2. Семья")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        parent_interest = st.slider("Интерес родителей к вашей учебе", 1, 5, 3)
    with c2:
        family_support = st.slider("Эмоциональная поддержка со стороны семьи", 1, 5, 3)
    with c3:
        parent_control = st.slider("Контроль выполнения домашних заданий со стороны родителей", 1, 5, 3)
    with c4:
        parent_pressure = st.slider("Давление со стороны родителей относительно результатов ЕНТ", 1, 5, 3)

    st.subheader("3. Мотивация")
    m1, m2, m3 = st.columns(3)
    with m1:
        interest_study = st.slider("Насколько вам интересна учеба?", 1, 5, 3)
        goals = st.slider("Насколько ясно вы понимаете свои учебные цели?", 1, 5, 3)
    with m2:
        effort = st.slider("Насколько вы готовы прилагать дополнительные усилия для учебы?", 1, 5, 3)
        motivation_loss = st.slider("Как часто вы теряете мотивацию к учебе?", 1, 5, 3)
    with m3:
        independence = st.slider("Насколько вы считаете себя самостоятельным в учебе?", 1, 5, 3)
        planning = st.slider("Как часто вы планируете свою учебную деятельность?", 1, 5, 3)

    st.subheader("4. Академическое поведение")
    a1, a2, a3 = st.columns(3)
    with a1:
        hw_full = st.slider("Как часто вы выполняете домашние задания полностью?", 1, 5, 3)
        repeat_days = st.slider("Сколько дней в неделю вы дополнительно повторяете учебный материал?", 0, 7, 3)
    with a2:
        q1_grade = st.number_input("Средний балл за первую четверть", min_value=2.0, max_value=5.0, value=4.0, step=0.1)
        q2_grade = st.number_input("Средний балл за вторую четверть", min_value=2.0, max_value=5.0, value=4.0, step=0.1)
    with a3:
        stress = st.slider("Уровень учебного стресса", 1, 5, 3)
        anxiety = st.slider("Тревожность перед экзаменами", 1, 5, 3)

    st.subheader("5. Психоэмоциональное состояние")
    w1, w2, w3 = st.columns(3)
    with w1:
        fatigue = st.slider("Уровень усталости от учебной нагрузки", 1, 5, 3)
        confidence = st.slider("Уверенность в своих учебных способностях", 1, 5, 3)
    with w2:
        concentration = st.slider("Концентрация во время занятий", 1, 5, 3)
        overload = st.slider("Как часто вы ощущаете учебную перегрузку?", 1, 5, 3)
    with w3:
        pass

    st.subheader("6. Цифровые ресурсы")
    d1, d2, d3 = st.columns(3)
    with d1:
        online_use = st.selectbox(
            "Используете ли вы онлайн-ресурсы для подготовки к ЕНТ?",
            [1, 2, 3],
            format_func=lambda x: {1: "Нет", 2: "Иногда", 3: "Регулярно"}[x]
        )
        online_hours = st.selectbox(
            "Сколько часов в неделю вы используете онлайн-ресурсы для учебы?",
            [0, 2, 4, 6, 8, 10],
            format_func=lambda x: f"{x} часов"
        )
    with d2:
        search_materials = st.slider("Как часто вы самостоятельно ищете дополнительные материалы?", 1, 5, 3)
        extra_tests = st.slider("Как часто вы решаете дополнительные тесты или задания сверх школьной программы?", 1, 5, 3)
    with d3:
        weekly_tests = st.number_input("Сколько раз в неделю вы проходите онлайн-тесты или тренировочные задания?", min_value=0, max_value=20, value=3, step=1)
        videos = st.selectbox(
            "Используете ли вы образовательные видео или онлайн-лекции для подготовки?",
            [1, 2, 3],
            format_func=lambda x: {1: "Нет", 2: "Иногда", 3: "Регулярно"}[x]
        )
        online_help = st.slider("Насколько онлайн-ресурсы помогают вам лучше понимать учебный материал?", 1, 5, 3)

    submitted = st.form_submit_button("Рассчитать прогноз")

if submitted:
    user_input = {
        "1. Класс обучения": class_num,
        "2. Возраст": age,
        "3. Пол": gender,
        "4. Есть ли у вас отдельное рабочее место для учебы?": workplace,
        "5. Есть ли у вас репетитор по учебным предметам?": tutor,
        "6. Материальное положение семьи (доход)": income,
        "7, Количество часов дополнительной занятости в неделю (спорт, кружки, работа)": extra_activity,
        "8. Ваши профильные предметы на ЕНТ": profile_subjects,
        "9. Язык обучения": language,
        "10. Среднее количество часов сна в сутки": sleep,
        "11, Общее количество часов самостоятельной учебной работы в день (включая выполнение домашних заданий и подготовку к ЕНТ)": study_hours,
        "12. Количество пропусков занятий за последний месяц": absences,
        "13. Интерес родителей к вашей учебе": parent_interest,
        "14. Эмоциональная поддержка со стороны семьи": family_support,
        "15. Контроль выполнения домашних заданий со стороны родителей": parent_control,
        "16. Давление со стороны родителей относительно результатов ЕНТ": parent_pressure,
        "17. Насколько вам интересна учеба? 1 - совсем не интересна 5 - очень интересна": interest_study,
        "18. Насколько ясно вы понимаете свои учебные цели? 1 - совсем не понимаю 5 - полностью понимаю": goals,
        "19. Насколько вы готовы прилагать дополнительные усилия для учебы? 1 - совсем не готов 5 - полностью готов": effort,
        "20. Как часто вы теряете мотивацию к учебе?": motivation_loss,
        "21. Как часто вы выполняете домашние задания полностью?": hw_full,
        "22. Сколько дней в неделю вы дополнительно повторяете учебный материал?": repeat_days,
        "23. Уровень учебного стресса": stress,
        "24. Тревожность перед экзаменами": anxiety,
        "25. Уровень усталости от учебной нагрузки": fatigue,
        "26. Уверенность в своих учебных способностях": confidence,
        "27. Концентрация во время занятий": concentration,
        "28. Как часто вы ощущаете учебную перегрузку?": overload,
        "29. Используете ли вы онлайн-ресурсы для подготовки к ЕНТ (онлайн-курсы, тестовые платформы, образовательные видео)?": online_use,
        "30. Сколько часов в неделю вы используете онлайн-ресурсы для учебы?": online_hours,
        "31. Как часто вы самостоятельно ищете дополнительные материалы для изучения темы  (видео, статьи, тесты, конспекты)?": search_materials,
        "32. Как часто вы решаете дополнительные тесты или задания сверх школьной программы?": extra_tests,
        "33. Сколько раз в неделю вы проходите онлайн-тесты или тренировочные задания? (Укажите число)": weekly_tests,
        "34. Используете ли вы образовательные видео или онлайн-лекции для подготовки?": videos,
        "35. Насколько онлайн-ресурсы помогают вам лучше понимать учебный материал? 1 - совсем не помогают 5 - значительно помогают": online_help,
        "36. Насколько вы считаете себя самостоятельным в учебе? 1 - совсем не самостоятельный 5 - полностью самостоятельный": independence,
        "37. Как часто вы планируете свою учебную деятельность (расписание, подготовку к экзаменам)?": planning,
        "38. Средний балл за первую четверть (по 5-балльной шкале) (укажите число)": q1_grade,
        "39. Средний балл за вторую четверть (по 5-балльной шкале) (укажите число)": q2_grade,
        "41. Результат предыдущего пробного ЕНТ (ЕНТ который вы сдавали  до последнего)": current_trial,
    }

    try:
        student_df = build_student_df(user_input, model_columns)
        st.subheader("Отладка: данные, которые уходят в модель")
        st.dataframe(student_df.T)
        filled = student_df.notna().sum(axis=1).iloc[0]
        st.write("Количество заполненных признаков:", filled)
        empty_cols = student_df.columns[student_df.iloc[0].isna()].tolist()
        st.write("Пустые признаки:", empty_cols)
        prediction = round(float(model.predict(student_df)[0]), 2)
        risk = risk_level(prediction)
        issues = detect_issues_full(student_df.iloc[0])
        recommendations = generate_recommendations(issues, prediction)
        ai_text = get_ai_recommendation(prediction, risk, issues, recommendations)

        st.subheader("Результат")
        st.write(f"**Прогнозируемый балл ЕНТ:** {prediction}")

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

        st.subheader("Базовые рекомендации")
        if recommendations:
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.write("Базовые рекомендации не сформированы.")

        st.subheader("ИИ-рекомендации")
        st.write(ai_text)

    except Exception as e:
        st.error("Ошибка при расчёте прогноза.")
        st.code(str(e))
