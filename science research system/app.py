import streamlit as st
from multi_agent_system import multi_agent_answer

st.title("Мульти-агентная системa")

question = st.text_input("Введите ваш запрос:")

if st.button("Отправить"):
    if question:
        # Создадим область для вывода логов
        log_container = st.empty()
        logs = []

        def log_callback(log_message):
            logs.append(log_message)
            # Обновляем область с логами
            log_container.text_area("Логи", value="\\n".join(logs), height=200)

        # Вызываем нашу систему с log_callback
        answer = multi_agent_answer(question, verbose=False, log_callback=log_callback)

        # Выводим ответ
        st.subheader("Ответ:")
        st.text_area("", value=answer, height=300)
    else:
        st.warning("Пожалуйста, введите запрос.")