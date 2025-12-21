# app_with_react.py
import streamlit as st
import asyncio
import json
import os
from typing import Optional
from langchain_openai import ChatOpenAI
from react_coordinator import ReActCoordinator, ReActState, AgentType
from agent_factory import AgentFactory

# Инициализация LLM (используем ваш способ подключения)
@st.cache_resource
def initialize_system():
    # Получаем настройки из Streamlit secrets или переменных окружения
    MODEL_NAME = st.secrets.get("OPENAI_MODEL_NAME", os.getenv("OPENAI_MODEL_NAME", "qwen3-32b"))
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "sk-fhMGj3XMTsnLDUe__ClMLA"))
    OPENAI_API_BASE = st.secrets.get("OPENAI_API_BASE", os.getenv("OPENAI_API_BASE", "http://10.32.15.89:34000/v1"))
    
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        temperature=0.1,
        max_retries=3
    )
    
    # Создаем фабрику агентов и координатор
    agent_factory = AgentFactory(llm)
    agents = agent_factory.get_all_agents()
    react_coordinator = ReActCoordinator(llm, agents)
    
    return react_coordinator, agents, llm

# Основной интерфейс Streamlit
st.set_page_config(
    page_title="ReAct Loop Multi-Agent System",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 ReAct Loop Multi-Agent System")
st.markdown("""
Система использует ReAct Loop для итеративного улучшения ответов:
1. **Анализ** запроса и выбор агента
2. **Выполнение** агентом
3. **Оценка** качества ответа
4. **Уточнение** при необходимости
5. **Форматирование** финального ответа
""")

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки ReAct Loop")
    
    max_iterations = st.slider(
        "Максимум итераций",
        min_value=1,
        max_value=10,
        value=3,
        help="Сколько раз можно уточнять ответ"
    )
    
    quality_threshold = st.slider(
        "Порог качества (%)",
        min_value=50,
        max_value=100,
        value=80,
        help="Минимальный балл для принятия ответа"
    )
    
    show_reasoning = st.checkbox("Показать цепочку рассуждений", value=True)
    auto_mode = st.checkbox("Автоматический режим", value=True)
    
    st.divider()
    st.markdown("### 📊 Статистика")
    if 'react_state' in st.session_state:
        state: ReActState = st.session_state.react_state
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Итерации", state.current_iteration)
        with col2:
            if state.agent_responses:
                last_score = state.agent_responses[-1].confidence_score * 100
                st.metric("Качество", f"{last_score:.1f}%")

# Основная область
query = st.text_area(
    "Введите ваш запрос:",
    height=120,
    placeholder="Например: 'Объясни архитектуру нейронной сети и приведи пример реализации на PyTorch'",
    key="query_input"
)

# Кнопки управления
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    if st.button("🚀 Запустить ReAct Loop", type="primary", use_container_width=True):
        if not query.strip():
            st.error("Введите запрос")
        else:
            # Инициализируем систему
            with st.spinner("Инициализация системы..."):
                react_coordinator, agents, llm = initialize_system()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Создаем контейнеры для отображения процесса
            process_container = st.container()
            reasoning_container = st.container()
            result_container = st.container()
            
            # Запускаем ReAct Loop
            try:
                async def run_react():
                    # Устанавливаем максимальное количество итераций
                    react_coordinator.max_iterations = max_iterations
                    return await react_coordinator.run_react_loop(query)
                
                # Запуск асинхронной функции
                state = asyncio.run(run_react())
                st.session_state.react_state = state
                
                # Отображение процесса
                with process_container:
                    st.subheader("📈 Процесс ReAct Loop")
                    
                    for i, response in enumerate(state.agent_responses):
                        with st.expander(f"Итерация {i+1} - {response.agent_type.value}", 
                                       expanded=i == len(state.agent_responses)-1):
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.markdown("**Ответ агента:**")
                                if response.agent_type == AgentType.CODING:
                                    st.code(response.content)
                                else:
                                    st.markdown(response.content)
                            
                            with col_b:
                                st.metric("Уверенность", f"{response.confidence_score*100:.1f}%")
                                st.markdown("**Проблемы:**")
                                for issue in response.issues:
                                    st.caption(f"• {issue}")
                            
                            if i < len(state.agent_responses) - 1:
                                st.info(f"➡️ Ответ требует доработки. Переход к итерации {i+2}")
                
                # Цепочка рассуждений
                if show_reasoning and state.reasoning_chain:
                    with reasoning_container:
                        st.subheader("🤔 Цепочка рассуждений")
                        for i, reasoning in enumerate(state.reasoning_chain):
                            st.write(f"{i+1}. {reasoning}")
                
                # Финальный результат
                with result_container:
                    st.divider()
                    st.subheader("✅ Финальный ответ")
                    
                    if state.final_answer:
                        # Определяем тип контента для форматирования
                        if "```" in state.final_answer or "import " in state.final_answer:
                            st.code(state.final_answer, language="python")
                        else:
                            st.markdown(state.final_answer)
                        
                        # Метрики
                        col_met1, col_met2, col_met3 = st.columns(3)
                        with col_met1:
                            st.metric("Всего итераций", state.current_iteration)
                        with col_met2:
                            st.metric("Использовано агентов", 
                                    len(set(r.agent_type for r in state.agent_responses)))
                        with col_met3:
                            if state.agent_responses:
                                final_confidence = state.agent_responses[-1].confidence_score * 100
                                st.metric("Итоговое качество", f"{final_confidence:.1f}%")
                    
                    # Кнопка для сохранения истории
                    if st.button("💾 Сохранить историю процесса"):
                        history = {
                            "query": state.original_query,
                            "iterations": state.current_iteration,
                            "final_answer": state.final_answer,
                            "reasoning_chain": state.reasoning_chain,
                            "agent_responses": [
                                {
                                    "iteration": i+1,
                                    "agent": r.agent_type.value,
                                    "content": r.content,
                                    "confidence": r.confidence_score,
                                    "issues": r.issues
                                }
                                for i, r in enumerate(state.agent_responses)
                            ]
                        }
                        
                        st.download_button(
                            label="Скачать JSON",
                            data=json.dumps(history, ensure_ascii=False, indent=2),
                            file_name="react_loop_history.json",
                            mime="application/json"
                        )
            
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")
                st.info("Попробуйте упростить запрос или проверьте подключение к LLM")

with col2:
    if st.button("⏹️ Остановить", type="secondary", use_container_width=True):
        st.info("Остановка процесса...")
        if 'react_state' in st.session_state:
            st.session_state.react_state.is_complete = True

with col3:
    if st.button("🧹 Очистить", use_container_width=True):
        for key in ['react_state', 'query_input']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Панель с примерами запросов

# Информация о системе

st.caption("""
🔄 ReAct Loop Multi-Agent System | ФТИИ | ИИвП | Гончаренко Данила | J4250
""")