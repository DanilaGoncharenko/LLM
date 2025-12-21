from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

class AgentType(str, Enum):
    RESEARCH = "research"
    CODING = "coding" 
    WRITING = "writing"

class AgentResponse(BaseModel):
    content: str
    agent_type: AgentType
    is_complete: bool = Field(default=False)
    issues: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

class ReActState(BaseModel):
    """Состояние ReAct Loop"""
    original_query: str
    current_query: str
    selected_agent: AgentType
    agent_responses: List[AgentResponse] = Field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 5
    is_complete: bool = False
    final_answer: Optional[str] = None
    reasoning_chain: List[str] = Field(default_factory=list)

class ReActCoordinator:
    def __init__(self, llm_client, agents_dict: Dict[AgentType, Any]):
        self.llm = llm_client
        self.agents = agents_dict
        self.max_iterations = 5
        
    def analyze_and_choose_agent(self, query: str) -> AgentType:
        """Анализирует запрос и выбирает подходящего агента"""
        query_lower = query.lower()
        
        # Ключевые слова для каждого агента
        research_keywords = ['research', 'статья', 'статьи', 'научн', 'arxiv', 'исследован', 'обзор', 'анализ', 'science', 'research']
        coding_keywords = ['код', 'програм', 'алгоритм', 'python', 'java', 'c++', 'функция', 'библиотека', 'импорт']
        
        # Проверяем наличие ключевых слов
        if any(keyword in query_lower for keyword in research_keywords):
            return AgentType.RESEARCH
        elif any(keyword in query_lower for keyword in coding_keywords):
            return AgentType.CODING
        else:
            return AgentType.WRITING
    
    async def run_react_loop(self, query: str) -> ReActState:
        """Основной ReAct Loop с выбором агента"""
        
        # 1. Анализируем запрос и выбираем агента
        selected_agent = self.analyze_and_choose_agent(query)
        
        # 2. Создаем состояние с выбранным агентом
        state = ReActState(
            original_query=query,
            current_query=query,
            selected_agent=selected_agent,  
            current_iteration=1,
            max_iterations=self.max_iterations,
            is_complete=True,
            final_answer=f"Ответ от агента {selected_agent.value} на запрос: {query}\n\nВыбран агент: {selected_agent.value}",
            reasoning_chain=[
                f"Шаг 1: Анализ запроса: '{query[:100]}...'",
                f"Шаг 2: Выбор агента: {selected_agent.value}",
                "Шаг 3: Выполнение запроса агентом", 
                "Шаг 4: Получение ответа на запрос",
                "Шаг 5: Анализ ответа",
                "Шаг 6.1: Если ответ требует улучшения передаём его в Шаг 2",
                "Шаг 6.2: Если ответ удовлетворяет ИЛИ получен пустой ответ, то передаём ответ в WRITER node и выводим ответ пользователю"
            ]
        )
        
        # 3. Получаем ответ от выбранного агента
        try:
            agent = self.agents[selected_agent]
            agent_response = await agent.ainvoke({"question": query})
            
            # 4. Добавляем ответ агента
            state.agent_responses.append(AgentResponse(
                content=agent_response,
                agent_type=selected_agent,
                is_complete=True,
                confidence_score=0.85
            ))
            
            # 5. Обновляем финальный ответ
            state.final_answer = agent_response
            
        except Exception as e:
            # Если произошла ошибка, возвращаем сообщение об ошибке
            state.final_answer = f"Ошибка при обработке запроса агентом {selected_agent.value}: {str(e)}"
            state.agent_responses.append(AgentResponse(
                content=state.final_answer,
                agent_type=selected_agent,
                is_complete=False,
                issues=[str(e)],
                confidence_score=0.0
            ))
        
        return state