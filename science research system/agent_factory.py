# agent_factory.py (обновленный)
from typing import Dict, Any
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class AgentType(str, Enum):
    RESEARCH = "research"
    CODING = "coding"
    WRITING = "writing"

class AgentFactory:
    def __init__(self, llm_client):  # Теперь принимаем ChatOpenAI напрямую
        self.llm = llm_client
        self.agents = {}
        
    def create_research_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты - научный исследовательский ассистент. 
            Твои ответы должны быть:
            1. Структурированными и хорошо организованными
            2. Содержать релевантные ссылки на источники
            3. Основанными на актуальных исследованиях
            4. С четкими выводами и рекомендациями"""),
            ("human", "Исследовательский запрос: {question}")
        ])
        return prompt | self.llm | StrOutputParser()
    
    def create_coding_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты - эксперт по программированию.
            Твои ответы должны:
            1. Содержать рабочий, готовый к использованию код
            2. Включать объяснения ключевых моментов
            3. Учитывать лучшие практики и производительность
            4. Предлагать альтернативные решения если уместно"""),
            ("human", "Запрос на код: {question}")
        ])
        return prompt | self.llm | StrOutputParser()
    
    def create_writing_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты - профессиональный писатель и редактор.
            Твои ответы должны быть:
            1. Четкими и понятными
            2. Хорошо структурированными
            3. Адаптированными под целевую аудиторию
            4. С правильной грамматикой и стилем"""),
            ("human", "Текст для обработки: {question}")
        ])
        return prompt | self.llm | StrOutputParser()
    
    def get_all_agents(self) -> Dict[AgentType, Any]:
        if not self.agents:
            self.agents = {
                AgentType.RESEARCH: self.create_research_agent(),
                AgentType.CODING: self.create_coding_agent(),
                AgentType.WRITING: self.create_writing_agent(),
            }
        return self.agents