import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
import json

# Модель
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "qwen3-32b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-fhMGj3XMTsnLDUe__ClMLA")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://10.32.15.89:34000/v1")

llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
    temperature=0.1,
    max_retries=3
)

# Промпты и цепочки
science_research_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a scientific research assistant. Your task is to find relevant research articles on a given topic and provide summaries and links."),
    ("human", "Topic: {question}\n\nProvide a structured research summary with relevant article links.")
])

code_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI coding assistant. You can use tools to help you. Write a code correctly. Test your code and give workable code."),
        ("human", "Question: {question}\nIf there is no code to write, say this is not a code writing question."),
    ]
)

writing_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a **writing assistant**. Explain things clearly or rewrite text in a more understandable way."),
        ("human", "User request: {question}\nExplain, rephrase or elaborate in a strict and correct way."),
    ]
)

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a router between three agents: 'science reasearch', 'coding' and 'writing'. Decide which agent is more suitable for the user question.\n- Use 'science reasearch' if there are names of science litrature, research themes\n- Use 'coding' for writing code.\n- Use 'writing' for explanations, rewriting, conceptual questions.\n\nRespond with a JSON object containing 'next_agent' (either 'science reaserch', 'coding' or 'writing') and 'reason' (a short explanation)."),
        ("human", "User question: {question}\n\nRespond in JSON format: {{\"next_agent\": \"science research\" or \"coding\" or \"writing\", \"reason\": \"explanation\"}}"),
    ]
)
science_research_agent = science_research_prompt | llm | StrOutputParser()
code_agent = code_prompt | llm | StrOutputParser()
writing_agent = writing_prompt | llm | StrOutputParser()

class RoutingDecision(BaseModel):
    """LLM output schema for router."""
    next_agent: str = Field(
        description="Which agent should answer: 'science reserch','coding' or 'writer'."
    )
    reason: str = Field(description="Short explanation of the decision.")


json_parser = JsonOutputParser(pydantic_object=RoutingDecision)
router_chain = router_prompt | llm | json_parser

def multi_agent_answer(question: str, verbose: bool = True, log_callback=None) -> str:
    """Top-level function: router -> specialized agent."""
    def log(message):
        if log_callback is not None:
            log_callback(message)
        elif verbose:
            print(message)

    try:
        decision_dict = router_chain.invoke({"question": question})
        # Convert dict to RoutingDecision object if needed
        if isinstance(decision_dict, dict):
            decision = RoutingDecision(**decision_dict)
        else:
            decision = decision_dict
    except Exception as e:
        # Fallback: if parsing fails, try to extract agent from raw response
        log(f"[Router] Error parsing decision: {e}")
        log("[Router] Using fallback: checking if question contains science research keywords...")
        # Simple keyword-based fallback
        science_keywords = [ "science", "research", "search", "science research"]
        coding_keywords = ["a code", "Python", "C++", "C#", "coding", "programme"]
        if any(keyword in question.lower() for keyword in science_keywords):
            decision = RoutingDecision(next_agent="science research", reason="Fallback: detected science litrature keywords")
        elif any(keyword in question.lower() for keyword in coding_keywords):
            decision = RoutingDecision(next_agent="coding", reason="Fallback: default to coding agent")
        else:
            decision = RoutingDecision(next_agent="writing", reason="Fallback: default to writing agent")
    
    log(f"[Router] next_agent = {decision.next_agent!r}")
    log(f"[Router] reason    = {decision.reason}\\n")

    if decision.next_agent.lower().strip() == "science research":
        answer = science_research_agent.invoke({"question": question})
        agent_name = "SCIENCE RESEARCH"

    elif decision.next_agent.lower().strip() == 'coding':
        answer = code_agent.invoke({"question": question})
        agent_name = "CODER"
    else:
        answer = writing_agent.invoke({"question": question})
        agent_name = "WRITING"

    log(f"[{agent_name} AGENT ANSWER]")
    log(answer)
    return answer