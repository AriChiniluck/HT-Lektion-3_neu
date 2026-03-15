from typing import List, Dict, Any
import concurrent.futures

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from tools import (
    extract_keywords,
    generate_filename_from_query,
    search_tool,
    read_tool,
    save_report_tool,
    list_files_tool,
    read_file_tool,
    calculate_tool,
)
from config import settings, SYSTEM_PROMPT


# ---------------------------------------------------------
# Debug helper
# ---------------------------------------------------------
def debug_print(*args):
    if settings.debug:
        print("[DEBUG]", *args)


# ---------------------------------------------------------
# LangChain tools wrappers
# ---------------------------------------------------------
@tool
def search_tool_lc(query: str) -> Any:
    """Виконує пошук у DuckDuckGo."""
    return search_tool(query)

@tool
def read_tool_lc(url: str) -> Any:
    """Завантажує та витягує текст зі сторінки."""    
    return read_tool(url)

@tool
def save_report_tool_lc(filename: str, content: str) -> Any:
    """Зберігає текст у файл."""
    return save_report_tool(filename, content)

@tool
def list_files_tool_lc(directory: str) -> Any:
    """Показує файли в директорії."""
    return list_files_tool(directory)

@tool
def read_file_tool_lc(path: str) -> Any:
    """Читає файл і повертає його вміст."""
    return read_file_tool(path)

@tool
def calculate_tool_lc(expression: str) -> Any:
    """Обчислює математичний вираз."""
    return calculate_tool(expression)


TOOLS_LC = [
    search_tool_lc,
    read_tool_lc,
    save_report_tool_lc,
    list_files_tool_lc,
    read_file_tool_lc,
    calculate_tool_lc,
]

TOOLS_MAP = {
    "search_tool_lc": search_tool,
    "read_tool_lc": read_tool,
    "save_report_tool_lc": save_report_tool,
    "list_files_tool_lc": list_files_tool,
    "read_file_tool_lc": read_file_tool,
    "calculate_tool_lc": calculate_tool,
}


# ---------------------------------------------------------
# LLMs
# ---------------------------------------------------------
llm_tools = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    temperature=0.2,
).bind_tools(TOOLS_LC)

llm_plain = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    temperature=0.2,
)


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{input}"),
])


# ---------------------------------------------------------
# State
# ---------------------------------------------------------
class AgentState(dict):
    messages: List[BaseMessage]
    step_count: int


# ---------------------------------------------------------
# Agent node
# ---------------------------------------------------------
def agent_node(state: AgentState) -> AgentState:
    debug_print("\n=== AGENT NODE ===")

    if state.get("step_count", 0) > 12:
        return {
            "messages": state["messages"] + [
                AIMessage(content="⚠️ Зупиняю цикл: забагато кроків.")
            ],
            "step_count": 0,
        }

    last_user = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        "Опиши свій запит."
    )

    formatted = prompt.format_messages(input=last_user)

    # Примушуємо модель викликати інструмент
    for _ in range(3):
        response = llm_tools.invoke(formatted)
        if getattr(response, "tool_calls", None):
            break
        formatted = formatted + [
            HumanMessage(content="Ти МАЄШ викликати хоча б один інструмент.")
        ]

    return {
        "messages": state["messages"] + [response],
        "step_count": state.get("step_count", 0) + 1,
    }


# ---------------------------------------------------------
# Tool node
# ---------------------------------------------------------
def tool_node(state: AgentState) -> AgentState:
    debug_print("\n=== TOOL NODE ===")

    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return state

    new_messages = state["messages"][:]
    tool_results = []

    # Виконуємо інструменти
    for call in last.tool_calls:
        tool_name = call["name"]
        args = call.get("args", {})

        tool_fn = TOOLS_MAP.get(tool_name)
        if not tool_fn:
            new_messages.append(AIMessage(content=f"Невідомий інструмент: {tool_name}"))
            continue

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(tool_fn, **args)
            try:
                result = future.result(timeout=12)
            except concurrent.futures.TimeoutError:
                result = "Tool timeout."

        tool_results.append(str(result))
        new_messages.append(AIMessage(content=str(result)))

    # Генеруємо фінальну відповідь
    summary_prompt = [
        HumanMessage(content="Сформуй підсумкову відповідь на основі результатів:"),
        HumanMessage(content="\n\n".join(tool_results)),
    ]
    final_answer_msg = llm_plain.invoke(summary_prompt)
    final_answer = final_answer_msg.content

    new_messages.append(final_answer_msg)

    # Зберігаємо у файл за ключовими словами відповіді
    filename = generate_filename_from_query(final_answer)
    save_result = save_report_tool(filename=filename, content=final_answer)

    new_messages.append(AIMessage(content=save_result))

    return {
        "messages": new_messages,
        "step_count": state.get("step_count", 0),
    }


# ---------------------------------------------------------
# Graph
# ---------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tool", tool_node)

workflow.set_entry_point("agent")
workflow.add_edge("agent", "tool")

memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)
