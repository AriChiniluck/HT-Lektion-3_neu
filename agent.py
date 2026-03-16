from typing import List, Dict, Any, Literal
import concurrent.futures

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
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
def search_tool_lc(query: str) -> str:
    """Виконує пошук у DuckDuckGo."""
    return search_tool(query)

@tool
def read_tool_lc(url: str) -> str:
    """Завантажує та витягує текст зі сторінки."""    
    return read_tool(url)

@tool
def save_report_tool_lc(filename: str, content: str) -> str:
    """Зберігає текст у файл."""
    return save_report_tool(filename, content)

@tool
def list_files_tool_lc(directory: str) -> str:
    """Показує файли в директорії."""
    return list_files_tool(directory)

@tool
def read_file_tool_lc(path: str) -> str:
    """Читає файл і повертає його вміст."""
    return read_file_tool(path)

@tool
def calculate_tool_lc(expression: str) -> str:
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

# Правильна map для реальних назв LangChain tools
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
# Prompt - ВИПРАВЛЕНО: Тільки {input}
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
# Agent node - вирішує, викликати інструмент чи ні
# ---------------------------------------------------------
def agent_node(state: AgentState) -> AgentState:
    debug_print("\n=== AGENT NODE ===")

    if state.get("step_count", 0) > 10:
        debug_print("⚠️ Зупиняю цикл: забагато кроків.")
        return {
            "messages": state["messages"] + [
                AIMessage(content="⚠️ Зупиняю ��икл: забагато кроків.")
            ],
            "step_count": state.get("step_count", 0) + 1,
        }

    last_user = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        "Опиши свій запит."
    )

    # ВИПРАВЛЕНО: format_messages замість format
    formatted = prompt.format_messages(input=last_user)
    response = llm_tools.invoke(formatted)

    debug_print(f"Response type: {type(response)}")
    debug_print(f"Tool calls: {getattr(response, 'tool_calls', None)}")

    return {
        "messages": state["messages"] + [response],
        "step_count": state.get("step_count", 0) + 1,
    }


# ---------------------------------------------------------
# Conditional edge - визначає, чи викликати tool_node
# ---------------------------------------------------------
def should_continue(state: AgentState) -> Literal["tool", "end"]:
    """Перевіряємо, потрібно ли виконувати інструменти."""
    last = state["messages"][-1]
    
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        debug_print("→ Переходимо до TOOL NODE")
        return "tool"
    
    debug_print("→ Завершуємо граф (END)")
    return "end"


# ---------------------------------------------------------
# Tool node - виконує інструменти
# ---------------------------------------------------------
def tool_node(state: AgentState) -> AgentState:
    debug_print("\n=== TOOL NODE ===")

    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return state

    new_messages = list(state["messages"])

    # Виконуємо інструменти
    for call in last.tool_calls:
        tool_name = call["name"]
        args = call.get("args", {})
        tool_id = call.get("id")

        debug_print(f"Executing tool: {tool_name} with args: {args}")

        tool_fn = TOOLS_MAP.get(tool_name)
        if not tool_fn:
            new_messages.append(ToolMessage(
                tool_call_id=tool_id,
                content=f"❌ Невідомий інструмент: {tool_name}"
            ))
            continue

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(tool_fn, **args)
                result = future.result(timeout=15)
        except concurrent.futures.TimeoutError:
            result = "⏳ Timeout: інструмент не відповідає за 15 секунд."
        except Exception as e:
            result = f"❌ Помилка інструменту: {str(e)}"

        # Правильно додаємо ToolMessage
        new_messages.append(ToolMessage(
            tool_call_id=tool_id,
            content=str(result)
        ))
        debug_print(f"Tool result: {str(result)[:100]}...")

    return {
        "messages": new_messages,
        "step_count": state.get("step_count", 0),
    }


# ---------------------------------------------------------
# Summarizer node - генерує фінальну відповідь
# ---------------------------------------------------------
def summarizer_node(state: AgentState) -> AgentState:
    debug_print("\n=== SUMMARIZER NODE ===")

    # Збираємо результати інструментів
    tool_results = [
        msg.content for msg in state["messages"]
        if isinstance(msg, ToolMessage)
    ]

    if not tool_results:
        debug_print("Немає результатів інструментів.")
        return state

    summary_prompt = [
        HumanMessage(content="На основі результатів пошуку, сформуй детальну та структуровану відповідь українською мовою:"),
        HumanMessage(content="\n\n".join(tool_results[:5])),  # Обмежуємо до 5 результатів
    ]

    try:
        final_answer_msg = llm_plain.invoke(summary_prompt)
        debug_print(f"Final answer: {final_answer_msg.content[:100]}...")
        return {
            "messages": state["messages"] + [final_answer_msg],
            "step_count": state.get("step_count", 0),
        }
    except Exception as e:
        debug_print(f"Error in summarizer: {e}")
        error_msg = AIMessage(content=f"❌ Помилка при формуванні відповіді: {str(e)}")
        return {
            "messages": state["messages"] + [error_msg],
            "step_count": state.get("step_count", 0),
        }


# ---------------------------------------------------------
# Save node - зберігає результат у файл
# ---------------------------------------------------------
def save_node(state: AgentState) -> AgentState:
    debug_print("\n=== SAVE NODE ===")

    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.content:
        return state

    try:
        filename = generate_filename_from_query(last.content)
        result = save_report_tool(filename=filename, content=last.content)
        debug_print(f"Saved to: {result}")
    except Exception as e:
        debug_print(f"❌ Error saving file: {e}")

    return state


# ---------------------------------------------------------
# Build Graph
# ---------------------------------------------------------
workflow = StateGraph(AgentState)

# Додаємо всі nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tool", tool_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("save", save_node)

# Визначаємо потік
workflow.set_entry_point("agent")

# Умовний перехід: agent → tool або end
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tool": "tool",
        "end": "summarizer"  # Якщо немає tool_calls, йдемо до summarizer
    }
)

# tool → summarizer (завжди)
workflow.add_edge("tool", "summarizer")

# summarizer → save (завжди)
workflow.add_edge("summarizer", "save")

# save → END
workflow.add_edge("save", END)

memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)
