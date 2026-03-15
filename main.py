from agent import agent
from langchain_core.messages import HumanMessage
from config import settings
import threading


def run_agent(user_input, output_container):
    try:
        result = agent.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "step_count": 0,
            },
            config={"thread_id": "session-1"},
        )

        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            # LangGraph повертає BaseMessage, тип перевіряємо через isinstance
            if hasattr(last, "content"):
                output_container.append(last.content)

    except Exception as e:
        output_container.append(f"Помилка агента: {e}")


def main():
    print("Research Agent (type 'exit' to quit)")
    print("Debug mode:", "ON" if settings.debug else "OFF")
    print("-" * 40)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "debug on":
            settings.debug = True
            print("Debug mode: ON")
            continue

        if user_input.lower() == "debug off":
            settings.debug = False
            print("Debug mode: OFF")
            continue

        output = []
        t = threading.Thread(target=run_agent, args=(user_input, output))
        t.start()
        t.join(timeout=10)

        if t.is_alive():
            print("⏳ Агент завис або не відповідає. Можливо, проблема з інтернетом або інструментами.")
            print("Спробуй інше формулювання або перевір з’єднання.")
            continue

        if output:
            print("Agent:", output[-1])
        else:
            print("⚠️ Агент не повернув відповідь.")


if __name__ == "__main__":
    main()