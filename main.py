rom agent import agent
from langchain_core.messages import HumanMessage
from config import settings
import threading
import logging

logger = logging.getLogger(__name__)

def run_agent(user_input, output_container, debug=False):
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
            if hasattr(last, "content") and last.content:
                output_container.append(last.content)
            else:
                output_container.append("⚠️ Агент повернув порожню відповідь.")
        else:
            output_container.append("⚠️ Немає повідомлень у відповіді.")

    except Exception as e:
        error_msg = f"Помилка агента: {str(e)}"
        logger.exception(error_msg)
        output_container.append(error_msg)


def main():
    print("Research Agent (type 'exit' to quit)")
    print("Debug mode:", "ON" if settings.debug else "OFF")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                print("⚠️ Будь ласка, введіть запит.")
                continue

            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            if user_input.lower() == "debug on":
                settings.debug = True
                print("✅ Debug mode: ON")
                continue

            if user_input.lower() == "debug off":
                settings.debug = False
                print("✅ Debug mode: OFF")
                continue

            output = []
            # Daemon потік - буде завершений при виході з програми
            t = threading.Thread(
                target=run_agent, 
                args=(user_input, output, settings.debug),
                daemon=True
            )
            t.start()
            t.join(timeout=10)

            if t.is_alive():
                print("⏳ Агент завис або не відповідає. Можливо, проблема з інтернетом або інструментами.")
                print("Спробуй інше формулювання або перевір з'єднання.")
                continue

            if output:
                print(f"Agent: {output[-1]}")
            else:
                print("⚠️ Агент не повернув відповідь.")

        except KeyboardInterrupt:
            print("\n\nПрограма завершена користувачем.")
            break
        except Exception as e:
            logger.exception("Неочікувана помилка в main loop")
            print(f"❌ Неочікувана помилка: {e}")


if __name__ == "__main__":
    main()
