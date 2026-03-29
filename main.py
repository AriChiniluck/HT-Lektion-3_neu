from agent import agent
from langchain_core.messages import HumanMessage
from config import settings
import threading
import logging

logger = logging.getLogger(__name__)

def run_agent(user_input, output_container):
    try:
        result = agent.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "step_count": 0,
            },
            config={"configurable": {"thread_id": "session-1"}},
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
            # Daemon потік з більшим timeout
            t = threading.Thread(
                target=run_agent,
                args=(user_input, output),
                daemon=True
            )
            t.start()
            # ✅ ВИПРАВЛЕНО: Збільшено timeout до 30 секунд
            t.join(timeout=30)

            if t.is_alive():
                print("⏳ Агент все ще обробляє запит...")
                # Чекаємо ще 20 секунд
                t.join(timeout=20)
                
                if t.is_alive():
                    print("⏳ Агент завис або не відповідає. Можливо, проблема з інтернетом або інструментами.")
                    print("Спробуй інше формулювання або перевір з'єднання.")
                    continue

            # ✅ ВИПРАВЛЕНО: Перевіряємо результат ПІСЛЯ того, як потік завершився
            if output:
                print(f"Agent: {output[-1]}\n")
            else:
                print("⚠️ Агент не повернув відповідь.\n")

        except KeyboardInterrupt:
            print("\n\nПрограма завершена користувачем.")
            break
        except Exception as e:
            logger.exception("Неочікувана помилка в main loop")
            print(f"❌ Неочікувана помилка: {e}")


if __name__ == "__main__":
    main()
