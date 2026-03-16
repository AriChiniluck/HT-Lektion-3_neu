import os
import re
from datetime import datetime
import trafilatura
from duckduckgo_search import DDGS
from config import settings


# ---------------------------------------------------------
# Debug helper
# ---------------------------------------------------------
def debug_print(*args):
    if settings.debug:
        print("[DEBUG]", *args)


# ---------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------
def extract_keywords(text: str, max_words: int = 3) -> str:
    """Виділяє ключові слова з тексту для формування імені файлу."""
    cleaned = re.sub(r"[^a-zA-Zа-яА-ЯїієґЇІЄҐ0-9 ]", " ", text)
    words = cleaned.lower().split()

    stopwords = {"що", "як", "коли", "де", "про", "та", "і", "або", "чи", "будь", "будь-який"}
    keywords = [w for w in words if w not in stopwords]

    if not keywords:
        keywords = ["agent_answer"]

    return "_".join(keywords[:max_words])


def generate_filename_from_query(text: str) -> str:
    """Створює ім'я файлу на основі ключових слів."""
    topic = extract_keywords(text)
    date_str = datetime.now().strftime("%Y-%m-%d")
    return f"{topic}_{date_str}.txt"


# ---------------------------------------------------------
# Search tool
# ---------------------------------------------------------
def search_tool(query: str):
    """Пошук у DuckDuckGo."""
    debug_print(f"search_tool called with query: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=settings.max_search_results))
        return results
    except Exception as e:
        return f"search_tool error: {e}"


# ---------------------------------------------------------
# Read URL tool
# ---------------------------------------------------------
def read_tool(url: str):
    """Завантажує та витягує текст зі сторінки."""
    debug_print(f"read_tool called with url: {url}")
    try:
        downloaded = trafilatura.fetch_url(url, timeout=10)
        if not downloaded:
            return "read_tool: failed to download content"

        text = trafilatura.extract(downloaded)
        if not text:
            return "read_tool: failed to extract content"

        return text[:settings.max_url_content_length]
    except Exception as e:
        return f"read_tool error: {e}"


# ---------------------------------------------------------
# Save report tool
# ---------------------------------------------------------
def save_report_tool(filename: str, content: str):
    """Зберігає текст у файл."""
    debug_print(f"save_report_tool called with filename={filename}")
    try:
        os.makedirs(settings.output_dir, exist_ok=True)
        path = os.path.join(settings.output_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Файл збережено: {path}"
    except Exception as e:
        return f"save_report_tool error: {e}"


# ---------------------------------------------------------
# List files tool
# ---------------------------------------------------------
def list_files_tool(directory: str):
    """Показує файли в директорії."""
    try:
        return os.listdir(directory)
    except Exception as e:
        return f"list_files_tool error: {e}"


# ---------------------------------------------------------
# Read file tool
# ---------------------------------------------------------
def read_file_tool(path: str):
    """Читає файл."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"read_file_tool error: {e}"


# ---------------------------------------------------------
# Calculator tool
# ---------------------------------------------------------
def calculate_tool(expression: str):
    """Обчислює математичний вираз."""
    try:
        return eval(expression, {"__builtins__": {}})
    except Exception as e:
        return f"calculate_tool error: {e}"
