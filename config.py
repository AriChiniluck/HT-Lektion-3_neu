from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import SecretStr, Field, field_validator
import os


class Settings(BaseSettings):
    """Application configuration loaded from .env file."""
    
    # --- OpenAI / LM Studio ---
    openai_api_key: SecretStr = Field(
        ...,  # Обов'язкове поле
        description="OpenAI API key (must start with 'sk-')"
    )
    
    model_name: str = Field(
        default="gpt-4o",
        description="LLM model name",
    )

    # --- Output directory for reports ---
    output_dir: str = Field(
        default="output",
        description="Directory for saving reports"
    )

    # --- Search settings ---
    max_search_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of search results (1-50)"
    )
    
    max_url_content_length: int = Field(
        default=5000,
        ge=100,
        le=100000,
        description="Maximum content length from URLs (100-100000 bytes)"
    )

    # --- Debug mode ---
    debug: bool = Field(
        default=False,
        description="Enable debug logging"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: SecretStr) -> SecretStr:
        """Перевіряєм��, чи OpenAI API ключ має правильний формат."""
        key_str = v.get_secret_value()
        if not key_str.startswith("sk-"):
            raise ValueError("❌ OpenAI API ключ має починатися з 'sk-'")
        if len(key_str) < 40:
            raise ValueError("❌ OpenAI API ключ занадто короткий (мін. 40 символів)")
        return v

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Перевіряємо, чи директорія безпечна."""
        abs_path = os.path.abspath(v)
        
        # Заборонені директорії
        forbidden_dirs = {"/", "/etc", "/sys", "/proc", "/root", "/home", "/bin", "/sbin"}
        
        if abs_path in forbidden_dirs:
            raise ValueError(f"❌ Директорія '{v}' заборонена з причин безпеки")
        
        # Перевіримо, чи можемо писати в цю директорію
        try:
            os.makedirs(abs_path, exist_ok=True)
            test_file = os.path.join(abs_path, ".write_test")
            Path(test_file).touch()
            os.remove(test_file)
        except (PermissionError, OSError):
            raise ValueError(f"❌ Немає дозволу на запис у директорію '{v}'")
        
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Перевіряємо назву моделі."""
        if not v or len(v) < 3:
            raise ValueError("❌ Назва моделі занадто коротка")
        if len(v) > 100:
            raise ValueError("❌ Назва моделі занадто довга")
        return v


# Завантажуємо конфігурацію
try:
    settings = Settings()
    
    # Перевіримо наявність .env файлу
    env_path = Path(".env")
    if not env_path.exists():
        raise FileNotFoundError(
            "\n"
            "❌ .env файл не знайдений!\n"
            "\n"
            "Створіть файл '.env' у кореневому каталозі проекту з наступним вмістом:\n"
            "\n"
            "───────────────────────────────────────\n"
            "openai_api_key=sk-your-api-key-here\n"
            "model_name=gpt-4o\n"
            "output_dir=output\n"
            "debug=false\n"
            "───────────────────────────────────────\n"
            "\n"
            "Отримайте API ключ на: https://platform.openai.com/api-keys\n"
        )
except Exception as e:
    print(f"❌ Помилка конфігурації: {e}")
    exit(1)


# =========================================
# SYSTEM PROMPT - БЕЗ {topic}!
# =========================================
SYSTEM_PROMPT = """You are a Research Agent powered by ReAct (Reasoning + Acting).

Your task is to answer user questions by gathering information, analyzing it, and providing comprehensive responses.

CAPABILITIES:

You have access to the following tools:

1. search_tool(query: str) — Search the internet using DuckDuckGo
2. read_tool(url: str) — Extract and read text from a webpage
3. save_report_tool(filename: str, content: str) — Save content to a text file
4. list_files_tool(directory: str) — List files in a directory
5. read_file_tool(path: str) — Read content from a saved file
6. calculate_tool(expression: str) — Perform mathematical calculations

WORKFLOW (ReAct Pattern):

1. THINK: Understand the user's request
2. REASON: Determine which tools you need to use
3. ACT: Use appropriate tools to gather information
4. OBSERVE: Analyze the results
5. RESPOND: Provide a comprehensive answer

WHEN TO USE EACH TOOL:

- Use search_tool when you need to find current information, research topics, or answer "what is" questions
- Use read_tool when you have a specific URL and need detailed information from it
- Use save_report_tool when explicitly asked to save, create a report, or export content
- Use calculate_tool for mathematical calculations and expressions
- Use list_files_tool and read_file_tool to work with existing files

CRITICAL RULES:

DO:
�� Use tools when you need external information
✅ Search for information you don't have
✅ Be explicit about what tools you're using
✅ Provide accurate, well-researched answers
✅ Admit limitations and ask for clarification when needed

DON'T:
❌ Fabricate information without using tools
❌ Skip tool usage when explicitly requested
❌ Make up statistics, quotes, or references
❌ Pretend to know current information you don't have

OUTPUT PREFERENCES:
- Respond in the same language as the user (Ukrainian or English)
- Format responses clearly with proper structure
- Cite sources when providing information from tools
- Provide detailed explanations, not just brief answers

Remember: Your goal is to be a helpful research assistant that provides accurate, well-researched answers.
"""
