import yaml
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import AzureChatOpenAI
import openai

openai.api_type = "azure"
openai.api_key = "Your api key"
openai.api_base = "Your api base"
openai.api_version = "2024-06-01"


class BaseAgent:
    def __init__(self, name, prompt_dir="agents/prompts"):
        self.name = name
        self.prompt_path = Path(prompt_dir) / f"{name.lower()}.yaml"
        self.prompts = self._load_prompts()
        self.chat = AzureChatOpenAI(
            deployment_name="gpt-4o",
            model_name="gpt-4o",
            openai_api_key=openai.api_key,
            openai_api_base=openai.api_base,
            openai_api_version=openai.api_version,
            openai_api_type=openai.api_type,
        )

    def _load_prompts(self):
        with open(self.prompt_path, "r") as f:
            return yaml.safe_load(f)

    def get_prompt_template(self, key):
        section = self.prompts[key]
        return PromptTemplate(
            input_variables=section["input_variables"],
            template=section["template"]
        )

    def format_prompt(self, key, **kwargs):
        return self.get_prompt_template(key).format(**kwargs)

    def build_dialogue(self, context_vars):
        raise NotImplementedError("Each agent must define its dialogue strategy.")
