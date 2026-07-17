import os
from typing import Dict, Any

class PromptManager:
    """
    Manages loading, caching, and formatting prompt templates from disk.
    Ensures no prompt text templates are hardcoded in modules.
    """
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self._cache: Dict[str, str] = {}

    def _load_prompt_file(self, name: str) -> str:
        """
        Loads prompt content from file.
        """
        if name in self._cache:
            return self._cache[name]
        
        # Normalize name (append extension if missing)
        filename = name if name.endswith(".txt") else f"{name}.txt"
        filepath = os.path.join(self.prompts_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prompt template file not found: {filepath}")
            
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        self._cache[name] = content
        return content

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Retrieves the prompt template and formats it with keywords.
        """
        template = self._load_prompt_file(prompt_name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Missing required parameter '{e.args[0]}' for formatting prompt '{prompt_name}'")
