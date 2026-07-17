import os
import pytest
from src.prompt_manager import PromptManager

def test_prompt_manager(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    
    # Write a dummy prompt template
    prompt_file = prompts_dir / "Prompt_test.txt"
    prompt_file.write_text("Hello {name}, your task is {task}.", encoding="utf-8")
    
    manager = PromptManager(prompts_dir=str(prompts_dir))
    formatted = manager.get_prompt("Prompt_test", name="Researcher", task="repro")
    
    assert formatted == "Hello Researcher, your task is repro."

    # Verify formatting key checks fail gracefully
    with pytest.raises(KeyError):
        manager.get_prompt("Prompt_test", name="Researcher")
