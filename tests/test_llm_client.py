import pytest
from src.llm_client import MockClient, get_llm_client, BaseLLMClient

def test_mock_client():
    client = MockClient(response_text="Test completed successfully")
    response = client.generate(prompt="Hello, world!", system_prompt="Sys")
    assert response == "Test completed successfully"
    assert client.last_prompt == "Hello, world!"

def test_factory_fallback():
    config = {"provider": "mock"}
    client = get_llm_client(config, "Qwen3-8B")
    assert isinstance(client, MockClient)
