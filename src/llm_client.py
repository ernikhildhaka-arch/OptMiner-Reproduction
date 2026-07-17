import abc
import os
import requests
import time
from typing import Dict, Any, Optional

class BaseLLMClient(abc.ABC):
    """
    Abstract base class for all LLM clients.
    All inference components in the repository must interact ONLY with this interface.
    """
    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generates text completion based on the given prompt and parameters.
        """
        pass

class OpenAIClient(BaseLLMClient):
    """
    OpenAI API compatibility client.
    """
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.endpoint = endpoint or "https://api.openai.com/v1/chat/completions"
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if not self.api_key:
            raise ValueError("OpenAI API Key is missing.")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p", 0.8),
            "max_tokens": kwargs.get("max_tokens", 2048)
        }
        response = requests.post(self.endpoint, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek API client.
    """
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.endpoint = endpoint or "https://api.deepseek.com/v1/chat/completions"
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if not self.api_key:
            raise ValueError("DeepSeek API Key is missing.")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p", 0.8),
            "max_tokens": kwargs.get("max_tokens", 2048)
        }
        response = requests.post(self.endpoint, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

class vLLMClient(BaseLLMClient):
    """
    vLLM hosting client with retry support and configuration handling.
    """
    def __init__(self, endpoint: str, model: str, api_key: Optional[str] = None, timeout: int = 60, max_retries: int = 3):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key or ""
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p", 0.8),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": False
        }
        if "stop" in kwargs:
            data["stop"] = kwargs["stop"]
            
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                print(f"[vLLMClient] Attempt {attempt+1}/{self.max_retries} failed: {e}")
                time.sleep(2 ** attempt)
                
        raise last_error

class HuggingFaceClient(BaseLLMClient):
    """
    Local HuggingFace model client using transformers.
    Supports 4-bit and 8-bit quantization via bitsandbytes.
    """
    def __init__(self, model_name: str, load_in_8bit: bool = False, load_in_4bit: bool = False, device: Optional[str] = None):
        self.model_name = model_name
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.device_override = device
        self.tokenizer = None
        self.model = None
        self.device = None

    def _lazy_init(self):
        if self.model is not None:
            return
            
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.device = self.device_override or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[HuggingFaceClient] Loading {self.model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        kwargs = {}
        if self.device == "cuda":
            if self.load_in_4bit:
                kwargs["load_in_4bit"] = True
                kwargs["device_map"] = "auto"
            elif self.load_in_8bit:
                kwargs["load_in_8bit"] = True
                kwargs["device_map"] = "auto"
            else:
                kwargs["torch_dtype"] = torch.float16
                kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **kwargs
        )
        if self.device == "cuda" and not self.load_in_4bit and not self.load_in_8bit:
            self.model = self.model.to(self.device)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        import torch
        self._lazy_init()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        temp = kwargs.get("temperature", 0.0)
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", 2048),
            "top_p": kwargs.get("top_p", 0.8),
        }
        if temp > 0.0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
            
        input_length = inputs['input_ids'].shape[-1]
        generated_ids = outputs[0][input_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

class MockClient(BaseLLMClient):
    """
    Mock LLM client for local verification and testing.
    """
    def __init__(self, response_text: str = "Mock response"):
        self.response_text = response_text
        self.last_prompt = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        self.last_prompt = prompt
        return self.response_text

def get_llm_client(config_dict: Dict[str, Any], model_name: str) -> BaseLLMClient:
    """
    Factory function to retrieve LLM client based on configuration.
    """
    provider = config_dict.get("provider", "mock")
    endpoint = config_dict.get("endpoint")
    api_key_env = config_dict.get("api_key", "")
    
    # Resolve env placeholder
    api_key = api_key_env
    if api_key_env and api_key_env.startswith("${") and api_key_env.endswith("}"):
        env_var = api_key_env[2:-1]
        api_key = os.getenv(env_var, "")

    if provider == "openai":
        return OpenAIClient(api_key=api_key, endpoint=endpoint, model=model_name)
    elif provider == "deepseek":
        return DeepSeekClient(api_key=api_key, endpoint=endpoint, model=model_name)
    elif provider == "vllm":
        if not endpoint:
            raise ValueError("Endpoint is required for vLLM provider")
        max_retries = config_dict.get("max_retries", 3)
        timeout = config_dict.get("timeout_seconds", 60)
        return vLLMClient(endpoint=endpoint, model=model_name, api_key=api_key, timeout=timeout, max_retries=max_retries)
    elif provider == "huggingface":
        load_in_8bit = config_dict.get("load_in_8bit", False)
        load_in_4bit = config_dict.get("load_in_4bit", False)
        return HuggingFaceClient(model_name=model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
    elif provider == "mock":
        return MockClient()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
