import asyncio
import logging
import time
from typing import Any, Dict


class LLMClient:
    """Unified LLM client supporting Gemini, vLLM, and Ollama backends"""

    def __init__(self, config: Dict[str, Any]):
        self.provider = config.get('llm_provider', 'ollama').lower()
        self.logger = logging.getLogger("llm_client")
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 2)
        self.cache_manager = config.get('cache_manager')
        if self.cache_manager is None:
            from utils.cache_manager import get_shared_cache
            self.cache_manager = get_shared_cache()

        if self.provider == 'gemini':
            import google.generativeai as genai
            api_key = config.get('google_api_key') or config.get('api_key')
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                config.get('gemini_model', config.get('model', 'gemini-2.5-flash'))
            )
        elif self.provider in ('vllm', 'openai'):
            from openai import AsyncOpenAI
            base_url = config.get('vllm_base_url', config.get('base_url', 'http://localhost:8000/v1'))
            api_key = config.get('vllm_api_key', config.get('api_key', 'not-needed'))
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            self.model_name = config.get('vllm_model', config.get('model', 'meta-llama/Llama-3.1-8B-Instruct'))
        elif self.provider == 'ollama':
            from openai import AsyncOpenAI
            base_url = config.get('ollama_base_url', config.get('base_url', 'http://localhost:11434/v1'))
            self.client = AsyncOpenAI(base_url=base_url, api_key='ollama')
            self.model_name = config.get('ollama_model', config.get('model', 'llama3.1:8b-instruct-q4_K_M'))
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}. Use 'gemini', 'vllm', 'ollama', or 'openai'.")

    async def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get('temperature', 0.0)
        cache_key = None

        if self.cache_manager:
            cache_key = self.cache_manager.make_key(
                self.provider, getattr(self, 'model_name', ''), prompt, temperature
            )
            found, cached = self.cache_manager.get_llm(cache_key)
            if found:
                self.logger.debug("LLM cache hit")
                return cached

        start = time.time()
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if self.provider == 'gemini':
                    result = await self._generate_gemini(prompt, **kwargs)
                else:
                    result = await self._generate_openai(prompt, **kwargs)

                if cache_key and self.cache_manager:
                    self.cache_manager.set_llm(
                        cache_key, result, getattr(self, 'model_name', '')
                    )
                    elapsed_ms = (time.time() - start) * 1000
                    self.cache_manager.record_time_saved("llm", elapsed_ms)

                return result
            except Exception as e:
                last_error = e
                self.logger.warning(f"LLM call failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        raise last_error

    async def _generate_gemini(self, prompt: str, **kwargs) -> str:
        import google.generativeai as genai
        generation_config = {}
        if 'temperature' in kwargs:
            generation_config['temperature'] = kwargs.pop('temperature')
        if 'max_tokens' in kwargs:
            generation_config['max_output_tokens'] = kwargs.pop('max_tokens')

        gen_config = genai.types.GenerationConfig(**generation_config) if generation_config else None

        response = await asyncio.to_thread(
            self.model.generate_content,
            prompt,
            generation_config=gen_config,
            **kwargs
        )
        return response.text

    async def _generate_openai(self, prompt: str, **kwargs) -> str:
        params = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
        }
        if 'temperature' in kwargs:
            params['temperature'] = kwargs.pop('temperature')
        if 'max_tokens' in kwargs:
            params['max_tokens'] = kwargs.pop('max_tokens')
        if 'response_format' in kwargs:
            params['response_format'] = kwargs.pop('response_format')

        response = await self.client.chat.completions.create(**params)
        return response.choices[0].message.content
