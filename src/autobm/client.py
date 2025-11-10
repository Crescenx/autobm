import time
from datetime import datetime
import threading
from typing import Literal, TypeVar, Generic, TypedDict, Dict, Any
from abc import ABC, abstractmethod
from openai import OpenAI
from anthropic import Anthropic
from openai.types.chat import ChatCompletion
from anthropic.types import Message as AnthropicMessage
import os
import toml

T = TypeVar('T')

class ExtractionResult(TypedDict):
    content: str
    token: int
    react_time: float

class Provider(ABC, Generic[T]):
    _last_api_call_duration: float

    @abstractmethod
    def create_completion(self, model: str, system_prompt: str, user_prompt: str) -> T:
        pass

    @abstractmethod
    def extract_output(self, completion: T) -> ExtractionResult:
        pass

class OpenAIService(Provider[ChatCompletion]):
    def __init__(self, base_url: str, api_key: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self._last_api_call_duration = 0.0

    def create_completion(self, model: str, system_prompt: str, user_prompt: str) -> ChatCompletion:
        start_time = time.time()
        completion_response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        end_time = time.time()
        self._last_api_call_duration = end_time - start_time
        return completion_response

    def extract_output(self, completion: ChatCompletion) -> ExtractionResult:
        content_text = ""
        if completion.choices and completion.choices[0].message:
            content_text = completion.choices[0].message.content or ""
        
        total_tokens_count = 0
        if completion.usage:
            total_tokens_count = completion.usage.total_tokens or 0

        return {
            "content": content_text,
            "token": total_tokens_count,
            "react_time": self._last_api_call_duration
        }

class AnthropicService(Provider[AnthropicMessage]):
    def __init__(self, base_url: str, api_key: str):
        self.client = Anthropic(base_url=base_url, api_key=api_key)
        self._last_api_call_duration = 0.0

    def create_completion(self, model: str, system_prompt: str, user_prompt: str) -> AnthropicMessage:
        start_time = time.time()
        message = self.client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=4096 # Required by Anthropic API
        )
        end_time = time.time()
        self._last_api_call_duration = end_time - start_time
        return message

    def extract_output(self, completion: AnthropicMessage) -> ExtractionResult:
        content_text = ""
        if completion.content and isinstance(completion.content, list) and len(completion.content) > 0:
            if hasattr(completion.content[0], 'text'):
                content_text = completion.content[0].text

        input_tokens = completion.usage.input_tokens if completion.usage else 0
        output_tokens = completion.usage.output_tokens if completion.usage else 0
        total_tokens = input_tokens + output_tokens

        return {
            "content": content_text,
            "token": total_tokens,
            "react_time": self._last_api_call_duration
        }

class LLMClient:
    SERVICE_CLASSES = {
        'aliyun_openai_compatible': OpenAIService,
        'anthropic_claude': AnthropicService
    }

    def __init__(self, log_file_override=None, csv_file_override=None, config_path="pyproject.toml"):
        self.services: Dict[str, Provider] = {}
        self.model_configs: Dict[str, Any] = {}
        self.system_prompts: Dict[str, str] = {}
        self.global_config: Dict[str, Any] = {}

        self._load_config_and_initialize_services(config_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.global_config.get('default_log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = log_file_override or f"{log_dir}/llm_responses_{timestamp}.log"

        csv_dir = self.global_config.get('default_csv_dir', 'csvs')
        os.makedirs(csv_dir, exist_ok=True)
        self.csv_file = csv_file_override or f"{csv_dir}/llm_metrics_{timestamp}.csv"

    def _load_config_and_initialize_services(self, config_path: str):
        try:
            pyproject_data = toml.load(config_path)
            self.global_config = pyproject_data.get("tool", {}).get("llm_client_config", {})
            if not self.global_config:
                raise ValueError("[tool.llm_client_config] section not found in pyproject.toml")

            model_mapping_config = self.global_config.get("model_mapping", {})
            self.system_prompts = self.global_config.get("system_prompts", {})

            for mode_name, mode_config in model_mapping_config.items():
                if mode_name not in ["quest", "code"]:
                    continue

                provider_type = mode_config.get('provider_type')
                model_name = mode_config.get('model')
                base_url_env_var = mode_config.get('base_url_env')
                api_key_env_var = mode_config.get('api_key_env')

                # Try to get base_url and api_key directly from config first
                base_url = mode_config.get('base_url')
                api_key = mode_config.get('api_key')

                # If not set directly, try to get from environment variables
                if not base_url and base_url_env_var:
                    base_url = os.getenv(base_url_env_var)
                if not api_key and api_key_env_var:
                    api_key = os.getenv(api_key_env_var)

                if not all([provider_type, model_name, base_url, api_key]):
                    print(f"Warning: Configuration for mode '{mode_name}' is incomplete (missing provider_type, model, base_url, or api_key). Skipping.")
                    continue

                service_class = self.SERVICE_CLASSES.get(provider_type)
                if not service_class:
                    print(f"Warning: Unknown provider type '{provider_type}' for mode '{mode_name}'. Skipping.")
                    continue

                self.services[mode_name] = service_class(base_url=base_url, api_key=api_key)
                self.model_configs[mode_name] = {'model': model_name}
                print(f"Successfully configured service for mode: {mode_name} using {provider_type}")


        except FileNotFoundError:
            print(f"Error: Configuration file '{config_path}' not found.")
            raise
        except toml.TomlDecodeError:
            print(f"Error: Could not decode TOML from '{config_path}'.")
            raise
        except Exception as e:
            print(f"Error loading configuration or initializing services: {e}")
            raise

    def generate(
        self,
        prompt: str,
        mode: Literal["quest", "code"] = "quest",
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> str:
        if mode not in self.services:
            available_modes = ", ".join(self.services.keys())
            raise ValueError(f"Mode '{mode}' not configured or service initialization failed. Available: {available_modes}")

        service = self.services[mode]
        model_config = self.model_configs[mode]
        actual_model_name = model_config['model']
        system_prompt_text = self.system_prompts.get(mode, self.system_prompts.get("default", "You are a helpful assistant."))
        
        for attempt in range(max_retries + 1):
            request_start_time = time.perf_counter()
            print(f"Attempt {attempt + 1}/{max_retries + 1} for mode '{mode}' (model: {actual_model_name})...")

            stop_timer_event = threading.Event()
            timer_thread = threading.Thread(
                target=self._display_elapsed_time,
                args=(request_start_time, stop_timer_event)
            )
            timer_thread.daemon = True
            timer_thread.start()

            try:
                completion = service.create_completion(
                    model=actual_model_name,
                    system_prompt=system_prompt_text,
                    user_prompt=prompt
                )
                stop_timer_event.set()
                timer_thread.join(timeout=0.5) 
                
                total_request_time = time.perf_counter() - request_start_time
                print(f"\rRequest for mode '{mode}' completed. Total time: {total_request_time:.2f}s" + " "*20)

                output_data = service.extract_output(completion)
                content = output_data['content']
                token_usage = output_data['token']
                api_call_duration = output_data['react_time']

                self._save_log(prompt, content, mode, actual_model_name, attempt, total_request_time, api_call_duration)
                self._write_csv(mode, token_usage, api_call_duration, total_request_time)
                return content if content else ""

            except Exception as e:
                stop_timer_event.set()
                timer_thread.join(timeout=0.5)
                total_request_time = time.perf_counter() - request_start_time
                print(f"\rRequest for mode '{mode}' failed. Total time: {total_request_time:.2f}s" + " "*20)
                
                error_msg = f"{type(e).__name__}: {str(e)}"
                self._save_log(prompt, error_msg, mode, actual_model_name, attempt, total_request_time, error=True)
                
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay}s... ({attempt + 2}/{max_retries + 1})")
                    time.sleep(retry_delay)
                else:
                    print(f"API request failed for mode '{mode}' after {max_retries + 1} attempts.")
                    raise RuntimeError(f"API request failed for mode '{mode}': {error_msg}") from e
        return "" # Should not be reached if max_retries >= 0

    def _display_elapsed_time(self, start_time: float, stop_event: threading.Event):
        while not stop_event.is_set():
            elapsed = time.perf_counter() - start_time
            print(f"\rProcessing... Time elapsed: {elapsed:.1f}s", end="", flush=True)
            if stop_event.wait(0.2): # Check event status frequently
                break
        print("\r" + " "*50 + "\r", end="", flush=True) # Clear line

    def _save_log(self, prompt: str, output: str, mode: str, model_name: str, retry_count: int, total_req_time: float, api_call_time: float = -1.0, error: bool = False):
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().isoformat()
                status = "ERROR" if error else "SUCCESS"
                log_entry = (
                    f"======= LLM Log =======\n"
                    f"Time: {timestamp}\n"
                    f"Status: {status}\n"
                    f"Mode: {mode}\n"
                    f"Model: {model_name}\n"
                    f"Attempt: {retry_count + 1}\n"
                    f"Total Request Time: {total_req_time:.3f}s\n"
                    f"API Call Duration: {api_call_time:.3f}s\n"
                    f"Prompt:\n{prompt}\n"
                    f"Response:\n{output}\n"
                    f"=======================\n\n"
                )
                f.write(log_entry)
        except Exception as e:
            print(f"Failed to write to log file: {e}")

    def _write_csv(self, mode: str, token: int, api_call_time: float, total_req_time: float):
        header = "timestamp,mode,token_usage,api_call_duration_seconds,total_request_duration_seconds\n"
        write_header = not os.path.exists(self.csv_file) or os.path.getsize(self.csv_file) == 0
        
        try:
            with open(self.csv_file, 'a', encoding='utf-8', newline='') as f:
                if write_header:
                    f.write(header)
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp},{mode},{token},{api_call_time:.3f},{total_req_time:.3f}\n")
        except Exception as e:
            print(f"Failed to write to CSV file: {e}")
