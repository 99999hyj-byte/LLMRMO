# -*- coding: utf-8 -*-
"""
LLM_lora.py

LangChain-compatible local LLM wrapper for LLaMA 3.1 with optional LoRA (PEFT) support.
This implementation aligns with the self-supervised adaptation strategy described in 
the MONLCD framework.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel

def _apply_stop(text: str, stop: Optional[List[str]]) -> str:
    """Helper to trim generated text at specified stop sequences."""
    if not stop:
        return text
    for s in stop:
        if s and s in text:
            text = text.split(s)[0]
    return text

def _get_input_device(model: torch.nn.Module) -> torch.device:
    """
    Identify the appropriate device for input tensors.
    For models utilizing device_map, inputs must match the first parameter's device.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        dev = getattr(model, "device", None)
        if dev is None:
            return torch.device("cpu")
        return dev

class _BaseChatLLM(LLM):
    """
    Shared base class for chat-style causal Language Models:
    - Renders chat templates
    - Handles tokenization and generation
    - Supports optional LoRA (PEFT) loading for adapted models
    """
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(
        self,
        model_name_or_path: str,
        *,
        lora_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = False,
        max_new_tokens: int = 512,
        use_fast: bool = False,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.lora_path = lora_path
        self.max_new_tokens = max_new_tokens

        print(f"Initializing tokenizer from: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast)

        print(f"Loading model: {model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config,
        ).eval()

        # Load LoRA adapter if path is provided (for self-supervised adaptation)
        if lora_path:
            print(f"Loading LoRA adapter from: {lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model, 
                model_id=lora_path, 
                is_trainable=False
            ).eval()

        # Ensure pad_token is configured to avoid generation errors
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _render_chat(self, prompt: str) -> str:
        """Apply the specific chat template for the model."""
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    def _generate_text(
        self,
        rendered: str,
        *,
        stop: Optional[List[str]] = None,
        **gen_kwargs: Any,
    ) -> str:
        """Perform inference and decode the generated output."""
        input_device = _get_input_device(self.model)
        model_inputs = self.tokenizer([rendered], return_tensors="pt").to(input_device)

        kwargs: Dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        kwargs.update(gen_kwargs or {})

        # Default generation parameters
        if "eos_token_id" not in kwargs:
            kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if "pad_token_id" not in kwargs:
            kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.get("attention_mask", None),
                **kwargs,
            )

        # Remove the prompt tokens from the result
        prompt_len = model_inputs.input_ids.shape[1]
        gen_only_ids = generated_ids[:, prompt_len:]
        text = self.tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)[0].strip()
        return _apply_stop(text, stop)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        rendered = self._render_chat(prompt)
        return self._generate_text(rendered, stop=stop, **kwargs)

class LLaMA3_1_LLM(_BaseChatLLM):
    """
    Local LLaMA 3.1 wrapper with optional LoRA support.
    Configured with a system prompt and high-performance inference settings.
    """
    def __init__(
        self,
        model_name_or_path: str,
        *,
        lora_path: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = False,
        max_new_tokens: int = 512,
        use_fast: bool = False,
    ):
        self.system_prompt = system_prompt
        super().__init__(
            model_name_or_path=model_name_or_path,
            lora_path=lora_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            max_new_tokens=max_new_tokens,
            use_fast=use_fast,
        )

    def _render_chat(self, prompt: str) -> str:
        """Render LLaMA 3.1 specific chat format with system prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    @property
    def _llm_type(self) -> str:
        return "LLaMA3_1_LLM"

__all__ = ["LLaMA3_1_LLM"]