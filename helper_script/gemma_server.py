"""
OpenAI-compatible FastAPI server for Gemma 4 E4B inference with tool call support.

Loads the model once at startup (with optional LoRA adapter), serves
/v1/chat/completions with Gemma-specific tool call parsing, and returns
structured OpenAI-format responses that litellm can consume directly.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/gemma_server.py \
        --model google/gemma-4-E4B-it \
        --port 8001

    # With LoRA adapter:
    CUDA_VISIBLE_DEVICES=0 python scripts/gemma_server.py \
        --model google/gemma-4-E4B-it \
        --adapter /path/to/lora/checkpoint \
        --port 8001
"""

import argparse
import asyncio
import gc
import json
import logging
import os
import re
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Monkey-patch PEFT weight conversion for transformers 5.x compat ───────
try:
    import peft.utils.transformers_weight_conversion as _twc
    import peft.utils.save_and_load as _sal

    _orig_convert = _twc.convert_peft_adapter_state_dict_for_transformers

    def _safe_convert(model, peft_config, adapter_state_dict, adapter_name="default"):
        try:
            return _orig_convert(model, peft_config, adapter_state_dict, adapter_name)
        except TypeError:
            return adapter_state_dict

    _twc.convert_peft_adapter_state_dict_for_transformers = _safe_convert
    _sal.convert_peft_adapter_state_dict_for_transformers = _safe_convert
except Exception:
    pass


# ── Global state ──────────────────────────────────────────────────────────

_model = None
_tokenizer = None
_model_name: str = ""
_device: str = "cuda:0"
_default_max_tokens: int = 128000
_inference_lock: asyncio.Lock = asyncio.Lock()


# ── Request/Response schemas ──────────────────────────────────────────────


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCallResponse(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCallResponse]] = None
    tool_call_id: Optional[str] = None


class ToolDefinition(BaseModel):
    type: str = "function"
    function: dict


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[dict]
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    top_p: float = 1.0


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ── Model loading ─────────────────────────────────────────────────────────


def _prepare_gemma4_for_lora(model):
    """Unwrap Gemma4ClippableLinear to expose nn.Linear for PEFT compatibility."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except ImportError:
        return model

    for name, module in list(model.named_modules()):
        if isinstance(module, Gemma4ClippableLinear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, module.linear)

    return model


def load_model(model_id: str, adapter_path: Optional[str], device: str):
    """Load model and tokenizer, optionally with LoRA adapter merge."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global _model, _tokenizer, _model_name, _device
    _device = device

    print(f"Loading model: {model_id} on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel

        model = _prepare_gemma4_for_lora(model)
        print(f"Applying LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("LoRA adapter merged")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model.eval()
    _model = model
    _tokenizer = tokenizer
    _model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    print(f"Model ready: {_model_name}")


# ── Gemma tool call parsing ──────────────────────────────────────────────


def _parse_gemma_argument_value(text: str):
    """Parse a single Gemma argument value from the custom format."""
    text = text.strip()
    if text.startswith('<|"|>') and text.endswith('<|"|>'):
        return text[5:-5]
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text == "true":
        return True
    if text == "false":
        return False
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        items = _split_gemma_top_level(inner)
        return [_parse_gemma_argument_value(item) for item in items]
    if text.startswith("{") and text.endswith("}"):
        return _parse_gemma_object(text)
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        pass
    return text


def _split_gemma_top_level(text: str) -> List[str]:
    """Split comma-separated values at the top level (respecting nesting)."""
    items = []
    depth = 0
    current = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in ("{", "["):
            depth += 1
            current.append(ch)
        elif ch in ("}", "]"):
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
        elif text[i : i + 5] == '<|"|>':
            end = text.find('<|"|>', i + 5)
            if end == -1:
                current.append(text[i:])
                i = len(text)
                continue
            current.append(text[i : end + 5])
            i = end + 5
            continue
        else:
            current.append(ch)
        i += 1
    if current:
        items.append("".join(current).strip())
    return items


def _parse_gemma_object(text: str) -> dict:
    """Parse a Gemma-format object {key:value,...}."""
    inner = text[1:-1].strip()
    if not inner:
        return {}
    result = {}
    items = _split_gemma_top_level(inner)
    for item in items:
        colon_idx = item.find(":")
        if colon_idx == -1:
            continue
        key = item[:colon_idx].strip()
        if key.startswith('<|"|>') and key.endswith('<|"|>'):
            key = key[5:-5]
        elif len(key) >= 2 and key.startswith('"') and key.endswith('"'):
            key = key[1:-1]
        value_str = item[colon_idx + 1 :].strip()
        result[key] = _parse_gemma_argument_value(value_str)
    return result


def _try_parse_json_arguments(args_str: str) -> Optional[dict]:
    """Try to parse tool call arguments as JSON (handles when Gemma outputs JSON-style args)."""
    # First try parsing as-is (handles case where args already include outer braces)
    stripped = args_str.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    # Then try wrapping in braces (handles bare key:value pairs)
    candidate = "{" + args_str + "}"
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def parse_gemma_tool_calls(text: str) -> Optional[List[ToolCallResponse]]:
    """Parse Gemma 4 tool call format: <|tool_call>call:NAME{args}<tool_call|>

    Returns OpenAI-compatible ToolCallResponse list, or None if no tool calls found.
    Handles both Gemma-native format (key: <|"|>value<|"|>) and JSON-style format.
    """
    if "<|tool_call>" not in text:
        return None

    calls = []
    for match in re.finditer(
        r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", text, re.DOTALL
    ):
        func_name = match.group(1)
        args_str = match.group(2).strip()
        if args_str:
            arguments = _try_parse_json_arguments(args_str)
            if arguments is None:
                arguments = _parse_gemma_object("{" + args_str + "}")
        else:
            arguments = {}
        calls.append(
            ToolCallResponse(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=func_name,
                    arguments=json.dumps(arguments),
                ),
            )
        )

    return calls if calls else None


def parse_json_text_tool_calls(text: str) -> Optional[List[ToolCallResponse]]:
    """Fallback: parse tool calls when the model outputs them as plain JSON text.

    Gemma sometimes outputs tool calls as:
        {"tool_calls": [{"function": "name", "args": {...}}]}
    instead of using <|tool_call> tokens.
    """
    text = text.strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    raw_calls = data.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        return None

    calls = []
    for raw in raw_calls:
        func_name = raw.get("function") or raw.get("name")
        arguments = raw.get("args") or raw.get("arguments") or {}
        if not func_name:
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                arguments = {}
        calls.append(
            ToolCallResponse(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=func_name,
                    arguments=json.dumps(arguments),
                ),
            )
        )

    return calls if calls else None


def extract_content_before_tool_calls(text: str) -> Optional[str]:
    """Extract text content that appears before the first tool call."""
    pre_tool = text.split("<|tool_call>", 1)[0].strip()
    return pre_tool if pre_tool else None


# ── Inference ─────────────────────────────────────────────────────────────


def run_inference(
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    top_p: float = 1.0,
) -> ChatCompletionResponse:
    """Run model inference and return an OpenAI-compatible response."""
    global _model, _tokenizer, _model_name, _device

    prepared_messages = []
    for msg in messages:
        msg = dict(msg)
        if msg.get("tool_calls"):
            msg["tool_calls"] = [
                {
                    **tc,
                    "function": {
                        **tc["function"],
                        "arguments": json.loads(tc["function"]["arguments"])
                        if isinstance(tc["function"]["arguments"], str)
                        else tc["function"]["arguments"],
                    },
                }
                for tc in msg["tool_calls"]
            ]
        prepared_messages.append(msg)

    template_kwargs: Dict[str, Any] = {
        "conversation": prepared_messages,
        "add_generation_prompt": True,
        "tokenize": True,
        "return_tensors": "pt",
    }
    if tools:
        template_kwargs["tools"] = tools

    input_ids = _tokenizer.apply_chat_template(**template_kwargs)
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor([input_ids])
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(_device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    # Stop token IDs
    stop_ids = []
    if _tokenizer.eos_token_id is not None:
        if isinstance(_tokenizer.eos_token_id, list):
            stop_ids.extend(_tokenizer.eos_token_id)
        else:
            stop_ids.append(_tokenizer.eos_token_id)
    turn_end_id = _tokenizer.convert_tokens_to_ids("<turn|>")
    if turn_end_id is not None and turn_end_id != _tokenizer.unk_token_id:
        stop_ids.append(turn_end_id)
    stop_ids = list(set(stop_ids)) if stop_ids else None

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "repetition_penalty": 1.05,
        "attention_mask": attention_mask,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    if stop_ids:
        gen_kwargs["eos_token_id"] = stop_ids
    if _tokenizer.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = _tokenizer.pad_token_id
    elif _tokenizer.eos_token_id is not None:
        eos = _tokenizer.eos_token_id
        gen_kwargs["pad_token_id"] = eos if not isinstance(eos, list) else eos[0]
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        output_ids = _model.generate(input_ids, **gen_kwargs)

    new_tokens = output_ids[0][prompt_len:]
    response_text = _tokenizer.decode(new_tokens, skip_special_tokens=False)

    del input_ids, attention_mask, output_ids
    torch.cuda.empty_cache()

    for stop_token in (_tokenizer.eos_token or "", "<turn|>"):
        if stop_token and response_text.endswith(stop_token):
            response_text = response_text[: -len(stop_token)]

    prompt_tokens = prompt_len
    completion_tokens = len(new_tokens)

    print(f"[GEMMA RAW] ({completion_tokens} tokens): {response_text[:500]!r}")

    # Parse tool calls from Gemma output
    tool_calls_parsed = None
    content = response_text.strip()
    if tools:
        tool_calls_parsed = parse_gemma_tool_calls(response_text)
        if tool_calls_parsed:
            content = extract_content_before_tool_calls(response_text)
        else:
            tool_calls_parsed = parse_json_text_tool_calls(response_text)
            if tool_calls_parsed:
                content = None
        if tool_calls_parsed:
            print(f"[GEMMA PARSED] {len(tool_calls_parsed)} tool call(s): "
                  f"{[tc.function.name for tc in tool_calls_parsed]}")
            for tc in tool_calls_parsed:
                print(f"  {tc.function.name}: {tc.function.arguments[:200]}")

    finish_reason = "tool_calls" if tool_calls_parsed else "stop"

    message = ChatMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls_parsed,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=_model_name,
        choices=[Choice(message=message, finish_reason=finish_reason)],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ── FastAPI app ───────────────────────────────────────────────────────────


def create_app(model_id: str, adapter_path: Optional[str], device: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _inference_lock
        _inference_lock = asyncio.Lock()
        load_model(model_id, adapter_path, device)
        yield
        global _model, _tokenizer
        del _model, _tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    app = FastAPI(title="Gemma 4 E4B Server", lifespan=lifespan)

    @app.get("/v1/models")
    async def list_models() -> ModelListResponse:
        return ModelListResponse(
            data=[ModelInfo(id=_model_name)]
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
        max_tokens = request.max_tokens or _default_max_tokens
        async with _inference_lock:
            try:
                response = await asyncio.to_thread(
                    run_inference,
                    messages=request.messages,
                    tools=request.tools,
                    temperature=request.temperature,
                    max_tokens=max_tokens,
                    seed=request.seed,
                    top_p=request.top_p,
                )
            except Exception as e:
                logging.error("Inference failed:\n%s", traceback.format_exc())
                torch.cuda.empty_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=str(e))
        return response

    return app


# ── CLI entry point ───────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 4 E4B OpenAI-compatible server")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-4-E4B-it",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional LoRA adapter path to merge",
    )
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("CUDA_DEVICE", "cuda:0"),
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _default_max_tokens = args.max_tokens
    app = create_app(args.model, args.adapter, args.device)
    uvicorn.run(app, host=args.host, port=args.port)
