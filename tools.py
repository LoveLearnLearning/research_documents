import inspect
from inspect import Parameter, Signature, getdoc, signature
from typing import Any, Callable, TypeVar, Union, get_args, get_origin

ToolFn = Callable[..., Any]
F = TypeVar("F", bound=ToolFn)

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


class ToolRegistry:
    def __init__(self) -> None:
        self.functions: dict[str, ToolFn] = {}
        self.schemas: list[dict[str, Any]] = []

    def _annotation_to_schema(self, annotation: Any) -> dict[str, Any]:
        if annotation is Parameter.empty:
            return {"type": "string"}

        origin = get_origin(annotation)
        args = get_args(annotation)

        # Optional[X] / X | None -> unwrap to X
        if origin is Union:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return self._annotation_to_schema(non_none[0])
            # 多类型 union，退化为 string，可按需扩展成 anyOf
            return {"type": "string"}

        if origin in (list, tuple):
            item_type = args[0] if args else Any
            item_schema = (
                {} if item_type is Any else self._annotation_to_schema(item_type)
            )
            return {"type": "array", "items": item_schema or {"type": "string"}}

        if origin is dict:
            return {"type": "object"}

        if annotation is Any:
            return {}

        if annotation in _TYPE_MAP:
            return {"type": _TYPE_MAP[annotation]}

        # 兜底：未知类型当 string 处理，而不是直接崩
        return {"type": "string"}

    def build_schema(self, fn: ToolFn, name: str, desc: str) -> dict[str, Any]:
        sig: Signature = signature(fn)
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                continue  # *args / **kwargs 不放进 schema

            prop_schema = self._annotation_to_schema(param.annotation)

            # 从 docstring 提不到参数说明，可选：留空 description 字段
            properties[param_name] = prop_schema

            if param.default is Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            tool_name = name or fn.__name__
            tool_desc = description or str(getdoc(fn)) or ""
            schema = self.build_schema(fn=fn, name=tool_name, desc=tool_desc)
            self.functions[tool_name] = fn
            self.schemas.append(schema)
            return fn

        return decorator


class VLLMToolCaller:
    """
    基于 vllm.LLM() 离线引擎的 tool call 封装。

    离线模式没有 OpenAI server 那层 tool parser，
    需要自己：
      1. apply_chat_template(tools=...) 拼 prompt
      2. generate
      3. 从模型输出文本里手动抠出 tool_call json

    不同模型的 tool_call 输出格式不同（Hermes 用 <tool_call>...</tool_call>，
    Llama3 直接吐 json，Qwen2.5 类似 Hermes），下面默认按 Hermes/Qwen 风格解析，
    可以按你实际用的模型换 parser。
    """

    def __init__(
        self,
        model: str,
        registry: ToolRegistry,
        tokenizer_mode: str = "auto",
        **llm_kwargs: Any,
    ) -> None:
        from vllm import LLM  # 延迟 import，避免没装 vllm 时脚本直接炸

        self.registry = registry
        self.llm = LLM(model=model, tokenizer_mode=tokenizer_mode, **llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def build_prompt(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tools=self.registry.schemas,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        sampling_params: Any = None,
    ) -> str:
        from vllm import SamplingParams

        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

        prompt = self.build_prompt(messages)
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    @staticmethod
    def parse_tool_calls(text: str) -> list[dict[str, Any]]:
        """
        解析 Hermes / Qwen2.5 风格的 <tool_call>{"name": ..., "arguments": {...}}</tool_call>。
        如果你用的是别的模型（比如 Llama3 直接吐纯 json，没有标签），
        改这个函数的正则/解析逻辑即可。
        """
        import json
        import re

        calls: list[dict[str, Any]] = []
        for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
            try:
                obj = json.loads(m.group(1))
                calls.append(obj)
            except json.JSONDecodeError:
                continue
        return calls

    def run_tool_calls(self, text: str) -> list[Any]:
        results = []
        for call in self.parse_tool_calls(text):
            fn_name = call.get("name")
            args = call.get("arguments", {})
            fn = self.registry.functions.get(fn_name)
            if fn is None:
                results.append({"error": f"unknown tool: {fn_name}"})
                continue
            results.append(fn(**args))
        return results


if __name__ == "__main__":
    registry = ToolRegistry()

    @registry.tool(description="加载某个技能的详细操作说明")
    def load_skill(name: str, verbose: bool = False) -> str:
        return f"loading {name}, verbose={verbose}"

    import json

    print(json.dumps(registry.schemas, ensure_ascii=False, indent=2))

    # 需要装了 vllm 且有本地/可下载模型才能跑下面这段
    # caller = VLLMToolCaller(model="Qwen/Qwen2.5-7B-Instruct", registry=registry)
    # messages = [{"role": "user", "content": "帮我加载 discrete-math 这个技能"}]
    # text = caller.generate(messages)
    # print(text)
    # print(caller.run_tool_calls(text))
