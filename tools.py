from inspect import Signature, getdoc, signature
from typing import Any, Callable, TypeVar

ToolFn = Callable[
    ...,
    Any,
]


F = TypeVar(
    "F",
    bound=ToolFn,
)


class ToolRegistry:
    def __init__(self) -> None:
        self.functions: dict[str, ToolFn]
        self.schemas: list[dict[str, Any]]

    def build_schema(self, fn: ToolFn, name: str, desc: str) -> dict[str, Any]:
        """
        {
            "type": "function",
            "function": {
                "name": "load_skill",
                "description": "加载某个技能的详细操作说明",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "技能名称",
                        },
                    },
                    "required": ["name"],
                },
            },
        }
        """
        sig: Signature = signature(fn)

        raise NotImplementedError

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            tool_name = name or fn.__name__
            tool_desc = description or str(getdoc(fn))

            schema = self.build_schema(
                fn=fn,
                name=tool_name,
                desc=tool_desc,
            )

            self.functions[tool_name] = fn
            self.schemas.append(schema)

            return fn

        return decorator
