from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class ToolResult:
    """result of tool execution"""
    success: bool
    message: str # msg for llm
    error: str | None = None
    metadata: dict | None = None # metadata (e.g. time or file size)

class BaseTool(ABC):
    """
    base class for tools
    name + description + input_schema + execute()

    subclass should override the following properties and methods:
    - name: str
    - description: str
    - input_schema: dict
    - execute(**kwargs) -> ToolResult
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """name of the tool"""
        ...
    
    @property
    @abstractmethod
    def description(self) -> str:
        """description of the tool"""
        ...
    
    @property
    @abstractmethod
    def input_schema(self) -> dict:
        """input schema of the tool"""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """execute the tool with given parameters"""
        ...

    @property
    def is_dangerous(self) -> bool:
        """whether the tool is dangerous (e.g. delete files)"""
        return False

    def to_schema(self) -> dict:
        """return the tool schema for llm"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }