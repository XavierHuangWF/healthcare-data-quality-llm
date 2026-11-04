from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from .microbiome import summarize_invalid_counts
from jsonschema import validate
from .column_inspection import (
    inspect_column,
    inspect_columns,
    compare_columns,
)
from ..duplicate_validation import (
    find_duplicate_ids,
    find_duplicate_conflicts,
)

from .microbiome import observe_matrix_structure
from .context import STRUCTURE_CONTEXT


@dataclass(frozen=True)
class ToolSpec:
    description: str
    function: Callable[..., list[dict]]
    requires_context: str
    arguments_schema: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    )


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "find_duplicate_ids": ToolSpec(
        description="Detect repeated identifiers in metadata.",
        function=find_duplicate_ids,
        requires_context="metadata with SampleID and NPDNum columns",

    ),
    "find_duplicate_conflicts": ToolSpec(
        description="Compare conflicting values in duplicate records.",
        function=find_duplicate_conflicts,
        requires_context="metadata with SampleID and NPDNum columns",

    ),
    "summarize_invalid_counts": ToolSpec(
        description=(
            "Summarize missing, negative, and fractional values "
            "in confirmed raw microbiome count matrices."
        ),
        function=summarize_invalid_counts,
        requires_context="confirmed raw microbiome count matrix",
    ),
    "observe_matrix_structure": ToolSpec(
        description=(
            "Inspect CSV matrix structure, numeric-value prevalence, "
            "and representative column characteristics without "
            "assuming measurement semantics."
        ),
        function=observe_matrix_structure,
        requires_context=STRUCTURE_CONTEXT,
    ),

    "inspect_column": ToolSpec(
        description=(
            "Inspect detailed statistics for a CSV column "
            "using its 1-based position."
        ),
        function=inspect_column,
        requires_context=STRUCTURE_CONTEXT,
        arguments_schema={
            "type": "object",
            "properties": {
                "position": {
                    "type": "integer",
                    "minimum": 1,
                }
            },
            "required": ["position"],
            "additionalProperties": False,
        },
    ),

    "inspect_columns": ToolSpec(
        description=(
            "Inspect detailed statistics for up to four "
            "specified CSV columns in a single call. "
            "Use 1-based column positions."
        ),
        function=inspect_columns,
        requires_context=STRUCTURE_CONTEXT,
        arguments_schema={
            "type": "object",
            "properties": {
                "positions": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "minItems": 1,
                    "maxItems": 4,
                    "uniqueItems": True,
                }
            },
            "required": ["positions"],
            "additionalProperties": False,
        },
    ),

    "compare_columns": ToolSpec(
        description=(
            "Compare two CSV columns row by row using "
            "their 1-based positions. Detect identical "
            "values, missing-value differences, and "
            "conflicting measurements. Useful for "
            "investigating duplicate column headers."
        ),
        function=compare_columns,
        requires_context=STRUCTURE_CONTEXT,
        arguments_schema={
            "type": "object",
            "properties": {
                "left": {
                    "type": "integer",
                    "minimum": 1,
                },
                "right": {
                    "type": "integer",
                    "minimum": 1,
                },
            },
            "required": ["left", "right"],
            "additionalProperties": False,
        },
    ),

}


def describe_tools() -> dict[str, str]:
    return {
        name: spec.description
        for name, spec in TOOL_REGISTRY.items()
    }



def execute_tool(
    name: str,
    path: Path,
    verified_contexts: set[str],
    arguments: dict | None = None,
) -> list[dict]:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")

    spec = TOOL_REGISTRY[name]

    if spec.requires_context not in verified_contexts:
        raise PermissionError(
            f"Tool '{name}' requires verified context: "
            f"{spec.requires_context}"
        )

    if arguments is None:
        arguments = {}

    validate(
        instance=arguments,
        schema=spec.arguments_schema,
    )

    return spec.function(path, **arguments)
