import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import CONFIG_MAPPING, AutoConfig, AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import Qwen2Config  # type: ignore
except ImportError:  # pragma: no cover
    Qwen2Config = None


class RouterError(Exception):
    pass


@dataclass
class ExpertModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    operation: str


_OPERATION_TABLE = {
    "addition": {"ast": ast.Add, "symbol": "+"},
    "subtraction": {"ast": ast.Sub, "symbol": "-"},
    "multiplication": {"ast": ast.Mult, "symbol": "*"},
    "division": {"ast": ast.Div, "symbol": "/"},
}

_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


class RouterEngine:
    def __init__(
        self,
        expert_paths: Dict[str, str],
        device: Optional[str] = None,
        max_new_tokens: int = 16,
        fallback_when_invalid: bool = True,
    ) -> None:
        if not expert_paths:
            raise RouterError("expert_paths must not be empty")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.fallback_when_invalid = fallback_when_invalid
        self._experts: Dict[str, ExpertModel] = {}
        self._trace: List[Dict[str, object]] = []
        self._load_experts(expert_paths)

    @property
    def trace(self) -> List[Dict[str, object]]:
        return list(self._trace)

    def predict(self, expression: str) -> float:
        value, history = self._evaluate_expression(expression)
        self._trace = history
        return value

    def _load_experts(self, expert_paths: Dict[str, str]) -> None:
        for operation, path in expert_paths.items():
            if operation not in _OPERATION_TABLE:
                raise RouterError(f"Unsupported operation '{operation}'")
            try:
                config = AutoConfig.from_pretrained(path)
            except ValueError as exc:
                if "layer types" not in str(exc):
                    raise
                config_path = os.path.join(path, "config.json")
                if not os.path.exists(config_path):
                    raise RouterError(
                        f"Config mismatch for '{operation}' and file '{config_path}' not found"
                    ) from exc
                with open(config_path, "r", encoding="utf-8") as cfg_file:
                    config_dict = json.load(cfg_file)
                num_layers = config_dict.get("num_hidden_layers")
                layer_types = config_dict.get("layer_types", [])
                if num_layers is None:
                    raise RouterError(
                        f"Config for '{operation}' missing 'num_hidden_layers'"
                    ) from exc
                if not isinstance(layer_types, list):
                    raise RouterError(
                        f"Config for '{operation}' has invalid 'layer_types'"
                    ) from exc
                if len(layer_types) > num_layers:
                    layer_types = layer_types[:num_layers]
                elif len(layer_types) < num_layers:
                    if not layer_types:
                        raise RouterError(
                            f"Config for '{operation}' has empty 'layer_types' but requires {num_layers} entries"
                        ) from exc
                    layer_types = layer_types + [layer_types[-1]] * (num_layers - len(layer_types))
                config_dict["layer_types"] = layer_types
                model_type = config_dict.get("model_type")
                if not model_type:
                    raise RouterError(
                        f"Config for '{operation}' missing 'model_type'"
                    ) from exc
                config_class = CONFIG_MAPPING.get(model_type)
                if config_class is None and model_type == "qwen2" and Qwen2Config is not None:
                    config_class = Qwen2Config
                if config_class is None:
                    raise RouterError(
                        f"Unsupported model_type '{model_type}' for operation '{operation}'"
                    ) from exc
                config = config_class.from_dict(config_dict)
            model = AutoModelForCausalLM.from_pretrained(path, config=config).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(path)
            model.eval()
            self._experts[operation] = ExpertModel(model=model, tokenizer=tokenizer, operation=operation)
        if set(self._experts) != set(expert_paths):
            missing = set(expert_paths) - set(self._experts)
            raise RouterError(f"Failed to load experts for: {sorted(missing)}")

    def _evaluate_expression(self, expression: str) -> Tuple[float, List[Dict[str, object]]]:
        try:
            node = ast.parse(expression, mode="eval").body
        except SyntaxError as exc:
            raise RouterError(f"Invalid expression '{expression}': {exc}") from exc
        value, history = self._visit(node)
        return value, history

    def _visit(self, node: ast.AST) -> Tuple[float, List[Dict[str, object]]]:
        if isinstance(node, ast.BinOp):
            return self._handle_binop(node)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value, history = self._visit(node.operand)
            return -value, history
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value), []
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                return float(node.value), []
            except ValueError as exc:
                raise RouterError(f"Unsupported literal '{node.value}'") from exc
        raise RouterError(f"Unsupported expression node: {ast.dump(node)}")

    def _handle_binop(self, node: ast.BinOp) -> Tuple[float, List[Dict[str, object]]]:
        left_value, left_history = self._visit(node.left)
        right_value, right_history = self._visit(node.right)
        operation = self._operation_from_ast(node.op)
        symbol = _OPERATION_TABLE[operation]["symbol"]
        prompt = f"{left_value} {symbol} {right_value}"
        prediction = self._call_expert(operation, prompt)
        record = {
            "operation": operation,
            "prompt": prompt,
            "prediction": prediction,
        }
        return prediction, left_history + right_history + [record]

    def _operation_from_ast(self, node: ast.operator) -> str:
        for name, entry in _OPERATION_TABLE.items():
            if isinstance(node, entry["ast"]):
                if name not in self._experts:
                    raise RouterError(f"Expert missing for operation '{name}'")
                return name
        raise RouterError(f"Unsupported operator: {node.__class__.__name__}")

    def _call_expert(self, operation: str, prompt: str) -> float:
        expert = self._experts[operation]
        tokenizer = expert.tokenizer
        model = expert.model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        answer = text[len(prompt):].strip()
        value = self._extract_numeric(answer) or self._extract_numeric(text)
        if value is None:
            if self.fallback_when_invalid:
                return self._fallback_compute(operation, prompt)
            raise RouterError(f"Expert '{operation}' produced non-numeric output: {text}")
        return value

    def _extract_numeric(self, text: str) -> Optional[float]:
        match = _NUMBER_PATTERN.search(text)
        if not match:
            return None
        try:
            return float(match.group())
        except ValueError:
            return None

    def _fallback_compute(self, operation: str, prompt: str) -> float:
        lhs_str, rhs_str = prompt.split(_OPERATION_TABLE[operation]["symbol"], 1)
        lhs = float(lhs_str.strip())
        rhs = float(rhs_str.strip())
        if operation == "addition":
            return lhs + rhs
        if operation == "subtraction":
            return lhs - rhs
        if operation == "multiplication":
            return lhs * rhs
        if operation == "division":
            if rhs == 0:
                raise RouterError("Division by zero during fallback computation")
            return lhs / rhs
        raise RouterError(f"Fallback not implemented for '{operation}'")

    def predict_with_trace(self, expression: str) -> Tuple[float, List[Dict[str, object]]]:
        value, history = self._evaluate_expression(expression)
        self._trace = history
        return value, history
