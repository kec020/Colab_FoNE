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

from number_encoders.FNE import FNE
from number_encoders.XVAL import XVAL
from number_encoders.vanilla import VanillaEmbedding
from train.utils import get_regular_embeddings


class RouterError(Exception):
    pass


@dataclass
class ExpertModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    operation: str
    number_encoder: Optional[torch.nn.Module] = None
    number_encoder_type: Optional[str] = None
    number_encoder_config: Optional[dict] = None


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
            number_encoder, encoder_type, encoder_config = self._load_number_encoder(path, model)
            model.eval()
            if number_encoder is not None:
                number_encoder.eval()
            self._experts[operation] = ExpertModel(
                model=model,
                tokenizer=tokenizer,
                operation=operation,
                number_encoder=number_encoder,
                number_encoder_type=encoder_type,
                number_encoder_config=encoder_config,
            )
        if set(self._experts) != set(expert_paths):
            missing = set(expert_paths) - set(self._experts)
            raise RouterError(f"Failed to load experts for: {sorted(missing)}")

    def _load_number_encoder(
        self,
        base_path: str,
        model: AutoModelForCausalLM,
    ) -> Tuple[Optional[torch.nn.Module], Optional[str], Optional[dict]]:
        config_path = os.path.join(base_path, "number_encoder_config.json")
        weights_path = os.path.join(base_path, "number_encoder.pt")
        if not (os.path.exists(config_path) and os.path.exists(weights_path)):
            return None, None, None

        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RouterError(f"Failed to read number encoder config at '{config_path}': {exc}") from exc

        encoder_type = config.get("type")
        embedding_dim = config.get("embedding_dim")
        if encoder_type is None or embedding_dim is None:
            raise RouterError(
                f"Number encoder config at '{config_path}' missing required fields"
            )

        if encoder_type == "fne":
            encoder = FNE(
                embedding_dim,
                int_digit_len=config.get("int_digit_len", 0),
                frac_digit_len=config.get("frac_digit_len", 0),
                period_base_list=config.get("period_base_list", []),
                add_linear=config.get("add_linear", True),
                device=self.device,
            )
        elif encoder_type == "vanilla":
            encoder = VanillaEmbedding(
                embedding_dim,
                int_digit_len=config.get("int_digit_len", 0),
                frac_digit_len=config.get("frac_digit_len", 0),
                device=self.device,
            )
        elif encoder_type == "xval":
            encoder = XVAL(
                embedding_dim=embedding_dim,
                max_num=config.get("max_num", 0),
                device=self.device,
            )
        else:
            raise RouterError(
                f"Unsupported number encoder type '{encoder_type}' at '{config_path}'"
            )

        try:
            state = torch.load(weights_path, map_location=self.device)
            encoder.load_state_dict(state)
        except (OSError, RuntimeError) as exc:
            raise RouterError(
                f"Failed to load number encoder weights at '{weights_path}': {exc}"
            ) from exc

        encoder.to(self.device)
        return encoder, encoder_type, config

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
        if expert.number_encoder is not None and expert.number_encoder_type is not None:
            value = self._call_number_encoder_expert(expert, prompt)
        else:
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
        if value is None:
            if self.fallback_when_invalid:
                return self._fallback_compute(operation, prompt)
            raise RouterError(f"Expert '{operation}' produced non-numeric output")
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

    def _call_number_encoder_expert(self, expert: ExpertModel, prompt: str) -> Optional[float]:
        encoder_type = expert.number_encoder_type
        if encoder_type == "fne":
            return self._predict_with_fne(expert, prompt)
        if encoder_type == "vanilla":
            return self._predict_with_vanilla(expert, prompt)
        if encoder_type == "xval":
            return self._predict_with_xval(expert, prompt)
        raise RouterError(f"Unsupported number encoder type '{encoder_type}'")

    def _prepare_number_encoder_inputs(
        self,
        expert: ExpertModel,
        prompt: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
        tokenizer = expert.tokenizer
        numbers = [float(match.group()) for match in _NUMBER_PATTERN.finditer(prompt)]
        placeholder_text = _NUMBER_PATTERN.sub(" [NUM] ", prompt)
        placeholder_text = " ".join(placeholder_text.split())
        encoded = tokenizer(placeholder_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        scatter_tensor = torch.zeros_like(input_ids, dtype=torch.float64, device=self.device)
        num_token_id = tokenizer.convert_tokens_to_ids("[NUM]")
        if num_token_id is None or num_token_id == tokenizer.unk_token_id:
            raise RouterError(
                f"Tokenizer for expert '{expert.operation}' does not contain the [NUM] token"
            )
        num_idx = 0
        for position, token_id in enumerate(input_ids[0].tolist()):
            if token_id == num_token_id and num_idx < len(numbers):
                scatter_tensor[0, position] = numbers[num_idx]
                num_idx += 1
        return input_ids, attention_mask, scatter_tensor, numbers

    def _compute_last_token_hidden_state(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        last_token_mask = torch.zeros_like(attention_mask, dtype=hidden_states.dtype)
        last_indices = attention_mask.sum(dim=1) - 1
        for batch_idx, index in enumerate(last_indices.tolist()):
            if index >= 0:
                last_token_mask[batch_idx, index] = 1.0
        return (hidden_states * last_token_mask.unsqueeze(-1)).sum(dim=1)

    def _predict_with_fne(self, expert: ExpertModel, prompt: str) -> Optional[float]:
        model = expert.model
        encoder = expert.number_encoder
        config = expert.number_encoder_config or {}
        if encoder is None:
            return None
        int_digit_len = config.get("int_digit_len", 0)
        frac_digit_len = config.get("frac_digit_len", 0)
        input_ids, attention_mask, scatter_tensor, _ = self._prepare_number_encoder_inputs(expert, prompt)
        with torch.no_grad():
            regular_embeddings = get_regular_embeddings(model, input_ids)
            fourier_embeddings = encoder(scatter_tensor)
            input_embeddings = regular_embeddings + fourier_embeddings.to(regular_embeddings.dtype)
            outputs = model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1]
            pooled = self._compute_last_token_hidden_state(last_hidden_state, attention_mask)
            prediction = encoder.fourier_compute_prediction(pooled, int_digit_len, frac_digit_len)
        if prediction.numel() == 0:
            return None
        return float(prediction.squeeze(0).cpu().item())

    def _predict_with_vanilla(self, expert: ExpertModel, prompt: str) -> Optional[float]:
        model = expert.model
        encoder = expert.number_encoder
        if encoder is None:
            return None
        input_ids, attention_mask, scatter_tensor, _ = self._prepare_number_encoder_inputs(expert, prompt)
        with torch.no_grad():
            regular_embeddings = get_regular_embeddings(model, input_ids)
            vanilla_embeddings = encoder(scatter_tensor)
            input_embeddings = regular_embeddings + vanilla_embeddings.to(regular_embeddings.dtype)
            outputs = model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1]
            pooled = self._compute_last_token_hidden_state(last_hidden_state, attention_mask)
            prediction = encoder.compute_prediction(pooled)
        if prediction.numel() == 0:
            return None
        return float(prediction.squeeze(0).cpu().item())

    def _predict_with_xval(self, expert: ExpertModel, prompt: str) -> Optional[float]:
        model = expert.model
        encoder = expert.number_encoder
        if encoder is None:
            return None
        input_ids, attention_mask, scatter_tensor, _ = self._prepare_number_encoder_inputs(expert, prompt)
        with torch.no_grad():
            regular_embeddings = get_regular_embeddings(model, input_ids)
            input_embeddings = encoder(scatter_tensor, regular_embeddings)
            outputs = model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1]
            pooled = self._compute_last_token_hidden_state(last_hidden_state, attention_mask)
            prediction = encoder.compute_prediction(pooled)
        if prediction.numel() == 0:
            return None
        return float(prediction.squeeze(0).cpu().item())
