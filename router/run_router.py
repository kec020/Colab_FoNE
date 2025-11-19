import argparse
import json
import logging
from typing import Dict

from .engine import RouterEngine, RouterError


def parse_expert_paths(args: argparse.Namespace) -> Dict[str, str]:
    expert_paths = {}
    for operation in ("addition", "subtraction", "multiplication", "division"):
        path = getattr(args, f"{operation}_model", None)
        if path:
            expert_paths[operation] = path
    if not expert_paths:
        raise RouterError("At least one expert model path must be provided")
    return expert_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Route arithmetic expressions across expert models")
    parser.add_argument("expression", help="Arithmetic expression to evaluate")
    parser.add_argument("--addition-model", dest="addition_model", help="Path to the addition expert model")
    parser.add_argument("--subtraction-model", dest="subtraction_model", help="Path to the subtraction expert model")
    parser.add_argument("--multiplication-model", dest="multiplication_model", help="Path to the multiplication expert model")
    parser.add_argument("--division-model", dest="division_model", help="Path to the division expert model")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=16, help="Maximum tokens to generate per expert call")
    parser.add_argument("--device", default=None, help="Torch device override (e.g. cuda, cpu)")
    parser.add_argument("--disable-fallback", action="store_true", help="Disable numeric fallback when an expert output is invalid")
    parser.add_argument("--trace", action="store_true", help="Print routing trace information")
    args = parser.parse_args()

    try:
        expert_paths = parse_expert_paths(args)
        engine = RouterEngine(
            expert_paths=expert_paths,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            fallback_when_invalid=not args.disable_fallback,
        )
        value, trace = engine.predict_with_trace(args.expression)
    except RouterError as exc:
        logging.error("Router error: %s", exc)
        raise SystemExit(1) from exc

    print(value)
    if args.trace:
        print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
