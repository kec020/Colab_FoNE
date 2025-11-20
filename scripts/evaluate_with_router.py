import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from router.engine import RouterEngine, RouterError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a dataset with the arithmetic router")
    parser.add_argument("dataset", help="Path to the CSV file containing columns Question and Answer")
    parser.add_argument("output", help="Path to the output CSV file")
    parser.add_argument("--addition", dest="addition", help="Path to the addition expert directory")
    parser.add_argument("--subtraction", dest="subtraction", help="Path to the subtraction expert directory")
    parser.add_argument("--multiplication", dest="multiplication", help="Path to the multiplication expert directory")
    parser.add_argument("--division", dest="division", help="Path to the division expert directory")
    parser.add_argument("--device", default=None, help="Torch device override (e.g. cuda, cpu)")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=16, help="Maximum tokens generated per expert call")
    parser.add_argument("--disable-fallback", action="store_true", help="Disable numeric fallback when expert output is invalid")
    parser.add_argument("--include-trace", action="store_true", help="Store routing trace for each example")
    return parser.parse_args()


def build_expert_paths(args: argparse.Namespace) -> Dict[str, str]:
    expert_paths: Dict[str, str] = {}
    for name in ("addition", "subtraction", "multiplication", "division"):
        path = getattr(args, name, None)
        if path:
            expert_paths[name] = path
    if not expert_paths:
        raise RouterError("At least one expert path must be provided")
    return expert_paths


def load_dataset(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = {"Question", "Answer"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset {path} missing required columns: {sorted(missing)}")
        return list(reader)


def evaluate_dataset(router: RouterEngine, rows: List[Dict[str, str]], include_trace: bool) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        question = row.get("Question", "").strip()
        if not question:
            continue
        try:
            prediction, trace = router.predict_with_trace(question)
        except RouterError as exc:
            prediction = None
            trace = [{"error": str(exc)}]
        answer_raw = row.get("Answer", "")
        try:
            label = float(answer_raw)
        except ValueError:
            label = answer_raw
        abs_error = None
        correct = None
        if isinstance(label, (int, float)) and isinstance(prediction, (int, float)):
            abs_error = abs(prediction - label) if prediction is not None else None
            correct = abs_error == 0 if abs_error is not None else None
        results.append({
            "index": idx,
            "question": question,
            "label": label,
            "prediction": prediction,
            "abs_error": abs_error,
            "correct": correct,
            "trace": json.dumps(trace) if include_trace else None,
        })
    return results


def write_results(path: Path, rows: List[Dict[str, object]], include_trace: bool) -> None:
    fieldnames = ["index", "question", "label", "prediction", "abs_error", "correct"]
    if include_trace:
        fieldnames.append("trace")
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not include_trace:
                row = {key: value for key, value in row.items() if key in fieldnames}
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    expert_paths = build_expert_paths(args)
    router = RouterEngine(
        expert_paths=expert_paths,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        fallback_when_invalid=not args.disable_fallback,
    )

    rows = load_dataset(dataset_path)
    results = evaluate_dataset(router, rows, args.include_trace)
    write_results(output_path, results, args.include_trace)

if __name__ == "__main__":
    main()
