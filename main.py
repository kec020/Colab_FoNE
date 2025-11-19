import argparse
import csv
import json
import os
import re
import logging
import torch
import wandb

from utils.logger_utils import setup_logger, get_output_folder
from utils.model_utils import load_model_and_tokenizer
from train.train_pipeline import (
    parse_period_base_list,
    create_dataloader_and_train
)


def sanitize_filename(value: str) -> str:
    """Convert an arbitrary string into a filesystem-friendly name."""
    sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', value.strip())
    return sanitized or 'output'


def write_test_results_csv(filepath: str, records):
    """Persist evaluation records into a CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = ["index", "question", "label", "prediction", "abs_error", "correct"]
    with open(filepath, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, record in enumerate(records or []):
            writer.writerow({
                "index": idx,
                "question": record.get("question", ""),
                "label": record.get("label", ""),
                "prediction": record.get("prediction", ""),
                "abs_error": record.get("abs_error", ""),
                "correct": record.get("correct", "")
            })


def sanitize_qwen_config(output_dir: str) -> None:
    """Ensure Qwen configs have consistent layer_types entries."""
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        return
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config_data = json.load(config_file)
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning(f"Failed to read config.json in {output_dir}: {exc}")
        return

    model_type = config_data.get("model_type")
    num_layers = config_data.get("num_hidden_layers")
    layer_types = config_data.get("layer_types")
    if model_type != "qwen2" or not isinstance(num_layers, int):
        return

    if not isinstance(layer_types, list):
        logging.warning(
            f"config.json in {output_dir} has invalid layer_types: {type(layer_types)}"
        )
        return

    if len(layer_types) == num_layers:
        return

    if len(layer_types) > num_layers:
        layer_types = layer_types[:num_layers]
    elif len(layer_types) == 0:
        logging.warning(
            f"config.json in {output_dir} has empty layer_types but needs {num_layers} entries"
        )
        return
    else:
        layer_types = layer_types + [layer_types[-1]] * (num_layers - len(layer_types))

    config_data["layer_types"] = layer_types
    try:
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(config_data, config_file, indent=2, sort_keys=True)
        logging.info(
            f"Sanitized config.json in {output_dir} so layer_types matches num_hidden_layers"
        )
    except OSError as exc:
        logging.warning(f"Failed to write sanitized config.json in {output_dir}: {exc}")

# Set environment variables for Hugging Face and WandB tokens
#os.environ["HF_TOKEN"] = "{your_HF_token}"
#os.environ["WANDB_API_KEY"] = "{your_WandB_API_key}"
#wandb.login()

def main():
    parser = argparse.ArgumentParser(
        description="Training LLMs with custom embeddings and Fourier loss"
    )
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--int_digit_len', type=int, default=7, help='Number of digits for integer part')
    parser.add_argument('--frac_digit_len', type=int, default=5, help='Number of digits for fractional part')
    parser.add_argument('--len_gen_size', type=int, default=0, help='FNE: add k 0s after numbers to len gen')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--name', type=str, default='', help='Log name')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Model name')
    parser.add_argument('--dataset', type=str, default='openai/gsm8k', help='Dataset name')
    parser.add_argument('--train_from_scratch', action='store_true', help='Train the model from scratch without pre-trained weights')
    parser.add_argument('--use_digit_wise_tokenizer', action='store_true', help='Whether to use digit-wise tokenizer')
    parser.add_argument('--num_train_samples', type=int, default=None, help='Number of training samples to use')
    parser.add_argument('--num_test_samples', type=int, default=None, help='Number of test samples to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model_size_level', type=int, default=-1, help='From 1 to 8, choose the model size for training from scratch')
    parser.add_argument('--method', type=str, choices=['regular', 'fne', 'xval', 'vanilla'], default='fne', help='Training method: regular, fne, xval, or vanilla')
    parser.add_argument('--scheduler_name', type=str, default='cosine', help='Name of the learning rate scheduler (e.g., linear, constant, cosine, etc.)')
    parser.add_argument('--period_base_list', type=str, nargs='+', default=['10'], help='List of period bases for Fourier embedding (e.g., 2, 5, 1/3)')
    parser.add_argument('--clip', action='store_true', help='Enable clipping')
    parser.add_argument('--not_add_linear', action='store_true', help='Do not add linear layer after FNE')
    parser.add_argument('--test_results_filename', type=str, default='', help='Filename to use when saving test predictions to CSV')
    parser.add_argument('--model_save_name', type=str, default='', help='Directory name for saving the trained model artifacts')
    
    # Added flag argument to enable LoRA to reduce memory usage
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for training')

    args = parser.parse_args()
    
    # Convert period_base_list strings to floats (handling fractions)
    args.add_linear = not args.not_add_linear
    args.period_base_list = parse_period_base_list(args.period_base_list)
    
    run_name = f"{args.name}{args.method}_{args.model}_{args.dataset}_seed{args.seed}"
    sanitized_run_name = sanitize_filename(run_name)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_results_dir = os.path.join(base_dir, "test_results")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(test_results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    results_filename = args.test_results_filename.strip()
    if results_filename:
        results_filename = sanitize_filename(results_filename)
        if not results_filename.lower().endswith('.csv'):
            results_filename += '.csv'
    else:
        results_filename = f"{sanitized_run_name}.csv"
    test_results_path = os.path.join(test_results_dir, results_filename)

    model_dir_name = args.model_save_name.strip() or sanitized_run_name
    model_dir_name = sanitize_filename(model_dir_name)
    model_save_path = os.path.join(models_dir, model_dir_name)

    args.test_results_filename = results_filename
    args.test_results_path = test_results_path
    args.model_save_name = model_dir_name
    args.model_save_path = model_save_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    wandb.init(
        project="FoNE_Qwen",
        config=vars(args),
        name=run_name
    )
    
    output_folder = get_output_folder(args)
    setup_logger(output_folder)
    
    logging.info(output_folder)
    logging.info(args)
    logging.info(f"Test results will be saved to: {args.test_results_path}")
    logging.info(f"Model artifacts will be saved to: {args.model_save_path}")
    
    if ',' in args.dataset:
        args.dataset = tuple(args.dataset.split(','))
        logging.info(f"Dataset specified as tuple: {args.dataset}")
    else:
        logging.info(f"Dataset specified as single name: {args.dataset}")
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        cache_dir="./hg_cache", # user-agnostive cache directory
        device=device,
        train_from_scratch=args.train_from_scratch,
        size_level=args.model_size_level,
        use_digit_wise_tokenizer=args.use_digit_wise_tokenizer,
        use_lora=args.use_lora # pass new argument
    )
    
    # Run the training pipeline
    training_summary = create_dataloader_and_train(args, model, tokenizer, device)

    records = (training_summary or {}).get("records", [])
    write_test_results_csv(args.test_results_path, records)
    logging.info(f"Saved test predictions CSV with {len(records)} rows to {args.test_results_path}")

    existing_artifacts = os.path.isdir(args.model_save_path) and os.listdir(args.model_save_path)
    if existing_artifacts:
        logging.warning(f"Model save directory '{args.model_save_path}' already exists and will be overwritten.")
    os.makedirs(args.model_save_path, exist_ok=True)
    model.save_pretrained(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)
    logging.info(f"Saved model and tokenizer to {args.model_save_path}")
    sanitize_qwen_config(args.model_save_path)

    number_encoder_state = (training_summary or {}).get("number_encoder_state")
    number_encoder_config = (training_summary or {}).get("number_encoder_config")
    if number_encoder_state is not None:
        number_encoder_weights_path = os.path.join(args.model_save_path, "number_encoder.pt")
        torch.save(number_encoder_state, number_encoder_weights_path)
        logging.info(f"Saved number encoder weights to {number_encoder_weights_path}")
        if number_encoder_config is not None:
            number_encoder_config_path = os.path.join(args.model_save_path, "number_encoder_config.json")
            with open(number_encoder_config_path, 'w', encoding='utf-8') as config_file:
                json.dump(number_encoder_config, config_file, indent=2, sort_keys=True)
            logging.info(f"Saved number encoder config to {number_encoder_config_path}")
    
    wandb.finish()

if __name__ == "__main__":
    main()
