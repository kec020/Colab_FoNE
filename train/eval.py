import logging
from typing import Any, Optional

import torch

from .utils import is_numeric, get_regular_embeddings


DEFAULT_ROUND_DECIMALS = 4


def _safe_decimals(decimals: Optional[int]) -> int:
    return max(int(decimals) if decimals is not None else DEFAULT_ROUND_DECIMALS, 0)


def _round_tensor(tensor: torch.Tensor, decimals: Optional[int] = None) -> torch.Tensor:
    """Round tensor values to the requested decimal places."""
    decimals = _safe_decimals(decimals)
    factor = 10 ** decimals if decimals else 1
    return torch.round(tensor * factor) / factor if decimals else torch.round(tensor)


def _round_value(value: Any, decimals: Optional[int] = None):
    """Round plain Python numeric values while leaving other types untouched."""
    decimals = _safe_decimals(decimals)
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return value


def _format_numeric_str(value: Any, decimals: Optional[int] = None) -> Any:
    """Represent numeric values with fixed decimal precision for logs/records."""
    decimals = _safe_decimals(decimals)
    if isinstance(value, (int, float)):
        return f"{float(value):.{decimals}f}"
    return value

def evaluate_fne(model, test_loader, fne, int_digit_len, frac_digit_len, device, print_labels=False, max_print=10):
    """Evaluation loop for Fourier Neural Embedding (FNE) based models."""
    logging.info('Evaluation start')
    model.eval()
    fne.eval()

    round_decimals = _safe_decimals(frac_digit_len + 1)
    rounded_atol = float(10 ** (-round_decimals)) if round_decimals else 1.0

    total_correct = 0
    total_samples = 0
    total_loss = 0
    total_squared_error = 0
    total_digits = 0
    correct_digits = 0
    mispredictions = []
    records = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            scatter_tensor = batch['scatter_tensor'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            last_token_mask = batch['last_token_mask'].to(device)
            raw_questions = batch.get('raw_questions', [''] * labels.size(0))

            regular_embeddings = get_regular_embeddings(model, input_ids)
            fourier_embeddings = fne(scatter_tensor)
            input_embeddings = regular_embeddings + fourier_embeddings

            outputs = model(
                inputs_embeds=input_embeddings.to(dtype=model.dtype),
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            before_decoder = outputs.hidden_states[-1]
            last_token_hidden_state = (before_decoder * last_token_mask.unsqueeze(-1)).sum(dim=1)

            predicted_numbers = fne.fourier_compute_prediction(
                last_token_hidden_state,
                int_digit_len,
                frac_digit_len
            )

            tolerance = 10 ** (-frac_digit_len)
            rounded_preds = _round_tensor(predicted_numbers, round_decimals)
            rounded_labels = _round_tensor(labels, round_decimals)
            exact_match = torch.abs(predicted_numbers - labels) < tolerance
            rounded_match = torch.isclose(rounded_preds, rounded_labels, atol=rounded_atol)
            correct_predictions = torch.logical_or(exact_match, rounded_match)
            total_correct += correct_predictions.sum().item()
            total_samples += labels.size(0)

            all_labels.append(rounded_labels.cpu())

            for i in range(labels.size(0)):
                actual_value = _format_numeric_str(rounded_labels[i].item(), round_decimals)
                predicted_value = _format_numeric_str(rounded_preds[i].item(), round_decimals)
                min_len = len(actual_value)
                correct_digits += sum(1 for a, p in zip(actual_value[:min_len], predicted_value[:min_len]) if a == p)
                total_digits += len(actual_value)

            for i in range(labels.size(0)):
                if not correct_predictions[i]:
                    mispredictions.append((rounded_preds[i].item(), rounded_labels[i].item()))

            for i in range(labels.size(0)):
                pred_value = rounded_preds[i].item()
                label_value = rounded_labels[i].item()
                raw_abs_error = abs(predicted_numbers[i].item() - labels[i].item())
                abs_error = _round_value(raw_abs_error, round_decimals)
                question_text = raw_questions[i] if i < len(raw_questions) else ''
                records.append({
                    'question': question_text,
                    'label': label_value,
                    'prediction': pred_value,
                    'abs_error': abs_error,
                    'correct': bool(correct_predictions[i].item()),
                    'round_decimals': round_decimals,
                })

            total_squared_error += torch.sum((rounded_preds - rounded_labels) ** 2).item()
            loss = fne.fourier_compute_loss(
                last_token_hidden_state,
                labels,
                int_digit_len,
                frac_digit_len
            )
            total_loss += loss.item()

    all_labels_tensor = torch.cat(all_labels)
    mean_label = all_labels_tensor.mean().item()
    total_variance = torch.sum((all_labels_tensor - mean_label) ** 2).item()

    avg_loss = total_loss / len(test_loader)
    whole_number_accuracy = total_correct / total_samples if total_samples else 0.0
    digit_wise_accuracy = correct_digits / total_digits if total_digits else 0.0
    mse = total_squared_error / total_samples if total_samples else float('nan')
    r2 = 1 - (total_squared_error / total_variance) if total_variance > 0 else float('nan')

    if print_labels:
        if mispredictions:
            log_count = min(len(mispredictions), max_print)
            logging.info(f"Mispredictions (up to {log_count} examples):")
            for i in range(log_count):
                predicted_val, actual_val = mispredictions[i]
                logging.info(
                    f"Predicted: {_format_numeric_str(predicted_val, round_decimals)}, Actual: {_format_numeric_str(actual_val, round_decimals)}"
                )
        else:
            logging.info("No mispredictions found!")

    return avg_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2, records

def evaluate_regular(model, dataloader, tokenizer, device, print_labels=False, max_print_examples=10):
    """Evaluation loop for regular models."""
    logging.info('Evaluation start')
    model.eval()

    round_decimals = _safe_decimals(DEFAULT_ROUND_DECIMALS)

    total_loss = 0
    total_examples = 0
    total_correct_examples = 0
    total_characters = 0
    correct_characters = 0
    total_squared_error = 0
    all_labels = []
    printed_examples = 0
    records = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            raw_questions = batch.get('raw_questions', [''] * input_ids.size(0))

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            examplelist = []

            for i in range(len(input_ids)):
                label_indices = (labels[i] != -100).nonzero(as_tuple=True)[0]
                actual_tokens = input_ids[i, label_indices].cpu().numpy()
                predicted_tokens = predictions[i, label_indices - 1].cpu().numpy()
                actual_label = tokenizer.decode(actual_tokens, skip_special_tokens=True).strip()
                predicted_label = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip()

                numeric_actual = is_numeric(actual_label)
                numeric_pred = is_numeric(predicted_label)
                rounded_actual = actual_label
                rounded_pred = predicted_label

                if actual_label == predicted_label:
                    total_correct_examples += 1
                total_examples += 1

                if numeric_pred and numeric_actual:
                    actual_value = float(actual_label)
                    predicted_value = float(predicted_label)
                    rounded_actual = _round_value(actual_value, round_decimals)
                    rounded_pred = _round_value(predicted_value, round_decimals)
                    total_squared_error += (rounded_actual - rounded_pred) ** 2
                    all_labels.append(rounded_actual)

                max_len = max(len(actual_label), len(predicted_label))
                padded_actual = actual_label.ljust(max_len)
                padded_predicted = predicted_label.ljust(max_len)
                correct_characters += sum(1 for a, p in zip(padded_actual, padded_predicted) if a == p)
                total_characters += max_len

                abs_error = ''
                if numeric_pred and numeric_actual:
                    abs_error = _round_value(float(predicted_label) - float(actual_label), round_decimals)
                    abs_error = abs(abs_error)

                display_label = (_format_numeric_str(rounded_actual, round_decimals)
                                  if numeric_actual else actual_label)
                display_prediction = (_format_numeric_str(rounded_pred, round_decimals)
                                      if numeric_pred else predicted_label)
                record_label = float(rounded_actual) if numeric_actual else actual_label
                record_prediction = float(rounded_pred) if numeric_pred else predicted_label

                records.append({
                    'question': raw_questions[i] if i < len(raw_questions) else '',
                    'label': record_label,
                    'prediction': record_prediction,
                    'abs_error': abs_error,
                    'correct': actual_label == predicted_label,
                    'round_decimals': round_decimals,
                })

                if print_labels and printed_examples < max_print_examples:
                    examplelist.append(
                        f"({display_prediction}, {display_label})"
                    )
                    printed_examples += 1

            if print_labels and examplelist:
                logging.info(" ".join(examplelist))

    avg_loss = total_loss / len(dataloader)
    whole_number_accuracy = total_correct_examples / total_examples if total_examples else 0.0
    digit_wise_accuracy = correct_characters / total_characters if total_characters else 0.0

    if all_labels:
        mean_label = sum(all_labels) / len(all_labels)
        total_variance = sum((label - mean_label) ** 2 for label in all_labels)
        mse = total_squared_error / len(all_labels) if all_labels else float('nan')
        r2 = 1 - (total_squared_error / total_variance) if total_variance > 0 else float('nan')
    else:
        mse = -1
        r2 = float('nan')

    return avg_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2, records

def evaluate_xval(model, test_loader, xval, device, print_labels=False, max_print=10):
    """Evaluation loop for models using the xval module."""
    logging.info('Evaluation start')
    model.eval()
    xval.eval()

    round_decimals = _safe_decimals(DEFAULT_ROUND_DECIMALS)
    rounded_atol = float(10 ** (-round_decimals)) if round_decimals else 1.0

    total_correct = 0
    total_samples = 0
    total_loss = 0
    total_squared_error = 0
    total_digits = 0
    correct_digits = 0
    printed_examples = 0
    all_labels = []
    records = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            scatter_tensor = batch['scatter_tensor'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            last_token_mask = batch['last_token_mask'].to(device)
            raw_questions = batch.get('raw_questions', [''] * labels.size(0))

            regular_embeddings = get_regular_embeddings(model, input_ids)
            input_embeddings = xval(scatter_tensor, regular_embeddings)

            outputs = model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            before_decoder = outputs.hidden_states[-1]
            last_token_hidden_state = (before_decoder * last_token_mask.unsqueeze(-1)).sum(dim=1)

            predicted_numbers = xval.compute_prediction(last_token_hidden_state)

            tolerance = 0.5  # Example tolerance value
            rounded_preds = _round_tensor(predicted_numbers, round_decimals)
            rounded_labels = _round_tensor(labels, round_decimals)
            exact_match = torch.abs(predicted_numbers - labels) < tolerance
            rounded_match = torch.isclose(rounded_preds, rounded_labels, atol=rounded_atol)
            correct_predictions = torch.logical_or(exact_match, rounded_match)

            total_correct += correct_predictions.sum().item()
            total_samples += labels.size(0)
            all_labels.extend(rounded_labels.cpu().numpy())

            for i in range(labels.size(0)):
                actual_value = _format_numeric_str(rounded_labels[i].item(), round_decimals)
                predicted_value = _format_numeric_str(rounded_preds[i].item(), round_decimals)
                min_len = len(actual_value)
                correct_digits += sum(1 for a, p in zip(actual_value[:min_len], predicted_value[:min_len]) if a == p)
                total_digits += len(actual_value)

            loss = xval.compute_loss(last_token_hidden_state, labels)
            total_loss += loss.item()
            total_squared_error += torch.sum((rounded_preds - rounded_labels) ** 2).item()

            for i in range(labels.size(0)):
                pred_value = rounded_preds[i].item()
                label_value = rounded_labels[i].item()
                abs_error = _round_value(predicted_numbers[i].item() - labels[i].item(), round_decimals)
                abs_error = abs(abs_error)
                question_text = raw_questions[i] if i < len(raw_questions) else ''
                records.append({
                    'question': question_text,
                    'label': label_value,
                    'prediction': pred_value,
                    'abs_error': abs_error,
                    'correct': bool(correct_predictions[i].item()),
                    'round_decimals': round_decimals,
                })

            if print_labels and printed_examples < max_print:
                output_pairs = []
                for i in range(len(labels)):
                    if printed_examples >= max_print:
                        break
                    actual_label = _format_numeric_str(rounded_labels[i].item(), round_decimals)
                    predicted_label = _format_numeric_str(rounded_preds[i].item(), round_decimals)
                    output_pairs.append((predicted_label, actual_label))
                    printed_examples += 1
                logging.info("Predictions and Labels: " + " ".join(f"({pred},{lbl})" for pred, lbl in output_pairs))

    avg_loss = total_loss / len(test_loader)
    whole_number_accuracy = total_correct / total_samples if total_samples else 0.0
    digit_wise_accuracy = correct_digits / total_digits if total_digits else 0.0
    mse = total_squared_error / total_samples if total_samples else float('nan')

    if total_samples > 1:
        mean_label = sum(all_labels) / len(all_labels)
        total_variance = sum((label - mean_label) ** 2 for label in all_labels)
        r2 = 1 - (total_squared_error / total_variance) if total_variance > 0 else float('nan')
    else:
        r2 = float('nan')

    return avg_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2, records

def evaluate_vanilla(model, test_loader, vanilla_model, device, print_labels=False, max_print=10):
    """Evaluation loop for models using the vanilla embedding module."""
    model.eval()
    vanilla_model.eval()

    round_decimals = _safe_decimals(vanilla_model.frac_digit_len + 1)
    rounded_atol = float(10 ** (-round_decimals)) if round_decimals else 1.0

    total_correct = 0
    total_samples = 0
    total_loss = 0
    total_squared_error = 0
    total_digits = 0
    correct_digits = 0
    mispredictions = []
    records = []
    label_values = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            scatter_tensor = batch['scatter_tensor'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            last_token_mask = batch['last_token_mask'].to(device)
            raw_questions = batch.get('raw_questions', [''] * labels.size(0))

            regular_embeddings = get_regular_embeddings(model, input_ids)
            vanilla_embeddings = vanilla_model(scatter_tensor)
            input_embeddings = regular_embeddings + vanilla_embeddings

            outputs = model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            last_hidden_state = outputs.hidden_states[-1]
            last_token_hidden_state = (last_hidden_state * last_token_mask.unsqueeze(-1)).sum(dim=1)

            predicted_numbers = vanilla_model.compute_prediction(last_token_hidden_state)

            tolerance = 10 ** (-vanilla_model.frac_digit_len)
            rounded_preds = _round_tensor(predicted_numbers, round_decimals)
            rounded_labels = _round_tensor(labels, round_decimals)
            exact_correct = torch.abs(predicted_numbers - labels) < tolerance
            rounded_match = torch.isclose(rounded_preds, rounded_labels, atol=rounded_atol)
            correct = torch.logical_or(exact_correct, rounded_match)
            total_correct += correct.sum().item()
            total_samples += labels.size(0)

            scale_factor = 10 ** vanilla_model.frac_digit_len
            scaled_labels = torch.round(rounded_labels * scale_factor).long()
            scaled_preds = torch.round(rounded_preds * scale_factor).long()

            for i in range(labels.size(0)):
                label_digits = []
                pred_digits = []
                num = scaled_labels[i]
                for p in vanilla_model.powers_of_ten:
                    label_digits.append((num // p) % 10)
                num = scaled_preds[i]
                for p in vanilla_model.powers_of_ten:
                    pred_digits.append((num // p) % 10)
                for l, p in zip(label_digits, pred_digits):
                    if l == p:
                        correct_digits += 1
                    total_digits += 1

            for i in range(labels.size(0)):
                pred_value = rounded_preds[i].item()
                label_value = rounded_labels[i].item()
                raw_abs_error = abs(predicted_numbers[i].item() - labels[i].item())
                abs_error = _round_value(raw_abs_error, round_decimals)
                label_values.append(label_value)
                question_text = raw_questions[i] if i < len(raw_questions) else ''
                records.append({
                    'question': question_text,
                    'label': label_value,
                    'prediction': pred_value,
                    'abs_error': abs_error,
                    'correct': bool(correct[i].item()),
                    'round_decimals': round_decimals,
                })

                if not correct[i]:
                    mispredictions.append((pred_value, label_value))

            loss = vanilla_model.compute_loss(last_token_hidden_state, labels)
            total_loss += loss.item()
            total_squared_error += torch.sum((rounded_preds - rounded_labels) ** 2).item()

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples if total_samples else 0.0
    digit_accuracy = correct_digits / total_digits if total_digits else 0.0
    mse = total_squared_error / total_samples if total_samples else float('nan')

    if label_values:
        mean_label = sum(label_values) / len(label_values)
        total_variance = sum((value - mean_label) ** 2 for value in label_values)
        r2 = 1 - (total_squared_error / total_variance) if total_variance != 0 else float('nan')
    else:
        r2 = float('nan')

    if print_labels and mispredictions:
        logging.info(f"Mispredictions (first {max_print}):")
        for pred, true in mispredictions[:max_print]:
            logging.info(
                f"Predicted: {_format_numeric_str(pred, round_decimals)}, True: {_format_numeric_str(true, round_decimals)}"
            )

    return avg_loss, (accuracy, digit_accuracy), mse, r2, records
