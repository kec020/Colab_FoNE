import torch
import logging
from .utils import is_numeric, get_regular_embeddings

def evaluate_fne(model, test_loader, fne, int_digit_len, frac_digit_len, device, print_labels=False, max_print=10):
    """Evaluation loop for Fourier Neural Embedding (FNE) based models."""
    logging.info('Evaluation start')
    model.eval()
    fne.eval()

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
            correct_predictions = torch.abs(predicted_numbers - labels) < tolerance
            total_correct += correct_predictions.sum().item()
            total_samples += labels.size(0)

            all_labels.append(labels.cpu())

            for i in range(labels.size(0)):
                actual_value = str(labels[i].item())
                predicted_value = str(predicted_numbers[i].item())
                min_len = len(actual_value)
                correct_digits += sum(1 for a, p in zip(actual_value[:min_len], predicted_value[:min_len]) if a == p)
                total_digits += len(actual_value)

            for i in range(labels.size(0)):
                if not correct_predictions[i]:
                    mispredictions.append((predicted_numbers[i].item(), labels[i].item()))

            for i in range(labels.size(0)):
                pred_value = predicted_numbers[i].item()
                label_value = labels[i].item()
                abs_error = abs(pred_value - label_value)
                question_text = raw_questions[i] if i < len(raw_questions) else ''
                records.append({
                    'question': question_text,
                    'label': label_value,
                    'prediction': pred_value,
                    'abs_error': abs_error,
                    'correct': bool(correct_predictions[i].item())
                })

            total_squared_error += torch.sum((predicted_numbers - labels) ** 2).item()
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
                logging.info(f"Predicted: {predicted_val}, Actual: {actual_val}")
        else:
            logging.info("No mispredictions found!")

    return avg_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2, records

def evaluate_regular(model, dataloader, tokenizer, device, print_labels=False, max_print_examples=10):
    """Evaluation loop for regular models."""
    logging.info('Evaluation start')
    model.eval()

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

                if actual_label == predicted_label:
                    total_correct_examples += 1
                total_examples += 1

                if is_numeric(predicted_label) and is_numeric(actual_label):
                    actual_value = float(actual_label)
                    predicted_value = float(predicted_label)
                    total_squared_error += (actual_value - predicted_value) ** 2
                    all_labels.append(actual_value)

                max_len = max(len(actual_label), len(predicted_label))
                padded_actual = actual_label.ljust(max_len)
                padded_predicted = predicted_label.ljust(max_len)
                correct_characters += sum(1 for a, p in zip(padded_actual, padded_predicted) if a == p)
                total_characters += max_len

                abs_error = ''
                if is_numeric(predicted_label) and is_numeric(actual_label):
                    abs_error = abs(float(predicted_label) - float(actual_label))

                records.append({
                    'question': raw_questions[i] if i < len(raw_questions) else '',
                    'label': actual_label,
                    'prediction': predicted_label,
                    'abs_error': abs_error,
                    'correct': actual_label == predicted_label
                })

                if print_labels and printed_examples < max_print_examples:
                    examplelist.append(f"({predicted_label}, {actual_label})")
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
            correct_predictions = torch.abs(predicted_numbers - labels) < tolerance

            total_correct += correct_predictions.sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())

            for i in range(labels.size(0)):
                actual_value = str(labels[i].item())
                predicted_value = str(predicted_numbers[i].item())
                min_len = len(actual_value)
                correct_digits += sum(1 for a, p in zip(actual_value[:min_len], predicted_value[:min_len]) if a == p)
                total_digits += len(actual_value)

            loss = xval.compute_loss(last_token_hidden_state, labels)
            total_loss += loss.item()
            total_squared_error += torch.sum((predicted_numbers - labels) ** 2).item()

            for i in range(labels.size(0)):
                pred_value = predicted_numbers[i].item()
                label_value = labels[i].item()
                abs_error = abs(pred_value - label_value)
                question_text = raw_questions[i] if i < len(raw_questions) else ''
                records.append({
                    'question': question_text,
                    'label': label_value,
                    'prediction': pred_value,
                    'abs_error': abs_error,
                    'correct': bool(correct_predictions[i].item())
                })

            if print_labels and printed_examples < max_print:
                output_pairs = []
                for i in range(len(labels)):
                    if printed_examples >= max_print:
                        break
                    actual_label = labels[i].cpu().numpy()
                    predicted_label = predicted_numbers[i].cpu().numpy()
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
            correct = torch.abs(predicted_numbers - labels) < tolerance
            total_correct += correct.sum().item()
            total_samples += labels.size(0)

            scaled_labels = (labels * (10 ** vanilla_model.frac_digit_len)).long()
            scaled_preds = (predicted_numbers * (10 ** vanilla_model.frac_digit_len)).long()

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
                pred_value = predicted_numbers[i].item()
                label_value = labels[i].item()
                abs_error = abs(pred_value - label_value)
                label_values.append(label_value)
                question_text = raw_questions[i] if i < len(raw_questions) else ''
                records.append({
                    'question': question_text,
                    'label': label_value,
                    'prediction': pred_value,
                    'abs_error': abs_error,
                    'correct': bool(correct[i].item())
                })

                if not correct[i]:
                    mispredictions.append((pred_value, label_value))

            loss = vanilla_model.compute_loss(last_token_hidden_state, labels)
            total_loss += loss.item()
            total_squared_error += torch.sum((predicted_numbers - labels) ** 2).item()

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
            logging.info(f"Predicted: {pred:.5f}, True: {true:.5f}")

    return avg_loss, (accuracy, digit_accuracy), mse, r2, records
