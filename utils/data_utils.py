import re
import logging
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset


def split_dataset(dataset, test_size=0.1, seed=42):
    """
    Splits a dataset into train and test sets.
    """
    return dataset.train_test_split(test_size=test_size, seed=seed)


def create_scatter_tensor(batch_input_ids, batch_numbers, num_token_id):
    """
    Creates a tensor that maps the [NUM] token positions to the corresponding numeric values.
    """
    scatter_tensor_batch = []
    for input_ids, numbers in zip(batch_input_ids, batch_numbers):
        scatter_tensor = torch.zeros(len(input_ids), dtype=torch.float64)
        num_index = 0
        for i, token_id in enumerate(input_ids):
            if token_id == num_token_id:
                if num_index < len(numbers):
                    scatter_tensor[i] = numbers[num_index]
                    num_index += 1
        scatter_tensor_batch.append(scatter_tensor)
    return torch.stack(scatter_tensor_batch)


def preprocess(entry, is_tabular, method='regular'):
    """
    Preprocesses an entry based on its format and the selected method.
    """
    def preprocess_entry(entry):
        keys = list(entry.keys())
        features, label_key = keys[:-1], keys[-1]
        question = ' '.join(f'{entry[key]},' for key in features).strip(',')
        label = entry[label_key]
        return question, label

    if is_tabular:
        question, label = preprocess_entry(entry)
        original_question = question
    else: # adjusted for gsm8k dataset
            # question, label = str(entry["question"]), float(entry["label"])
            question = str(entry['question'])
            original_question = question
            answer_str = str(entry['answer'])
            matches = re.findall(r'\d+\.?\d*', answer_str)
            label = matches[-1] if matches else float('nan')
            # label = float(answer_str.split('####')[-1].strip().replace(',', '')) 

    if method in ['fne', 'xval', 'vanilla']:
        label = float(label)
        numbers = [float(num) for num in re.findall(r'\d+\.?\d*', question)]
        question_with_num = re.sub(r'\d+\.?\d*', ' [NUM] ', question)
        return {
            'question_with_num': question_with_num,
            'numbers': numbers,
            'label': label,
            'raw_question': original_question
        }
    elif method == 'regular':
        return {'question': question, 'label': label, 'raw_question': original_question}


def load_and_preprocess_dataset(dataset_name, tokenizer, num_train_samples=None, num_test_samples=None, method='regular'):
    """
    Loads and preprocesses a dataset.
    """

    is_tabular = 'tabular' in dataset_name if isinstance(dataset_name, str) else 'tabular' in dataset_name[0]
    dataset = (
        load_dataset('csv', data_files={'train': dataset_name})
        if isinstance(dataset_name, str) and dataset_name.endswith('.csv')
        else load_dataset(dataset_name) if isinstance(dataset_name, str) else load_dataset(*dataset_name)
    )
    # dataset = load_dataset(dataset_name) if isinstance(dataset_name, str) else load_dataset(*dataset_name)

    if 'test' not in dataset:
        logging.info("Splitting train set into train and test splits.")
        dataset = split_dataset(dataset['train'])

    logging.info(f"Train dataset length: {len(dataset['train'])}, Test dataset length: {len(dataset['test'])}")

    def preprocess_entry(entry):
        result = preprocess(entry, is_tabular, method)
        raw_question = result.get('raw_question', '')
        if method in ['fne', 'xval', 'vanilla']:
            input_text = result['question_with_num']
            input_ids = tokenizer.encode(input_text, return_tensors="pt").squeeze(0)
            return {
                'input_ids': input_ids,
                'numbers': result.get('numbers', []),
                'label': result['label'],
                'raw_question': raw_question,
                'raw_label': result['label']
            }
        elif method == 'regular':
            question_text = result['question']
            input_text = question_text + ' ' + str(result['label'])
            input_ids = tokenizer.encode(input_text, return_tensors="pt").squeeze(0)
            question_len = len(tokenizer.encode(question_text, return_tensors="pt").squeeze())
            return {
                'input_ids': input_ids,
                'question_len': question_len,
                'raw_question': raw_question,
                'raw_label': result['label']
            }

    train_data = (
        [preprocess_entry(entry) for entry in dataset['train'].select(range(num_train_samples))]
        if num_train_samples is not None else
        [preprocess_entry(entry) for entry in dataset['train']]
    )
    test_data = (
        [preprocess_entry(entry) for entry in dataset['test'].select(range(num_test_samples))]
        if num_test_samples is not None else
        [preprocess_entry(entry) for entry in dataset['test']]
    )

    return train_data, test_data


def collate_fn(batch, tokenizer, num_token_id=None, max_length=128, method='regular'):
    """
    Collates a batch of data for the given preprocessing method.
    """
    input_ids = [item['input_ids'] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    input_ids_padded = input_ids_padded[:, :max_length]
    attention_mask = (input_ids_padded != tokenizer.pad_token_id).long()

    if method in ['fne', 'xval', 'vanilla']:
        scatter_tensor = create_scatter_tensor(input_ids_padded, [item['numbers'] for item in batch], num_token_id)
        last_token_mask = torch.zeros_like(input_ids_padded, dtype=torch.float32)
        for i, seq in enumerate(input_ids_padded):
            last_non_pad_idx = (seq != tokenizer.pad_token_id).nonzero()[-1].item()
            last_token_mask[i, last_non_pad_idx] = 1
        raw_questions = [item.get('raw_question', '') for item in batch]
        raw_labels = [item.get('raw_label') for item in batch]
        return {
            'input_ids': input_ids_padded,
            'scatter_tensor': scatter_tensor,
            'attention_mask': attention_mask,
            'labels': torch.tensor([item['label'] for item in batch], dtype=torch.float64),
            'last_token_mask': last_token_mask,
            'raw_questions': raw_questions,
            'raw_labels': raw_labels
        }

    elif method == 'regular':
        question_lens = [item['question_len'] for item in batch]
        batch_size, seq_len = input_ids_padded.shape
        indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        question_lens_tensor = torch.tensor(question_lens, dtype=torch.long).unsqueeze(1)
        loss_mask = (indices >= question_lens_tensor).float() * attention_mask.float()
        labels = input_ids_padded.clone()
        labels[loss_mask == 0] = -100  # Ignore tokens during loss calculation
        raw_questions = [item.get('raw_question', '') for item in batch]
        raw_labels = [item.get('raw_label') for item in batch]
        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask,
            'labels': labels,
            'raw_questions': raw_questions,
            'raw_labels': raw_labels
        }
