import os
import re
import torch
import logging

def is_numeric(s):
    """
    Check if a string represents a valid float.
    """
    return bool(re.match(r"^-?\d+(\.\d+)?$", s))

def handle_nan_loss(batch, before_decoder, data_idx, model, save_path="debug_nan_loss.pt"):
    """
    Handles NaN loss by saving the problematic batch and model state before the decoder.
    """
    debug_data = {
        "batch": batch,
        "before_decoder": before_decoder.cpu() if before_decoder is not None else None,
        "data_idx": data_idx,
        "model_state": model.state_dict(),
    }
    save_path = os.path.join("fail_case_log", save_path)
    torch.save(debug_data, save_path)
    logging.info(f"Saved debug data to {save_path}. Stopping training due to NaN loss.")

def get_regular_embeddings(model, input_ids):
    """
    Returns the token embeddings based on the model type, handling PEFT wrappers.
    """
    return model.get_input_embeddings()(input_ids)