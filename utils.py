# import torch
# import tensorflow as tf
# import os
# import logging


# def restore_checkpoint(ckpt_dir, state, device):
#   if not tf.io.gfile.exists(ckpt_dir):
#     tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
#     logging.warning(f"No checkpoint found at {ckpt_dir}. "
#                     f"Returned the same state as input")
#     return state
#   else:
#     loaded_state = torch.load(ckpt_dir, map_location=device)
#     state['optimizer'].load_state_dict(loaded_state['optimizer'])
#     state['model'].load_state_dict(loaded_state['model'], strict=False)
#     state['ema'].load_state_dict(loaded_state['ema'])
#     state['step'] = loaded_state['step']
#     return state


# def save_checkpoint(ckpt_dir, state):
#   saved_state = {
#     'optimizer': state['optimizer'].state_dict(),
#     'model': state['model'].state_dict(),
#     'ema': state['ema'].state_dict(),
#     'step': state['step']
#   }
#   torch.save(saved_state, ckpt_dir)

import torch
import os
import logging


def restore_checkpoint(ckpt_path, state, device):
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_path}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_path, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_path, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_path)


def log_allocated_memory():
    allocated = torch.mps.current_allocated_memory()
    driver = torch.mps.driver_allocated_memory()
    recommended = torch.mps.recommended_max_memory()
    gigabyte_size = 1024 * 1024 * 1024

    logging.debug("--------- MEMORY SUMMARY ---------")
    logging.debug("Current allocated: %.2fG", allocated / gigabyte_size)
    logging.debug("Driver allocated: %.2fG", driver / gigabyte_size)
    logging.debug("Recommended max: %.2fG", recommended / gigabyte_size)
    logging.debug("----------------------------------")
