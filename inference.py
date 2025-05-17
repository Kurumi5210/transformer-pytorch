from pathlib import Path
import torch
import torch.nn as nn
from config import get_config, get_weight_file_path
from train import get_model, get_dataset, run_validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用{device}：推理")
config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

model_filename = get_weight_file_path(config, '1000')
# model_filename = '/home/xiao/AIandML/transformer-pytorch/weights/transformer_model_1000.pt'
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])

run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: print(msg), 0, None, num_examples=10)
