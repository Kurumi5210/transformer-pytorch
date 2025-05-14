from pathlib import Path

def get_config():
    return {
        "batch_size":32,
        "num_epochs":20,
        "lr":1e-4,
        "seq_len":512,
        "d_model":512,
        "lang_src":"en",
        "lang_tgt":"ru",
        "model_folder": "weights",
        "model_basename":"transformer_model_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "run/transformer_model" 
    }

def get_weight_file_path(config, epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.')/model_folder / model_filename)