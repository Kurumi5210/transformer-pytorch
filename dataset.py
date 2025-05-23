import torch
import torch.nn as nn
from torch.utils.data import Dataset

def causal_mask(size):  # 返回所有对角线以下的值
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0 

class BuildDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len)->None:
        super().__init__()

        self.ds = ds
        print(ds)
        assert(0)
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_toeken = torch.tensor([tokenizer_src.token_to_id('[SOS]')],dtype=torch.int64)
        self.eos_toeken = torch.tensor([tokenizer_src.token_to_id('[EOS]')],dtype=torch.int64)
        self.pad_toeken = torch.tensor([tokenizer_src.token_to_id('[PAD]')],dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index:any)->any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_src.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1   # decode补开头 label补结束

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('原始句子太长了')
        
        encoder_input = torch.cat(
            [
                self.sos_toeken,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_toeken,
                torch.tensor([self.pad_toeken]*enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_toeken,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_toeken]*dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_toeken,
                torch.tensor([self.pad_toeken]*dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_toeken).unsqueeze(0).unsqueeze(0),    #(1, 1, seq_len)
            # decode的因果掩码  (1, seq_len) & (1, seq_len, seq_len)
            "decoder_mask": (encoder_input != self.pad_toeken).unsqueeze(0).unsqueeze(0) & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
