import gc
import torch
torch.cuda.empty_cache()
gc.collect()

import os
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, AutoConfig
from transformers import BitsAndBytesConfig
import torch

data_num=-1
max_source_len = 512
max_target_len=512
#cache_data = 'cache_data/summarize_python'
cache_data = 'batchsize_experiments/pandu'
load = 'Salesforce/codet5p-16b'
# load = "HuggingFaceH4/starchat-alpha"
# Training
epochs=1
lr=5e-3
lr_warmup_steps=200
batch_size_per_replica=1
grad_acc_steps=16
local_rank=-1
deepspeed="ds.json"
fp16=True

# Logging and stuff
save_dir="saved_models/pandas_tune"
log_freq=10
save_freq=500
os.makedirs(save_dir, exist_ok=True)


def run_training(model, train_data, tokenizer):
    print(f"Starting main loop")

    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',

        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size_per_replica,
        gradient_accumulation_steps=grad_acc_steps,
        learning_rate=lr,
        weight_decay=0.05,
        warmup_steps=lr_warmup_steps,

        logging_dir=save_dir,
        logging_first_step=True,
        logging_steps=log_freq,
        save_total_limit=1,

        dataloader_drop_last=True,
        dataloader_num_workers=2,

        local_rank=local_rank,
#         deepspeed=deepspeed,
#         fp16=fp16,
        bf16 = True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    trainer.train()

    if local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')
        tokenizer.save_pretrained(final_checkpoint_dir)

#for deepspeed
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9994' # modify if RuntimeError: Address already in use
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1"

tokenizer = AutoTokenizer.from_pretrained(load)
config = AutoConfig.from_pretrained(load, trust_remote_code=True, revision="main")
config.decoder_start_token_id = tokenizer.bos_token_id
config.pad_token_id = tokenizer.pad_token_id
#for deepspeed
config.max_position_embeddings = max_source_len
# print('Model hidden size: ', config.cross_attention_hidden_size)

import math
import numpy as np
def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))

def freeze_decoder_except_xattn_codegen(model):
    print(f'Para before freezing: {model.num_parameters()}, trainable para: {get_model_size(model)}')
    for param in model.decoder.parameters():
        param.requires_grad = False

    num_decoder_layers = model.decoder.config.n_layer
    for i in range(num_decoder_layers):
        each_decoder_layer = model.decoder.transformer.h[i]
        if hasattr(each_decoder_layer, 'crossattention'):
            for param in each_decoder_layer.crossattention.parameters():
                param.requires_grad = True
            each_decoder_layer.crossattention.to(torch.float32)

        if hasattr(each_decoder_layer, 'alpha_xattn'):
            each_decoder_layer.alpha_xattn.requires_grad = True
    print(f'Para after freezing: {model.num_parameters()}, trainable para: {get_model_size(model)}')

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def preprocess_function(examples):
    source = [ex for ex in examples["question"]]
#     source = [ex for ex in examples["input"]]
    target = [ex for ex in examples["answer"]]

    model_inputs = tokenizer(source, max_length=max_source_len, padding="max_length", truncation=True)
    labels = tokenizer(target, max_length=max_target_len, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"].copy()
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
    ]
    return model_inputs


def load_tokenize_data():
#     if os.path.exists(cache_data):
#         train_data = load_from_disk(cache_data)
#         print(f'  ==> Loaded {len(train_data)} samples')
# #         res =  convert_size(train_data.size_in_bytes)
# #         print('Dataset Size:', res)
#         return train_data, config
#     else:
        train_data = load_from_disk("/home/unnati/batchsize_experiments/pandu", )
        
#         print(datasets)
# #         datasets = load_dataset("semeru/text-code-codesummarization", split="validation")
#         # datasets = datasets.select(range(len(datasets)))
#         # res =  convert_size(datasets.size_in_bytes)
#         # print('Dataset Size:', res)
#         train_data = datasets.map(
#             preprocess_function,
#             # batched=True,
#             remove_columns=datasets.column_names,
#             # num_proc=64,
#             load_from_cache_file=False,
#         )
#         print(f'  ==> Loaded {len(train_data)} samples')
#         # train_data.save_to_disk(cache_data)
        # print(f'  ==> Saved to {cache_data}')

        return train_data, config

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

train_data, config = load_tokenize_data()

#LORA
from peft import LoraConfig, get_peft_model, TaskType
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj","v_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type=TaskType.SEQ_2_SEQ_LM
# )


#QLORA

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

qlora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=["q_proj","v_proj"]
)


if data_num != -1:
    train_data = train_data.select([i for i in range(data_num)])

model = AutoModelForSeq2SeqLM.from_pretrained(load,config=config,trust_remote_code=True,
                                              revision="main", 
                                              low_cpu_mem_usage=True, 
                                              quantization_config=bnb_config)
# freeze_decoder_except_xattn_codegen(model)
print('Model: ', convert_size(model.get_memory_footprint()))
model = get_peft_model(model, qlora_config)
print('PEFT Model: ',convert_size(model.get_memory_footprint()))
print(model.print_trainable_parameters())

print(f"  ==> Loaded model from {load}, model size {model.num_parameters()}")

run_training(model, train_data, tokenizer)
