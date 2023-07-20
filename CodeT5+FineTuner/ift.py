import torch
import os
import pprint
import argparse
import numpy as np
import math
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, AutoConfig
from transformers import BitsAndBytesConfig
from accelerate import Accelerator
import copy

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def get_model_param_count(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))

def convert_size(size_bytes, cnt=False):
   if size_bytes == 0:
       return "0B" 
   if cnt:
       size_name = (".", "K", "M", "B", "T", "P", "E", "Z", "Y")
   else:
       size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {convert_size(trainable_params, cnt=True)} || all params: {convert_size(all_param, cnt=True)} || trainable%: {100 * trainable_params / all_param}")

def freeze_decoder_except_xattn_codegen(model):
    print(f'Para before freezing: {convert_size(model.num_parameters(), cnt=True)}, trainable para: {get_model_param_count(model)}')
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
    print(f'Para after freezing: {model.num_parameters()}, trainable para: {get_model_param_count(model)}')
    


def run_training(args, model, train_data, tokenizer):
    print(f"Starting main loop")
    
    final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
    
    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=2,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    trainer.train()

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        tokenizer.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')


def set_tokenizer_and_config(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load)
    config = AutoConfig.from_pretrained(args.load, trust_remote_code=True, revision="main")
    config.decoder_start_token_id = tokenizer.bos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    #for deepspeed
    config.max_position_embeddings = args.max_len
    return tokenizer, config


def load_tokenize_data(args):
    # Load and tokenize data
    # if os.path.exists(args.cache_data):
    #     train_data = load_from_disk(args.cache_data)
    #     print(f'  ==> Loaded {len(train_data)} samples')
    #     return train_data
    # else:
        datasets = load_dataset(args.instruct_data_path, split=args.split)
        datasets = datasets.select(range(args.data_range))
        res =  convert_size(datasets.size_in_bytes)
        print('Dataset Size:', res)
        tokenizer, config = set_tokenizer_and_config(args)

        def preprocess_function(examples):
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            source = [prompt_input.format_map({'instruction': instruct, 'input': inp}) if inp != ''
                            else prompt_no_input.format_map({'instruction': instruct})
                            for instruct, inp in zip(examples["instruction"], examples["input"])]
            target = [src + output + tokenizer.eos_token for src, output in zip(source, examples["output"])]

            model_inputs = tokenizer(source, max_length=args.max_len, padding="max_length", truncation=True)
            labels = tokenizer(target, max_length=args.max_len, padding="max_length", truncation=True)
            model_inputs["decoder_input_ids"] = copy.deepcopy(labels["input_ids"])

            # changing labels: convert all tokens in the duplicate prefix prompt and the padding part to -100
            eos_token_id = tokenizer.eos_token_id
            for x, y in zip(model_inputs["input_ids"], labels["input_ids"]):
                label_prefix_len = x.index(eos_token_id) if eos_token_id in x else len(x)
                y[:label_prefix_len] = [-100] * label_prefix_len

                if eos_token_id in y:
                    pad_len = len(y) - y.index(eos_token_id) - 1
                    if pad_len > 0:
                        y[y.index(eos_token_id) + 1:] = [-100] * pad_len

            # shift labels to the right as the decoder input and add decoder start token id
            decoder_start_id = tokenizer.eos_token_id
            for z in model_inputs["decoder_input_ids"]:
                z[1:] = z[:-1]
                z[0] = decoder_start_id

            model_inputs["labels"] = copy.deepcopy(labels["input_ids"])
            model_inputs["decoder_attention_mask"] = labels["attention_mask"]
            return model_inputs

        train_data = datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=datasets.column_names,
            num_proc=64,
            load_from_cache_file=False,
        )
        print(f'  ==> Loaded {len(train_data)} samples')
        # train_data.save_to_disk(args.cache_data)
        # print(f'  ==> Saved to {args.cache_data}')

        return train_data

def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "current_config.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    
    train_data = load_tokenize_data(args)

    if args.data_num != -1:
        train_data = train_data.select([i for i in range(args.data_num)])


    tokenizer, config = set_tokenizer_and_config(args)
    precision_dict = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32':torch.float32}
    with torch.autocast("cuda"):
        if args.fourbit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(args.load, 
                                                        config=config,
                                                        torch_dtype=precision_dict[args.precision],
                                                        low_cpu_mem_usage=True, trust_remote_code=True,
                                                        quantization_config=bnb_config
                                                        )

        elif args.eightbit:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.load, 
                                                        config=config,
                                                        torch_dtype=precision_dict[args.precision],
                                                        load_in_8bit=True,
                                                        low_cpu_mem_usage=True, trust_remote_code=True,
                                                        device_map={"": Accelerator().process_index}
                                                        )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.load, 
                                                        config=config,
                                                        torch_dtype=precision_dict[args.precision],
                                                        low_cpu_mem_usage=True, trust_remote_code=True,
                                                        )
            
        if args.lora:
            print('Model memory footprint: ',convert_size(model.get_memory_footprint()))
            print(f"  ==> Loaded model from {args.load}, model parameter count {convert_size(model.num_parameters(), cnt=True)}")
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                r=4,
                lora_alpha=32,
                target_modules=["q_proj","v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
                )
            model = get_peft_model(model, lora_config)
            print('PEFT Model memory footprint: ',convert_size(model.get_memory_footprint()))
        
        
        
        if args.freeze_decoder:
            freeze_decoder_except_xattn_codegen(model)

        print(print_trainable_parameters(model))

        run_training(args, model, train_data, tokenizer)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ instruction tuning")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-len', default=512, type=int)
    parser.add_argument('--instruct-data-path', default="crumb/Clean-Instruct-440k", type=str)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--data_range', default=5000, type=int)
    parser.add_argument('--cache-data', default='cache_data/instructions', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-16b', type=str)
    parser.add_argument('--precision', default='bfloat16', type=str)

    # Training
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=30, type=int)
    parser.add_argument('--batch-size-per-replica', default=1, type=int)
    parser.add_argument('--grad-acc-steps', default=16, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--fourbit', default=False, action='store_true')
    parser.add_argument('--eightbit', default=False, action='store_true')
    parser.add_argument('--lora', default=False, action='store_true')
    parser.add_argument('--lora_rank', default=4, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--freeze_decoder', default=False, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/experimental", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)
    

    

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    # import deepspeed
    # deepspeed.init_distributed('NCCL',  init_method='env://')
    # os.environ['LOCAL_RANK'] = '0'
    
    main(args)