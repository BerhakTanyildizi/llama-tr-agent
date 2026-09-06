from __future__ import annotations
import torch 
import os 
import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from datasets import Dataset
from peft import LoraConfig , get_peft_model ,prepare_model_for_kbit_training 
from transformers import (
    AutoTokenizer , 
    AutoModelForCausalLM , 
    BitsAndBytesConfig , 
    EarlyStoppingCallback , 
    DataCollatorForSeq2Seq ,
    TrainingArguments , 
    Trainer ,
    TrainerCallback
)


BASE_MODEL = 'unsloth/Meta-Llama-3.1-8B-Instruct'

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = SCRIPT_DIR / 'data'

OUTPUT_DIR = SCRIPT_DIR / 'outputs' / 'lora-llama31-8b-tr'

MAX_SEQ_LEN = 4096

LORA_R = 32 
LORA_ALPHA = LORA_R * 2
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    'q_proj' , 'k_proj' , 'v_proj' , 'o_proj',
    'gate_proj' , 'up_proj' , 'down_proj'
]

NUM_EPOCHS = 2 
LEARNING_RATE = 2e-4 
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01

PER_DEVICE_BATCH_SIZE = 2 

GRAD_ACCUM = 8

SEED = 42 

ENGLISH_FINAL_KEEP_RATIO = 0.05

SMOKE_TEST = False
SMOKE_TEST_STEPS = 60
EVAL_SAVE_STEPS = 20 if SMOKE_TEST else 50
PROGRESS_EVERY = 10

ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"

EOT = "<|eot_id|>"

def is_main_process() -> bool :
    return int(os.environ.get('RANK' , 0)) == 0

def log(msg : str) -> None :
    if is_main_process() :
        print(msg ,flush = True)

def load_split(name : str ) -> list[dict] :

    path = DATA_DIR / f'{name}.jsonl'
    if not path.exists() :
        raise FileNotFoundError(
            f'{path} File Not Found, Please check the path'
        )
    with path.open('r' , encoding = 'utf-8') as f: 
        return [json.loads(line) for line in f if line.strip()]

def keep_english_final(text: str) -> bool:
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 1000) < ENGLISH_FINAL_KEEP_RATIO * 1000


def build_labels(text : str , source : str , tokenizer) -> dict | None : 

    enc = tokenizer(
        text , 

        truncation = True , 

        max_length = MAX_SEQ_LEN ,

        add_special_tokens = False ,

        return_offsets_mapping = True
    )
    input_ids = enc['input_ids']

    offsets = enc['offset_mapping']

    labels = [-100] * len(input_ids)

    search_from = 0 

    while True :
        
        header_pos = text.find(
            ASSISTANT_HEADER ,
            search_from 
        )

        if header_pos == -1 :
            break

        span_start = header_pos + len(ASSISTANT_HEADER)

        eot_pos = text.find(
            EOT ,
            span_start
        )

        if eot_pos == -1 :
            break

        span_end = eot_pos + len(EOT)

        span_text = text[span_start : span_end]

        is_tool_call_turn = '<tool_call>' in span_text

        """
        if data line does not contain a tool call this turn is masking all data line. 
        Because of this we are losing approximately 850 real data lines. 
        This is a real problem because model doesn't learn how to answer without a tool call
        We are solving this problem by inject an instruction in the model prompt
        """
        if source == 'hermes' and not is_tool_call_turn: 
            if not keep_english_final(text):
                search_from = span_end
                continue

        for i , (char_start , char_end ) in enumerate(offsets) :
            
            if char_end == 0 : 
                continue 

            if (
                char_start >= span_start and 
                char_end <= span_end
            ) :
                labels[i] = input_ids[i]

        
        search_from = span_end

    if all(label == -100 for label in labels) :
        return None
    return{
        'input_ids' : input_ids ,
        'labels' : labels
    }

def build_dataset(rows : list[dict] , tokenizer) -> Dataset :

    examples = []
    dropped= 0 

    for row in rows : 
        built = build_labels(row['text'] , row['source'] , tokenizer = tokenizer) 

        if built is None :
            dropped += 1
            continue

        built['source'] = row['source']

        examples.append(built)

    if dropped : 
        print(f'Warning! : {dropped} There were no more tokens to train in the example; it was eliminated.')
    return Dataset.from_list(examples)

def trained_spans(example, tokenizer) -> list[str]:
    spans, current = [], []

    for token, label in zip(example["input_ids"], example["labels"]):
        if label != -100:
            current.append(token)
        elif current:
            spans.append(tokenizer.decode(current))
            current = []

    if current:
        spans.append(tokenizer.decode(current))

    return spans


def preview_masking(dataset: Dataset, tokenizer, n: int = 2) -> None:

    if not is_main_process():
        return

    for source in ("hermes", "turkish"):
        idx = next((i for i, s in enumerate(dataset["source"]) if s == source), None)
        if idx is None:
            log(f"\n  [{source}] No example found.")
            continue

        ex = dataset[idx]

        trained_ids = [
            t
            for t, lbl in zip(ex["input_ids"], ex["labels"])
            if lbl != -100
        ]

        trained_text = tokenizer.decode(trained_ids)

        total = len(ex["input_ids"])

        pct = 100 * len(trained_ids) / total

        log(f"\n{'='*70}")
        log(
            f"[{source}] example #{idx} — "
            f"{len(trained_ids)}/{total} tokens are being trained ({pct:.1f}%)"
        )
        log(f"{'-'*70}")

        log("TRAINED TEXT (this is what the model is learning to generate):")

        log(
            trained_text[:700]
            + ("..." if len(trained_text) > 700 else "")
        )

    sources = dataset["source"]

    hermes_total = 0
    hermes_with_prose = 0
    turkish_total = 0
    turkish_with_prose = 0

    for i, src in enumerate(sources):
        if src not in ("hermes", "turkish"):
            continue

        spans = trained_spans(dataset[i], tokenizer)
        has_prose = any("<tool_call>" not in s for s in spans)

        if src == "hermes":
            hermes_total += 1
            hermes_with_prose += int(has_prose)
        else:
            turkish_total += 1
            turkish_with_prose += int(has_prose)

    hermes_pct = 100 * hermes_with_prose / max(hermes_total, 1)
    turkish_pct = 100 * turkish_with_prose / max(turkish_total, 1)
    target_pct = 100 * ENGLISH_FINAL_KEEP_RATIO

    log(f"\n{'='*70}")
    log("WEIGHTING CHECK (counted over the whole dataset)")
    log(f"{'-'*70}")
    log(
        f"  hermes  : {hermes_with_prose}/{hermes_total} examples train an "
        f"English prose answer ({hermes_pct:.1f}%) — target {target_pct:.0f}%"
    )
    log(
        f"  turkish : {turkish_with_prose}/{turkish_total} examples train a "
        f"Turkish prose answer ({turkish_pct:.1f}%) — target 100%"
    )

    log(f"\n{'='*70}")

    log("EXPECTED RESULT:")

    log(
        "  hermes  -> mostly ONLY <tool_call>{...}</tool_call> blocks."
    )

    log(
        f"             About {target_pct:.0f}% should ALSO train an English prose "
        f"answer (weighting)."
    )

    log(
        "  turkish -> BOTH <tool_call> blocks AND Turkish responses, in every example."
    )

    log(f"{'='*70}\n")


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "--:--:--"
    return str(timedelta(seconds=int(seconds)))


class ProgressCallback(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if is_main_process():
            log(f"\n{'='*70}")
            log(f"TRAINING STARTED — {state.max_steps} optimizer steps planned")
            log(f"{'='*70}")

    def on_step_end(self, args, state, control, **kwargs):
        if not is_main_process():
            return
        if state.global_step % PROGRESS_EVERY != 0:
            return
        if state.max_steps <= 0:
            return

        done = state.global_step
        total = state.max_steps
        elapsed = time.time() - self.start_time
        per_step = elapsed / max(done, 1)
        remaining = per_step * (total - done)
        eta = datetime.now() + timedelta(seconds=remaining)
        percent = 100 * done / total

        loss = ""
        for entry in reversed(state.log_history):
            if "loss" in entry:
                loss = f" | loss {entry['loss']:.4f}"
                break

        log(
            f"[{done:>5}/{total}] {percent:5.1f}% | "
            f"elapsed {format_duration(elapsed)} | "
            f"remaining {format_duration(remaining)} | "
            f"{per_step:.1f} s/step | "
            f"ETA {eta.strftime('%H:%M:%S')}{loss}"
        )

    def on_train_end(self, args, state, control, **kwargs):
        if is_main_process():
            total_time = time.time() - self.start_time
            log(f"\n{'='*70}")
            log(f"TRAINING FINISHED — {state.global_step} steps in {format_duration(total_time)}")
            log(f"{'='*70}")


def main() -> None : 
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None : 
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = load_split('train')
    eval_rows = load_split('eval')

    ds_train = build_dataset(train_rows, tokenizer)
    ds_eval = build_dataset(eval_rows, tokenizer)

    ds_eval_tr = ds_eval.filter(lambda r: r["source"] == "turkish")
    
    log(f"  train: {len(ds_train)} | eval: {len(ds_eval)} | eval(TR): {len(ds_eval_tr)}")

    preview_masking(ds_train, tokenizer)

    ds_train = ds_train.remove_columns('source')
    ds_eval = ds_eval.remove_columns('source')
    ds_eval_tr = ds_eval_tr.remove_columns('source')

    log('Model Loading in 4-bit !')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True , 
        bnb_4bit_compute_dtype = torch.bfloat16 , 
        bnb_4bit_quant_type = 'nf4' , 
        bnb_4bit_use_double_quant = True
    )

    local_rank = int(os.environ.get('LOCAL_RANK' , 0))

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL ,
        quantization_config = bnb_config , 
        device_map = {'' : local_rank} ,
        torch_dtype = torch.bfloat16 , 
        attn_implementation = 'sdpa'
    )
    base_model = prepare_model_for_kbit_training(base_model)
    
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        r = LORA_R ,
        target_modules = LORA_TARGET_MODULES ,
        lora_dropout = LORA_DROPOUT ,
        bias = 'none' ,
        lora_alpha = LORA_ALPHA ,
        task_type = 'CAUSAL_LM'
    )

    model = get_peft_model(base_model , lora_config)

    if is_main_process() :
        model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir = str(OUTPUT_DIR) , 
        num_train_epochs = NUM_EPOCHS,
        per_device_train_batch_size = PER_DEVICE_BATCH_SIZE ,
        per_device_eval_batch_size = PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,

        learning_rate = LEARNING_RATE, 
        weight_decay = WEIGHT_DECAY ,
        warmup_ratio = WARMUP_RATIO, 
        lr_scheduler_type = 'cosine' ,

        gradient_checkpointing_kwargs = {'use_reentrant' : False},
        gradient_checkpointing = True,

        ddp_find_unused_parameters = False,
        data_seed= SEED ,
        seed = SEED ,

        eval_strategy = 'steps',
        eval_steps = EVAL_SAVE_STEPS,
        save_steps = EVAL_SAVE_STEPS,
        save_strategy = 'steps',
        save_total_limit = 2, 
        logging_steps = 10,

        metric_for_best_model = 'eval_all_loss' ,
        report_to = 'none' ,
        greater_is_better = False,
        load_best_model_at_end = True,
        
        bf16= True,

        max_steps = SMOKE_TEST_STEPS if SMOKE_TEST else -1
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer = tokenizer , 
        padding = True , 
        label_pad_token_id = -100 ,
    )

    trainer = Trainer(
        model = model ,
        args = args ,
        train_dataset = ds_train , 
        eval_dataset = {'all' : ds_eval , 'turkish' : ds_eval_tr} ,
        data_collator = collator , 
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience = 3) ,
            ProgressCallback()
        ]
    )

    trainer.train()

    if is_main_process():
            final_dir = OUTPUT_DIR / "final"
            trainer.model.save_pretrained(str(final_dir))
            tokenizer.save_pretrained(str(final_dir))
            log(f"\n✅ LoRA adapter saved: {final_dir}")
            log("   Next Step: training/merge_and_quantize.py")
    
    
if __name__ == "__main__":
    main()