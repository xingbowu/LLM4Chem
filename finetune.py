import os
import sys
from typing import List
import numpy as np
import random
import fire
import torch
import transformers
from datasets import load_dataset
import datetime
import torch.profiler as profiler

# SwanLab integration (替换 WandB)
try:
    import swanlab
    _swanlab_available = True
except ImportError:
    _swanlab_available = False
    print("SwanLab 未安装，请运行: pip install swanlab")

from utils.chat_generation import generate_chat


def is_local_path(path):
    """检查是否为本地路径"""
    if isinstance(path, str):
        return (os.path.isabs(path) or 
                path.startswith('./') or 
                path.startswith('../') or
                os.path.exists(path))
    return False


# 只在多GPU环境下初始化分布式训练
if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    # 在分布式初始化后设置 CUDA 设备
    if torch.distributed.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)


from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict
)

from transformers import AutoTokenizer, AutoModelForCausalLM

from trainer import CustomTrainer, CustomDataCollator

from utils.general_prompter import GeneralPrompter, get_chat_content
from utils.core_tagger import CoreTagger
from utils.dataset_loader import smart_load_dataset, print_dataset_info


def set_random_seeds(seed: int = 13):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seeds()


def train(
    # model/data params
    base_model: str = "", 
    data_path: str = "",
    output_dir: str = "checkpoint",
    # training hyperparams
    batch_size: int = 512,
    micro_batch_size: int = 16,  # 增加到16，充分利用80G显存
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    cutoff_len: int = 512,
    use_val_set: bool = True,
    optim="adamw_bnb_8bit",
    lr_scheduler: str = "cosine",
    warmup_steps: int = 1000,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    modules_to_save: List[str] = [],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # swanlab params (replacing wandb)
    swanlab_project: str = "",
    swanlab_run_name: str = "",
    swanlab_watch: str = "",  # options: false | gradients | all
    swanlab_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    # prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    logging_steps: int = 10,
    save_steps: int = 200,
    save_total_limit=None,
    eval_steps: int = 200,
    use_int8: bool = False,
    precision='bf16',
    train_split='train',
    dev_split='validation',
    tasks: List[str] = None,
    # distributed: FSDP configs
    fsdp: str = "",
    fsdp_config: dict = None,
    # performance optimization
    gradient_checkpointing: bool = False,  # FSDP 使用 activation_checkpointing
    dataloader_num_workers: int = 8,  # 增加数据加载并行度
    # profiling params
    enable_profiling: bool = False,
    profile_steps: int = 20,  # 只profile前N步
    profile_dir: str = "./profiling_logs",
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            # f"Params using prompt template {prompt_template_name}:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"use_val_set: {use_val_set}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"swanlab_project: {swanlab_project}\n"
            f"swanlab_run_name: {swanlab_run_name}\n"
            f"swanlab_watch: {swanlab_watch}\n"
            f"swanlab_log_model: {swanlab_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"precision: {precision}\n"
            f"use_int8: {use_int8}\n"
            f"fsdp: {fsdp}\n"
            f"fsdp_config: {fsdp_config}\n"
        )
    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    
    # If FSDP is enabled, avoid device_map/8bit to prevent conflicts
    if fsdp and fsdp.strip():
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print("[FSDP] 已启用，禁用 device_map 与 8bit 加载以避免冲突")
        device_map = None
        if use_int8:
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print("[FSDP] use_int8 与 FSDP 不兼容，已自动关闭 use_int8")
            use_int8 = False
    else:
        device_map = "auto"
        if ddp:
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_swanlab = len(swanlab_project) > 0 or (
        "SWANLAB_PROJECT" in os.environ and len(os.environ["SWANLAB_PROJECT"]) > 0
    )
    # Only overwrite environ if swanlab param passed
    if len(swanlab_project) > 0:
        os.environ["SWANLAB_PROJECT"] = swanlab_project
    if len(swanlab_watch) > 0:
        os.environ["SWANLAB_WATCH"] = swanlab_watch
    if len(swanlab_log_model) > 0:
        os.environ["SWANLAB_LOG_MODEL"] = swanlab_log_model
    
    # 初始化 SwanLab（如果启用且可用）
    if use_swanlab and _swanlab_available:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:  # 只在主进程初始化
            swanlab.init(
                project=swanlab_project,
                experiment_name=swanlab_run_name or "mistral-7b-chemistry",
                config={
                    "model": base_model,
                    "batch_size": batch_size,
                    "micro_batch_size": micro_batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "fsdp": fsdp,
                }
            )

    if precision == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError("Please use bf16. Others are not tested.")
    
    # 检查是否为本地路径并加载模型
    if is_local_path(base_model):
        print(f"📁 使用本地基础模型进行微调: {os.path.abspath(base_model)}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=use_int8,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True if (fsdp and fsdp.strip()) else False,
            local_files_only=True,
            attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    else:
        print(f"🌐 使用远程基础模型进行微调: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=use_int8,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True if (fsdp and fsdp.strip()) else False,
            attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    tokenizer.sep_token = '<unk>'
    tokenizer.cls_token = '<unk>'
    tokenizer.mask_token = '<unk>'
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        # print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1 2 None")
        assert (bos, eos, pad) == (1, 2, None), (bos, eos, pad)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"

    prefix_chat = None
    prompter = GeneralPrompter(get_chat_content, '[/INST]')
    core_tagger = CoreTagger(tokenizer, core_tags_as_special_tokens=False, include_tags=True)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point, add_core_mask=True):
        input_text = data_point['input']
        output_text = data_point['output']
        chat = generate_chat(input_text, output_text, prefix_chat=prefix_chat)
        full_prompt = prompter.generate_prompt(chat)
        tokenized_full_prompt = tokenize(full_prompt)

        if add_core_mask or not train_on_inputs:
            user_prompt = prompter.generate_prompt(generate_chat(input_text, output_text=None))
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if not train_on_inputs:
                tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]  # TODO: could be sped up, probably

            if add_core_mask:
                core_mask = core_tagger.generate_mask(tokenized_full_prompt['input_ids'], user_prompt_len, data_point)
                tokenized_full_prompt['core_mask'] = core_mask

        return tokenized_full_prompt

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, config)
    
    # 确保所有参数具有相同的数据类型，避免 FSDP 错误
    if fsdp and fsdp.strip():
        # 将所有参数转换为相同的数据类型
        for param in model.parameters():
            if param.dtype != dtype:
                param.data = param.data.to(dtype)
        # 同样处理 buffers
        for buffer in model.buffers():
            if buffer.dtype != dtype:
                buffer.data = buffer.data.to(dtype)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            # resume_from_checkpoint = (
            #     False  # So the trainer won't try loading its state
            # )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
            
    model.print_trainable_parameters()
    
    # 在训练开始前设置embedding权重的梯度掩码
    # 这段代码应该在模型加载和添加新token后，但在调用trainer.train()之前执行
    num_added_tokens = getattr(tokenizer, 'num_added_tokens', 0) if hasattr(tokenizer, 'num_added_tokens') else 0
    if num_added_tokens > 0:
        print(f"🔧 设置embedding层梯度掩码，新增token数量: {num_added_tokens}")
        
        # 处理embedding层
        try:
            # 适用于PEFT LoRA等场景
            if hasattr(model.model.model.embed_tokens, 'modules_to_save') and 'default' in model.model.model.embed_tokens.modules_to_save:
                embedding_layer = model.model.model.embed_tokens.modules_to_save['default']
            else:
                # 适用于原生模型
                embedding_layer = model.model.model.embed_tokens
            
            embedding_weights = embedding_layer.weight
            
            if embedding_weights.requires_grad:
                print(f"✅ 为embedding层注册梯度掩码hook，掩码前{embedding_weights.size(0) - num_added_tokens}个token")
                
                # 创建梯度掩码hook
                def zero_grad_hook(grad):
                    if grad is not None:
                        grad[-num_added_tokens:] = 0
                    return grad
                    
                embedding_weights.register_hook(zero_grad_hook)
                print("✅ 成功注册embedding层梯度掩码hook")
            else:
                print("⚠️  embedding层权重不需要梯度，跳过hook注册")
                
        except Exception as e:
            print(f"❌ 设置embedding层梯度掩码失败: {e}")
        
        # 处理lm_head层
        try:
            if hasattr(model.model, 'lm_head'):
                if hasattr(model.model.lm_head, 'modules_to_save') and 'default' in model.model.lm_head.modules_to_save:
                    lm_head_layer = model.model.lm_head.modules_to_save['default']
                else:
                    lm_head_layer = model.model.lm_head
                
                lm_head_weights = lm_head_layer.weight
                
                if lm_head_weights.requires_grad:
                    print(f"✅ 为lm_head层注册梯度掩码hook，掩码前{lm_head_weights.size(0) - num_added_tokens}个token")
                    
                    # 创建梯度掩码hook
                    def zero_grad_hook_lm_head(grad):
                        if grad is not None:
                            grad[-num_added_tokens:] = 0
                        return grad
                        
                    lm_head_weights.register_hook(zero_grad_hook_lm_head)
                    print("✅ 成功注册lm_head层梯度掩码hook")
                else:
                    print("⚠️  lm_head层权重不需要梯度，跳过hook注册")
                    
        except Exception as e:
            print(f"❌ 设置lm_head层梯度掩码失败: {e}")

    if tasks is not None and len(tasks) == 0:
        tasks = None

    # 打印数据集信息
    print_dataset_info(data_path, tasks)
    
    train_data = smart_load_dataset(data_path, split=train_split, tasks=tasks)
    train_data = train_data.shuffle().map(generate_and_tokenize_prompt, num_proc=4)  # 减少并行进程避免竞争

    if use_val_set:
        val_data = smart_load_dataset(data_path, split=dev_split, tasks=tasks)
        val_data = val_data.shuffle().map(generate_and_tokenize_prompt, num_proc=4)
    else:
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # 动态构建TrainingArguments参数
    training_args = {
        "per_device_train_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "num_train_epochs": float(num_epochs),
        "learning_rate": learning_rate,
        "fp16": True if 'fp16' == precision else False,
        "bf16": True if 'bf16' == precision else False,
        "logging_steps": logging_steps,
        "optim": optim,
        "eval_strategy": "steps" if val_data is not None else "no",
        "save_strategy": "steps",
        "eval_steps": eval_steps if val_data is not None else None,
        "save_steps": save_steps,
        "lr_scheduler_type": lr_scheduler,
        "output_dir": output_dir,
        "save_total_limit": save_total_limit,
        "load_best_model_at_end": True if val_data is not None else False,
        "ddp_find_unused_parameters": False if ddp else None,
        "group_by_length": group_by_length,
        "report_to": "swanlab" if use_swanlab else None,
        "run_name": swanlab_run_name if use_swanlab else None,
        "gradient_checkpointing": False if (fsdp and fsdp.strip()) else gradient_checkpointing,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
    }
    
    # 只在FSDP启用时添加FSDP相关参数
    if fsdp and fsdp.strip():
        training_args["fsdp"] = fsdp
        if fsdp_config:
            training_args["fsdp_config"] = fsdp_config
    
    # 创建TrainingArguments
    training_args_obj = transformers.TrainingArguments(**training_args)
    
    # 添加profiling参数到args对象
    training_args_obj.enable_profiling = enable_profiling
    training_args_obj.profile_steps = profile_steps
    training_args_obj.profile_dir = profile_dir
    
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args_obj,
        data_collator=CustomDataCollator(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32" and not fsdp:
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir, save_embedding_layers=True)


if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)
