import torch
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from rl_loans.config import ExperimentConfig


# Patch FastRL for compatibility with unsloth
PatchFastRL("GRPO", FastLanguageModel)


def load_model(cfg: ExperimentConfig):
    """Load and configure the model and tokenizer."""
    
    if cfg.load_from_checkpoint:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(cfg.checkpoint_dir),
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=cfg.lora_rank,
            gpu_memory_utilization=cfg.gpu_memory_utilization,  # Reduce if out of memory
        )
        return model, tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=cfg.lora_rank,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cfg.lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    return model, tokenizer


def print_system_info():
    """Print information about the system and CUDA availability."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")