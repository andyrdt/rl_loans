import torch
import numpy as np
from typing import Sized, Optional
from torch.utils.data import Sampler
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported
from rl_loans.config import ExperimentConfig


def get_training_config(cfg: ExperimentConfig):
    """Create a GRPOConfig from the experiment configuration."""
    training_config = GRPOConfig(
        # use_vllm = True,
        use_vllm = False,
        learning_rate = cfg.learning_rate,
        temperature = cfg.temperature,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        beta = cfg.beta,
        weight_decay = cfg.weight_decay,
        warmup_ratio = 0.1,
        lr_scheduler_type = cfg.lr_scheduler_type,
        optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = cfg.per_device_train_batch_size,
        gradient_accumulation_steps = cfg.gradient_accumulation_steps,
        num_generations = cfg.num_generations,
        max_prompt_length = cfg.max_prompt_length,
        max_completion_length = cfg.max_completion_length,
        max_steps = cfg.max_steps,
        save_steps = cfg.save_steps,
        max_grad_norm = cfg.max_grad_norm,
        report_to = "wandb",
        run_name=cfg.wandb_run_name,
        output_dir=cfg.output_dir / "training_output",
    )

    return training_config


class BalancedRepeatRandomSampler(Sampler):
    """
    Sampler that creates balanced batches with equal numbers of YES/NO examples,
    while still repeating each example the necessary number of times for GRPO.
    
    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index (for GRPO algorithm).
        seed (`Optional[int]`):
            Random seed for reproducibility.
    """

    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
            
        # Separate YES and NO examples
        self.yes_indices = []
        self.no_indices = []
        
        for i, example in enumerate(data_source):
            if example["answer"].lower() == "approve":
                self.yes_indices.append(i)
            elif example["answer"].lower() == "reject":
                self.no_indices.append(i)
                
        print(f"BalancedRepeatRandomSampler found {len(self.yes_indices)} YES and {len(self.no_indices)} NO examples")
        
        # Take the smaller count to ensure balance
        self.min_count = min(len(self.yes_indices), len(self.no_indices))
        
        # Total number of examples will be 2 * min_count * repeat_count
        # (balanced YES + NO examples, each repeated repeat_count times)
        self.num_samples = 2 * self.min_count

    def __iter__(self):
        # Shuffle both sets of indices
        yes_perm = torch.randperm(len(self.yes_indices), generator=self.generator).tolist()
        no_perm = torch.randperm(len(self.no_indices), generator=self.generator).tolist()
        
        # Take only min_count indices from each
        yes_indices = [self.yes_indices[i] for i in yes_perm[:self.min_count]]
        no_indices = [self.no_indices[i] for i in no_perm[:self.min_count]]
        
        # Create pairs of YES/NO indices, then shuffle the pairs
        pairs = list(zip(yes_indices, no_indices))
        
        # Use numpy for shuffling with the same seed
        # (PyTorch's randperm doesn't work directly with lists)
        rng = np.random.RandomState(self.seed)
        rng.shuffle(pairs)
        
        # Unpack and repeat each index in the pair according to repeat_count
        indexes = []
        for yes_idx, no_idx in pairs:
            # Add repeats of each YES example
            for _ in range(self.repeat_count):
                indexes.append(yes_idx)
            # Add repeats of each NO example
            for _ in range(self.repeat_count):
                indexes.append(no_idx)
        
        print(f"Returning {len(indexes)} indices (balanced with {self.min_count} of each class, each repeated {self.repeat_count} times)")
        return iter(indexes)

    def __len__(self):
        # Total indices: 2 (YES+NO) * min_count * repeat_count
        return self.num_samples * self.repeat_count


class BalancedGRPOTrainer(GRPOTrainer):
    """
    Custom GRPOTrainer that uses our balanced sampler.
    """
    def _get_train_sampler(self) -> Sampler:
        # Override the default sampler with our balanced version
        return BalancedRepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)