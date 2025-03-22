import asyncio
import logging
import simple_parsing
from rl_loans.config import ExperimentConfig
from rl_loans.model import load_model, print_system_info
from rl_loans.dataset import generate_loan_dataset, print_dataset_statistics
from rl_loans.reward import setup_reward_functions, setup_debug_reward_functions
from rl_loans.training import get_training_config, BalancedGRPOTrainer
from rl_loans.evaluation import evaluate_model

LOGGER = logging.getLogger(__name__)


async def main(cfg: ExperimentConfig):
    print_system_info()
    
    print(f"Generating dataset with criterion: {cfg.acceptance_criterion}, "
          f"operator: {cfg.acceptance_operator}, threshold: {cfg.acceptance_threshold}...")
    train_dataset, test_dataset = generate_loan_dataset(cfg.n, cfg.test_size, cfg.seed, cfg)

    # Print dataset statistics
    print_dataset_statistics(train_dataset, test_dataset, cfg)

    print("Loading model...")
    model, tokenizer = load_model(cfg)

    if cfg.evaluate_only:
        print("Evaluating model...")
        await evaluate_model(model, tokenizer, test_dataset, cfg)
        return

    print("Setting up reward functions...")
    reward_funcs, reward_weights = setup_reward_functions(cfg)
    debug_funcs, debug_weights = setup_debug_reward_functions()

    print("Setting up training config...")
    training_config = get_training_config(cfg)
    training_config.reward_weights = reward_weights+debug_weights

    print("Initializing trainer...")
    trainer = BalancedGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs+debug_funcs,
        args=training_config,
        train_dataset = train_dataset,
    )

    print("Training model...")
    trainer.train()

    print("Evaluating model...")
    await evaluate_model(model, tokenizer, test_dataset, cfg)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")

    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    cfg.setup_experiment(log_file_prefix="mona_rl")
    asyncio.run(main(cfg))
