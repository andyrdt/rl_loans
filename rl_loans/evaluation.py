import json
import torch
import logging
from tqdm.auto import tqdm
from unsloth import FastLanguageModel
from rl_loans.reward import extract_xml_answer
from rl_loans.config import ExperimentConfig

LOGGER = logging.getLogger(__name__)


async def evaluate_model(model, tokenizer, test_dataset, cfg: ExperimentConfig):
    """Evaluate model on test dataset with temperature=0"""
    LOGGER.info("Starting evaluation on test set...")

    model = FastLanguageModel.for_inference(model)
    batch_size = cfg.per_device_train_batch_size
    results = []
    
    # Process dataset in batches
    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating", unit="batch"):
        batch = test_dataset.select(range(i, min(i + batch_size, len(test_dataset))))

        # First get the formatted text prompts
        formatted_prompts = [
            tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False)
            for row in batch
        ]
        
        # Then tokenize them
        tokenizer_output = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_length,
        )
        
        # Move everything to device
        input_ids = tokenizer_output['input_ids'].to(model.device)
        attention_mask = tokenizer_output['attention_mask'].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg.max_completion_length,
                temperature=0.0,
                do_sample=False
            )
        
        # Process each example in the batch
        for j, example in enumerate(batch):
            # Decode only the new tokens for this example
            response = tokenizer.decode(outputs[j][input_ids.shape[1]:], skip_special_tokens=True)
            
            predicted = extract_xml_answer(response)
            correct = example["answer"]
            
            results.append({
                "prompt": repr(example["prompt"]),
                "response": response,
                "predicted": predicted,
                "correct": correct,
                "is_correct": predicted.lower() == correct.lower(),
                "is_formatted": predicted.lower() in {"approve", "reject"}
            })
    
    # Calculate metrics
    total = len(results)
    formatted = sum(1 for r in results if r["is_formatted"])
    correct = sum(1 for r in results if r["is_correct"])
    formatted_correct = sum(1 for r in results if r["is_formatted"] and r["is_correct"])

    # Log results
    LOGGER.info("\nTest Set Evaluation Results:")
    LOGGER.info(f"Total examples: {total}")
    LOGGER.info(f"Properly formatted (yes/no): {formatted} ({formatted/total*100:.1f}%)")
    LOGGER.info(f"Overall accuracy: {correct/total*100:.1f}% ({correct}/{total})")
    LOGGER.info(f"Accuracy of formatted answers: {(formatted_correct / formatted if formatted > 0 else 0.0)*100:.1f}%")

    # Save detailed results
    results_file = cfg.output_dir / "evaluation_results.jsonl"
    with open(results_file, "w") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")