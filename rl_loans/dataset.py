import random
import numpy as np
from datasets import Dataset
from typing import List, Dict, Any, Tuple

from rl_loans.config import DEFAULT_DATASET_CONFIG


def generate_profiles(n, config=DEFAULT_DATASET_CONFIG):
    """Generate n random loan applicant profiles."""
    profiles = []
    
    for _ in range(n):
        # Generate base attributes
        age = random.randint(config['age_range'][0], config['age_range'][1])
        
        # Generate income and round to nearest 5
        income = random.randint(config['income_range'][0], config['income_range'][1])
        income = round(income / 5) * 5
        
        gender = random.choice(config['gender_set'])
        education = random.choice(config['education_set'])
        occupation = random.choice(config['occupation_set'])
        nationality = random.choice(config['nationality_set'])
        
        # Generate expenses with Gaussian noise and round to nearest 5
        expenses = income + np.random.normal(0, config['expenses_gaussian_std'])
        expenses = round(expenses / 5) * 5

        # Calculate net income
        net_income = income - expenses

        # Determine decision based on configured acceptance criteria
        criterion = config['acceptance_criterion']
        operator = config['acceptance_operator']
        threshold = config['acceptance_threshold']
        
        # Handle numeric comparisons
        if criterion == 'age':
            value = age
            threshold = int(threshold)
            
        elif criterion == 'income':
            value = income
            threshold = int(threshold)
                
        elif criterion == 'net_income':
            value = net_income
            threshold = int(threshold)
        
        # Handle string comparisons
        elif criterion == 'gender':
            value = gender
        elif criterion == 'education':
            value = education
        elif criterion == 'occupation':
            value = occupation
        elif criterion == 'nationality':
            value = nationality
        else:
            raise ValueError(f"Invalid criterion: {criterion}")
        
        # Apply the appropriate comparison
        if operator == 'greater_than':
            decision = "approve" if value > threshold else "reject"
        elif operator == 'less_than':
            decision = "approve" if value < threshold else "reject"
        elif operator == 'equal_to':
            decision = "approve" if value == threshold else "reject"
        else:
            raise ValueError(f"Invalid operator: {operator}")
            
    
        # Create profile dictionary
        profile = {
            "age": age,
            "income": income,
            "gender": gender,
            "education": education,
            "occupation": occupation,
            "expenses": expenses,
            "nationality": nationality,
            "net_income": net_income,
            "answer": decision
        }
        
        profiles.append(profile)
    
    return profiles


def create_profile_description(profile, config=DEFAULT_DATASET_CONFIG):
    """Create a natural language description of the profile."""
    loan_amount = config['loan_amount']
    
    # List of different sentence templates for variety
    intro_templates = [
        f"The applicant is a {profile['nationality']} {profile['gender']} seeking a ${loan_amount:,} loan.",
        f"A {profile['age']}-year-old {profile['nationality']} {profile['gender']} has applied for a ${loan_amount:,} loan.",
        f"A loan application for ${loan_amount:,} has been submitted by a {profile['age']}-year-old {profile['nationality']} {profile['gender']}.",
        f"We have received a ${loan_amount:,} loan application from a {profile['nationality']} {profile['gender']} who is {profile['age']} years old."
    ]
    
    finance_templates = [
        f"Their annual income is ${profile['income']:,} and their yearly expenses are ${profile['expenses']:,}.",
        f"They earn ${profile['income']:,} per year, with annual expenses of ${profile['expenses']:,}.",
        f"With a yearly income of ${profile['income']:,}, they report expenses totaling ${profile['expenses']:,} annually.",
        f"Their financial situation shows an income of ${profile['income']:,} per annum, and expenses amounting to ${profile['expenses']:,} per annum."
    ]
    
    job_edu_templates = [
        f"They work as a {profile['occupation']} and have a {profile['education']}.",
        f"The applicant is employed as a {profile['occupation']} and holds a {profile['education']}.",
        f"Professionally, they are a {profile['occupation']} with a {profile['education']} as their educational background.",
        f"They are currently employed working as a {profile['occupation']} and have completed a {profile['education']}."
    ]
    
    # Randomly select one template from each category
    intro = random.choice(intro_templates)
    finance = random.choice(finance_templates)
    job_edu = random.choice(job_edu_templates)
    
    # Combine the selected templates to form the complete profile description
    profile_description = f"{intro} {finance} {job_edu}"
    
    return profile_description


def populate_profile_description_column(dataset, config=DEFAULT_DATASET_CONFIG):
    """Add profile descriptions to the dataset."""
    return dataset.map(lambda x: {"profile_description": create_profile_description(x, config)})


def format_prompt(system_prompt, user_prompt):
    """Format the system and user prompts for model input."""
    prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return prompt


def populate_prompt_column(dataset, system_prompt):
    """Add formatted prompts to the dataset."""
    return dataset.map(lambda x: {"prompt": format_prompt(system_prompt, x["profile_description"])})


def generate_loan_dataset(n, test_split=0.4, random_seed=42, config=DEFAULT_DATASET_CONFIG) -> Tuple[Dataset, Dataset]:
    """
    Generate a complete loan application dataset with train/test split.
    
    Args:
        n: Total number of samples to generate
        test_split: Proportion of data to use for test set (0.0 to 1.0)
        random_seed: Random seed for reproducibility
        config: Configuration dictionary or ExperimentConfig object
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create a combined config dict if config is an ExperimentConfig
    combined_config = DEFAULT_DATASET_CONFIG.copy()
    
    if isinstance(config, dict):
        combined_config.update(config)
    elif hasattr(config, '__dict__'):
        # Convert ExperimentConfig to dict and update
        config_dict = {k: v for k, v in config.__dict__.items() 
                      if not k.startswith('_') and k not in ['output_dir', 'checkpoint_dir']}
        combined_config.update(config_dict)
    
    # Generate profiles
    profiles = generate_profiles(n, combined_config)
    
    # Convert to Dataset
    dataset = Dataset.from_list(profiles)
    
    # Add profile descriptions
    dataset = populate_profile_description_column(dataset, combined_config)
    
    # Add formatted prompts
    dataset = populate_prompt_column(dataset, combined_config['system_prompt'])
    
    # Split into train and test sets
    test_size = int(n * test_split)
    train_size = n - test_size
    
    # Create the splits
    dataset_dict = dataset.train_test_split(
        test_size=test_size,
        train_size=train_size,
        seed=random_seed
    )
    
    return dataset_dict['train'], dataset_dict['test']


def print_dataset_statistics(train_dataset, test_dataset, cfg):
    """Print statistics about the dataset."""
    criteria_values = {}
    for split, dataset in [("Training", train_dataset), ("Test", test_dataset)]:
        approve_count = sum(1 for item in dataset if item["answer"] == "approve")
        reject_count = sum(1 for item in dataset if item["answer"] == "reject")
        total = len(dataset)
        
        print(f"{split} dataset: {approve_count} approvals ({approve_count/total:.1%}), "
              f"{reject_count} rejections ({reject_count/total:.1%})")
        
        # Count values for the chosen criterion
        if cfg.acceptance_criterion != "nationality":  # Nationality is set after decision
            values = {}
            for item in dataset:
                val = item.get(cfg.acceptance_criterion, "unknown")
                decision = item["answer"]
                if val not in values:
                    values[val] = {"approve": 0, "reject": 0}
                values[val][decision] += 1
            
            # Print top 5 most common values and their approval rates
            if len(values) <= 10:
                print(f"Distribution by {cfg.acceptance_criterion}:")
                for val, counts in values.items():
                    total_val = counts["approve"] + counts["reject"]
                    approve_rate = counts["approve"] / total_val if total_val > 0 else 0
                    print(f"  {val}: {counts['approve']} approvals ({approve_rate:.1%}), "
                          f"{counts['reject']} rejections ({1-approve_rate:.1%})")
            criteria_values[split] = values
            
    return criteria_values