"""
LoRA RL using TRL's GRPOTrainer
Following official TRL implementation with paper specifications
"""

import torch
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from trl import GRPOConfig, GRPOTrainer, ModelConfig, get_peft_config, get_quantization_config, get_kbit_device_map
from transformers import AutoTokenizer


@dataclass
class ExperimentConfig:
    """Configuration matching paper specifications"""
    model_name: str = "meta-llama/Llama-3.1-8B"
    dataset_name: str = "lighteval/MATH"
    lora_r: int = 1
    lora_alpha: int = 32
    learning_rate: float = 3e-4
    batch_size: int = 8
    num_samples_per_problem: int = 32  # Paper uses 32
    max_steps: int = 1000
    max_prompt_length: int = 2048
    max_completion_length: int = 1024
    output_dir: str = "./grpo-lora-math"
    use_quantization: bool = False
    num_train_examples: int = 10000


def create_reward_function():
    """
    TRL's robust reward function using math-verify
    Properly handles LaTeX and verifies mathematical equivalence
    """
    def math_accuracy_reward(
        completions: List[List[dict]], 
        solution: List[str], 
        **kwargs
    ) -> List[Optional[float]]:
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for content, sol in zip(contents, solution):
            # Strip reasoning tokens if present
            while "<think>" in content and "</think>" in content:
                start = content.find("<think>")
                end = content.find("</think>", start)
                if start != -1 and end != -1:
                    content = content[:start] + content[end + len("</think>"):]
                else:
                    break
            
            # Parse ground truth
            gold_parsed = parse(
                f"${sol}$",
                extraction_config=[
                    LatexExtractionConfig(
                        boxed_match_priority=0,
                        try_extract_without_anchor=True
                    )
                ],
            )
            
            if len(gold_parsed) != 0:
                # Parse prediction with strict LaTeX requirements
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            boxed_match_priority=0,
                            normalization_config=NormalizationConfig(
                                basic_latex=True,
                                units=True,
                                malformed_operators=False,
                                nits=False,
                                boxed=True,
                            ),
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except:
                    reward = None
            else:
                reward = None
            
            rewards.append(reward)
        
        return rewards
    
    return math_accuracy_reward


def prepare_dataset(dataset_name: str, num_examples: int):
    """Load and format dataset"""
    if "math" in dataset_name.lower():
        dataset = load_dataset("lighteval/MATH", split="train")
    else:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    # Select subset
    dataset = dataset.select(range(min(num_examples, len(dataset))))
    
    # Format for GRPO
    def make_conversation(example):
        problem = example.get("problem", example.get("question", ""))
        solution = example.get("solution", example.get("answer", ""))
        return {
            "prompt": [{"role": "user", "content": problem}],
            "solution": solution
        }
    
    dataset = dataset.map(make_conversation)
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["prompt", "solution"]]
    )
    
    return dataset


def create_grpo_trainer(config: ExperimentConfig):
    """Initialize GRPO trainer with paper specifications"""
    
    # Model configuration
    model_config = ModelConfig(
        model_name_or_path=config.model_name,
        torch_dtype="bfloat16",
        use_peft=True,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_target_modules="all-linear",  # All layers as per paper
        load_in_4bit=config.use_quantization,
    )
    
    # Training configuration
    training_args = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=config.max_steps,
        gradient_checkpointing=True,
        num_generations=config.num_samples_per_problem,
        generation_batch_size=config.num_samples_per_problem,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        logging_steps=10,
        save_steps=100,
        report_to=["none"],  # Disable wandb/trackio for simplicity
        bf16=True,
    )
    
    # Setup model initialization
    dtype = torch.bfloat16
    if config.use_quantization:
        training_args.model_init_kwargs = {
            "torch_dtype": dtype,
            "device_map": get_kbit_device_map(),
            "quantization_config": get_quantization_config(model_config),
        }
    else:
        training_args.model_init_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
        }
    
    # Prepare dataset
    dataset = prepare_dataset(config.dataset_name, config.num_train_examples)
    
    # Get PEFT config
    peft_config = get_peft_config(model_config)
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        reward_funcs=[create_reward_function()],
        train_dataset=dataset,
        peft_config=peft_config,
    )
    
    return trainer, model_config


def train_single_experiment(config: ExperimentConfig):
    """Train single experiment"""
    print(f"\n{'='*70}")
    print(f"Training LoRA Rank={config.lora_r}, LR={config.learning_rate}")
    print('='*70)
    
    trainer, model_config = create_grpo_trainer(config)
    trainer.train()
    trainer.save_model(config.output_dir)
    
    # Evaluate
    metrics = trainer.state.log_history
    rewards = [m.get("train/reward", 0) for m in metrics if "train/reward" in m]
    
    print(f"\nFinal average reward: {np.mean(rewards[-10:]) if rewards else 0:.4f}")
    
    return {
        "config": config,
        "metrics": metrics,
        "final_reward": np.mean(rewards[-10:]) if rewards else 0
    }


def test_trained_model(model_path: str, model_name: str, problem: str):
    """Test trained model on a problem"""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    
    messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return response


def run_lr_sweep(
    ranks=[1, 2, 4, 8, 16, 32],
    learning_rates=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
):
    """
    Reproduce Figure 6: LR sweep across ranks
    Key finding: Rank=1 matches full fine-tuning
    """
    results = {}
    
    for rank in ranks:
        results[rank] = {}
        for lr in learning_rates:
            config = ExperimentConfig(
                lora_r=rank,
                learning_rate=lr,
                max_steps=1000,
                num_samples_per_problem=32,
                output_dir=f"./grpo-lora-r{rank}-lr{lr}"
            )
            
            result = train_single_experiment(config)
            results[rank][lr] = result["final_reward"]
    
    return results


def quick_test():
    """Quick test with small model and dataset"""
    config = ExperimentConfig(
        model_name="Qwen/Qwen3-0.6B",  # Smaller for testing
        lora_r=1,
        learning_rate=1e-6,  # Scaled for smaller model
        max_steps=100,
        num_samples_per_problem=8,
        batch_size=1,
        num_train_examples=500,
        use_quantization=True,
        output_dir="./grpo-test"
    )
    
    return train_single_experiment(config)


if __name__ == "__main__":
    # Paper specification: Rank=1 on 8B model
    config = ExperimentConfig(
        model_name="meta-llama/Llama-3.1-8B",
        dataset_name="lighteval/MATH",
        lora_r=1,
        lora_alpha=32,
        learning_rate=3e-4,
        max_steps=1000,
        num_samples_per_problem=32,
        batch_size=8,
        num_train_examples=10000,
        use_quantization=False,
        output_dir="./grpo-lora-llama8b-r1"
    )
    
    result = train_single_experiment(config)
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"Final Reward: {result['final_reward']:.4f}")
    print(f"Model saved to: {config.output_dir}")
    print('='*70)
