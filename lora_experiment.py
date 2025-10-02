"""
LoRA Without Regret: Experiment Replication
Implementation of supervised fine-tuning experiments with LoRA
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import wandb
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ExperimentConfig:
    """Configuration for LoRA experiments"""
    model_name: str = "meta-llama/Llama-3.1-8B"
    dataset_name: str = "allenai/tulu-3-sft-mixture"
    
    # LoRA hyperparameters
    lora_r: int = 128  # Rank: varies from 1 to 512
    lora_alpha: int = 32  # Scaling factor α
    lora_dropout: float = 0.0
    target_modules: List[str] = None  # ["q_proj", "v_proj"] or "all"
    
    # Training hyperparameters
    learning_rate: float = 3e-4  # Will sweep this
    batch_size: int = 32
    max_steps: int = 10000
    eval_steps: int = 100
    max_seq_length: int = 2048
    
    # Experiment settings
    use_full_ft: bool = False
    apply_lora_to_all_layers: bool = True
    
    # Logging
    use_wandb: bool = False
    log_dir: str = "./logs"


class LoRAExperiment:
    """Main experiment class for LoRA vs Full Fine-Tuning comparison"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = self._setup_model()
        
        # Load dataset
        self.train_dataset, self.eval_dataset = self._load_datasets()
        
    def _setup_model(self):
        """Setup model with or without LoRA"""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        if not self.config.use_full_ft:
            # Configure LoRA
            target_modules = self._get_target_modules()
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                # Key: initialize A with uniform distribution, B with zeros
                init_lora_weights=True
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            # Full fine-tuning - all parameters trainable
            for param in model.parameters():
                param.requires_grad = True
                
        return model.to(self.device)
    
    def _get_target_modules(self):
        """Determine which modules to apply LoRA to"""
        if self.config.apply_lora_to_all_layers:
            # Apply to all linear layers (attention + MLP)
            # For Llama: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
            return [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj"       # MLP
            ]
        else:
            # Attention-only (original LoRA paper recommendation)
            return ["q_proj", "v_proj"]
    
    def _load_datasets(self):
        """Load and preprocess datasets"""
        dataset = load_dataset(self.config.dataset_name, split="train")
        
        # Split into train/eval
        split = dataset.train_test_split(test_size=0.05, seed=42)
        
        return split["train"], split["test"]
    
    def preprocess_batch(self, examples):
        """Tokenize and format examples"""
        # Format as instruction-response pairs
        texts = []
        for ex in examples:
            # Assuming dataset has 'messages' field with user/assistant turns
            formatted = self.tokenizer.apply_chat_template(
                ex.get("messages", []),
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(formatted)
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        
        # Labels are input_ids shifted
        encodings["labels"] = encodings["input_ids"].clone()
        
        return encodings
    
    def compute_loss(self, batch):
        """Compute cross-entropy loss (log loss)"""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        outputs = self.model(**batch)
        return outputs.loss
    
    def train_epoch(self, optimizer, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        losses = []
        
        pbar = tqdm(dataloader, desc="Training")
        for step, batch in enumerate(pbar):
            loss = self.compute_loss(batch)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            losses.append(loss.item())
            
            if step % 10 == 0:
                pbar.set_postfix({"loss": loss.item()})
            
            if step >= self.config.max_steps:
                break
        
        return losses
    
    def evaluate(self, dataloader):
        """Evaluate on held-out data"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def run_experiment(self):
        """Run full training experiment"""
        # Setup optimizer with constant LR (no warmup/cooldown)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Prepare dataloaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.preprocess_batch
        )
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.preprocess_batch
        )
        
        # Training loop (single epoch as per paper)
        print(f"\n{'='*60}")
        print(f"Training with LoRA rank={self.config.lora_r}, LR={self.config.learning_rate}")
        print(f"{'='*60}\n")
        
        train_losses = self.train_epoch(optimizer, train_loader)
        
        # Final evaluation
        eval_loss = self.evaluate(eval_loader)
        
        return {
            "train_losses": train_losses,
            "final_eval_loss": eval_loss,
            "config": self.config
        }


def sweep_learning_rates_and_ranks():
    """
    Main experiment: sweep over learning rates and ranks
    Reproduces Figure 1 from the paper
    """
    ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    # Key finding: LoRA optimal LR is ~10x FullFT optimal LR
    lora_learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    full_ft_learning_rates = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
    
    results = {
        "lora": [],
        "full_ft": None
    }
    
    # LoRA experiments
    for rank in ranks:
        for lr in lora_learning_rates:
            config = ExperimentConfig(
                lora_r=rank,
                learning_rate=lr,
                use_full_ft=False
            )
            
            experiment = LoRAExperiment(config)
            result = experiment.run_experiment()
            results["lora"].append(result)
    
    # Full fine-tuning baseline
    for lr in full_ft_learning_rates:
        config = ExperimentConfig(
            learning_rate=lr,
            use_full_ft=True
        )
        
        experiment = LoRAExperiment(config)
        result = experiment.run_experiment()
        if results["full_ft"] is None:
            results["full_ft"] = []
        results["full_ft"].append(result)
    
    return results


def batch_size_experiment():
    """
    Reproduces Figure 3: Batch size effects
    Tests LoRA tolerance to large batch sizes
    """
    batch_sizes = [32, 64, 128, 256]
    rank = 128
    
    results = {"lora": [], "full_ft": []}
    
    for bs in batch_sizes:
        # LoRA
        config = ExperimentConfig(
            lora_r=rank,
            batch_size=bs,
            learning_rate=3e-4,
            use_full_ft=False
        )
        experiment = LoRAExperiment(config)
        results["lora"].append(experiment.run_experiment())
        
        # Full FT
        config.use_full_ft = True
        config.learning_rate = 3e-5  # 10x lower for FullFT
        experiment = LoRAExperiment(config)
        results["full_ft"].append(experiment.run_experiment())
    
    return results


def layer_application_experiment():
    """
    Reproduces Figure 4: Which layers to apply LoRA to
    Tests attention-only vs MLP-only vs all layers
    """
    rank = 256
    
    configs = [
        ("attention_only", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ("mlp_only", ["gate_proj", "up_proj", "down_proj"]),
        ("all_layers", ["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"])
    ]
    
    results = {}
    
    for name, target_modules in configs:
        config = ExperimentConfig(
            lora_r=rank,
            learning_rate=3e-4
        )
        config.target_modules = target_modules
        
        experiment = LoRAExperiment(config)
        results[name] = experiment.run_experiment()
    
    return results


if __name__ == "__main__":
    # Example: Run a single experiment
    config = ExperimentConfig(
        lora_r=128,
        learning_rate=3e-4,
        max_steps=1000,  # Reduced for testing
        batch_size=4  # Small for local testing
    )
    
    experiment = LoRAExperiment(config)
    result = experiment.run_experiment()
    
    print(f"\nFinal Evaluation Loss: {result['final_eval_loss']:.4f}")
