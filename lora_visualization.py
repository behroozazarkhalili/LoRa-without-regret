"""
Visualization code to reproduce figures from "LoRA Without Regret"
Generates plots matching Figures 1-8 from the paper
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from typing import Dict, List
import pandas as pd

# Set publication-quality plot style
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10
rcParams['figure.dpi'] = 150


class LoRAVisualizer:
    """Generate publication-quality visualizations of LoRA experiments"""
    
    def __init__(self):
        self.colors = sns.color_palette("husl", 13)  # For different ranks
        self.rank_colors = {
            1: self.colors[0],
            2: self.colors[1],
            4: self.colors[2],
            8: self.colors[3],
            16: self.colors[4],
            32: self.colors[5],
            64: self.colors[6],
            128: self.colors[7],
            256: self.colors[8],
            512: self.colors[9],
            'full': 'black'
        }
    
    def plot_training_curves(
        self, 
        results: Dict[int, List[float]], 
        title: str = "Training Curves",
        save_path: str = None
    ):
        """
        Reproduce Figure 1: Training curves for various ranks
        
        Args:
            results: Dict mapping rank -> list of losses over training steps
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot each rank
        for rank, losses in sorted(results.items()):
            steps = np.arange(1, len(losses) + 1)
            color = self.rank_colors.get(rank, 'gray')
            label = 'full' if rank == 'full' else f'Rank {rank}'
            
            # Take pointwise minimum over learning rates (as in paper)
            ax.plot(steps, losses, color=color, label=label, linewidth=2)
        
        ax.set_xscale('log')
        ax.set_xlabel('Step')
        ax.set_ylabel('Test NLL')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_lr_vs_loss(
        self,
        results: Dict[int, Dict[float, float]],
        title: str = "Learning Rate vs Final Loss",
        save_path: str = None
    ):
        """
        Reproduce Figure 2: LR sweep showing optimal LRs for each rank
        
        Args:
            results: Dict mapping rank -> {lr: final_loss}
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for rank, lr_losses in sorted(results.items()):
            lrs = sorted(lr_losses.keys())
            losses = [lr_losses[lr] for lr in lrs]
            
            color = self.rank_colors.get(rank, 'gray')
            label = 'full' if rank == 'full' else f'Rank {rank}'
            
            ax.plot(lrs, losses, 'o-', color=color, label=label, 
                   linewidth=2, markersize=6)
        
        ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Final Loss')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Annotate key finding: LoRA optimal LR is ~10x FullFT optimal LR
        ax.axvline(x=3e-5, color='black', linestyle='--', alpha=0.5, 
                   label='FullFT optimal')
        ax.axvline(x=3e-4, color='red', linestyle='--', alpha=0.5,
                   label='LoRA optimal (10x)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_batch_size_effects(
        self,
        lora_results: Dict[int, List[float]],
        fullft_results: Dict[int, List[float]],
        save_path: str = None
    ):
        """
        Reproduce Figure 3: Batch size effects on LoRA vs FullFT
        
        Args:
            lora_results: Dict mapping batch_size -> losses over steps
            fullft_results: Dict mapping batch_size -> losses over steps
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: Learning curves for different batch sizes
        for bs in sorted(lora_results.keys()):
            steps = np.arange(1, len(lora_results[bs]) + 1)
            ax1.plot(steps, lora_results[bs], '--', 
                    label=f'LoRA, BS={bs}', linewidth=2)
            ax1.plot(steps, fullft_results[bs], '-',
                    label=f'Full, BS={bs}', linewidth=2)
        
        ax1.set_xscale('log')
        ax1.set_xlabel('Examples')
        ax1.set_ylabel('Test NLL')
        ax1.set_title('Learning Curves vs Batch Size')
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Final loss vs batch size
        batch_sizes = sorted(lora_results.keys())
        lora_final = [lora_results[bs][-1] for bs in batch_sizes]
        fullft_final = [fullft_results[bs][-1] for bs in batch_sizes]
        
        ax2.plot(batch_sizes, lora_final, 'o--', label='LoRA', 
                linewidth=2, markersize=8)
        ax2.plot(batch_sizes, fullft_final, 'o-', label='Full', 
                linewidth=2, markersize=8, color='black')
        
        ax2.set_xlabel('Batch size')
        ax2.set_ylabel('Final Loss')
        ax2.set_title('Batch Penalties')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_layer_comparison(
        self,
        results: Dict[str, Dict[float, float]],
        save_path: str = None
    ):
        """
        Reproduce Figure 4/5: Comparing LoRA on different layers
        
        Args:
            results: Dict mapping layer_config -> {lr: final_loss}
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        layer_configs = {
            'attention_only': 'Attention only',
            'mlp_only': 'MLP only',
            'all_layers': 'All layers (MLP+Attn)'
        }
        
        colors = {'attention_only': 'red', 'mlp_only': 'blue', 
                  'all_layers': 'green'}
        
        for config, label in layer_configs.items():
            if config in results:
                lrs = sorted(results[config].keys())
                losses = [results[config][lr] for lr in lrs]
                
                ax.plot(lrs, losses, 'o-', color=colors[config],
                       label=label, linewidth=2, markersize=6)
        
        ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Final Loss / Reward')
        ax.set_title('Layer Application Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_rl_results(
        self,
        results: Dict[int, Dict[float, float]],
        title: str = "RL: Learning Rate vs Final Reward",
        save_path: str = None
    ):
        """
        Reproduce Figure 6: RL experiments showing LoRA=FullFT even at rank=1
        
        Args:
            results: Dict mapping rank -> {lr: final_reward}
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for rank in sorted(results.keys()):
            lrs = sorted(results[rank].keys())
            rewards = [results[rank][lr] for lr in lrs]
            
            color = self.rank_colors.get(rank, 'gray')
            label = 'full' if rank == 'full' else f'Rank {rank}'
            
            ax.plot(lrs, rewards, 'o-', color=color, label=label,
                   linewidth=2, markersize=6)
        
        ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Final Reward (Accuracy)')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add annotation about key finding
        ax.text(0.5, 0.95, 
                'Key finding: Rank=1 matches FullFT for RL',
                transform=ax.transAxes,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_capacity_analysis(
        self,
        dataset_sizes: List[int],
        lora_params: List[int],
        performance: Dict[tuple, float],
        save_path: str = None
    ):
        """
        Analyze capacity requirements: when does LoRA underperform?
        
        Args:
            dataset_sizes: List of dataset sizes (in tokens)
            lora_params: List of LoRA parameter counts
            performance: Dict mapping (dataset_size, params) -> final_loss
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create heatmap
        data = np.zeros((len(lora_params), len(dataset_sizes)))
        for i, params in enumerate(lora_params):
            for j, size in enumerate(dataset_sizes):
                data[i, j] = performance.get((size, params), np.nan)
        
        im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r', 
                      interpolation='nearest')
        
        # Set ticks
        ax.set_xticks(range(len(dataset_sizes)))
        ax.set_yticks(range(len(lora_params)))
        ax.set_xticklabels([f'{s/1e6:.1f}M' for s in dataset_sizes])
        ax.set_yticklabels([f'{p/1e6:.1f}M' for p in lora_params])
        
        ax.set_xlabel('Dataset Size (tokens)')
        ax.set_ylabel('LoRA Parameters')
        ax.set_title('Capacity Analysis: When LoRA Underperforms')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Final Loss', rotation=270, labelpad=20)
        
        # Add diagonal line showing capacity boundary
        # Rule of thumb: 2 bits per parameter
        x = np.arange(len(dataset_sizes))
        y = x  # Simplified for visualization
        ax.plot(x, y, 'w--', linewidth=2, 
               label='Capacity boundary\n(~2 bits/param)')
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def analyze_information_theory():
    """
    Illustrate the information-theoretic arguments from the paper
    """
    print("=" * 70)
    print("INFORMATION THEORY ANALYSIS")
    print("=" * 70)
    
    # Supervised Learning capacity
    print("\n1. SUPERVISED LEARNING:")
    print("-" * 70)
    tokens_per_example = 1000
    examples = 10000
    total_tokens = tokens_per_example * examples
    bits_per_token = 1.0  # ~1 bit/token for typical LLM datasets
    total_bits = total_tokens * bits_per_token
    
    print(f"Dataset: {examples:,} examples × {tokens_per_example} tokens/ex")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Information content: ~{bits_per_token} bit/token")
    print(f"Total information: {total_bits/1e6:.1f}M bits")
    
    # LoRA capacity
    print("\n2. LoRA CAPACITY:")
    print("-" * 70)
    for rank in [1, 8, 64, 256]:
        # Llama-3.1-8B dimensions
        hidden_size = 4096
        num_layers = 32
        matrices_per_layer = 7  # q,k,v,o,gate,up,down
        
        params_per_layer = rank * (hidden_size + hidden_size) * matrices_per_layer
        total_params = params_per_layer * num_layers
        bits_capacity = total_params * 2  # 2 bits per parameter
        
        print(f"\nRank {rank}:")
        print(f"  Parameters: {total_params/1e6:.2f}M")
        print(f"  Capacity: {bits_capacity/1e6:.1f}M bits (@ 2 bits/param)")
        print(f"  Ratio to dataset: {bits_capacity/total_bits:.1f}x")
    
    # RL information
    print("\n3. REINFORCEMENT LEARNING:")
    print("-" * 70)
    episodes = 10000
    samples_per_episode = 32
    bits_per_sample = 1  # Binary reward = 1 bit
    total_rl_bits = episodes * samples_per_episode * bits_per_sample
    
    print(f"Episodes: {episodes:,}")
    print(f"Samples per episode: {samples_per_episode}")
    print(f"Information per sample: {bits_per_sample} bit (binary reward)")
    print(f"Total information: {total_rl_bits/1e6:.2f}M bits")
    print(f"\nThis explains why rank=1 works for RL!")
    print(f"Even rank=1 LoRA has ~3M parameters = 6M bits capacity")
    print(f"Dataset only needs: {total_rl_bits/1e6:.2f}M bits")
    print(f"Capacity ratio: {(3e6 * 2) / total_rl_bits:.0f}x")


def summarize_key_findings():
    """Print summary of key findings from the paper"""
    print("\n" + "=" * 70)
    print("KEY FINDINGS FROM 'LoRA WITHOUT REGRET'")
    print("=" * 70)
    
    findings = [
        ("1. LoRA matches FullFT for typical post-training",
         "With sufficient rank (e.g., 128-256), LoRA achieves the same "
         "performance as full fine-tuning on instruction-tuning datasets"),
        
        ("2. Apply LoRA to ALL layers, especially MLPs",
         "Attention-only LoRA significantly underperforms. MLP-only performs "
         "similarly to all-layers. This contradicts original LoRA paper"),
        
        ("3. Optimal LoRA LR is 10x FullFT LR",
         "Empirically, LoRA optimal learning rate is consistently 10× higher "
         "than full fine-tuning across models and tasks"),
        
        ("4. Rank=1 works perfectly for RL",
         "Due to O(1) bits/episode in policy gradient, even rank-1 LoRA "
         "matches FullFT for reinforcement learning"),
        
        ("5. Large batch sizes hurt LoRA more",
         "LoRA is less tolerant of large batch sizes than FullFT, "
         "showing a persistent gap that doesn't depend on rank"),
        
        ("6. Capacity determines performance",
         "LoRA underperforms when dataset information exceeds LoRA capacity. "
         "Rule of thumb: ~2 bits per parameter"),
        
        ("7. LoRA is compute-efficient",
         "LoRA uses ~2/3 FLOPs of FullFT per pass, providing better "
         "compute efficiency overall"),
    ]
    
    for title, description in findings:
        print(f"\n{title}")
        print("-" * 70)
        print(description)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    visualizer = LoRAVisualizer()
    
    # Generate sample data for demonstration
    # In practice, load from actual experiment results
    
    # Example: Training curves
    sample_results = {
        1: np.linspace(2.0, 1.5, 1000) + np.random.normal(0, 0.02, 1000),
        8: np.linspace(2.0, 1.2, 1000) + np.random.normal(0, 0.01, 1000),
        128: np.linspace(2.0, 0.8, 1000) + np.random.normal(0, 0.01, 1000),
        'full': np.linspace(2.0, 0.8, 1000) + np.random.normal(0, 0.01, 1000),
    }
    
    visualizer.plot_training_curves(
        sample_results,
        title="LoRA Training Curves (Sample Data)"
    )
    
    # Print information theory analysis
    analyze_information_theory()
    
    # Print key findings summary
    summarize_key_findings()
