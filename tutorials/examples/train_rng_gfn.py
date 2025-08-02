
"""
Tutorial: Training a GFlowNet to finetune an LLM for random number generation.

This tutorial demonstrates how to use TorchGFN to finetune a language model (e.g., GPT-2)
to generate random integers between 0 and 100. The GFlowNet learns to sample from a 
uniform distribution over these numbers by using trajectory balance training. 

Features:
- Supports both parameter-efficient fine-tuning with LoRA and full fine-tuning
- Uses AdamW optimizer with weight decay (standard for transformer models)
- Configurable learning rate scheduling (cosine, linear, or constant)
- Gradient clipping for training stability
- Warmup steps for better convergence
- Optional replay buffer for improved training stability

Usage:
    # LoRA training with cosine scheduler (recommended)
    python train_rng_gfn.py --use_lora --n_steps 1000 --batch_size 16 --warmup_steps 100
    
    # Full fine-tuning with linear scheduler
    python train_rng_gfn.py --n_steps 500 --scheduler_type linear --lr 1e-5 --weight_decay 0.01
    
    # Training with replay buffer for better stability
    python train_rng_gfn.py --use_lora --use_buffer --n_steps 1000 --batch_size 16
    
    # Custom LoRA configuration with different scheduler
    python train_rng_gfn.py --lora_r 16 --use_lora --lora_alpha 32 --target_modules c_attn c_proj --scheduler_type constant
"""

import torch
from typing import cast
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import os
from peft.tuners.lora import LoraConfig
from peft import get_peft_model
from peft.utils.peft_types import TaskType
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from gfn.env import DiscreteEnv
from gfn.actions import Actions
from gfn.states import DiscreteStates
from gfn.modules import DiscretePolicyEstimator, GFNModule
from gfn.preprocessors import IdentityPreprocessor, Preprocessor
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.samplers import Sampler
from gfn.containers import ReplayBuffer


class RNGEnv(DiscreteEnv):
    """Environment that builds a number token-by-token after a fixed prompt.

    A state is a fixed-length tensor of token ids consisting of the prompt
    followed by up to ``max_length`` generated tokens. Padding is done with the
    tokenizer pad token id. The episode terminates when an ``eos`` token is
    generated **or** when the maximum length is reached (in which case only the
    ``eos`` token is allowed). The reward is the uniform probability over the
    integers 0-100.
    """

    def __init__(self, tokenizer, prompt, max_length: int = 5, device: str | torch.device = "cuda"):
        self.tokenizer = tokenizer
        self.prompt = prompt
        device = torch.device(device)

        # Prompt tokens (1, prompt_len)
        self.prompt_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = self.prompt_tokens.shape[1]

        # Fixed state length (prompt + generated tokens).
        self.max_length = max_length
        self.total_length = prompt_len + max_length  #  total length of the tensor

        # Initial state: prompt followed by padding.
        s0 = torch.nn.functional.pad(
            self.prompt_tokens.squeeze(0),
            (0, max_length),
            value=tokenizer.pad_token_id,
        )  # (state_shape,)

        # Sink state: only the EOS token followed by padding.
        sf = torch.nn.functional.pad(
            torch.tensor([tokenizer.eos_token_id], device=device),
            (0, self.total_length - 1),
            value=tokenizer.pad_token_id,
        )

        # Use default action handling like HyperGrid
        super().__init__(
            n_actions=tokenizer.vocab_size,
            s0=s0,
            state_shape=(self.total_length,),
            sf=sf,
        )

    @property
    def device(self) -> torch.device:
        """Returns the device of the environment."""
        return self.s0.device

    # ---------------------------------------------------------------------
    # Masks helpers
    # ---------------------------------------------------------------------
    def _forward_action_masks(self, states: DiscreteStates) -> torch.Tensor:
        """Returns a bool mask (*batch, n_actions) of valid *forward* actions."""
        batch_size = states.batch_shape[0]
        # Use the device from the states tensor instead of self.device
        masks = torch.ones((batch_size, self.n_actions), dtype=torch.bool, device=states.tensor.device)

        # Current sequence length (non-pad tokens).
        seq_lens = (states.tensor != self.tokenizer.pad_token_id).sum(dim=1)

        # If the state is already full forbid everything but EOS.
        full_idx = seq_lens >= self.total_length - 1
        if full_idx.any():
            masks[full_idx] = False
            masks[full_idx, self.tokenizer.eos_token_id] = True
        
        return masks
    
    def _backward_action_masks(self, states: DiscreteStates) -> torch.Tensor:
        """Returns a bool mask (*batch, n_actions) of valid *backward* actions."""
        prompt_len = self.prompt_tokens.shape[1]
        seq_lens = (states.tensor != self.tokenizer.pad_token_id).sum(dim=1)
        at_initial = seq_lens <= prompt_len
        
        # Backward mask: only the last non-pad token can be removed (one True per row, rest False)
        batch_size = states.tensor.shape[0]
        backward_masks = torch.zeros((batch_size, self.n_actions - 1), dtype=torch.bool, device=states.tensor.device)
        
        # Find which sequences can go backward (not at initial state)
        can_go_back = ~at_initial
        if can_go_back.any():
            # Get the last token ID for each sequence that can go backward
            last_token_indices = seq_lens[can_go_back] - 1
            batch_indices = torch.arange(batch_size, device=states.tensor.device)[can_go_back]
            last_token_ids = states.tensor[batch_indices, last_token_indices]  
            backward_masks[batch_indices, last_token_ids] = True
        return backward_masks

    def update_masks(self, states: DiscreteStates) -> None:  # type: ignore[override]
        """Populate ``states.forward_masks`` and ``states.backward_masks`` in-place."""
        # Forward masks (valid next tokens).
        states.forward_masks = self._forward_action_masks(states)
        # Backward masks: can go backward unless we're at the initial state
        states.backward_masks = self._backward_action_masks(states)

    # ---------------------------------------------------------------------
    # Transitions
    # ---------------------------------------------------------------------
    def step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        # Insert the new token at the first padding position (keeps shape constant).
        new_states_tensor = states.tensor.clone()
        pad_token_id = self.tokenizer.pad_token_id
        for idx in range(len(states)):
            pad_positions = (new_states_tensor[idx] == pad_token_id).nonzero()
            if pad_positions.numel() == 0:
                continue  # already full, should not happen thanks to masks
            first_pad = pad_positions[0].item()
            new_states_tensor[idx, first_pad] = actions.tensor[idx, 0]
        out = self.States(new_states_tensor)
        self.update_masks(cast(DiscreteStates, out))
        return cast(DiscreteStates, out)

    def backward_step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        # Remove the last token (it should match ``actions``).
        pad_token_id = self.tokenizer.pad_token_id
        new_states_tensor = states.tensor.clone()
        for idx in range(len(states)):
            non_pad_positions = (new_states_tensor[idx] != pad_token_id).nonzero()
            if non_pad_positions.numel() == 0:
                continue
            last_idx = non_pad_positions[-1].item()
            assert new_states_tensor[idx, last_idx] == actions.tensor[idx, 0]
            new_states_tensor[idx, last_idx] = pad_token_id
        out = self.States(new_states_tensor)
        self.update_masks(cast(DiscreteStates, out))
        return cast(DiscreteStates, out)

    # ---------------------------------------------------------------------
    # Reward
    # ---------------------------------------------------------------------
    def log_reward(self, states: DiscreteStates) -> torch.Tensor:
        """Uniform log-probability for numbers 0-100; −∞ elsewhere."""
        rewards = torch.full((len(states),), -1e2, device=self.device)

        prompt_len = self.prompt_tokens.shape[1]
        for idx in range(len(states)):
            # Identify generated part (after prompt) ignoring padding and eos.
            seq = states.tensor[idx]
            # Determine where padding starts.
            pad_mask = seq == self.tokenizer.pad_token_id
            if pad_mask.all():
                continue  # empty

            # Extract tokens between prompt and eos.
            # First generated index after prompt.
            gen_start = prompt_len
            # Find eos position.
            eos_positions = (seq == self.tokenizer.eos_token_id).nonzero(as_tuple=False)
            if eos_positions.numel() == 0:
                continue  # not terminated yet
            eos_pos = eos_positions[0].item()
            generated_tokens = seq[gen_start:eos_pos]
            if len(generated_tokens) == 0:
                continue

            decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            try:
                number = int(decoded_text.strip())
                if 0 <= number <= 100:
                    rewards[idx] = 0.0
            except ValueError:
                pass

        return rewards


class LLMGFNModule(DiscretePolicyEstimator):
    """GFNModule wrapping a pretrained LLM to act as a policy."""

    def __init__(self, model, tokenizer, state_dim: int, is_backward: bool = False):
        super().__init__(
            module=model,
            n_actions=tokenizer.vocab_size,
            preprocessor=IdentityPreprocessor(state_dim),
            is_backward=is_backward
        )
        self.model = model
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # GFNModule interface
    # ------------------------------------------------------------------
    @property
    def expected_output_dim(self):
        return self.tokenizer.vocab_size

    def forward(self, states: DiscreteStates):  # type: ignore[override]
        input_ids = states.tensor
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Select logits corresponding to the last *non-pad* token.
        seq_lengths = attention_mask.sum(dim=1)  # (batch,)
        last_token_logits = outputs.logits[torch.arange(len(outputs.logits)), seq_lengths - 1, :]

        return last_token_logits

    def to_probability_distribution(
        self,
        states: DiscreteStates,
        module_output: torch.Tensor,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        sf_bias: float = 0.0,
        **kwargs,
    ):
        """Convert raw logits to a categorical distribution, respecting masks."""

        logits = module_output.clone()
        
        if self.is_backward:
            # Backward masks
            masks = states.backward_masks
            assert self.tokenizer.eos_token_id == logits.shape[1] - 1
            logits[:, :-1][~masks] = -float("inf")
            logits[:, -1] = -float("inf")
        else:
            # Forward masks include all actions
            masks = states.forward_masks
            logits[~masks] = -float("inf")

        # Create a custom distribution that samples with the right shape
        categorical = torch.distributions.Categorical(logits=logits)
        class ShapedCategorical:
            def __init__(self, cat_dist):
                self.cat_dist = cat_dist
            
            def sample(self):
                # Sample from categorical and reshape to (batch_size, 1)
                samples = self.cat_dist.sample()
                return samples.unsqueeze(-1)
            
            def log_prob(self, value):
                # value should have shape (batch_size, 1), squeeze for categorical
                if value.dim() > 1:
                    value = value.squeeze(-1)
                return self.cat_dist.log_prob(value)
        
        return ShapedCategorical(categorical)


def evaluate_model(env, trajectories, tokenizer, step_name="Evaluation"):
    """Evaluate the model by sampling trajectories and analyzing generated numbers."""
    final_states = trajectories.terminating_states
    
    numbers = []
    for i in range(len(final_states)):
        state_tensor = final_states.tensor[i]
        # Identify non-pad tokens
        non_pad = state_tensor != tokenizer.pad_token_id
        seq = state_tensor[non_pad]

        # Remove prompt and EOS
        prompt_len = env.prompt_tokens.shape[1]
        if len(seq) <= prompt_len + 1:
            continue
        generated_tokens = seq[prompt_len:-1]  # exclude prompt and eos
        decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        try:
            number = int(decoded_text.strip())
            if 0 <= number <= 100:  # Only count valid numbers in range
                numbers.append(number)
        except (ValueError, IndexError):
            pass
    
    results = {}
    total_trajectories = len(trajectories)
    
    if numbers:
        counts = np.bincount(numbers, minlength=101)
        valid_counts = counts[:101]  # Only 0-100
        success_rate = len(numbers) / total_trajectories
        unique_numbers = np.arange(101)[valid_counts > 0]
        unique_count = len(unique_numbers)
        mean_val = np.mean(numbers)
        std_val = np.std(numbers)
        
        results = {
            'valid_numbers': numbers,
            'success_rate': success_rate,
            'unique_count': unique_count,
            'unique_numbers': unique_numbers,
            'mean': mean_val,
            'std': std_val,
            'total_valid': len(numbers),
            'total_trajectories': total_trajectories,
            'step_name': step_name
        }
    else:
        results = {
            'valid_numbers': [],
            'success_rate': 0.0,
            'unique_count': 0,
            'mean': 0.0,
            'std': 0.0,
            'total_valid': 0,
            'total_trajectories': total_trajectories,
            'step_name': step_name
        }
    
    return results


def get_lambda_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def get_lr_mult_at_step(step):
        if step <= num_warmup_steps:
            return step / num_warmup_steps
        return max((num_training_steps - step) / (num_training_steps - num_warmup_steps), 0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_mult_at_step)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GFlowNet to generate random numbers 0-100")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--model_name", default="gpt2", help="Model name from HuggingFace")
    parser.add_argument("--n_steps", type=int, default=400, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=3, help="Max tokens to generate")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of samples for evaluation")
    
    # LoRA-specific arguments
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=512, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout probability")
    parser.add_argument("--target_modules", nargs="+", default=["c_attn", "c_proj"], 
                       help="Target modules for LoRA adaptation (default for GPT-2)")
    
    # Optimizer and scheduler arguments
    parser.add_argument("--weight_decay", type=float, default=0.00, help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--scheduler_type", default="lambda_lr", choices=["cosine", "linear", "constant", "lambda_lr"], 
                       help="Learning rate scheduler type")
    
    # Replay buffer arguments
    parser.add_argument("--use_buffer", action="store_true", default=True, help="Whether to use replay buffer for training stability")
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Model and tokenizer setup
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    
    if args.use_lora:
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias="none",
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(f"Applied LoRA with rank={args.lora_r}, alpha={args.lora_alpha}, target_modules={args.target_modules}")
    else:
        print("Using full fine-tuning (LoRA disabled)")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Environment setup
    prompt = "The following is a random integer drawn uniformly between 0 and 100: "
    env = RNGEnv(tokenizer, prompt, device=device, max_length=args.max_length)
    print(f"Environment set up with prompt: '{prompt}'")

    # GFlowNet setup
    state_dim = env.total_length
    pf_module = LLMGFNModule(model, tokenizer, state_dim, is_backward=False)
    pb_module = LLMGFNModule(model, tokenizer, state_dim, is_backward=True)

    gflownet = TBGFlowNet(pf_module, pb_module)
    sampler = Sampler(pf_module)
    
    # Initialize replay buffer if requested
    replay_buffer = None
    if args.use_buffer:
        replay_buffer = ReplayBuffer(
            env,
            capacity=args.batch_size * 4,  # Use 4x batch size as capacity
            prioritized_capacity=True,
            prioritized_sampling=True,
        )

    # Set up optimizer and scheduler for trainable parameters
    trainable_params = [p for p in gflownet.parameters() if p.requires_grad]
    
    # Use AdamW optimizer (preferred for transformers)
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Set up learning rate scheduler
    if args.scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.n_steps
        )
    elif args.scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.n_steps
        )
    elif args.scheduler_type == "lambda_lr":
        scheduler = get_lambda_lr_scheduler(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.n_steps
        )
    else:  # constant
        scheduler = None
    
    param_count = sum(p.numel() for p in trainable_params)
    print(f"Training for {args.n_steps} steps with batch size {args.batch_size}")
    print(f"Optimizing {param_count:,} trainable parameters across {len(trainable_params)} parameter groups")
    print(f"Using AdamW optimizer with lr={args.lr}, weight_decay={args.weight_decay}")
    print(f"Learning rate scheduler: {args.scheduler_type}, warmup_steps={args.warmup_steps}")
    print(f"Gradient clipping: max_norm={args.max_grad_norm}")
    if args.use_buffer:
        print(f"Using replay buffer with capacity: {args.batch_size * 4}")
    else:
        print("Not using replay buffer (--use_buffer flag not set)")

    # Pre-training evaluation
    print("Evaluating model performance before training...")
    with torch.no_grad():
        trajectories = sampler.sample_trajectories(env, n=args.eval_samples)
    pre_training_results = evaluate_model(env, trajectories, tokenizer, "Pre-training Evaluation")
    print(f"Generated valid numbers: {pre_training_results['valid_numbers'][:20]}...")  # Show first 20
    success_rate = pre_training_results['success_rate']
    unique_count = pre_training_results['unique_count']
    unique_numbers = pre_training_results['unique_numbers']
    mean_val = pre_training_results['mean']
    std_val = pre_training_results['std']
    total_valid = pre_training_results['total_valid']
    print(f"Number of valid samples: {total_valid} / {len(trajectories)} ({100*success_rate:.1f}%)")
    print(f"Unique numbers generated: {unique_count} / 101")
    print(f"Unique numbers: {unique_numbers}")
    print(f"Mean: {mean_val:.1f}, Std: {std_val:.1f}")
    
    # Initialize loss tracking
    loss_history = []
    
    # Training Loop
    for step in range(args.n_steps):
        # Sample trajectories using gflownet for compatibility with replay buffer
        trajectories = gflownet.sample_trajectories(env, n=args.batch_size, save_logprobs=True)
        training_samples = gflownet.to_training_samples(trajectories)
        
        # Use replay buffer if enabled
        if args.use_buffer and replay_buffer is not None:
            with torch.no_grad():
                replay_buffer.add(training_samples)
                # After some initial steps, use half fresh samples and half from buffer
                if step > 10:
                    training_samples = training_samples[: args.batch_size // 2]
                    buffer_samples = replay_buffer.sample(n_samples=args.batch_size // 2)
                    training_samples.extend(buffer_samples)  # type: ignore
        
        # Calculate loss with recalculated logprobs for buffer compatibility
        recalculate_logprobs = args.use_buffer and replay_buffer is not None
        loss = gflownet.loss(env, training_samples, recalculate_all_logprobs=recalculate_logprobs)
        
        # Store loss for plotting
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        if args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        
        optimizer.step()
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        if step % 1 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            training_results = evaluate_model(env, trajectories, tokenizer, "Training Evaluation")
            buffer_info = f", buffer_size = {len(replay_buffer) if replay_buffer else 0}" if args.use_buffer else ""
            print(f"Step {step:4d}: loss = {loss.item():.4f}, lr = {current_lr:.6f}, grad_norm = {grad_norm:.4f}, success_rate = {training_results['success_rate']:.1%}, unique_numbers = {training_results['unique_count']}, unique_numbers = {training_results['unique_numbers']}{buffer_info}")

    # Post-training evaluation
    with torch.no_grad():
        trajectories = sampler.sample_trajectories(env, n=args.eval_samples)
    post_training_results = evaluate_model(env, trajectories, tokenizer, "Post-training Evaluation")
    
    # Create and save loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_history)), loss_history, 'b-', linewidth=1.0)
    plt.title('Training Loss Over Time')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save the plot with a descriptive filename
    plot_filename = f'plots/loss_plot_steps{args.n_steps}_bs{args.batch_size}_lr{args.lr}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nLoss plot saved to: {plot_filename}")
    
    # Show basic loss statistics
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Minimum loss: {min(loss_history):.4f} (at step {loss_history.index(min(loss_history))})")
    print(f"Average loss: {np.mean(loss_history):.4f}")
    
    plt.close()  # Close the figure to free memory
    
    # Compare results
    print("\n--- Training Results Comparison ---")
    print(f"Success Rate: {pre_training_results['success_rate']:.1%} → {post_training_results['success_rate']:.1%} "
          f"(Δ: {post_training_results['success_rate'] - pre_training_results['success_rate']:+.1%})")
    print(f"Unique Numbers: {pre_training_results['unique_count']} → {post_training_results['unique_count']} "
          f"(Δ: {post_training_results['unique_count'] - pre_training_results['unique_count']:+d})")
    print(f"Mean Value: {pre_training_results['mean']:.1f} → {post_training_results['mean']:.1f} "
          f"(Δ: {post_training_results['mean'] - pre_training_results['mean']:+.1f})")
    print(f"Std Deviation: {pre_training_results['std']:.1f} → {post_training_results['std']:.1f} "
          f"(Δ: {post_training_results['std'] - pre_training_results['std']:+.1f})")



if __name__ == '__main__':
    main()
