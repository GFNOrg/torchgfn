
"""
Tutorial: Training a GFlowNet to finetune an LLM for random number generation.

This tutorial demonstrates how to use TorchGFN to finetune a language model (e.g., GPT-2)
to generate random integers between 0 and 100. The GFlowNet learns to sample from a 
uniform distribution over these numbers by using trajectory balance training.

Usage:
    python train_rng_gfn.py --help
    python train_rng_gfn.py --n_steps 1000 --batch_size 16
    python train_rng_gfn.py --model_name distilgpt2 --device cpu
"""

import torch
import torch.nn as nn
from typing import cast
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

from gfn.env import DiscreteEnv
from gfn.actions import Actions
from gfn.states import DiscreteStates
from gfn.modules import GFNModule
from gfn.preprocessors import Preprocessor
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.samplers import Sampler


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

    # ---------------------------------------------------------------------
    # Masks helpers
    # ---------------------------------------------------------------------
    def _forward_action_masks(self, states: DiscreteStates) -> torch.Tensor:
        """Returns a bool mask (*batch, n_actions) of valid *forward* actions."""
        batch_size = states.batch_shape[0]
        masks = torch.ones((batch_size, self.n_actions), dtype=torch.bool, device=self.device)

        # Current sequence length (non-pad tokens).
        seq_lens = (states.tensor != self.tokenizer.pad_token_id).sum(dim=1)

        # If the state is already full forbid everything but EOS.
        full_idx = seq_lens >= self.total_length
        if full_idx.any():
            masks[full_idx] = False
            masks[full_idx, self.tokenizer.eos_token_id] = True
        
        # Debug: make sure at least some actions are valid
        valid_actions = masks.sum(dim=1)
        if torch.any(valid_actions == 0):
            print(f"Warning: Some states have no valid actions. seq_lens: {seq_lens}, total_length: {self.total_length}")
            # Allow at least EOS token
            invalid_states = valid_actions == 0
            masks[invalid_states, self.tokenizer.eos_token_id] = True

        return masks

    def update_masks(self, states: DiscreteStates) -> None:  # type: ignore[override]
        """Populate ``states.forward_masks`` and ``states.backward_masks`` in-place."""

        # Forward masks (valid next tokens).
        states.forward_masks = self._forward_action_masks(states)
        
        # Backward masks: can go backward unless we're at the initial state
        prompt_len = self.prompt_tokens.shape[1]
        seq_lens = (states.tensor != self.tokenizer.pad_token_id).sum(dim=1)
        at_initial = seq_lens <= prompt_len
        states.backward_masks = ~at_initial.unsqueeze(-1).expand(-1, self.n_actions - 1)

    # ---------------------------------------------------------------------
    # Transitions
    # ---------------------------------------------------------------------
    def step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        # Insert the new token at the first padding position (keeps shape constant).
        new_states_tensor = states.tensor.clone()
        pad_token_id = self.tokenizer.pad_token_id
        for idx in range(len(states)):
            pad_positions = (new_states_tensor[idx] == pad_token_id).nonzero(as_tuple=False)
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
            non_pad_positions = (new_states_tensor[idx] != pad_token_id).nonzero(as_tuple=False)
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
        rewards = torch.full((len(states),), float("-inf"), device=self.device)

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
            try:
                eos_pos = (seq == self.tokenizer.eos_token_id).nonzero(as_tuple=False)[0].item()
            except IndexError:
                continue  # not terminated yet

            # Only consider when eos is present and sequence beyond prompt.
            if eos_pos <= gen_start:
                continue

            generated_tokens = seq[gen_start:eos_pos]
            if len(generated_tokens) == 0:
                continue

            decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            try:
                number = int(decoded_text.strip())
                if 0 <= number <= 100:
                    rewards[idx] = torch.log(torch.tensor(1.0 / 101.0, device=self.device))
            except ValueError:
                pass

        return rewards


class PassThroughPreprocessor(Preprocessor):
    """Returns the raw tensor representation of states (no preprocessing)."""

    def __init__(self, output_dim: int):
        super().__init__(output_dim=output_dim)

    def preprocess(self, states):  # type: ignore[override]
        return states.tensor


class LLMGFNModule(GFNModule):
    """GFNModule wrapping a pretrained LLM to act as a policy."""

    def __init__(self, model, tokenizer, state_dim: int, is_backward: bool = False):
        super().__init__(module=model, preprocessor=PassThroughPreprocessor(output_dim=state_dim), is_backward=is_backward)
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
        # Build an attention mask: 1 for non-pad.
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
            # Backward masks exclude the exit action (last action)
            masks = states.backward_masks
            # Apply masks to all actions except the last one (exit action)
            logits[:, :-1][~masks] = -float("inf")
        else:
            # Forward masks include all actions
            masks = states.forward_masks
            logits[~masks] = -float("inf")

        # Check for any completely invalid states
        if self.is_backward:
            valid_mask_counts = masks.sum(dim=-1)
        else:
            valid_mask_counts = masks.sum(dim=-1)
        
        if torch.any(valid_mask_counts == 0):
            print(f"Warning: Found states with no valid actions in {'backward' if self.is_backward else 'forward'} mode")
            print(f"Valid action counts: {valid_mask_counts}")
            # Force at least one action to be valid (EOS for forward, or some action for backward)
            if self.is_backward:
                invalid_idx = valid_mask_counts == 0
                if torch.any(invalid_idx):
                    # For backward, allow the first non-exit action
                    masks[invalid_idx, 0] = True
            else:
                invalid_idx = valid_mask_counts == 0
                if torch.any(invalid_idx):
                    # For forward, allow EOS
                    logits[invalid_idx, self.tokenizer.eos_token_id] = 0.0

        if not self.is_backward and sf_bias != 0.0:
            logits[:, -1] -= sf_bias  # usually bias exit action

        if temperature != 1.0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)
        
        # Ensure probabilities are valid
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Create a custom distribution that samples with the right shape
        categorical = torch.distributions.Categorical(probs=probs)
        
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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GFlowNet to generate random numbers 0-100")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--model_name", default="gpt2", help="Model name from HuggingFace")
    parser.add_argument("--n_steps", type=int, default=500, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=5, help="Max tokens to generate")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of samples for evaluation")
    
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
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

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

    optimizer = torch.optim.Adam(gflownet.parameters(), lr=args.lr)
    print(f"Training for {args.n_steps} steps with batch size {args.batch_size}")

    # Training Loop
    for step in range(args.n_steps):
        trajectories = sampler.sample_trajectories(env, n=args.batch_size, save_logprobs=False)
        loss = gflownet.loss(env, trajectories)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step:4d}: loss = {loss.item():.4f}")

    # Evaluation
    print("\n--- Evaluation ---")
    print(f"Sampling {args.eval_samples} trajectories for evaluation...")
    with torch.no_grad():
        trajectories = sampler.sample_trajectories(env, n=args.eval_samples)
        final_states = trajectories.last_states
        
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
                
        print(f"Generated valid numbers: {numbers[:20]}...")  # Show first 20
        if numbers:
            counts = np.bincount(numbers, minlength=101)
            valid_counts = counts[:101]  # Only 0-100
            print(f"Number of valid samples: {len(numbers)} / {args.eval_samples} ({100*len(numbers)/args.eval_samples:.1f}%)")
            print(f"Unique numbers generated: {np.count_nonzero(valid_counts)} / 101")
            print(f"Mean: {np.mean(numbers):.1f}, Std: {np.std(numbers):.1f}")
        else:
            print("No valid numbers generated.")

if __name__ == '__main__':
    main()
