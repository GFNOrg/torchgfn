
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
from gfn.gflownet import TBGFlowNet


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

        self.state_shape = self.total_length  # keep name for clarity before tuple wrap

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

        # We'll treat actions as scalars (shape = ()), so configure accordingly.
        super().__init__(
            s0=s0,
            sf=sf,
            n_actions=tokenizer.vocab_size,
            state_shape=(self.total_length,),
            action_shape=(),
            dummy_action=torch.tensor(-1, device=device),
            exit_action=torch.tensor(tokenizer.eos_token_id, device=device),
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

        return masks

    def update_masks(self, states: DiscreteStates) -> None:  # type: ignore[override]
        """Populate ``states.forward_masks`` and ``states.backward_masks`` in-place."""

        # Forward masks (valid next tokens).
        states.forward_masks = self._forward_action_masks(states)

        # Backward masks exclude the exit action (EOS).
        backward_masks = torch.ones((*states.batch_shape, self.n_actions), dtype=torch.bool, device=self.device)
        backward_masks[..., self.tokenizer.eos_token_id] = False
        states.backward_masks = backward_masks

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
            new_states_tensor[idx, first_pad] = actions.tensor[idx]
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
            assert new_states_tensor[idx, last_idx] == actions.tensor[idx]
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

        masks = states.backward_masks if self.is_backward else states.forward_masks
        logits = module_output.clone()

        # Apply masks by setting invalid logits to −inf.
        logits[~masks] = -float("inf")

        if not self.is_backward and sf_bias != 0.0:
            logits[:, -1] -= sf_bias  # usually bias exit action

        if temperature != 1.0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)

        if epsilon != 0.0:
            uniform = torch.where(
                masks.sum(dim=-1, keepdim=True) == 0,
                torch.zeros_like(masks, dtype=probs.dtype),
                masks.float() / masks.sum(dim=-1, keepdim=True),
            )
            probs = (1 - epsilon) * probs + epsilon * uniform

        return torch.distributions.Categorical(probs=probs)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

    prompt = "The following is a random integer drawn uniformly between 0 and 100: "
    env = RNGEnv(tokenizer, prompt, device=device, max_length=5)

    state_dim = env.state_shape[0]
    pf_module = LLMGFNModule(model, tokenizer, state_dim, is_backward=False)
    pb_module = LLMGFNModule(model, tokenizer, state_dim, is_backward=True)

    gflownet = TBGFlowNet(pf_module, pb_module)

    optimizer = torch.optim.Adam(gflownet.parameters(), lr=1e-4)

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    for step in range(501):  # quick demo run
        trajectories = gflownet.sample_trajectories(env, n=8, save_logprobs=False)
        loss = gflownet.loss(env, trajectories)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    # 5. Evaluation
    print("\n--- Evaluation ---")
    with torch.no_grad():
        trajectories = gflownet.sample_trajectories(env, n=100)
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
                numbers.append(int(decoded_text.strip()))
            except (ValueError, IndexError):
                pass
                
        print(f"Generated numbers: {numbers}")
        if numbers:
            counts = np.bincount(numbers, minlength=101)
            print(f"Counts of numbers 0-100: {counts}")
            print(f"Number of valid samples: {len(numbers)}")
        else:
            print("No valid numbers generated.")

if __name__ == '__main__':
    main()
