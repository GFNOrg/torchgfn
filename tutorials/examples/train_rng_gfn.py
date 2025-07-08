
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

from gfn.env import Env
from gfn.states import DiscreteStates
from gfn.modules import GFNModule
from gfn.gflownet import GFlowNet, SubTrajectoryBalance
from gfn.samplers import Sampler



# 1. Environment Definition
class RNGEnv(Env):
    def __init__(self, tokenizer, prompt, max_length=5, device='cuda'):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
        self.max_length = max_length
        self.device = device
        
        s0 = DiscreteStates(self.prompt_tokens.squeeze(0))
        sf = DiscreteStates(torch.tensor([self.tokenizer.eos_token_id], device=self.device))
        
        super().__init__(s0=s0, sf=sf, n_actions=tokenizer.vocab_size)

    def get_actions_masks(self, states: DiscreteStates) -> torch.Tensor:
        masks = torch.ones(len(states), self.n_actions, dtype=torch.bool, device=self.device)
        for i in range(len(states)):
            if states.masks[i].sum() >= self.max_length + self.prompt_tokens.shape[1]:
                masks[i, :] = False
                masks[i, self.tokenizer.eos_token_id] = True
        return masks

    def forward_step(self, states: DiscreteStates, actions: torch.Tensor) -> DiscreteStates:
        new_states_list = []
        for i in range(len(states)):
            state_tensor = states.tensor[i][states.masks[i]]
            action = actions[i]
            new_state_tensor = torch.cat([state_tensor, action.unsqueeze(0)])
            new_states_list.append(new_state_tensor)
        return DiscreteStates(new_states_list)

    def log_reward(self, states: DiscreteStates) -> torch.Tensor:
        rewards = torch.full((len(states),), -20.0, device=self.device)
        
        prompt_len = self.prompt_tokens.shape[1]
        
        sequences_to_decode = []
        valid_indices = []
        for i in range(len(states)):
            if states.masks[i].sum() > prompt_len:
                seq = states.tensor[i][states.masks[i]]
                # Only reward terminal states
                if seq[-1] == self.tokenizer.eos_token_id:
                    sequences_to_decode.append(seq[prompt_len:-1])
                    valid_indices.append(i)

        if not sequences_to_decode:
            return rewards

        decoded_texts = self.tokenizer.batch_decode(sequences_to_decode, skip_special_tokens=True)
        
        for i, decoded_text in enumerate(decoded_texts):
            original_index = valid_indices[i]
            try:
                number = int(decoded_text.strip())
                if 0 <= number <= 100:
                    rewards[original_index] = torch.log(torch.tensor(1.0 / 101.0, device=self.device))
            except (ValueError, IndexError):
                pass
            
        return rewards

    def reset(self, batch_size: int = 1) -> DiscreteStates:
        return DiscreteStates([self.s0.tensor for _ in range(batch_size)])

# 2. GFN Module (wrapping the LLM)
class LLMGFNModule(GFNModule):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.log_z = nn.Parameter(torch.tensor(0.0))

    def forward(self, states: States):
        input_ids = states.tensor
        attention_mask = states.masks
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_logits = outputs.logits[torch.arange(len(outputs.logits)), sequence_lengths - 1, :]
        
        return last_token_logits, self.log_z

# 3. Training Setup
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

    prompt = "The following is a random integer drawn uniformly between 0 and 100: "
    env = RNGEnv(tokenizer, prompt, device=device, max_length=5)
    gfn_module = LLMGFNModule(model, tokenizer)
    gflownet = GFlowNet(gfn_module, loss=SubTrajectoryBalance())
    sampler = Sampler(gfn_module, env)

    optimizer = torch.optim.Adam(gfn_module.parameters(), lr=1e-4)

    print("Starting training...")
    # 4. Training Loop
    for i in range(501): # A short training loop for demonstration
        trajectories = sampler.sample_trajectories(n_trajectories=8)
        loss = gflownet.loss(trajectories)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss.item()}")

    # 5. Evaluation
    print("\n--- Evaluation ---")
    with torch.no_grad():
        trajectories = sampler.sample_trajectories(n_trajectories=100)
        final_states = trajectories.last_states
        
        numbers = []
        for i in range(len(final_states)):
            state_tensor = final_states.tensor[i][final_states.masks[i]]
            generated_tokens = state_tensor[len(env.prompt_tokens[0]):-1] # Exclude prompt and EOS
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
