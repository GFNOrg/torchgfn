from __future__ import annotations

from typing import Any, Optional

import torch
from torch.distributions import Categorical, Distribution

from gfn.chunking.policy import ChunkedPolicy
from gfn.env import DiscreteEnv
from gfn.samplers import AdapterContext, EstimatorAdapter
from gfn.states import DiscreteStates


class ChunkedAdapter(EstimatorAdapter):
    """EstimatorAdapter that produces macro-level distributions using ChunkedPolicy.

    Forward-only in this PR. TODO(backward): support backward chunking by switching
    stepping and termination criteria to the backward direction.
    """

    def __init__(self, env: DiscreteEnv, policy: ChunkedPolicy, library: Any) -> None:
        self.env = env
        self.policy = policy
        self.library = library
        self._is_backward = False  # TODO(backward): allow backward chunking

    @property
    def is_backward(self) -> bool:
        return self._is_backward

    def init_context(
        self,
        batch_size: int,
        device: torch.device,
        conditioning: Optional[torch.Tensor] = None,
    ) -> AdapterContext:
        ctx = AdapterContext(
            batch_size=batch_size, device=device, conditioning=conditioning
        )
        ctx.extras["macro_log_probs"] = []  # List[(N,)]
        return ctx

    def _strict_macro_mask(self, states_active: DiscreteStates) -> torch.Tensor:
        """Strict mask by simulating each macro sequentially on each active state.

        Invalidates a macro if any sub-action is invalid or if sink is reached before
        the sequence completes. Guarantees EXIT macro is valid if no macro is valid.
        """
        B = states_active.batch_shape[0]
        N = self.library.n_actions
        device = states_active.device
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        for b in range(B):
            s_curr = states_active[b : b + 1]
            for j, seq in enumerate(self.library.id_to_sequence):
                ok = True
                s = s_curr
                for k, a in enumerate(seq):
                    a_tensor = self.env.actions_from_tensor(
                        torch.tensor([[a]], device=device)
                    )
                    if not self.env.is_action_valid(s, a_tensor):
                        ok = False
                        break
                    s_next = self.env._step(s, a_tensor)
                    if s_next.is_sink_state.item() and k != len(seq) - 1:
                        ok = False
                        break
                    s = s_next
                mask[b, j] = ok

        # Ensure EXIT macro is available when none is valid
        try:
            exit_id = self.library.id_to_sequence.index([self.env.exit_action.item()])
        except ValueError:
            exit_id = N - 1
        no_valid = ~mask.any(dim=1)
        if no_valid.any():
            mask[no_valid] = False
            mask[no_valid, exit_id] = True
        return mask

    def compute(
        self,
        states_active: DiscreteStates,
        ctx: Any,
        step_mask: torch.Tensor,
        **policy_kwargs: Any,
    ) -> tuple[Distribution, Any]:
        logits = self.policy.forward_logits(states_active)  # (B_active, N)
        macro_mask = self._strict_macro_mask(states_active)
        masked_logits = torch.where(
            macro_mask, logits, torch.full_like(logits, -float("inf"))
        )
        dist = Categorical(logits=masked_logits)
        ctx.current_estimator_output = None
        return dist, ctx

    def record_step(
        self,
        ctx: Any,
        step_mask: torch.Tensor,
        sampled_actions: torch.Tensor,
        dist: Distribution,
        save_logprobs: bool,
        save_estimator_outputs: bool,
    ) -> None:
        if save_logprobs:
            lp_masked = dist.log_prob(sampled_actions)
            step_lp = torch.full((ctx.batch_size,), 0.0, device=ctx.device)
            step_lp[step_mask] = lp_masked
            ctx.extras["macro_log_probs"].append(step_lp)
        # No estimator outputs for macros by default
        return

    def finalize(self, ctx: Any) -> dict[str, Optional[torch.Tensor]]:
        out: dict[str, Optional[torch.Tensor]] = {
            "log_probs": None,
            "estimator_outputs": None,
        }
        macro_log_probs = ctx.extras.get("macro_log_probs", [])
        if macro_log_probs:
            out["macro_log_probs"] = torch.stack(macro_log_probs, dim=0)
        else:
            out["macro_log_probs"] = None
        return out

    def get_current_estimator_output(self, ctx: Any):
        return None
