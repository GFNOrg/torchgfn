"""GFlowNet environment for chip placement."""

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple, cast

import torch

from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.gym.helpers.chip_design import (
    SAMPLE_INIT_PLACEMENT,
    SAMPLE_NETLIST_FILE,
)
from gfn.gym.helpers.chip_design import utils as placement_util
from gfn.gym.helpers.chip_design.utils import cost_info_function, optimize_orientations
from gfn.states import DiscreteStates


@dataclass
class CostStats:
    """Pre-computed cost statistics for a netlist."""

    mean: float
    std: float
    min: float
    range: float  # max - min

    @staticmethod
    def precompute(
        netlists: List[Tuple[str, str]],
        n_samples: int = 100,
        wirelength_weight: float = 1.0,
        density_weight: float = 1.0,
        congestion_weight: float = 0.5,
        singularity_image: Optional[str] = None,
    ) -> Dict[str, "CostStats"]:
        """Pre-compute cost statistics for multiple netlists.

        Args:
            netlists: List of ``(netlist_file, init_placement)`` tuples.
            n_samples: Number of random placements per netlist.
            wirelength_weight: Wirelength weight for cost computation.
            density_weight: Density weight for cost computation.
            congestion_weight: Congestion weight for cost computation.
            singularity_image: Optional singularity image path.

        Returns:
            Dict mapping netlist file paths to their ``CostStats``.
        """
        stats = {}
        for netlist_file, init_placement in netlists:
            env = ChipDesign(
                netlist_file=netlist_file,
                init_placement=init_placement,
                wirelength_weight=wirelength_weight,
                density_weight=density_weight,
                congestion_weight=congestion_weight,
                singularity_image=singularity_image,
                cd_finetune=False,
                reward_norm=None,
            )
            env._estimate_cost_stats(n_samples)
            stats[netlist_file] = CostStats(
                mean=env._cost_mean,
                std=env._cost_std,
                min=env._cost_min,
                range=env._cost_max,
            )
            env.close()
        return stats


class ChipDesignStates(DiscreteStates):
    """A class to represent the states of the chip design environment."""

    state_shape: ClassVar[tuple[int, ...]]
    s0: ClassVar[torch.Tensor]
    sf: ClassVar[torch.Tensor]
    n_actions: ClassVar[int]

    def __init__(
        self,
        tensor: torch.Tensor,
        current_node_idx: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        debug: bool = False,
    ):
        super().__init__(
            tensor=tensor, conditions=conditions, device=device, debug=debug
        )
        if current_node_idx is None:
            is_unplaced = tensor == -1
            is_unplaced_padded = torch.cat(
                [
                    is_unplaced,
                    torch.ones_like(is_unplaced[..., :1]),
                ],
                dim=-1,
            )
            current_node_idx = is_unplaced_padded.long().argmax(dim=-1)

        self.current_node_idx = current_node_idx

    def clone(self) -> "ChipDesignStates":
        """Creates a copy of the states."""
        cloned = self.__class__(
            self.tensor.clone(),
            current_node_idx=self.current_node_idx.clone(),
            conditions=(
                self.conditions.clone() if self.conditions is not None else None
            ),
            device=self.tensor.device,
            debug=self.debug,
        )
        if self._forward_masks_cache is not None:
            cloned.forward_masks = self._forward_masks_cache.clone()
        if self._backward_masks_cache is not None:
            cloned.backward_masks = self._backward_masks_cache.clone()
        return cloned

    def __getitem__(self, index) -> "ChipDesignStates":
        """Gets a subset of the states."""
        subset = self.__class__(
            self.tensor[index],
            current_node_idx=self.current_node_idx[index],
        )
        if self._forward_masks_cache is not None:
            subset.forward_masks = self._forward_masks_cache[index]
        if self._backward_masks_cache is not None:
            subset.backward_masks = self._backward_masks_cache[index]
        return subset

    def __setitem__(self, index, value: "ChipDesignStates") -> None:
        """Sets a subset of the states."""
        super().__setitem__(index, value)
        self.current_node_idx[index] = value.current_node_idx

    def extend(self, other: "ChipDesignStates") -> None:
        """Extends the states with another states."""
        super().extend(other)
        self.current_node_idx = torch.cat(
            (self.current_node_idx, other.current_node_idx),
            dim=len(self.batch_shape) - 1,
        )

    @classmethod
    def stack(cls, states: Sequence["ChipDesignStates"]) -> "ChipDesignStates":
        """Stacks the states with another states."""
        stacked = super().stack(states)
        stacked.current_node_idx = torch.stack(
            [s.current_node_idx for s in states],
            dim=0,
        )
        return cast(ChipDesignStates, stacked)


class ChipDesign(DiscreteEnv):
    """
    GFlowNet environment for chip placement.

    The state is a vector of length `n_macros`, where `state[i]` is the grid
    cell location of the i-th macro to be placed. Unplaced macros have a
    location of -1.

    Actions are integers from `0` to `n_grid_cells - 1`, representing the
    grid cell to place the current macro on. Action `n_grid_cells` is the
    exit action.
    """

    def __init__(
        self,
        netlist_file: str = SAMPLE_NETLIST_FILE,
        init_placement: str = SAMPLE_INIT_PLACEMENT,
        std_cell_placer_mode: str = "fd",
        wirelength_weight: float = 1.0,
        density_weight: float = 1.0,
        congestion_weight: float = 0.5,
        device: str | None = None,
        debug: bool = False,
        singularity_image: Optional[str] = None,
        cd_finetune: bool = True,
        reward_norm: Optional[str] = None,
        reward_temper: float = 1.0,
        reward_norm_samples: int = 100,
        cost_stats: Optional[CostStats] = None,
    ):
        """Initializes the chip design environment.

        Args:
            reward_norm: Reward normalization mode. ``None`` for raw
                ``-cost``, ``"zscore"`` for z-score normalization, or
                ``"minmax"`` for min-max normalization. Statistics are
                estimated from ``reward_norm_samples`` random placements
                unless ``cost_stats`` is provided.
            reward_temper: Temperature scaling applied after normalization.
                Higher values sharpen the reward distribution.
            reward_norm_samples: Number of random placements used to
                estimate normalization statistics.
            cost_stats: Pre-computed cost statistics. Use
                ``CostStats.precompute()`` to compute these once for
                multiple netlists, then pass them here to avoid
                recomputation.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.plc = placement_util.create_placement_cost(
            netlist_file=netlist_file,
            init_placement=init_placement,
            singularity_image=singularity_image,
        )
        self.std_cell_placer_mode = std_cell_placer_mode
        self.cd_finetune = cd_finetune
        self.reward_norm = reward_norm
        self.reward_temper = reward_temper

        self.wirelength_weight = wirelength_weight
        self.density_weight = density_weight
        self.congestion_weight = congestion_weight

        self._grid_cols, self._grid_rows = self.plc.get_grid_num_columns_rows()
        self.n_grid_cells = self._grid_cols * self._grid_rows

        self._sorted_node_indices = placement_util.get_ordered_node_indices(
            mode="descending_size_macro_first", plc=self.plc
        )
        self._hard_macro_indices = [
            m for m in self._sorted_node_indices if not self.plc.is_node_soft_macro(m)
        ]
        self.n_macros = len(self._hard_macro_indices)

        # Cost statistics for reward normalization.
        if cost_stats is not None:
            self._cost_mean = cost_stats.mean
            self._cost_std = cost_stats.std
            self._cost_min = cost_stats.min
            self._cost_max = cost_stats.range
        elif reward_norm is not None:
            self._cost_mean = 0.0
            self._cost_std = 1.0
            self._cost_min = 0.0
            self._cost_max = 1.0
            self._estimate_cost_stats(reward_norm_samples)
        else:
            self._cost_mean = 0.0
            self._cost_std = 1.0
            self._cost_min = 0.0
            self._cost_max = 1.0

        s0 = torch.full((self.n_macros,), -1, dtype=torch.long, device=device)
        sf = torch.full((self.n_macros,), -2, dtype=torch.long, device=device)
        n_actions = self.n_grid_cells + 1

        super().__init__(
            n_actions=n_actions,
            s0=s0,
            state_shape=(self.n_macros,),
            sf=sf,
            debug=debug,
        )
        self.States: type[ChipDesignStates] = self.make_states_class()

    def make_states_class(self) -> type[ChipDesignStates]:
        """Creates the ChipDesignStates class."""
        env = self

        class BaseChipDesignStates(ChipDesignStates):
            state_shape = env.state_shape
            s0 = env.s0
            sf = env.sf
            n_actions = env.n_actions

        return BaseChipDesignStates

    def close(self) -> None:
        """Closes the PlacementCost subprocess to free resources."""
        if hasattr(self, "plc") and self.plc is not None:
            self.plc.close()

    def __del__(self) -> None:
        self.close()

    def _apply_state_to_plc(self, state_tensor: torch.Tensor):
        """Applies a single state tensor to the plc object."""
        assert state_tensor.shape == (self.n_macros,)

        self.plc.unplace_all_nodes()
        for i in range(self.n_macros):
            loc = state_tensor[i].item()
            if loc >= 0:
                node_index = self._hard_macro_indices[i]
                self.plc.place_node(node_index, loc)

    def update_masks(self, states: ChipDesignStates) -> None:
        """Updates the forward and backward masks of the states."""
        batch_shape = states.batch_shape
        n = states.tensor.shape[:-1].numel()  # flat batch size

        flat_tensor = states.tensor.reshape(n, -1)
        flat_node_idx = states.current_node_idx.reshape(n)
        flat_is_sink = states.is_sink_state.reshape(n)

        fwd = torch.zeros(n, self.n_actions, dtype=torch.bool, device=self.device)
        bwd = torch.zeros(n, self.n_actions - 1, dtype=torch.bool, device=self.device)

        for i in range(n):
            if flat_is_sink[i]:
                continue

            state_tensor = flat_tensor[i]
            current_node_idx = flat_node_idx[i].item()

            if current_node_idx >= self.n_macros:  # All macros placed
                fwd[i, -1] = True  # Only exit is possible
            else:
                self._apply_state_to_plc(state_tensor)
                node_to_place = self._hard_macro_indices[int(current_node_idx)]
                mask = self.plc.get_node_mask(node_to_place)
                mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
                fwd[i, : self.n_grid_cells] = mask

            if current_node_idx > 0:
                last_placed_loc = state_tensor[int(current_node_idx - 1)].item()
                assert last_placed_loc >= 0, "Last placed location should be >= 0"
                bwd[i, int(last_placed_loc)] = True

        states.forward_masks = fwd.reshape(*batch_shape, self.n_actions)
        states.backward_masks = bwd.reshape(*batch_shape, self.n_actions - 1)

    def reset(
        self,
        batch_shape: int | Tuple[int, ...],
        random: bool = False,
        sink: bool = False,
        seed: Optional[int] = None,
        conditions: Optional[torch.Tensor] = None,
    ) -> ChipDesignStates:
        """Resets the environment and computes initial masks."""
        states = super().reset(batch_shape, random, sink, seed, conditions=conditions)
        states = cast(ChipDesignStates, states)
        self.update_masks(states)
        return states

    def _step(self, states: DiscreteStates, actions: Actions) -> ChipDesignStates:
        """Wraps parent _step and updates masks."""
        new_states = super()._step(states, actions)
        new_states = cast(ChipDesignStates, new_states)
        self.update_masks(new_states)
        return new_states

    def _backward_step(
        self, states: DiscreteStates, actions: Actions
    ) -> ChipDesignStates:
        """Wraps parent _backward_step and updates masks."""
        new_states = super()._backward_step(states, actions)
        new_states = cast(ChipDesignStates, new_states)
        self.update_masks(new_states)
        return new_states

    def step(self, states: ChipDesignStates, actions: Actions) -> ChipDesignStates:
        """Performs a forward step in the environment."""
        batch_shape = states.batch_shape
        n = states.tensor.shape[:-1].numel()

        if n == 0:
            return self.States(
                tensor=states.tensor.clone(),
                current_node_idx=states.current_node_idx.clone(),
            )

        flat_tensor = states.tensor.reshape(n, self.n_macros).clone()
        flat_node_idx = states.current_node_idx.reshape(n).clone()
        flat_actions = actions.tensor.reshape(n, actions.tensor.shape[-1])
        flat_is_exit = actions.is_exit.reshape(n)

        non_exit = ~flat_is_exit
        if torch.any(non_exit):
            rows = torch.arange(n, device=self.device)[non_exit]
            cols = flat_node_idx[non_exit]
            flat_tensor[rows, cols] = flat_actions[non_exit].squeeze(-1)

        if torch.any(flat_is_exit):
            flat_tensor[flat_is_exit] = self.sf

        flat_node_idx[non_exit] += 1

        return self.States(
            tensor=flat_tensor.reshape(*batch_shape, self.n_macros),
            current_node_idx=flat_node_idx.reshape(*batch_shape),
        )

    def backward_step(
        self, states: ChipDesignStates, actions: Actions
    ) -> ChipDesignStates:
        """Performs a backward step in the environment."""
        batch_shape = states.batch_shape
        n = states.tensor.shape[:-1].numel()

        flat_tensor = states.tensor.reshape(n, self.n_macros).clone()
        flat_node_idx = states.current_node_idx.reshape(n)

        rows = torch.arange(n, device=self.device)
        cols = flat_node_idx - 1
        flat_tensor[rows, cols] = -1

        new_node_idx = flat_node_idx - 1
        return self.States(
            tensor=flat_tensor.reshape(*batch_shape, self.n_macros),
            current_node_idx=new_node_idx.reshape(*batch_shape),
        )

    def _estimate_cost_stats(self, n_samples: int) -> None:
        """Estimates cost distribution from random placements."""
        import random as pyrandom

        costs = []
        for _ in range(n_samples):
            self.plc.unplace_all_nodes()
            grid_cells = list(range(self.n_grid_cells))
            for macro_idx in self._hard_macro_indices:
                pyrandom.shuffle(grid_cells)
                for cell in grid_cells:
                    if self.plc.can_place_node(macro_idx, cell):
                        self.plc.place_node(macro_idx, cell)
                        break

            self.analytical_placer()
            cost, _ = cost_info_function(
                plc=self.plc,
                done=True,
                wirelength_weight=self.wirelength_weight,
                density_weight=self.density_weight,
                congestion_weight=self.congestion_weight,
            )
            costs.append(cost)

        self.plc.unplace_all_nodes()

        costs_t = torch.tensor(costs, dtype=torch.float64)
        self._cost_mean = costs_t.mean().item()
        self._cost_std = max(costs_t.std().item(), 1e-8)
        self._cost_min = costs_t.min().item()
        self._cost_max = max(costs_t.max().item() - costs_t.min().item(), 1e-8)

    def analytical_placer(self):
        """Places standard cells using an analytical placer."""
        if self.std_cell_placer_mode == "fd":
            placement_util.fd_placement_schedule(self.plc)
        else:
            raise ValueError(
                f"{self.std_cell_placer_mode} is not a supported std_cell_placer_mode."
            )

    def _normalize_cost(self, cost: float) -> float:
        """Applies reward normalization to a raw cost value."""
        if self.reward_norm == "zscore":
            return (cost - self._cost_mean) / self._cost_std
        elif self.reward_norm == "minmax":
            return (cost - self._cost_min) / self._cost_max
        return cost

    def log_reward(self, final_states: ChipDesignStates) -> torch.Tensor:
        """Computes the log reward of the final states."""
        rewards = torch.zeros(len(final_states), device=self.device)
        for i in range(len(final_states)):
            state_tensor = final_states.tensor[i]
            self._apply_state_to_plc(state_tensor)

            self.analytical_placer()

            if self.cd_finetune:
                optimize_orientations(
                    self.plc,
                    wirelength_weight=self.wirelength_weight,
                    density_weight=self.density_weight,
                    congestion_weight=self.congestion_weight,
                )

            cost, _ = cost_info_function(
                plc=self.plc,
                done=True,
                wirelength_weight=self.wirelength_weight,
                density_weight=self.density_weight,
                congestion_weight=self.congestion_weight,
            )
            rewards[i] = -self._normalize_cost(cost) * self.reward_temper
        return rewards
