"""GFlowNet environment for chip placement."""
import torch
from typing import Optional

from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates
from gfn.actions import Actions

# Assuming chip_design is in the python path
from .helpers.chip_design import utils as placement_util
from .helpers.chip_design.utils import cost_info_function


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
        netlist_file: str,
        init_placement: str,
        std_cell_placer_mode: str = "fd",
        wirelength_weight: float = 1.0,
        density_weight: float = 1.0,
        congestion_weight: float = 0.5,
        device: str = "cpu",
        check_action_validity: bool = True,
    ):
        self.device = torch.device(device)
        self.plc = placement_util.create_placement_cost(
            netlist_file=netlist_file, init_placement=init_placement
        )
        self.std_cell_placer_mode = std_cell_placer_mode

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

        s0 = torch.full((self.n_macros,), -1, dtype=torch.long, device=self.device)
        sf = torch.full((self.n_macros,), -2, dtype=torch.long, device=self.device)
        n_actions = self.n_grid_cells + 1

        super().__init__(
            n_actions=n_actions,
            s0=s0,
            state_shape=(self.n_macros,),
            sf=sf,
            check_action_validity=check_action_validity,
        )
        self.States: "type[ChipDesignStates]" = self.make_states_class()

    def make_states_class(self) -> "type[ChipDesignStates]":
        """Creates the ChipDesignStates class."""

        class ChipDesignStates(DiscreteStates):
            """A class to represent the states of the chip design environment."""

            def __init__(
                self,
                tensor: torch.Tensor,
                current_node_idx: Optional[torch.Tensor] = None,
                **kwargs,
            ):
                super().__init__(tensor=tensor, **kwargs)
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
                return self.__class__(
                    self.tensor.clone(),
                    self.current_node_idx.clone(),
                    forward_masks=self.forward_masks.clone(),
                    backward_masks=self.backward_masks.clone(),
                )

            def __getitem__(self, index) -> "ChipDesignStates":
                """Gets a subset of the states."""
                return self.__class__(
                    self.tensor[index],
                    self.current_node_idx[index],
                    forward_masks=self.forward_masks[index],
                    backward_masks=self.backward_masks[index],
                )

        return ChipDesignStates

    def _apply_state_to_plc(self, state_tensor: torch.Tensor):
        """Applies a single state tensor to the plc object."""
        self.plc.unplace_all_nodes()
        for i in range(self.n_macros):
            loc = state_tensor[i].item()
            if loc != -1:
                node_index = self._hard_macro_indices[i]
                self.plc.place_node(node_index, loc)

    def update_masks(self, states: "ChipDesignStates") -> None:
        """Updates the forward and backward masks of the states."""
        states.forward_masks.zero_()
        states.backward_masks.zero_()

        for i in range(len(states)):
            state_tensor = states.tensor[i]
            current_node_idx = states.current_node_idx[i].item()

            if current_node_idx >= self.n_macros:  # All macros placed
                states.forward_masks[i, -1] = True  # Only exit is possible
            else:
                # Apply partial placement to plc to get mask for next node
                self._apply_state_to_plc(state_tensor)
                node_to_place = self._hard_macro_indices[current_node_idx]
                mask = self.plc.get_node_mask(node_to_place)
                mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
                states.forward_masks[i, : self.n_grid_cells] = mask
                states.forward_masks[i, -1] = False  # No exit

            if current_node_idx > 0:
                last_placed_loc = state_tensor[current_node_idx - 1].item()
                if last_placed_loc != -1:
                    states.backward_masks[i, last_placed_loc] = True

    def step(self, states: "ChipDesignStates", actions: Actions) -> "ChipDesignStates":
        """Performs a forward step in the environment."""
        new_tensor = states.tensor.clone()

        non_exit_mask = ~actions.is_exit
        if torch.any(non_exit_mask):
            rows = torch.arange(len(states), device=self.device)[non_exit_mask]
            cols = states.current_node_idx[non_exit_mask]
            new_tensor[rows, cols] = actions.tensor[non_exit_mask].squeeze(-1)

        if torch.any(actions.is_exit):
            new_tensor[actions.is_exit] = self.sf

        new_current_node_idx = states.current_node_idx.clone()
        new_current_node_idx[non_exit_mask] += 1

        return self.States(tensor=new_tensor, current_node_idx=new_current_node_idx)

    def backward_step(
        self, states: "ChipDesignStates", actions: Actions
    ) -> "ChipDesignStates":
        """Performs a backward step in the environment."""
        new_tensor = states.tensor.clone()
        rows = torch.arange(len(states), device=self.device)
        cols = states.current_node_idx - 1
        new_tensor[rows, cols] = -1

        new_current_node_idx = states.current_node_idx - 1
        return self.States(tensor=new_tensor, current_node_idx=new_current_node_idx)

    def analytical_placer(self):
        """Places standard cells using an analytical placer."""
        if self.std_cell_placer_mode == "fd":
            placement_util.fd_placement_schedule(self.plc)
        else:
            raise ValueError(
                f"{self.std_cell_placer_mode} is not a supported std_cell_placer_mode."
            )

    def log_reward(self, final_states: "ChipDesignStates") -> torch.Tensor:
        """Computes the log reward of the final states."""
        rewards = torch.zeros(len(final_states), device=self.device)
        for i in range(len(final_states)):
            state_tensor = final_states.tensor[i]
            self._apply_state_to_plc(state_tensor)

            self.analytical_placer()

            cost, _ = cost_info_function(
                plc=self.plc,
                done=True,
                wirelength_weight=self.wirelength_weight,
                density_weight=self.density_weight,
                congestion_weight=self.congestion_weight,
            )
            rewards[i] = -cost
        return rewards
