# coding=utf-8
# Copyright 2021 The Circuit Training Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# flake8: noqa
# pyright: reportCallIssue=false, reportArgumentType=false

"""A collection of non-prod utility functions for placement.

All the dependencies in this files should be non-prod.
"""

import datetime
import logging
import re
import textwrap
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from . import plc_client

COST_COMPONENTS = ["wirelength", "congestion", "density"]


def cost_info_function(
    plc: plc_client.PlacementCost,
    done: bool,
    infeasible_state: bool = False,
    wirelength_weight: float = 1.0,
    density_weight: float = 1.0,
    congestion_weight: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Returns the RL cost and info."""
    del infeasible_state
    proxy_cost = 0.0
    info = {cost: -1.0 for cost in COST_COMPONENTS}

    if not done:
        return proxy_cost, info

    if wirelength_weight > 0.0:
        info["wirelength"] = plc.get_cost()
        proxy_cost += wirelength_weight * info["wirelength"]

    if congestion_weight > 0.0:
        info["congestion"] = plc.get_congestion_cost()
        proxy_cost += congestion_weight * info["congestion"]

    if density_weight > 0.0:
        info["density"] = plc.get_density_cost()
        proxy_cost += density_weight * info["density"]

    return proxy_cost, info


def nodes_of_types(plc: plc_client.PlacementCost, type_list: List[str]) -> Iterator[int]:
    """Yields the index of a node of certain types."""
    i = 0
    while True:
        node_type = plc.get_node_type(i)
        if not node_type:
            break
        if node_type in type_list:
            yield i
        i += 1


def get_node_xy_coordinates(
    plc: plc_client.PlacementCost,
) -> Dict[int, Tuple[float, float]]:
    """Returns all node x,y coordinates (canvas) in a dict."""
    node_coords = dict()
    for node_index in nodes_of_types(plc, ["MACRO", "STDCELL", "PORT"]):
        if plc.is_node_placed(node_index):
            node_coords[node_index] = plc.get_node_location(node_index)
    return node_coords


def get_macro_orientations(plc: plc_client.PlacementCost) -> Dict[int, int]:
    """Returns all macros' orientations in a dict."""
    macro_orientations = dict()
    for node_index in nodes_of_types(plc, ["MACRO"]):
        macro_orientations[node_index] = plc.get_macro_orientation(node_index)
    return macro_orientations


def restore_node_xy_coordinates(
    plc: plc_client.PlacementCost, node_coords: Dict[int, Tuple[float, float]]
) -> None:
    for node_index, coords in node_coords.items():
        if not plc.is_node_fixed(node_index):
            plc.update_node_coords(node_index, coords[0], coords[1])


def restore_macro_orientations(
    plc: plc_client.PlacementCost, macro_orientations: Dict[int, int]
) -> None:
    for node_index, orientation in macro_orientations.items():
        plc.update_macro_orientation(node_index, orientation)


def extract_attribute_from_comments(
    attribute: str, filenames: List[str]
) -> Optional[str]:
    """Parses the files' comments section, tries to extract the attribute."""
    for filename in filenames:
        if filename:
            f = filename.split(",")[0]
            if f:
                with open(f, "r") as infile:
                    for line in infile:
                        if line.startswith("#"):
                            match = re.search(rf"{attribute} : ([-\w]+)", line)
                            if match:
                                return match.group(1)
                        else:
                            break
    return None


def get_blockages_from_comments(
    filenames: Union[str, List[str]],
) -> Optional[List[List[float]]]:
    """Returns list of blockages if they exist in the file's comments section."""
    for filename in filenames:
        if not filename:
            continue
        blockages = []
        try:
            with open(filename, "r") as infile:
                for line in infile:
                    if line.startswith("# Blockage : "):
                        blockages.append([float(x) for x in line.split()[3:8]])
                    elif not line.startswith("#"):
                        break
        except OSError:
            logging.error("could not read file %s.", filename)
        if blockages:
            return blockages


def extract_sizes_from_comments(
    filenames: List[str],
) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """Parses the file's comments section, tries to extract canvas/grid sizes."""
    canvas_width, canvas_height = None, None
    grid_cols, grid_rows = None, None
    for filename in filenames:
        if not filename:
            continue
        with open(filename, "r") as infile:
            for line in infile:
                if line.startswith("#"):
                    fp_re = re.search(
                        r"FP bbox: {([\d\.]+) ([\d\.]+)} {([\d\.]+) ([\d\.]+)}", line
                    )
                    if fp_re:
                        canvas_width = float(fp_re.group(3))
                        canvas_height = float(fp_re.group(4))
                        continue
                    plc_wh = re.search(r"Width : ([\d\.]+)  Height : ([\d\.]+)", line)
                    if plc_wh:
                        canvas_width = float(plc_wh.group(1))
                        canvas_height = float(plc_wh.group(2))
                        continue
                    plc_cr = re.search(r"Columns : ([\d]+)  Rows : ([\d]+)", line)
                    if plc_cr:
                        grid_cols = int(plc_cr.group(1))
                        grid_rows = int(plc_cr.group(2))
                else:
                    break
    return canvas_width, canvas_height, grid_cols, grid_rows


def fix_port_coordinates(plc: plc_client.PlacementCost) -> None:
    """Find all ports and fix their coordinates."""
    for node in nodes_of_types(plc, ["PORT"]):
        plc.fix_node_coord(node)


def create_placement_cost(
    netlist_file: str,
    init_placement: str,
    overlap_threshold: float = 4e-3,
    congestion_smooth_range: int = 5,
    macro_macro_x_spacing: float = 0.1,
    macro_macro_y_spacing: float = 0.1,
    boundary_check: bool = False,
    horizontal_routes_per_micron: float = 70.33,
    vertical_routes_per_micron: float = 74.51,
    macro_horizontal_routing_allocation: float = 51.79,
    macro_vertical_routing_allocation: float = 51.79,
    routes_per_congestion_grid: int = 1000,
    blockages: Optional[List[List[float]]] = None,
    fixed_macro_names_regex: Optional[List[str]] = None,
    legacy_congestion_grid: bool = False,
) -> plc_client.PlacementCost:
    """Creates a placement_cost object."""
    if not netlist_file:
        raise ValueError("netlist_file should be provided.")

    block_name = extract_attribute_from_comments("Block", [init_placement, netlist_file])
    if not block_name:
        logging.warning(
            "block_name is not set. Please add the block_name in:\n%s\nor in:\n%s",
            netlist_file,
            init_placement,
        )

    plc = plc_client.PlacementCost(
        netlist_file,
        macro_macro_x_spacing=macro_macro_x_spacing,
        macro_macro_y_spacing=macro_macro_y_spacing,
    )

    plc.make_soft_macros_square()

    blockages = blockages or get_blockages_from_comments([netlist_file, init_placement])
    if blockages:
        for blockage in blockages:
            plc.create_blockage(*blockage)

    canvas_width, canvas_height, grid_cols, grid_rows = extract_sizes_from_comments(
        [netlist_file, init_placement]
    )
    if canvas_width and canvas_height:
        plc.set_canvas_size(canvas_width, canvas_height)
    if grid_cols and grid_rows:
        plc.set_placement_grid(grid_cols, grid_rows)
        if legacy_congestion_grid:
            plc.set_congestion_grid(grid_cols, grid_rows)

    plc.set_project_name("circuit_training")
    plc.set_block_name(block_name or "unset_block")
    plc.set_routes_per_micron(horizontal_routes_per_micron, vertical_routes_per_micron)
    plc.set_macro_routing_allocation(
        macro_horizontal_routing_allocation, macro_vertical_routing_allocation
    )
    plc.set_congestion_smooth_range(congestion_smooth_range)

    if not legacy_congestion_grid:
        congestion_grid_size = (
            2.0
            * routes_per_congestion_grid
            / (horizontal_routes_per_micron + vertical_routes_per_micron)
        )
        canvas_width, canvas_height = plc.get_canvas_width_height()
        congestion_grid_cols = max(1, int(canvas_width / congestion_grid_size))
        congestion_grid_rows = max(1, int(canvas_height / congestion_grid_size))
        plc.set_congestion_grid(congestion_grid_cols, congestion_grid_rows)

    plc.set_overlap_threshold(overlap_threshold)
    plc.set_canvas_boundary_check(boundary_check)
    if init_placement:
        plc.restore_placement(init_placement)
        fix_port_coordinates(plc)

    if fixed_macro_names_regex:
        logging.info("Fixing macro locations using regex.")
        fix_macros_by_regex(plc, fixed_macro_names_regex)

    return plc


def get_node_type_counts(plc: plc_client.PlacementCost) -> Dict[str, int]:
    """Returns number of each type of nodes in the netlist."""
    counts = {
        "MACRO": 0,
        "STDCELL": 0,
        "PORT": 0,
        "MACRO_PIN": 0,
        "SOFT_MACRO": 0,
        "HARD_MACRO": 0,
        "SOFT_MACRO_PIN": 0,
        "HARD_MACRO_PIN": 0,
    }

    for node_index in nodes_of_types(plc, ["MACRO", "STDCELL", "PORT", "MACRO_PIN"]):
        node_type = plc.get_node_type(node_index)
        counts[node_type] += 1
        if node_type == "MACRO":
            if plc.is_node_soft_macro(node_index):
                counts["SOFT_MACRO"] += 1
            else:
                counts["HARD_MACRO"] += 1
        if node_type == "MACRO_PIN":
            ref_id = plc.get_ref_node_id(node_index)
            if plc.is_node_soft_macro(ref_id):
                counts["SOFT_MACRO_PIN"] += 1
            else:
                counts["HARD_MACRO_PIN"] += 1
    return counts


def make_blockage_text(plc: plc_client.PlacementCost) -> str:
    ret = ""
    for blockage in plc.get_blockages():
        ret += "Blockage : {}\n".format(" ".join([str(b) for b in blockage]))
    return ret


def save_placement(
    plc: plc_client.PlacementCost, filename: str, user_comments: str = ""
) -> None:
    """Saves the placement file with some information in the comments section."""
    cols, rows = plc.get_grid_num_columns_rows()
    congestion_cols, congestion_rows = plc.get_congestion_grid_num_columns_rows()
    width, height = plc.get_canvas_width_height()
    hor_routes, ver_routes = plc.get_routes_per_micron()
    hor_macro_alloc, ver_macro_alloc = plc.get_macro_routing_allocation()
    smooth = plc.get_congestion_smooth_range()
    info = textwrap.dedent(
        """\
    Placement file for Circuit Training
    Source input file(s) : {src_filename}
    This file : {filename}
    Date : {date}
    Columns : {cols}  Rows : {rows}
    Congestion Columns : {congestion_cols}  Congestion Rows : {congestion_rows}
    Width : {width:.3f}  Height : {height:.3f}
    Area : {area}
    Wirelength : {wl:.3f}
    Wirelength cost : {wlc:.4f}
    Congestion cost : {cong:.4f}
    Density cost : {density:.4f}
    Project : {project}
    Block : {block_name}
    Routes per micron, hor : {hor_routes:.3f}  ver : {ver_routes:.3f}
    Routes used by macros, hor : {hor_macro_alloc:.3f}  ver : {ver_macro_alloc:.3f}
    Smoothing factor : {smooth}
    Overlap threshold : {overlap_threshold}
  """.format(
            src_filename=plc.get_source_filename(),
            filename=filename,
            date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cols=cols,
            rows=rows,
            width=width,
            congestion_cols=congestion_cols,
            congestion_rows=congestion_rows,
            height=height,
            area=plc.get_area(),
            wl=plc.get_wirelength(),
            wlc=plc.get_cost(),
            cong=plc.get_congestion_cost(),
            density=plc.get_density_cost(),
            project=plc.get_project_name(),
            block_name=plc.get_block_name(),
            hor_routes=hor_routes,
            ver_routes=ver_routes,
            hor_macro_alloc=hor_macro_alloc,
            ver_macro_alloc=ver_macro_alloc,
            smooth=smooth,
            overlap_threshold=plc.get_overlap_threshold(),
        )
    )

    info += "\n" + make_blockage_text(plc) + "\n"
    info += "\nCounts of node types:\n"
    node_type_counts = get_node_type_counts(plc)
    for node_type in sorted(node_type_counts):
        info += "{:<15} : {:>9}\n".format(node_type + "s", node_type_counts[node_type])
    if user_comments:
        info += "\nUser comments:\n" + user_comments + "\n"
    info += "\nnode_index x y orientation fixed"
    return plc.save_placement(filename, info)


def fd_placement_schedule(
    plc: plc_client.PlacementCost,
    num_steps: Tuple[int, ...] = (100, 100, 100),
    io_factor: float = 1.0,
    move_distance_factors: Tuple[float, ...] = (1.0, 1.0, 1.0),
    attract_factor: Tuple[float, ...] = (100.0, 1.0e-3, 1.0e-5),
    repel_factor: Tuple[float, ...] = (0.0, 1.0e6, 1.0e7),
    use_current_loc: bool = False,
    move_macros: bool = False,
) -> None:
    """A placement schedule that uses force directed method."""
    assert len(num_steps) == len(move_distance_factors)
    assert len(num_steps) == len(repel_factor)
    assert len(num_steps) == len(attract_factor)
    canvas_size = max(plc.get_canvas_width_height())
    max_move_distance = [
        f * canvas_size / s for s, f in zip(num_steps, move_distance_factors)
    ]
    move_stdcells = True
    log_scale_conns = False
    use_sizes = False
    plc.optimize_stdcells(
        use_current_loc,
        move_stdcells,
        move_macros,
        log_scale_conns,
        use_sizes,
        io_factor,
        num_steps,
        max_move_distance,
        attract_factor,
        repel_factor,
    )


def read_node_order_file(
    plc: plc_client.PlacementCost, node_order_file: str
) -> List[int]:
    """Reads the node order from a file."""
    with open(node_order_file, "r") as f:
        node_order = [plc.get_node_index(line.strip()) for line in f.readlines()]
    return node_order


def save_node_order_file(
    plc: plc_client.PlacementCost,
    node_order: List[int],
    node_order_file: str,
) -> None:
    """Saves the node order to a file."""
    with open(node_order_file, "w") as f:
        for node_index in node_order:
            if not plc.is_node_soft_macro(node_index):
                f.write(plc.get_node_name(node_index) + "\n")


def get_ordered_node_indices(
    mode: str,
    plc: plc_client.PlacementCost,
    seed: int = 111,
    node_order_file: str = "",
    exclude_fixed_nodes: bool = True,
) -> List[int]:
    """Returns an ordering of node indices according to the specified mode."""
    rng = np.random.default_rng(seed=seed)
    macro_indices = plc.get_macro_indices()
    hard_macro_indices = [m for m in macro_indices if not plc.is_node_soft_macro(m)]
    soft_macro_indices = [m for m in macro_indices if plc.is_node_soft_macro(m)]

    def macro_area(idx):
        if idx not in hard_macro_indices:
            return 0.0
        w, h = plc.get_node_width_height(idx)
        return w * h

    canvas_width, canvas_height = plc.get_canvas_width_height()

    def distance_to_edge(idx):
        x, y = plc.get_node_location(idx)
        return min(
            x, y, canvas_width - x - canvas_width, canvas_height - y - canvas_height
        )

    logging.info("node_order: %s", mode)
    if mode == "legalization_order":
        ordered_indices = sorted(
            hard_macro_indices,
            key=distance_to_edge,
        ) + sorted(
            soft_macro_indices,
            key=macro_area,
            reverse=True,
        )
    elif mode == "descending_size_macro_first":
        ordered_indices = sorted(
            hard_macro_indices,
            key=macro_area,
            reverse=True,
        ) + sorted(
            soft_macro_indices,
            key=macro_area,
            reverse=True,
        )
    elif mode == "random":
        rng.shuffle(macro_indices)
        ordered_indices = macro_indices
    elif mode == "random_macro_first":
        rng.shuffle(hard_macro_indices)
        logging.info("ordered hard macros: %s", hard_macro_indices)
        ordered_indices = hard_macro_indices + soft_macro_indices
    elif mode == "fake_net_topological":
        fake_net_adj = {}
        fake_nets = plc.get_fake_nets()
        nodes = (
            set([nm[0] for _, nm in fake_nets])
            .union(set([nm[1] for _, nm in fake_nets]))
            .union(set(hard_macro_indices))
        )
        is_port = {n: n not in hard_macro_indices for n in nodes}
        macro_with_fake_net = {n: False for n in nodes}
        for fake_net in fake_nets:
            weight = fake_net[0]
            if weight <= 0:
                continue
            node_0 = fake_net[1][0]
            node_1 = fake_net[1][1]
            fake_net_adj[(node_0, node_1)] = weight
            fake_net_adj[(node_1, node_0)] = weight
            if node_0 in hard_macro_indices:
                macro_with_fake_net[node_0] = True
            if node_1 in hard_macro_indices:
                macro_with_fake_net[node_1] = True

        closeness = {n: 0.0 for n in nodes}

        source = max(nodes, key=lambda n: (is_port[n], macro_area(n)))
        visited_nodes = [source]
        last_node = source
        del closeness[last_node]

        while len(visited_nodes) < len(nodes):
            for node in nodes:
                if node in visited_nodes:
                    continue
                if (node, last_node) in fake_net_adj:
                    closeness[node] += fake_net_adj[(node, last_node)]

            last_node = max(
                closeness,
                key=lambda n: (closeness[n], macro_with_fake_net[n], macro_area(n)),
            )
            visited_nodes.append(last_node)
            del closeness[last_node]

        ordered_indices = [n for n in visited_nodes if n in hard_macro_indices] + sorted(
            soft_macro_indices, key=macro_area
        )[::-1]
    elif mode == "file":
        ordered_indices = read_node_order_file(plc, node_order_file)
    else:
        raise ValueError("{} is an unsupported node placement mode.".format(mode))

    if exclude_fixed_nodes:
        ordered_indices = [m for m in ordered_indices if not plc.is_node_fixed(m)]
    return ordered_indices


def extract_blockages_from_file(
    filename: str, canvas_width: float, canvas_height: float
) -> Optional[List[List[float]]]:
    """Reads blockage information from a given file."""
    blockages = []
    try:
        with open(filename, "r") as infile:
            for line in infile:
                if line.startswith("#"):
                    continue
                items = line.split()
                if len(items) != 4:
                    raise ValueError(
                        "Blockage file does not meet expected format"
                        "Expected format <llx> <lly> <urx> <ury>"
                    )
                llx = float(items[0])
                lly = float(items[1])
                urx = float(items[2])
                ury = float(items[3])
                if llx >= urx:
                    raise ValueError(f"Illegal blockage llx {llx} >= urx {urx}")
                if lly >= ury:
                    raise ValueError(f"Illegal blockage lly {lly} >= ury {ury}")
                if llx < 0:
                    raise ValueError(f"Illegal blockage llx {llx} < 0")
                if urx > canvas_width:
                    raise ValueError(
                        f"Illegal blockage urx {urx} > canvas width {canvas_width}"
                    )
                if lly < 0:
                    raise ValueError(f"Illegal blockage lly {lly} < 0")
                if ury > canvas_height:
                    raise ValueError(
                        f"Illegal blockage ury {ury} > canvas height {canvas_height}"
                    )
                blockages.append([llx, lly, urx, ury, 0.99])
    except IOError:
        logging.error("Could not read file %s", filename)
    return blockages


def get_node_locations(plc: plc_client.PlacementCost) -> Dict[int, int]:
    """Returns all node grid locations (macros and stdcells) in a dict."""
    node_locations = dict()
    for i in nodes_of_types(plc, ["MACRO", "STDCELL"]):
        node_locations[i] = plc.get_grid_cell_of_node(i)
    return node_locations


def get_node_ordering_by_size(plc: plc_client.PlacementCost) -> List[int]:
    """Returns the list of nodes (macros and stdcells) ordered by area."""
    node_areas = dict()
    for i in nodes_of_types(plc, ["MACRO", "STDCELL"]):
        if plc.is_node_fixed(i):
            continue
        w, h = plc.get_node_width_height(i)
        node_areas[i] = w * h
    return sorted(node_areas, key=node_areas.get, reverse=True)


def grid_locations_near(
    plc: plc_client.PlacementCost, start_grid_index: int
) -> Iterator[int]:
    """Yields node indices closest to the start_grid_index."""
    cols, rows = plc.get_grid_num_columns_rows()
    start_col, start_row = start_grid_index % cols, int(start_grid_index / cols)
    for distance in range(cols + rows):
        for row_offset in range(-distance, distance + 1):
            for col_offset in range(-distance, distance + 1):
                if abs(row_offset) + abs(col_offset) != distance:
                    continue
                new_col = start_col + col_offset
                new_row = start_row + row_offset
                if new_col < 0 or new_row < 0 or new_col >= cols or new_row >= rows:
                    continue
                yield int(new_col + new_row * cols)


def place_near(plc: plc_client.PlacementCost, node_index: int, location: int) -> bool:
    """Places a node (legally) closest to the given location."""
    for loc in grid_locations_near(plc, location):
        if plc.can_place_node(node_index, loc):
            plc.place_node(node_index, loc)
            return True
    return False


def disconnect_high_fanout_nets(
    plc: plc_client.PlacementCost, max_allowed_fanouts: int = 500
) -> None:
    high_fanout_nets = []
    for i in nodes_of_types(plc, ["PORT", "STDCELL", "MACRO_PIN"]):
        num_fanouts = len(plc.get_fan_outs_of_node(i))
        if num_fanouts > max_allowed_fanouts:
            print(
                "Disconnecting node: {} with {} fanouts.".format(
                    plc.get_node_name(i), num_fanouts
                )
            )
            high_fanout_nets.append(i)
    plc.disconnect_nets(high_fanout_nets)


def legalize_placement(plc: plc_client.PlacementCost) -> bool:
    """Places the nodes to legal positions snapping to grid cells."""
    fix_port_coordinates(plc)
    node_locations = get_node_locations(plc)
    previous_xy_coords = get_node_xy_coordinates(plc)
    total_macro_displacement = 0
    total_macros = 0
    plc.unplace_all_nodes()
    ordered_nodes = get_node_ordering_by_size(plc)
    for node in ordered_nodes:
        if not place_near(plc, node, node_locations[node]):
            print("Could not place node")
            return False
        if node in previous_xy_coords and not plc.is_node_soft_macro(node):
            x, y = plc.get_node_location(node)
            px, py = previous_xy_coords[node]
            print(
                "x/y displacement: dx = {}, dy = {}, macro: {}".format(
                    x - px, y - py, plc.get_node_name(node)
                )
            )
            total_macro_displacement += abs(x - px) + abs(y - py)
            total_macros += 1
    print(
        "Total macro displacement: {}, avg: {}".format(
            total_macro_displacement, total_macro_displacement / total_macros
        )
    )
    return True


def fix_macros_by_regex(plc: plc_client.PlacementCost, macro_regex_str_list: List[str]):
    """Fix macro locations given a list of macro name regex strings."""
    regexs = []
    for regex_str in macro_regex_str_list:
        regexs.append(re.compile(regex_str))

    hard_macros = []
    for m in plc.get_macro_indices():
        if plc.is_node_soft_macro(m):
            continue
        hard_macros.append(m)

    total = 0
    for m in plc.get_macro_indices():
        if plc.is_node_soft_macro(m):
            continue
        macro_name = plc.get_node_name(m)
        for regex in regexs:
            if regex.fullmatch(macro_name):
                plc.fix_node_coord(m)
                total += 1
                logging.info("Fixed macro: %s", macro_name)
                continue
    logging.info("Total number of fixed macros: %d", total)


def create_blockages_by_spacing_constraints(
    canvas_width: float,
    canvas_height: float,
    macro_boundary_x_spacing: float = 0,
    macro_boundary_y_spacing: float = 0,
    rectilinear_blockages: Optional[List[List[float]]] = None,
) -> List[List[float]]:
    """Create blockages using macro-to-boundary spacing constraints."""
    blockages = []
    blockage_rate = 0.1
    if macro_boundary_x_spacing:
        assert 0 < macro_boundary_x_spacing <= canvas_width
        blockages.append([0, 0, macro_boundary_x_spacing, canvas_height, blockage_rate])
        blockages.append(
            [
                canvas_width - macro_boundary_x_spacing,
                0,
                canvas_width,
                canvas_height,
                blockage_rate,
            ]
        )
    if macro_boundary_y_spacing:
        assert 0 < macro_boundary_y_spacing <= canvas_height
        blockages.append([0, 0, canvas_width, macro_boundary_y_spacing, blockage_rate])
        blockages.append(
            [
                0,
                canvas_height - macro_boundary_y_spacing,
                canvas_width,
                canvas_height,
                blockage_rate,
            ]
        )
    for rectilinear_blockage in rectilinear_blockages or []:
        minx, miny, maxx, maxy, _ = rectilinear_blockage
        if macro_boundary_x_spacing:
            blockages.append(
                [
                    max(minx - macro_boundary_x_spacing, 0),
                    max(miny - macro_boundary_y_spacing, 0),
                    minx,
                    min(maxy + macro_boundary_y_spacing, canvas_height),
                    blockage_rate,
                ]
            )
            blockages.append(
                [
                    maxx,
                    max(miny - macro_boundary_y_spacing, 0),
                    min(maxx + macro_boundary_x_spacing, canvas_width),
                    min(maxy + macro_boundary_y_spacing, canvas_height),
                    blockage_rate,
                ]
            )
        if macro_boundary_y_spacing:
            blockages.append(
                [
                    max(minx - macro_boundary_x_spacing, 0),
                    max(miny - macro_boundary_y_spacing, 0),
                    min(maxx + macro_boundary_x_spacing, canvas_width),
                    miny,
                    blockage_rate,
                ]
            )
            blockages.append(
                [
                    max(minx - macro_boundary_x_spacing, 0),
                    maxy,
                    min(maxx + macro_boundary_x_spacing, canvas_width),
                    min(maxy + macro_boundary_y_spacing, canvas_height),
                    blockage_rate,
                ]
            )
    return blockages
