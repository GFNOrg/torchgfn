import torch


def check_shape(tensor: torch.Tensor, shape: str) -> None:
    """Checks that a tensor has the expected shape and dtype.

    Args:
        tensor: tensor to check
        shape: expected shape of the tensor
    """
    dims = shape.split(" ")
    assert tensor.ndim == len(dims), f"Expected shape {shape}, got {tensor.shape}."

    name_to_value = {}
    for dim_name, size in zip(dims, tensor.shape):
        try:
            expected_size = int(dim_name)
            assert size == expected_size, f"Expected shape {shape}, got {tensor.shape}."
        except ValueError:
            if dim_name in name_to_value:
                assert name_to_value[dim_name] == size, (
                    f"Expected shape {shape}, got {tensor.shape}."
                )
            else:
                name_to_value[dim_name] = size