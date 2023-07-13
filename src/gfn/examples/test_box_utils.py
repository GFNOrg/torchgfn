import torch

from gfn.envs import BoxEnv
from gfn.examples.box_utils import (
    QuarterCircle,
    QuarterCircleWithExit,
    QuarterDisk,
    BoxPFNeuralNet,
    BoxPBNeuralNet,
    BoxPBEstimator,
    BoxPFEstimator,
    split_PF_module_output,
)


def test_mixed_distributions():
    """Ensure DistributionWrapper functions correctly."""

    delta = 0.1
    hidden_dim = 10
    n_hidden_layers = 2
    n_components = 5
    n_components_s0 = 6

    environment = BoxEnv(
        delta=delta,
        R0=0.1,
        R1=0.5,
        R2=2.0,
        device_str="cpu",
    )
    States = environment.make_States_class()

    # Three cases: when all states are s0, some are s0, and none are s0.
    centers_start = States(torch.FloatTensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))
    centers_mixed = States(torch.FloatTensor([[0.03, 0.06], [0.0, 0.0], [0.0, 0.0]]))
    centers_intermediate = States(
        torch.FloatTensor([[0.03, 0.06], [0.2, 0.3], [0.95, 0.7]])
    )

    net_forward = BoxPFNeuralNet(
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        n_components=n_components,
        n_components_s0=n_components_s0,
    )
    net_backward = BoxPBNeuralNet(
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        n_components=n_components,
    )

    estimator_forward = BoxPFEstimator(
        env=environment,
        module=net_forward,
        n_components_s0=n_components_s0,
        n_components=n_components,
    )
    estimator_backward = BoxPBEstimator(
        env=environment, module=net_forward, n_components=n_components
    )

    out_start = net_forward(centers_start.tensor)
    out_mixed = net_forward(centers_mixed.tensor)
    out_intermediate = net_forward(centers_intermediate.tensor)

    # Check the mixed_distribution.
    assert torch.all(torch.sum(out_mixed == 0, -1)[1:])  # Second two elems are s_0.

    # Retrieve the non-s0 elem and split:
    (
        exit_probability,
        mixture_logits,
        alpha_theta,
        beta_theta,
        alpha_r,
        beta_r,
    ) = split_PF_module_output(
        out_mixed[0, :].unsqueeze(0), max(n_components_s0, n_components)
    )

    assert exit_probability > 0

    def _assert_correct_parameter_masking(x, mask_val):
        B, P = x.shape

        if n_components_s0 > n_components:
            assert (
                torch.sum(x == mask_val) == (n_components_s0 - n_components) * B
            )  # max - min == 1.
            assert torch.all(
                x[..., -1] == mask_val
            )  # One of the masked elements should be the final one.

    _assert_correct_parameter_masking(mixture_logits, 0)
    _assert_correct_parameter_masking(alpha_theta, 0.5)
    _assert_correct_parameter_masking(beta_theta, 0.5)

    # These are all 0.5, because they're only used at s_0.
    assert torch.sum(alpha_r == 0.5) == max(n_components_s0, n_components)
    assert torch.sum(beta_r == 0.5) == max(n_components_s0, n_components)

    # Now check the batch of all-intermediate states.
    B, P = out_intermediate.shape
    (
        exit_probability,
        mixture_logits,
        alpha_theta,
        beta_theta,
        alpha_r,
        beta_r,
    ) = split_PF_module_output(out_intermediate, max(n_components_s0, n_components))

    assert len(exit_probability > 0) == B  # All exit probabilities are non-zero.

    _assert_correct_parameter_masking(mixture_logits, 0)
    _assert_correct_parameter_masking(alpha_theta, 0.5)
    _assert_correct_parameter_masking(beta_theta, 0.5)

    assert torch.sum(alpha_r == 0.5) == B * max(n_components_s0, n_components)
    assert torch.sum(beta_r == 0.5) == B * max(n_components_s0, n_components)

if __name__ == "__main__":
    test_mixed_distributions()