"""Tests for gfn.utils.modules: MLP, Tabular, UniformModule, NoisyLinear."""

import pytest
import torch
import torch.nn as nn

from gfn.utils.modules import MLP, DiscreteUniform, NoisyLinear, Tabular, UniformModule

# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class TestMLP:
    def test_forward_basic(self):
        mlp = MLP(input_dim=10, output_dim=5, hidden_dim=32, n_hidden_layers=2)
        x = torch.randn(4, 10)
        out = mlp(x)
        assert out.shape == (4, 5)

    def test_forward_with_trunk(self):
        trunk = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32))
        trunk.hidden_dim = torch.tensor(32)
        mlp = MLP(input_dim=10, output_dim=5, trunk=trunk)
        x = torch.randn(4, 10)
        out = mlp(x)
        assert out.shape == (4, 5)

    def test_with_noisy_layers(self):
        mlp = MLP(
            input_dim=10,
            output_dim=5,
            hidden_dim=32,
            n_hidden_layers=2,
            n_noisy_layers=1,
        )
        x = torch.randn(4, 10)
        out = mlp(x)
        assert out.shape == (4, 5)

    def test_with_layer_norm(self):
        mlp = MLP(
            input_dim=10,
            output_dim=5,
            hidden_dim=32,
            n_hidden_layers=2,
            add_layer_norm=True,
        )
        x = torch.randn(4, 10)
        out = mlp(x)
        assert out.shape == (4, 5)

    @pytest.mark.parametrize("activation", ["relu", "leaky_relu", "tanh", "elu"])
    def test_activation_functions(self, activation):
        mlp = MLP(
            input_dim=10,
            output_dim=5,
            hidden_dim=32,
            n_hidden_layers=1,
            activation_fn=activation,
        )
        x = torch.randn(4, 10)
        out = mlp(x)
        assert out.shape == (4, 5)

    def test_integer_input_cast_to_float(self):
        """MLP should handle integer inputs by casting to float."""
        mlp = MLP(input_dim=10, output_dim=5, hidden_dim=32, n_hidden_layers=1)
        x = torch.randint(0, 10, (4, 10))
        out = mlp(x)
        assert out.shape == (4, 5)
        assert out.dtype == torch.get_default_dtype()

    def test_batched_3d_input(self):
        """MLP should handle 3D inputs (time x batch x features)."""
        mlp = MLP(input_dim=10, output_dim=5, hidden_dim=32, n_hidden_layers=1)
        x = torch.randn(3, 4, 10)
        out = mlp(x)
        assert out.shape == (3, 4, 5)


# ---------------------------------------------------------------------------
# Tabular
# ---------------------------------------------------------------------------


class TestTabular:
    def test_forward(self):
        tab = Tabular(n_states=100, output_dim=5)
        # EnumPreprocessor outputs integer indices
        indices = torch.tensor([[0], [1], [50], [99]])
        out = tab(indices)
        assert out.shape == (4, 5)

    def test_forward_values_from_table(self):
        tab = Tabular(n_states=10, output_dim=3)
        with torch.no_grad():
            tab.table[5] = torch.tensor([1.0, 2.0, 3.0])
        indices = torch.tensor([[5]])
        out = tab(indices)
        assert torch.allclose(out, torch.tensor([[1.0, 2.0, 3.0]]))


# ---------------------------------------------------------------------------
# UniformModule
# ---------------------------------------------------------------------------


class TestUniformModule:
    def test_forward(self):
        mod = UniformModule(output_dim=5, fill_value=0.0)
        x = torch.randn(4, 10)
        out = mod(x)
        assert out.shape == (4, 5)
        assert (out == 0.0).all()

    def test_fill_value(self):
        mod = UniformModule(output_dim=3, fill_value=1.5)
        x = torch.randn(2, 8)
        out = mod(x)
        assert torch.allclose(out, torch.full((2, 3), 1.5))

    def test_skip_normalization_flag(self):
        mod = UniformModule(output_dim=3, skip_normalization=True)
        assert mod.skip_normalization is True

    def test_input_dim_attr(self):
        mod = UniformModule(output_dim=3, input_dim=10)
        assert mod.input_dim == 10


class TestDiscreteUniform:
    def test_forward(self):
        mod = DiscreteUniform(output_dim=5)
        x = torch.randn(4, 10)
        out = mod(x)
        assert out.shape == (4, 5)
        assert (out == 0.0).all()  # uniform logits


# ---------------------------------------------------------------------------
# NoisyLinear
# ---------------------------------------------------------------------------


class TestNoisyLinear:
    def test_forward(self):
        layer = NoisyLinear(in_features=10, out_features=5)
        x = torch.randn(4, 10)
        out = layer(x)
        assert out.shape == (4, 5)

    def test_reset_noise(self):
        layer = NoisyLinear(in_features=10, out_features=5)
        # Should not raise
        layer.reset_noise()

    def test_noise_changes_output(self):
        """Two forward passes with different noise should produce different outputs."""
        torch.manual_seed(42)
        layer = NoisyLinear(in_features=10, out_features=5)
        x = torch.randn(4, 10)

        layer.reset_noise()
        out1 = layer(x).detach().clone()
        layer.reset_noise()
        out2 = layer(x).detach().clone()
        # Outputs should differ due to noise
        assert not torch.equal(out1, out2)

    def test_gradient_flows(self):
        layer = NoisyLinear(in_features=10, out_features=5)
        x = torch.randn(4, 10)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert getattr(layer.weight_mu, "grad", None) is not None
        assert getattr(layer.bias_mu, "grad", None) is not None
