"""Quick test to verify the GatedRecurrentMask LSTM implementation."""
import jax
import jax.numpy as jnp
from metaworld_algorithms.nn.moore import GatedRecurrentMask

# Test basic functionality
def test_lstm():
    # Create LSTM instance
    lstm = GatedRecurrentMask(hidden_size=64, output_size=128)

    # Initialize with dummy input
    batch_size = 32
    input_dim = 128
    dummy_input = jnp.ones((batch_size, input_dim))

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = lstm.init(key, dummy_input)

    print("LSTM Parameters:")
    print(f"  h0 shape: {params['params']['h0'].shape}")
    print(f"  c0 shape: {params['params']['c0'].shape}")
    print(f"  W_x shape: {params['params']['W_x'].shape}")
    print(f"  W_h shape: {params['params']['W_h'].shape}")
    print(f"  W_out shape: {params['params']['W_out'].shape}")

    # Test forward pass with None states (uses learned h0, c0)
    mask1, h1, c1 = lstm.apply(params, dummy_input)
    print(f"\nForward pass (with learned h0/c0):")
    print(f"  Mask shape: {mask1.shape}")
    print(f"  Hidden state shape: {h1.shape}")
    print(f"  Cell state shape: {c1.shape}")
    print(f"  Mask range: [{mask1.min():.4f}, {mask1.max():.4f}]")

    # Test forward pass with provided states (temporal update)
    mask2, h2, c2 = lstm.apply(params, dummy_input, h1, c1)
    print(f"\nForward pass (with previous states):")
    print(f"  Mask shape: {mask2.shape}")
    print(f"  Hidden state shape: {h2.shape}")
    print(f"  Cell state shape: {c2.shape}")
    print(f"  Mask range: [{mask2.min():.4f}, {mask2.max():.4f}]")

    # Verify states changed
    print(f"\nState changes:")
    print(f"  h changed: {not jnp.allclose(h1, h2)}")
    print(f"  c changed: {not jnp.allclose(c1, c2)}")
    print(f"  mask changed: {not jnp.allclose(mask1, mask2)}")

    print("\n✓ LSTM test passed!")

if __name__ == "__main__":
    test_lstm()
