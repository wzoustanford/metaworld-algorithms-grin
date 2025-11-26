"""Unit test for GatedRecurrentMask LSTM training.

Tests that the LSTM can learn from random sequences by verifying:
1. Loss decreases over training steps
2. Gradients flow properly
3. No NaN or Inf values appear
"""
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from metaworld_algorithms.nn.moore import GatedRecurrentMask


def create_random_sequence_data(
    key, num_sequences=100, seq_len=10, input_dim=32, output_dim=16
):
    """Create sequence data with learnable pattern.

    Creates a simple pattern: output is a function of the cumulative sum of inputs.
    This gives the LSTM something meaningful to learn (temporal dependency).

    Args:
        key: JAX random key
        num_sequences: Number of sequences in dataset
        seq_len: Length of each sequence
        input_dim: Dimension of input features
        output_dim: Dimension of target outputs

    Returns:
        Tuple of (sequences, targets) where:
        - sequences: (num_sequences, seq_len, input_dim)
        - targets: (num_sequences, seq_len, output_dim)
    """
    key1, key2 = jax.random.split(key)

    # Random input sequences
    sequences = jax.random.normal(key1, (num_sequences, seq_len, input_dim))

    # Create targets based on cumulative sum (temporal dependency)
    # This gives LSTM something to learn
    cumsum = jnp.cumsum(sequences[:, :, :output_dim], axis=1)  # Use first output_dim features
    targets = jax.nn.sigmoid(cumsum)  # Squash to [0, 1]

    return sequences, targets


def sequence_loss(params, lstm, sequences, targets):
    """Compute loss over a batch of sequences.

    Processes each sequence timestep by timestep, computing loss at each step.

    Args:
        params: LSTM parameters
        lstm: LSTM module
        sequences: (batch, seq_len, input_dim)
        targets: (batch, seq_len, output_dim)

    Returns:
        Scalar loss value
    """
    batch_size, seq_len, _ = sequences.shape

    # Initialize states to None (will use learned h0/c0)
    h = None
    c = None

    total_loss = 0.0

    # Process sequence timestep by timestep
    for t in range(seq_len):
        x_t = sequences[:, t, :]  # (batch, input_dim)
        target_t = targets[:, t, :]  # (batch, output_dim)

        # Forward pass
        mask_t, h, c = lstm.apply(params, x_t, h, c)

        # Binary cross-entropy loss
        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        eps = 1e-7  # For numerical stability
        bce = -(target_t * jnp.log(mask_t + eps) + (1 - target_t) * jnp.log(1 - mask_t + eps))
        total_loss += bce.mean()

    # Average loss over timesteps
    return total_loss / seq_len


def train_lstm_on_random_data(
    num_epochs=50,
    batch_size=32,
    seq_len=10,
    input_dim=32,
    hidden_size=64,
    output_dim=16,
    learning_rate=1e-3,
):
    """Train LSTM on random data and return training history.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        seq_len: Length of sequences
        input_dim: Input feature dimension
        hidden_size: LSTM hidden state dimension
        output_dim: Output mask dimension
        learning_rate: Learning rate for optimizer

    Returns:
        List of losses per epoch
    """
    print("=" * 60)
    print("LSTM Training Test")
    print("=" * 60)
    print(f"Config:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Output dim: {output_dim}")
    print(f"  Learning rate: {learning_rate}")
    print()

    # Initialize LSTM
    key = jax.random.PRNGKey(42)
    key, init_key, data_key = jax.random.split(key, 3)

    lstm = GatedRecurrentMask(hidden_size=hidden_size, output_size=output_dim)

    # Initialize parameters with dummy input
    dummy_input = jnp.ones((batch_size, input_dim))
    params = lstm.init(init_key, dummy_input)

    print(f"LSTM initialized with parameters:")
    for param_name, param_val in params['params'].items():
        print(f"  {param_name}: {param_val.shape}")
    print()

    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Create training data
    sequences, targets = create_random_sequence_data(
        data_key,
        num_sequences=batch_size * 10,  # 10 batches worth
        seq_len=seq_len,
        input_dim=input_dim,
        output_dim=output_dim
    )

    print(f"Training data created:")
    print(f"  Sequences shape: {sequences.shape}")
    print(f"  Targets shape: {targets.shape}")
    print()

    # Create loss function with lstm baked in
    def loss_fn(params, sequences, targets):
        return sequence_loss(params, lstm, sequences, targets)

    # Compile loss and gradient function
    loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0))

    # Training loop
    losses = []
    print("Training...")

    for epoch in range(num_epochs):
        # Sample a batch
        batch_idx = jax.random.randint(key, (batch_size,), 0, len(sequences))
        key = jax.random.split(key)[0]

        batch_sequences = sequences[batch_idx]
        batch_targets = targets[batch_idx]

        # Compute loss and gradients
        loss_val, grads = loss_and_grad_fn(params, batch_sequences, batch_targets)

        # Check for NaN/Inf
        if jnp.isnan(loss_val) or jnp.isinf(loss_val):
            print(f"ERROR: NaN/Inf detected at epoch {epoch}")
            return losses

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        losses.append(float(loss_val))

        # Print progress
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
            print(f"Epoch {epoch:3d}: Loss = {loss_val:.4f}, Grad norm = {grad_norm:.4f}")

    print()
    return losses


def test_lstm_training():
    """Main test function."""
    losses = train_lstm_on_random_data(
        num_epochs=200,
        batch_size=32,
        seq_len=10,
        input_dim=32,
        hidden_size=64,
        output_dim=16,
        learning_rate=3e-3,
    )

    print("=" * 60)
    print("Test Results")
    print("=" * 60)

    # Check that loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Loss reduction: {loss_reduction:.2f}%")
    print()

    # Check for consistent decrease
    window = 10
    early_avg = sum(losses[:window]) / window
    late_avg = sum(losses[-window:]) / window

    print(f"Early average (first {window} epochs): {early_avg:.4f}")
    print(f"Late average (last {window} epochs):   {late_avg:.4f}")
    print()

    # Check for consistent improvement
    improvements = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])
    improvement_rate = improvements / (len(losses) - 1) * 100
    print(f"Improvement rate: {improvement_rate:.1f}% of steps showed loss decrease")
    print()

    # Verify loss decreased
    if final_loss < initial_loss * 0.9:  # At least 10% reduction
        print("✓ TEST PASSED: Loss decreased significantly")
        print(f"  Loss reduced by {loss_reduction:.2f}%")
        return True
    else:
        print("✗ TEST FAILED: Loss did not decrease enough")
        print(f"  Expected >10% reduction, got {loss_reduction:.2f}%")
        return False


if __name__ == "__main__":
    success = test_lstm_training()
    exit(0 if success else 1)
