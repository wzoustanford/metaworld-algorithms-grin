import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Tuple

from metaworld_algorithms.config.nn import MOOREConfig

from .base import MLP

# --- OPTIMIZED FUNCTION USING JAX.LAX.SCAN ---

def orthogonal_1d(
    x: Float[Array, "batch_size num_experts dim"], num_experts: int
) -> Float[Array, "batch_size num_experts dim"]:
    
    #Optimized, JAX-idiomatic implementation of batch-wise Gram-Schmidt
    #orthogonalization using jax.lax.scan.

    #Args:
    #    x: Input vectors (batch_size, num_experts, dim).
    #    num_experts: The number of vectors/experts.

    #Returns:
    #    The resulting orthonormal basis (batch_size, num_experts, dim).
    
    chex.assert_rank(x, 3)
    batch_size, _, dim = x.shape

    # ------------------ 1. Initial State & Data Preparation ------------------

    # The first vector is calculated and stored separately.
    v0 = x[:, 0, :]
    norm_v0 = jnp.linalg.norm(v0, axis=1, keepdims=True) + 1e-8
    basis0 = v0 / norm_v0 # shape (batch_size, dim)
    
    # The carry is the growing basis, initialized with the first vector.
    # We use the full (batch_size, num_experts, dim) shape and fill it step-by-step.
    basis_init = jnp.zeros((batch_size, num_experts, dim))
    
    # Update the first slice of the 3D array:
    basis_init = jax.lax.dynamic_update_slice(
        basis_init, basis0[:, None, :], (0, 0, 0) # Update with (B, 1, D) at index (0, 0, 0)
    )

    # The remaining vectors to process are x[:, 1:, :]
    xs_to_scan = x[:, 1:, :]

    # The 'carry' will be the basis generated so far, plus the current index 'i'.
    carry_init = (basis_init, 1) # (current_basis_array, current_index)


    # ------------------ 2. Define the Scan Step Function (Runs inside VMAP) ------------------

    def gram_schmidt_step(carry: Tuple[Array, int], v_slice: Array) -> Tuple[Tuple[Array, int], None]:
        
        #Processes one vector 'v' against the current basis using jax.lax.scan.
        
        #NOTE: Inside vmap, basis_array is (num_experts, dim) and v_slice is (dim,).
        
        basis_array, i = carry 
        num_experts_static, dim = basis_array.shape # Get dim from the 2D array shape inside vmap
        
        # 1. Create a mask to zero out projections from basis vectors that are not yet filled (indices >= i)
        # The mask will be [1, 1, ..., 1, 0, 0, ...] where the first 'i' elements are 1.
        mask = jnp.arange(num_experts_static) < i # shape (num_experts,)
        mask = jnp.expand_dims(mask.astype(basis_array.dtype), axis=1) # shape (num_experts, 1)
        
        # The current_basis is the full basis_array, but scaled by the mask.
        current_basis = basis_array * mask # (num_experts, dim)
        
        # 2. Project v onto the subspace spanned by current_basis (all elements)
        # v_slice is (dim,). current_basis is (num_experts, dim).
        
        # Projection coefficient (v @ basis^T): coefficients = current_basis @ v_slice
        # (num_experts, D) @ (D,) -> (num_experts,)
        # Note: The coefficients for indices >= i will be near-zero because of the mask.
        coefficients = jnp.dot(current_basis, v_slice) 
        
        # Projection vector: sum(coefficients[j] * basis_j)
        # (num_experts, 1) * (num_experts, D) -> (num_experts, D)
        # Sum over num_experts -> (D,)
        projection_components = jnp.expand_dims(coefficients, axis=1) * current_basis
        projection = jnp.sum(projection_components, axis=0)
        
        # 3. Calculate orthogonal component w = v - projection
        w = v_slice - projection # (D,)
        
        # 4. Normalize w -> wnorm
        norm_w = jnp.linalg.norm(w) + 1e-8
        wnorm = w / norm_w # (D,)
        
        # 5. Update the basis_array with the new orthogonal vector
        # Store wnorm at the current index 'i'. wnorm is (D,), expanded to (1, D) for 2D update
        # The starting index (i, 0) is now valid because we are updating, not slicing with dynamic length.
        new_basis_array = jax.lax.dynamic_update_slice(
            basis_array, jnp.expand_dims(wnorm, axis=0), (i, 0)
        )

        # 6. Increment index and return new carry
        new_i = i + 1
        return (new_basis_array, new_i), None

    # ------------------ 3. Execute the Scan ------------------

    # Transpose data for scan iteration: (num_experts - 1, batch_size, dim)
    xs_to_scan_transposed = xs_to_scan.transpose(1, 0, 2)
    
    # vmap_scanned_fn takes the batch_size out of the carry and xs.
    # The carry is (basis_array_batch, index_scalar)
    vmap_scanned_fn = jax.vmap(
        lambda carry, xs: jax.lax.scan(gram_schmidt_step, carry, xs),
        in_axes=((0, None), 1) 
    )

    # Initialize the basis array for vmap (only need to slice it in vmap)
    vmap_carry_init = (basis_init, 1)
    
    # Perform the vmap and scan
    # Note: vmap_carry_init[0] has shape (B, N, D), vmap_carry_init[1] has shape ()
    # xs_to_scan_transposed has shape (B, N-1, D)
    final_carry, _ = vmap_scanned_fn(
        vmap_carry_init,
        xs_to_scan_transposed
    )

    # The final result is the first element of the final carry tuple (the fully built basis)
    final_basis, _ = final_carry
    
    chex.assert_equal_shape((x, final_basis))
    return final_basis


def non_jax_orthogonal_1d(
    x: Float[Array, "batch_size num_experts dim"], num_experts: int
) -> Float[Array, "batch_size num_experts dim"]:
    chex.assert_rank(x, 3)

    basis = jnp.expand_dims(
        x[:, 0, :] / (jnp.linalg.norm(x[:, 0, :], axis=1, keepdims=True) + 1e-8), axis=1
    )

    for i in range(1, num_experts):
        v = jnp.expand_dims(x[:, i, :], axis=1)  # (batch_size, 1, dim)
        w = v - ((v @ basis.transpose(0, 2, 1)) @ basis)
        wnorm = w / (jnp.linalg.norm(w, axis=2, keepdims=True) + 1e-8)
        basis = jnp.concatenate((basis, wnorm), axis=1)

    chex.assert_equal_shape((x, basis))
    return basis


class MOORENetwork(nn.Module):
    config: MOOREConfig

    head_dim: int  # o, 1 for Q networks and 2 * action_dim for policy networks
    head_kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    head_bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array | None = None, store: bool = False) -> jax.Array:
        """Forward pass through MOORE network with optional masking.

        Args:
            x: Input tensor (batch, feature_dim) - includes task IDs in last num_tasks dimensions
            mask: Optional mask from LSTM (batch, width) - applied to features_out
            store: If True, stores features_out to 'intermediates' collection for LSTM

        Returns:
            Output tensor (batch, head_dim) - task-specific predictions

        Design Notes:
            - mask parameter added for LSTM integration (element-wise multiplication)
            - mask is applied BEFORE storing to intermediates (temporal consistency)
            - store=True must be used with mutable=['intermediates'] in apply call
        """
        batch_dim = x.shape[0]
        assert self.config.num_tasks is not None, "Number of tasks must be provided."
        task_idx = x[..., -self.config.num_tasks :]
        x = x[..., : -self.config.num_tasks]

        # Task ID embedding
        task_embedding = nn.Dense(
            self.config.num_experts,
            use_bias=False,
            kernel_init=self.config.kernel_init(),
        )(task_idx)

        # MOORE torso - creates expert representations
        experts_out = nn.vmap(
            MLP,
            variable_axes={"params": 0, "intermediates": 1},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,  # pyright: ignore [reportArgumentType]
            out_axes=-2,
            axis_size=self.config.num_experts,
        )(
            self.config.width,
            self.config.depth - 1,
            self.config.width,
            self.config.activation,
            self.config.kernel_init(),
            self.config.bias_init(),
            self.config.use_bias,
            activate_last=False,
        )(x)
        # Orthogonalize expert outputs for better representation diversity
        experts_out = orthogonal_1d(experts_out, num_experts=self.config.num_experts)
        # Combine experts using task embedding weights
        features_out = jnp.einsum("bnk,bn->bk", experts_out, task_embedding)
        features_out = jax.nn.tanh(features_out)

        # LSTM Integration: Apply mask if provided
        # CRITICAL: Mask is applied BEFORE storing to intermediates
        # This ensures the next LSTM timestep sees the masked features it helped create
        # Mask shape: (batch, width) - same as features_out
        # Operation: element-wise multiplication (gating mechanism)
        if mask is not None:
            features_out = features_out * mask

        # Store features for LSTM temporal processing
        # CAUTION: Requires mutable=['intermediates'] in apply call to return this
        # Accessed later as: updated_vars['intermediates']['or']
        #print(f"DEBUG MOORENetwork: store={store}, features_out.shape={features_out.shape}")
        if store:
            #print("DEBUG MOORENetwork: Calling self.variable to store 'or'")
            self.variable('intermediates', 'or', lambda: features_out)
            #self.sow("intermediates", "or", features_out)
        # sow() is always called (doesn't require mutable flag)
        #self.sow("intermediates", "torso_output", features_out)

        # MH (multi-head) - task-specific output layers
        x = nn.vmap(
            nn.Dense,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,  # pyright: ignore [reportArgumentType]
            out_axes=1,
            axis_size=self.config.num_tasks,
        )(
            self.head_dim,
            kernel_init=self.head_kernel_init,
            bias_init=self.head_bias_init,
            use_bias=self.config.use_bias,
        )(features_out)

        # Select output for the current task
        task_indices = task_idx.argmax(axis=-1)
        x = x[jnp.arange(batch_dim), task_indices]

        return x


class GatedRecurrentMask(nn.Module):
    """Standard LSTM with tunable initial states for generating masks.

    This LSTM takes network activations (intermediate features) and produces
    masks that can be used to gate or modulate network behavior. The initial
    hidden and cell states (h0, c0) are learnable parameters.

    Attributes:
        hidden_size: Dimension of LSTM hidden state
        output_size: Dimension of output mask (typically matches input size)
        kernel_init: Initializer for weight matrices
        bias_init: Initializer for bias vectors
    """
    hidden_size: int
    output_size: int
    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.orthogonal()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "batch input_dim"],
        h: Float[Array, "batch hidden_size"] | None = None,
        c: Float[Array, "batch hidden_size"] | None = None
    ) -> Tuple[Float[Array, "batch output_size"], Float[Array, "batch hidden_size"], Float[Array, "batch hidden_size"]]:
        """Apply LSTM to input and return mask along with new hidden states.

        Args:
            x: Input features (typically network activations/intermediates)
            h: Hidden state (if None, uses learnable h0)
            c: Cell state (if None, uses learnable c0)

        Returns:
            Tuple of (mask, new_h, new_c)
            - mask: Output mask of shape (batch, output_size)
            - new_h: Updated hidden state
            - new_c: Updated cell state
        """
        batch_size = x.shape[0]
        input_dim = x.shape[-1]

        # Learnable initial states
        # DESIGN: h0 and c0 are trainable parameters optimized via gradients
        # CAUTION: Must use zeros initializer (orthogonal requires 2D shapes)
        # These are initialized once and reused for each sequence start
        h0_param = self.param(
            'h0',
            jax.nn.initializers.zeros,
            (self.hidden_size,)
        )
        c0_param = self.param(
            'c0',
            jax.nn.initializers.zeros,
            (self.hidden_size,)
        )

        print(f"hidden_size: {self.hidden_size}")

        # Use provided states or broadcast learned initial states
        # If h/c are None (first timestep), broadcast h0/c0 to batch size
        # Otherwise, use the provided states from previous timestep
        if h is None:
            h = jnp.tile(h0_param[None, :], (batch_size, 1))  # (batch, hidden_size)
        if c is None:
            c = jnp.tile(c0_param[None, :], (batch_size, 1))  # (batch, hidden_size)

        # Standard LSTM gates: input, forget, output, candidate
        # IMPLEMENTATION: Compute all 4 gates simultaneously for efficiency
        # Combined weight matrix structure: [W_i | W_f | W_o | W_g]
        # This is standard practice in LSTM implementations
        combined_dim = 4 * self.hidden_size

        # Input projection: x -> gates
        # Maps input features to all 4 gates
        W_x = self.param(
            'W_x',
            self.kernel_init,
            (input_dim, combined_dim)
        )

        # Hidden state projection: h -> gates
        # Recurrent connection from previous hidden state to all gates
        W_h = self.param(
            'W_h',
            self.kernel_init,
            (self.hidden_size, combined_dim)
        )

        # Bias for all gates
        b = self.param(
            'b',
            self.bias_init,
            (combined_dim,)
        )
        
        # Compute all gates at once: gates = xW_x + hW_h + b
        # Shape: (batch, 4*hidden_size)
        gates = x @ W_x + h @ W_h + b

        # Split into individual gates
        # Each gate has shape (batch, hidden_size)
        i_gate, f_gate, o_gate, g_gate = jnp.split(gates, 4, axis=-1)

        # Apply activations to gates
        # LSTM STANDARD: sigmoid for gates (range [0,1]), tanh for candidate (range [-1,1])
        i = jax.nn.sigmoid(i_gate)  # Input gate: controls what new info to add
        f = jax.nn.sigmoid(f_gate)  # Forget gate: controls what to discard from memory
        o = jax.nn.sigmoid(o_gate)  # Output gate: controls what to expose from memory
        g = jnp.tanh(g_gate)        # Candidate cell state: new information to add

        # Update cell state (memory)
        # FORMULA: c_new = f * c + i * g
        # INTUITION: Forget old memory (f*c) and remember new info (i*g)
        c_new = f * c + i * g

        # Update hidden state (output)
        # FORMULA: h_new = o * tanh(c_new)
        # INTUITION: Output gate controls how much of the cell state to expose
        h_new = o * jnp.tanh(c_new)
        
        # Generate output mask from hidden state
        # DESIGN: Project hidden state to output dimension (mask size)
        # This mask will be applied to MOORE network features
        W_out = self.param(
            'W_out',
            self.kernel_init,
            (self.hidden_size, self.output_size)
        )
        b_out = self.param(
            'b_out',
            self.bias_init,
            (self.output_size,)
        )

        # Mask output with sigmoid activation
        # CHOICE: sigmoid -> [0,1] range for gating (element-wise multiplication)
        # ALTERNATIVE: Could use tanh for [-1,1] modulation if needed
        # Current implementation: multiplicative gating of features
        mask = jax.nn.sigmoid(h_new @ W_out + b_out)

        # Return mask and updated states
        # USAGE: mask applied to critic features, h_new/c_new stored for next timestep
        return mask, h_new, c_new
