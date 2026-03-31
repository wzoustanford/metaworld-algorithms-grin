# LSTM + Critic Joint Optimization Integration Summary

## User Request Timeline

### Initial Request
User wanted to implement a recurrent LSTM that:
1. Remembers temporal activations of the network
2. Produces masks for the next timestep of the critic network
3. Has tunable h0 and c0 parameters

### Implementation Journey

1. **Understanding Phase**
   - Analyzed `jax.value_and_grad` usage for computing gradients (lines 341-350 in mtsac.py)
   - Reviewed how optimizer (Adam) connects to `apply_gradients` via TrainState
   - Examined buffer's `sample_sequences` function for temporal data

2. **LSTM Design Phase**
   - Implemented standard LSTM with input, forget, output gates
   - Made h0, c0 learnable parameters (initialized to zeros)
   - LSTM processes one timestep at a time (not full sequences)
   - Returns (mask, h_new, c_new) tuple

3. **Integration Strategy**
   - Combined critic + LSTM parameters into single pytree: `{'critic': ..., 'lstm': ...}`
   - Used `jax.value_and_grad` to compute joint gradients
   - Stored LSTM states (h, c) as class attributes in MTSACSequential
   - Updated states after each timestep for temporal continuity

4. **Critical Fixes**
   - Fixed intermediate variable capture using `mutable=['intermediates']`
   - Added mask parameter to MOORENetwork
   - Applied mask to features BEFORE storing to intermediates

## Implementation Components

### 1. LSTM Network (`metaworld_algorithms/nn/moore.py`)
**GatedRecurrentMask class (lines 104-220)**
- Standard LSTM architecture with input, forget, output, and candidate gates
- Learnable initial states (h0, c0) that are optimized during training
- Generates output masks via sigmoid activation
- Input: `(batch, input_dim)`, Hidden/Cell: `(batch, hidden_size)`
- Output: `(mask, h_new, c_new)` where mask is `(batch, output_size)`

### 2. MOORE Network Mask Support (`metaworld_algorithms/nn/moore.py:40-101`)
**Modified `MOORENetwork.__call__`**
- Added `mask: jax.Array | None = None` parameter
- Applies mask to features via element-wise multiplication: `features_out = features_out * mask`
- Mask is applied BEFORE storing to intermediates (line 76-77)
- Ensures next LSTM timestep sees masked features

### 3. Critic Loss Function (`metaworld_algorithms/rl/algorithms/mtsac.py:286-350`)
**Modified to accept combined parameters**
- Signature: `critic_loss(combined_params, data, ...)`
- `combined_params` can be `{'critic': ..., 'lstm': ...}` or just critic params
- Uses `self.lstm_h` and `self.lstm_c` from class attributes
- Returns: `(loss, (q_pred.mean(), h_new, c_new))`
- Captures intermediates via `mutable=['intermediates']` and `store=True`
- Extracts intermediates: `updated_vars['intermediates']['or'][0]`

### 4. Update Critic Function (`metaworld_algorithms/rl/algorithms/mtsac.py:352-449`)
**Joint gradient computation and application**
- Creates `combined_params = {'critic': self.critic.params, 'lstm': self.lstm.params}`
- Computes joint gradients: `jax.value_and_grad(critic_loss, ...)(combined_params, ...)`
- Extracts auxiliary outputs: `qf_values, h_new, c_new = aux`
- Updates LSTM states: `self.lstm_h = h_new`, `self.lstm_c = c_new`
- Applies critic gradients: `self.critic.apply_gradients(grads=critic_grads)`
- Applies LSTM gradients: `self.lstm.apply_gradients(grads=lstm_grads)` (line 441-449)

### 5. MTSACSequential Class (`metaworld_algorithms/rl/algorithms/mtsac.py:632-692`)
**LSTM initialization and state tracking**
- Added fields:
  - `grin_state_vars: list = []` - stores network intermediate states
  - `lstm: TrainState | None` - LSTM TrainState
  - `lstm_h: Array | None` - current hidden state
  - `lstm_c: Array | None` - current cell state
- In `initialize()`:
  - Asserts `lstm_config` is present
  - Creates `GatedRecurrentMask` network
  - Initializes LSTM TrainState with optimizer
  - Initializes `lstm_h` and `lstm_c` from learned `h0`, `c0` parameters

## Data Flow

```
Timestep t:
  1. LSTM receives previous features from grin_state_vars[-1]
  2. LSTM uses self.lstm_h and self.lstm_c (current states)
  3. LSTM outputs: (mask, h_new, c_new)

  4. Critic receives (obs, actions, mask)
  5. Critic applies mask to features: features_out = features_out * mask
  6. Critic stores masked features to intermediates
  7. Critic outputs: q_pred

  8. Loss = (q_pred - target)^2
  9. Compute gradients for both critic and LSTM params
  10. Apply gradients to update both networks
  11. Update LSTM states: self.lstm_h = h_new, self.lstm_c = c_new
  12. Store masked features: grin_state_vars.append(intermediates)

Timestep t+1:
  - Repeats with updated lstm_h, lstm_c, and grin_state_vars[-1]
```

## Key Design Decisions

1. **Combined Parameters**: Use dict structure `{'critic': ..., 'lstm': ...}` for joint optimization
   - JAX treats nested dicts as pytrees
   - Gradients maintain same structure as parameters
   - Allows single `value_and_grad` call for both networks

2. **State Storage**: Store LSTM h/c as class attributes, updated after each timestep
   - Avoids passing states through function arguments
   - JIT compilation friendly
   - Maintains temporal continuity across timesteps

3. **Intermediate Capture**: Use `mutable=['intermediates']` to capture network features
   - Required for Flax to return modified collections
   - `store=True` flag tells MOORENetwork to save features
   - Extracted as `updated_vars['intermediates']['or'][0]`

4. **Mask Application**: Apply mask before storing intermediates (temporal consistency)
   - Next LSTM sees the masked features it helped create
   - Creates feedback loop for learning

5. **Backward Compatibility**: critic_loss handles both combined and single params
   - Checks if params is dict with 'critic' key
   - Falls back to non-LSTM path if no LSTM params

## Required Config

MTSACSequential requires `lstm_config` with:
- `hidden_size`: LSTM hidden state dimension
- `output_size`: Mask output dimension (should match MOORE features_out dimension)
- `optimizer`: OptimizerConfig for LSTM (e.g., Adam with lr=1e-3)

Example:
```python
@dataclass
class LSTMConfig:
    hidden_size: int = 64
    output_size: int = 256  # Match MOORE network width
    optimizer: OptimizerConfig = OptimizerConfig(lr=1e-3)

MTSACSequentialConfig(
    ...,
    lstm_config=LSTMConfig()
)
```

## Testing Status
- ✅ LSTM unit test passed (35% loss reduction)
- ⏳ Full integration test pending

## Next Steps
1. Create MTSACSequentialConfig with lstm_config
2. Test with actual MT50 environment
3. Monitor LSTM gradient norms and state evolution
4. Tune LSTM hyperparameters (hidden_size, lr)

## Potential Issues & Debugging

### Common Errors
1. **"lstm_config not found"**: Ensure MTSACSequentialConfig has lstm_config attribute
2. **Shape mismatches**: Verify output_size matches MOORE network width
3. **Empty grin_state_vars**: Initialize with first forward pass before LSTM
4. **Gradient NaNs**: Check learning rates, may need lower LR for LSTM

### Verification Points
- Check `updated_vars['intermediates']` structure matches LSTM input expectations
- Verify h, c states are updating (not stuck at h0, c0)
- Monitor mask values (should be in [0, 1] range due to sigmoid)
- Ensure grin_state_vars grows with sequence length
