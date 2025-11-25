
# ✅ **ddsp_phasor_core.py**

```python
"""
ddsp_phasor_core.py

GammaJAX DDSP — Phasor Core (GDSP Style)
----------------------------------------

A fully differentiable, pure-functional JAX implementation of a phase
accumulator ("phasor") suitable for driving oscillators, wavetables,
FM/PM, envelopes, LFOs, sync systems, and any normalized-phase
timebase.

This module follows the GDSP architectural pattern:

    phasor_core_init(...)
    phasor_core_update_state(...)
    phasor_core_tick(x, state, params)
    phasor_core_process(x, state, params)

Each `tick` computes:

    • frequency smoothing (one-pole)
    • normalized phase increment
    • modulo wrapping into [0,1)
    • optional per-sample phase reset
    • optional phase offset

Everything is:
    – Pure JAX (jit-safe)
    – No classes, no dicts
    – State = tuple only
    – No Python branching inside jit
    – No dynamic allocation inside jit
    – No jnp.arange / jnp.zeros inside jit
    – All shapes determined outside jit
    – All control via jnp.where or lax.cond
    – Differentiable end-to-end
"""

from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax


# ============================================================================
# 1. phasor_core_init
# ============================================================================

def phasor_core_init(
    sample_rate: float,
    initial_phase: float = 0.0,
    initial_freq_hz: float = 440.0,
    phase_offset: float = 0.0,
    smooth_coeff: float = 0.0,
    interp_mode: int = 1,
    *,
    dtype=jnp.float32,
) -> Tuple[
        Tuple[jnp.ndarray, jnp.ndarray],           # state
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],  # params
]:
    """
    Initialize GDSP phasor core.

    Parameters:
        sample_rate   : sampling rate (Hz)
        initial_phase : starting phase ∈ [0,1)
        initial_freq_hz : smoothed frequency initial value
        phase_offset  : additive offset to output phase ∈ cycles
        smooth_coeff  : one-pole smoothing coefficient ∈ [0,1]
        interp_mode   : integer passed downstream for table interpolation
        dtype         : JAX dtype

    State = (phase, freq_smooth)
    Params = (sample_rate, phase_offset, smooth_coeff, interp_mode)
    """
    phase0 = jnp.asarray(initial_phase, dtype=dtype) % 1.0
    freq0 = jnp.asarray(initial_freq_hz, dtype=dtype)

    state = (phase0, freq0)

    params = (
        jnp.asarray(sample_rate, dtype=dtype),
        jnp.asarray(phase_offset, dtype=dtype),
        jnp.asarray(smooth_coeff, dtype=dtype),
        jnp.asarray(interp_mode, dtype=jnp.int32),
    )

    return state, params


# ============================================================================
# 2. phasor_core_update_state  (placeholder — included for GDSP structure)
# ============================================================================

def phasor_core_update_state(
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
):
    """
    No-op placeholder to match GDSP structure.

    State is only updated in phasor_core_tick().
    """
    return state


# ============================================================================
# 3. phasor_core_tick
# ============================================================================

@jax.jit
def phasor_core_tick(
    x,
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    One-sample phasor tick.

    Inputs:
        x      : (freq_hz, phase_reset)
                 freq_hz     : desired frequency (Hz)
                 phase_reset : 0 or 1 (reset phase to 0 when > 0.5)

        state  : (phase, freq_smooth)
        params : (sample_rate, phase_offset, smooth_coeff, interp_mode)

    Output:
        y         : new output phase ∈ [0,1)
        new_state : (phase_next, freq_smooth_next)

    Math:

    freq_smooth[n+1] = freq_smooth[n] + α (freq[n] − freq_smooth[n])

    inc[n] = freq_smooth[n] / sample_rate

    phase_raw[n+1] = phase[n] + inc[n]
    phase_wrap[n+1] = phase_raw[n+1] − floor(phase_raw[n+1])

    If reset:
        phase_next = 0
    Else:
        phase_next = phase_wrap

    y[n] = (phase_next + phase_offset) mod 1
    """

    freq_hz, reset_flag = x

    phase, freq_smooth = state
    sample_rate, phase_offset, smooth_coeff, interp_mode = params

    # Ensure matching dtype
    freq_hz = jnp.asarray(freq_hz, dtype=phase.dtype)
    reset_flag = jnp.asarray(reset_flag, dtype=phase.dtype)

    phase = jnp.asarray(phase, dtype=freq_hz.dtype)
    freq_smooth = jnp.asarray(freq_smooth, dtype=freq_hz.dtype)
    smooth_coeff = jnp.clip(smooth_coeff, 0.0, 1.0)

    # One-pole smoothing
    freq_smooth_next = freq_smooth + smooth_coeff * (freq_hz - freq_smooth)

    # Phase increment
    inc = freq_smooth_next / jnp.maximum(sample_rate, 1e-12)

    # Phase wrap
    phase_raw = phase + inc
    phase_wrapped = phase_raw - jnp.floor(phase_raw)

    # Reset
    reset_mask = jnp.where(reset_flag > 0.5, 1.0, 0.0)
    phase_next = phase_wrapped * (1.0 - reset_mask) + 0.0 * reset_mask

    # Output with offset
    y_raw = phase_next + phase_offset
    y = y_raw - jnp.floor(y_raw)

    new_state = (phase_next, freq_smooth_next)
    return y, new_state


# ============================================================================
# 4. phasor_core_process
# ============================================================================

@jax.jit
def phasor_core_process(
    x,
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
):
    """
    Process vector of (freq_hz, reset_flag) using lax.scan.

    Inputs:
        x      : tuple (freq_buf, reset_buf)
                 each shape = (T,)
        state  : (phase, freq_smooth)
        params : (sample_rate, phase_offset, smooth_coeff, interp_mode)

    Outputs:
        phase_buf   : (T,)
        final_state : (phase, freq_smooth)
    """

    freq_buf, reset_buf = x

    freq_buf = jnp.asarray(freq_buf)
    reset_buf = jnp.asarray(reset_buf)
    reset_buf = jnp.broadcast_to(reset_buf, freq_buf.shape)

    init_state = state

    def body(carry, xs):
        freq_t, reset_t = xs
        y_t, next_state = phasor_core_tick((freq_t, reset_t), carry, params)
        return next_state, y_t

    final_state, phase_buf = lax.scan(body, init_state, (freq_buf, reset_buf))
    return phase_buf, final_state


# ============================================================================
# 5. Smoke test + plotting + optional listening
# ============================================================================

if __name__ == "__main__":
    import numpy as onp
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

    sr = 48000.0
    duration = 0.01
    T = int(duration * sr)

    freq_buf = jnp.full((T,), 440.0, dtype=jnp.float32)
    reset_buf = jnp.zeros_like(freq_buf)

    state, params = phasor_core_init(
        sample_rate=sr,
        initial_phase=0.0,
        initial_freq_hz=440.0,
        phase_offset=0.0,
        smooth_coeff=0.01,
        interp_mode=1,
        dtype=jnp.float32,
    )

    phase_buf, state_out = phasor_core_process(
        (freq_buf, reset_buf),
        state,
        params
    )

    phase_np = onp.asarray(phase_buf)
    t = onp.arange(T) / sr

    # Plot phasor phase + sine
    sine_np = onp.sin(2 * onp.pi * phase_np)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(t, phase_np)
    axs[0].set_title("phasor_core phase output")
    axs[0].set_ylabel("phase")

    axs[1].plot(t, sine_np)
    axs[1].set_title("sine derived from phasor")
    axs[1].set_ylabel("amplitude")

    plt.tight_layout()
    plt.show()

    if HAVE_SD:
        print("Playing sine test tone…")
        sd.play(sine_np * 0.2, samplerate=int(sr), blocking=True)
        print("Done.")
    else:
        print("sounddevice not installed; skipping audio output.")
```

---

# ✅ NEXT MODULES TO BUILD

The natural continuation now is:

1. **ddsp_sine_from_phase**
   *Use phase → sine with optional BLEP/BLAMP correction, amplitude, DC correction.*

2. **ddsp_saw_blep_from_phasor**
   *BLEP-corrected saw from phase.*

3. **ddsp_square_blep_from_phasor**
   *BLEP-corrected band-limited square.*

4. **ddsp_pulse_blep_from_phasor**
   *Pulse wave with variable duty cycle and BLEP transitions.*

5. **ddsp_table_core**
   *Full wavetable table reader with interpolation kernels.*

Tell me which one you want:

👉 **“Generate ddsp_sine_from_phase.py”**
👉 **“Generate ddsp_saw_blep_from_phasor.py”**
👉 **“Generate ddsp_square_blep_from_phasor.py”**
👉 **“Generate ddsp_pulse_blep_from_phasor.py”**
👉 **“Generate ddsp_table_core.py”**

I will generate the full GDSP-style module instantly.

    Initialize linear ramp smoother.

    The ramp smoother tracks a value from current_value to target_value over
    duration_samples ticks. A new ramp is triggered whenever the incoming
    target (x) changes significantly or when the previous ramp finishes.

    Args:
        initial_value    : starting output value
        initial_target   : initial target to ramp toward
        duration_samples : number of samples to reach the target
        dtype            : JAX dtype

    Returns:
        state  : (current_value, target_value, increment, remaining_samples)
        params : (duration_samples,)
    """
    cv = jnp.asarray(initial_value, dtype=dtype)
    tv = jnp.asarray(initial_target, dtype=dtype)
    dur = jnp.asarray(duration_samples, dtype=dtype)
    dur = jnp.maximum(dur, 1.0)

    inc = (tv - cv) / dur
    remaining = dur

    state = (cv, tv, inc, remaining)
    params = (dur,)
    return state, params


def ramp_smoother_update_state(
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Placeholder update_state for ramp smoother.
    Ramp smoothing is driven by tick() based on target inputs.
    """
    return state


@jax.jit
def ramp_smoother_tick(
    x: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Linear ramp smoother single tick.

    Inputs:
        x      : new desired target value at this sample
        state  : (current_value, target_value, increment, remaining_samples)
        params : (duration_samples,)

    Behavior:
        - If remaining_samples <= 0 or the target has changed by more than
          a small threshold, start a new ramp from current_value to x over
          duration_samples ticks.
        - Otherwise, continue existing ramp.

    All control flow uses jnp.where masks, no Python branching.
    """
    (current_value, target_value, increment, remaining) = state
    (duration_samples,) = params

    x = jnp.asarray(x, dtype=current_value.dtype)
    duration_samples = jnp.asarray(duration_samples, dtype=current_value.dtype)
    duration_samples = jnp.maximum(duration_samples, 1.0)

    # Condition for starting a new ramp
    diff = jnp.abs(x - target_value)
    changed = diff > 1e-12
    finished = remaining <= 0.5

    changed_mask = jnp.where(changed, 1.0, 0.0)
    finished_mask = jnp.where(finished, 1.0, 0.0)
    new_ramp_flag = jnp.maximum(changed_mask, finished_mask)

    # Candidate new increment/target if ramp is (re)started
    inc_candidate = (x - current_value) / duration_samples
    increment_next = jnp.where(new_ramp_flag > 0.5, inc_candidate, increment)
    target_next = jnp.where(new_ramp_flag > 0.5, x, target_value)
    remaining_candidate = jnp.where(new_ramp_flag > 0.5,
                                    duration_samples,
                                    jnp.maximum(remaining - 1.0, 0.0))

    # Advance value
    value_candidate = current_value + increment_next

    # Determine whether current ramp is finished after this step
    finished_after = remaining_candidate <= 1.0
    finished_after_mask = jnp.where(finished_after, 1.0, 0.0)

    # Output is either the ramped value or the target when finished
    y = jnp.where(finished_after_mask > 0.5, target_next, value_candidate)

    current_value_next = y
    remaining_next = jnp.where(finished_after_mask > 0.5, 0.0, remaining_candidate)

    new_state = (current_value_next, target_next, increment_next, remaining_next)
    return y, new_state


@jax.jit
def ramp_smoother_process(
    xs: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Process a buffer with the ramp smoother.

    Args:
        xs     : buffer of target values, shape (T,)
        state  : (current_value, target_value, increment, remaining_samples)
        params : (duration_samples,), scalar or shape (T,)

    Returns:
        ys         : ramped buffer, shape (T,)
        final_state
    """
    xs = jnp.asarray(xs)
    (duration_samples,) = params
    duration_samples = jnp.asarray(duration_samples, dtype=xs.dtype)
    duration_samples = jnp.broadcast_to(duration_samples, xs.shape)

    init_state = state

    def body(carry, xs_t):
        st = carry
        x_t, dur_t = xs_t
        y_t, st_next = ramp_smoother_tick(x_t, st, (dur_t,))
        return st_next, y_t

    final_state, ys = lax.scan(body, init_state, (xs, duration_samples))
    return ys, final_state


# =============================================================================
# 3. Allpass smoother
# =============================================================================

def allpass_smoother_init(
    initial_value: float,
    a: float,
    *,
    dtype=jnp.float32,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray]]:
    """
    Initialize allpass smoother.

    Args:
        initial_value : starting output and previous input value
        a             : allpass coefficient (|a| < 1 recommended)
        dtype         : JAX dtype

    Returns:
        state  : (prev_x, prev_y)
        params : (a,)
    """
    v = jnp.asarray(initial_value, dtype=dtype)
    prev_x = v
    prev_y = v
    a_arr = jnp.asarray(a, dtype=dtype)
    state = (prev_x, prev_y)
    params = (a_arr,)
    return state, params


def allpass_smoother_update_state(
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Placeholder update_state for allpass smoother.
    The state is evolved in tick(), so this is a pass-through.
    """
    return state


@jax.jit
def allpass_smoother_tick(
    x: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    First-order allpass smoother single tick.

    y[n] = a * x[n] + prev_x - a * prev_y

    Args:
        x      : input sample
        state  : (prev_x, prev_y)
        params : (a,)

    Returns:
        y        : smoothed output
        new_state: (prev_x_next, prev_y_next)
    """
    prev_x, prev_y = state
    (a,) = params

    x = jnp.asarray(x)
    prev_x = jnp.asarray(prev_x, dtype=x.dtype)
    prev_y = jnp.asarray(prev_y, dtype=x.dtype)
    a = jnp.asarray(a, dtype=x.dtype)

    a = jnp.clip(a, -0.999, 0.999)

    y = a * x + prev_x - a * prev_y

    new_state = (x, y)
    return y, new_state


@jax.jit
def allpass_smoother_process(
    xs: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Process a buffer with the first-order allpass smoother.

    Args:
        xs     : input buffer, shape (T,)
        state  : (prev_x, prev_y)
        params : (a,), scalar or shape (T,)

    Returns:
        ys         : smoothed buffer, shape (T,)
        final_state
    """
    xs = jnp.asarray(xs)
    (a,) = params
    a = jnp.asarray(a, dtype=xs.dtype)
    a = jnp.broadcast_to(a, xs.shape)

    init_state = state

    def body(carry, xs_t):
        st = carry
        x_t, a_t = xs_t
        y_t, st_next = allpass_smoother_tick(x_t, st, (a_t,))
        return st_next, y_t

    final_state, ys = lax.scan(body, init_state, (xs, a))
    return ys, final_state


# =============================================================================
# 4. Smoke tests, plot, listen
# =============================================================================

if __name__ == "__main__":
    import numpy as onp
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

    sr = 48000
    T = 2000

    # ------------------------------
    # One-pole test: smoothing a step
    # ------------------------------
    step = jnp.concatenate([
        jnp.zeros((T // 2,), dtype=jnp.float32),
        jnp.ones((T - T // 2,), dtype=jnp.float32),
    ])

    op_state, op_params = onepole_smoother_init(initial_value=0.0, alpha=0.05)
    step_smoothed, op_state_out = onepole_smoother_process(step, op_state, op_params)

    # ------------------------------
    # Ramp test: ramp from 0 to 1 and back
    # ------------------------------
    ramp_targets = jnp.concatenate([
        jnp.zeros((T // 4,), dtype=jnp.float32),
        jnp.ones((T // 2,), dtype=jnp.float32),
        jnp.zeros((T - 3 * T // 4,), dtype=jnp.float32),
    ])

    rp_state, rp_params = ramp_smoother_init(
        initial_value=0.0,
        initial_target=0.0,
        duration_samples=200.0,
    )
    ramp_out, rp_state_out = ramp_smoother_process(ramp_targets, rp_state, rp_params)

    # ------------------------------
    # Allpass test: smooth white noise
    # ------------------------------
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, (T,), dtype=jnp.float32)

    ap_state, ap_params = allpass_smoother_init(initial_value=0.0, a=0.7)
    noise_smoothed, ap_state_out = allpass_smoother_process(noise, ap_state, ap_params)

    # ------------------------------
    # Plot results
    # ------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(onp.asarray(step), label="step")
    axs[0].plot(onp.asarray(step_smoothed), label="one-pole smoothed")
    axs[0].set_title("One-pole smoother")
    axs[0].legend()

    axs[1].plot(onp.asarray(ramp_targets), label="targets")
    axs[1].plot(onp.asarray(ramp_out), label="ramp output")
    axs[1].set_title("Ramp smoother")
    axs[1].legend()

    axs[2].plot(onp.asarray(noise), alpha=0.3, label="noise")
    axs[2].plot(onp.asarray(noise_smoothed), label="allpass smoothed")
    axs[2].set_title("Allpass smoother")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    # ------------------------------
    # Listen example (optional)
    # ------------------------------
    if HAVE_SD:
        print("Playing noise vs allpass-smoothed noise...")
        # Stack noise and smoothed noise sequentially
        playback = onp.concatenate([
            onp.asarray(noise) * 0.1,
            onp.asarray(noise_smoothed) * 0.1,
        ], axis=0)
        sd.play(playback, samplerate=sr, blocking=True)
        print("Done.")
    else:
        print("sounddevice not available; skipping audio playback.")
```

---

Next natural steps after this module:

* Wire **onepole_smoother** into `phasor_core` for frequency smoothing.
* Use **ramp_smoother** for continuous control changes (cutoff, gain, etc.).
* Use **allpass_smoother** for fractional delay smoothing in delay/reverb cores.
