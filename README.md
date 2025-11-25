# ✅ **ddsp_phasor_core.py**

MODULE NAME:
**ddsp_phasor_core**

DESCRIPTION:
A fully differentiable, pure-JAX **time-base core** that generates wrapped, continuous phase ramps from frequency inputs.
`ddsp_phasor_core` is the central phase generator for GammaJAX DDSP: it drives oscillators, wavetables, LFOs, and other phase-based modules. It implements GDSP-style `init`, `tick`, and `process` functions with internal frequency smoothing and through-zero capable phase accumulation.

INPUTS:

* **x (freq)** : input frequency in Hz (scalar in `tick`, array in `process`)
* **params.dt** : time step in seconds (`1 / sample_rate`)
* **params.alpha** : one-pole smoothing coefficient in `[0,1]` for the frequency

OUTPUTS:

* **y (phase)** : current phase sample, wrapped to `[0,1)`; this is the only output and can be fed into table lookup, sine, etc.

STATE VARIABLES:
`state = (phase, freq_smooth)`

* **phase** : current phase value in `[0,1)` (scalar or broadcastable array)
* **freq_smooth** : smoothed frequency in Hz (same shape as phase)

EQUATIONS / MATH:

Given:

* `x[n]` = input frequency in Hz
* `dt` = time step in seconds
* `alpha` = smoothing coefficient in `[0,1]`

One-pole smoothing of frequency:

* `freq_smooth[n+1] = freq_smooth[n] + alpha * (x[n] − freq_smooth[n])`

Phase integration (through-zero capable):

* `phase_unwrapped[n+1] = phase[n] + freq_smooth[n+1] * dt`

Phase wrapping:

* `phase[n+1] = (phase_unwrapped[n+1]) mod 1.0`
  implemented via `jnp.mod(phase_unwrapped, 1.0)`

Output:

* `y[n] = phase[n+1]`

State update:

* `state[n+1] = (phase[n+1], freq_smooth[n+1])`

through-zero rules:

* `x[n]` (and thus `freq_smooth`) may be **negative**, causing backward phase motion.
* No special branching is needed; `phase_unwrapped` may decrease and `jnp.mod` wraps into `[0,1)`.

phase wrapping rules:

* Always wrap via `jnp.mod(phase_unwrapped, 1.0)` to keep phase in `[0,1)`.

nonlinearities:

* None; everything is linear except for the wrap modulo operation, which is piecewise smooth and differentiable almost everywhere.

interpolation rules:

* `phasor_core` does **not** interpolate signals; it only produces a phase ramp.
* Interpolation of tables or buffers is delegated to `table_core` / `interp_core`.

any time-varying coefficient rules:

* The smoothing coefficient `alpha` is treated as a JAX scalar or array; it can be **time-varying** if passed as such into `process`.
* `dt` is usually constant, but may also be time-varying in principle if broadcastable.

NOTES:

* Stable parameter ranges:

  * `alpha ∈ [0,1]` (clamped inside jit).
  * `dt > 0`. Usually `dt = 1 / sample_rate`.
* Frequencies can be any real values; large values will wrap faster and can alias in downstream oscillators if beyond Nyquist.
* All operations are differentiable with respect to `freq` and `alpha` (modulo the saw-like discontinuity at wrap).

---

## Full `ddsp_phasor_core.py`

```python
"""
ddsp_phasor_core.py

GammaJAX DDSP – Phasor Core
---------------------------

This module implements a fully differentiable, pure-JAX phase generator
(time-base core) in GDSP style:

    phasor_core_init(...)
    phasor_core_update_state(...)
    phasor_core_tick(freq, state, params)
    phasor_core_process(freq_buf, state, params)

The phasor core:
    - Accepts frequency in Hz (scalar or per-sample).
    - Applies one-pole smoothing to the frequency.
    - Integrates phase with time step dt.
    - Wraps phase into [0,1) using jnp.mod.
    - Supports through-zero (negative frequency) behavior naturally.

Design constraints:
    - Pure functional JAX.
    - No classes, no dicts, no dataclasses.
    - State = tuple only (arrays, scalars).
    - tick() returns (y, new_state).
    - process() is a lax.scan wrapper around tick().
    - No Python branching inside @jax.jit.
    - No dynamic allocation or jnp.arange/jnp.zeros inside jit.
    - All shapes determined outside jit.
    - All control flow via jnp.where / lax.cond / lax.scan.
    - Everything jittable and differentiable.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax


# =============================================================================
# 1. GDSP-style API: init / update_state / tick / process
# =============================================================================

def phasor_core_init(
    initial_phase: float,
    initial_freq_hz: float,
    sample_rate: float,
    smooth_alpha: float = 0.0,
    *,
    dtype=jnp.float32,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Initialize phasor core.

    Args:
        initial_phase : starting phase in [0,1)
        initial_freq_hz : initial frequency in Hz
        sample_rate   : sample rate in Hz (used to compute dt = 1 / sample_rate)
        smooth_alpha  : one-pole smoothing coefficient for frequency in [0,1]
                        0  => no smoothing (freq_smooth follows freq exactly)
                        1  => heavy smoothing / slow response
        dtype         : JAX dtype for internal state

    Returns:
        state  : (phase, freq_smooth)
        params : (dt, smooth_alpha)
    """
    phase0 = jnp.asarray(initial_phase, dtype=dtype)
    phase0 = jnp.mod(phase0, 1.0)

    freq0 = jnp.asarray(initial_freq_hz, dtype=dtype)
    dt = jnp.asarray(1.0 / float(sample_rate), dtype=dtype)
    alpha = jnp.asarray(smooth_alpha, dtype=dtype)

    state = (phase0, freq0)
    params = (dt, alpha)
    return state, params


def phasor_core_update_state(
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optional non-IO state update.

    For this simple phasor, smoothing is applied inside tick(),
    so update_state is currently a pass-through.

    Args:
        state  : (phase, freq_smooth)
        params : (dt, smooth_alpha)  # unused here

    Returns:
        new_state: (phase, freq_smooth)
    """
    del params
    return state


@jax.jit
def phasor_core_tick(
    freq_hz: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Single-sample phasor tick.

    Inputs:
        freq_hz : input frequency in Hz (scalar)
        state   : (phase, freq_smooth)
        params  : (dt, smooth_alpha)

    Returns:
        y          : phase sample in [0,1)
        new_state  : (phase_next, freq_smooth_next)
    """
    phase, freq_smooth = state
    dt, alpha = params

    # Cast everything consistently
    freq_hz = jnp.asarray(freq_hz, dtype=phase.dtype)
    dt = jnp.asarray(dt, dtype=phase.dtype)
    alpha = jnp.asarray(alpha, dtype=phase.dtype)
    alpha = jnp.clip(alpha, 0.0, 1.0)

    # One-pole smoothing of frequency
    freq_smooth_next = freq_smooth + alpha * (freq_hz - freq_smooth)

    # Phase update (through-zero capable)
    phase_unwrapped = phase + freq_smooth_next * dt
    phase_next = jnp.mod(phase_unwrapped, 1.0)

    y = phase_next
    new_state = (phase_next, freq_smooth_next)
    return y, new_state


@jax.jit
def phasor_core_process(
    freq_hz_buf: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Process a buffer of frequencies into a buffer of phases.

    Args:
        freq_hz_buf : array of input frequencies in Hz, shape (T,)
        state       : (phase, freq_smooth)
        params      : (dt, smooth_alpha)

    Returns:
        phase_buf    : buffer of phases in [0,1), shape (T,)
        final_state
    """
    freq_hz_buf = jnp.asarray(freq_hz_buf)

    def body(carry, freq_t):
        st = carry
        y_t, st_next = phasor_core_tick(freq_t, st, params)
        return st_next, y_t

    final_state, phase_buf = lax.scan(body, state, freq_hz_buf)
    return phase_buf, final_state


# =============================================================================
# 2. Smoke test, plot, listen
# =============================================================================

if __name__ == "__main__":
    import numpy as onp
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

    print("=== ddsp_phasor_core: smoke test ===")

    # Parameters
    sample_rate = 48000.0
    duration_sec = 0.01  # short for plotting
    N = int(sample_rate * duration_sec)

    # Constant 440 Hz for the smoke test
    freq = 440.0
    freq_buf = jnp.full((N,), freq, dtype=jnp.float32)

    # Initialize phasor
    state, params = phasor_core_init(
        initial_phase=0.0,
        initial_freq_hz=freq,
        sample_rate=sample_rate,
        smooth_alpha=0.05,
        dtype=jnp.float32,
    )

    # Process
    phase_buf, state_out = phasor_core_process(freq_buf, state, params)
    phase_np = onp.asarray(phase_buf)

    # Plot phase ramp
    plt.figure(figsize=(10, 4))
    plt.plot(phase_np, label="phase")
    plt.title("ddsp_phasor_core: phase ramp at 440 Hz")
    plt.xlabel("Sample")
    plt.ylabel("Phase [0,1)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Listen example: convert phase to sine
    duration_sec_audio = 1.0
    N_audio = int(sample_rate * duration_sec_audio)
    freq_buf_audio = jnp.full((N_audio,), freq, dtype=jnp.float32)

    phase_audio, _ = phasor_core_process(freq_buf_audio, state, params)
    # Convert phase to sine wave
    two_pi = 2.0 * jnp.pi
    audio = jnp.sin(two_pi * phase_audio)
    audio_np = onp.asarray(audio) * 0.2  # scale down

    if HAVE_SD:
        print("Playing test sine generated from phasor_core...")
        sd.play(audio_np, samplerate=int(sample_rate), blocking=True)
        print("Done.")
    else:
        print("sounddevice not available; skipping audio playback.")
```

---

Next natural steps you can build on top of this:

* A **high-level phasor oscillator** wrapper that converts `freq_hz` to phase and directly drives:

  * `ddsp_interp_core` / `ddsp_table_core`
  * `ddsp_sine_from_phase`, SAW/SQUARE/PULSE BLEP
* A **multi-voice phasor bank** for additive/harmonic oscillators, all driven by this core.


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
