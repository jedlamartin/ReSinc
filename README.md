# ReSinc
[![CI](https://github.com/jedlamartin/ReSinc/actions/workflows/ci.yml/badge.svg)](https://github.com/jedlamartin/ReSinc/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Language](https://img.shields.io/badge/language-C%2B%2B17-blue.svg)

`ReSinc` is a lightweight, real-time safe, header-only C++ library for audio oversampling and asynchronous resampling.

It is designed for real-time audio applications (like VST/AU plugins) where running non-linear processes (distortion, saturation, compression) at a higher sample rate is necessary to prevent aliasing artifacts. The library uses a pre-calculated, windowed Sinc filter for high-fidelity interpolation and decimation.

## Features

* **Real-Time Safe:** All memory is pre-allocated during configuration. Processing functions are guaranteed not to allocate heap memory, lock, or block.
* **Flexible I/O:** Supports multiple data formats directly:
    * **Standard Containers:** `std::vector<float>`, `std::vector<std::vector<float>>`, `std::array`.
    * **JUCE Framework:** Directly accepts `juce::AudioBuffer<float>` / `juce::AudioBuffer<double>`.
    * **Raw Pointers:** Standard `float**` arrays.
* **High-Quality:** Uses a Kaiser-windowed Sinc filter for near-perfect reconstruction.
* **Header-Only:** Single file `ReSinc.hpp`. No linking required.

---

## Oversampler

The `Oversampler` is optimized for fixed integer-ratio upsampling (2x, 4x, etc.). It is the ideal choice for internal DSP effects.

### Quick Start

```cpp
// <TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>
Oversampler<float, 4, 32> oversampler;

void setup() {
    oversampler.configure(44100.0f, 2, 512);
}

void process(std::vector<std::vector<float>>& buffer) {
    oversampler.interpolate(buffer); // Input -> 4x Internal Buffer
    
    oversampler.process([&](std::vector<std::vector<float>>& upsampled) {
        // High-rate processing happens here
    });

    oversampler.decimate(buffer);    // 4x Internal Buffer -> Output
}
```

### API Summary
* `interpolate(input)`: Upsamples input data.
* `process(callback)`: Provides access to the high-rate internal buffer.
* `decimate(output)`: Applies anti-aliasing and downsamples to output.
* **Latency**: Fixed at `2 * SINC_RADIUS` samples.

## Resampler
The `Resampler` handles asynchronous rate conversion where the ratio is not an integer (e.g., converting a 48kHz file for 44.1kHz playback).

### Quick Start
```cpp
// <TYPE, SINC_RADIUS, PHASE_RESOLUTION>
Resampler<float, 32, 256> resampler;

void setup() {
    // 48kHz -> 44.1kHz
    resampler.configure(48000.0f, 44100.0f, 2, 512);
}

void process(std::vector<std::vector<float>>& input, 
             std::vector<std::vector<float>>& output) {
    
    // Returns the actual number of output samples produced
    int produced = resampler.resample(input, output);
}
```

### API Summary
* `resample(input, output)`: Performs the asynchronous conversion.
* `int` **Return Type**: Because fractional ratios can result in varying samples per block, the return value indicates how many samples in the output buffer are valid.
* **Gain Scaling**: Automatically applies compensation when downsampling to maintain unity gain.
* **Latency**: Fixed at `SINC_RADIUS` samples (at input rate).

## Latency & Phase
This library uses **linear-phase** (symmetric) filters.

* **Oversampler Latency**: `2 * SINC_RADIUS` (input samples).
* **Resampler Latency**: `SINC_RADIUS` (input samples).

To maintain perfect alignment in a DAW, you must report these values to the host for Latency Compensation.

## Installation
This is a **header-only** library.

1. Copy `include/ReSinc.hpp` to your project.

2. `#include "ReSinc.hpp"`

3. Compile with **C++17** or higher.

4. (Optional) For best performance, enable SIMD/Vectorization flags (`-O3 -march=native`).

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
