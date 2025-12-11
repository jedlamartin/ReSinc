# ReSinc
[![CI](https://github.com/jedlamartin/ReSinc/actions/workflows/ci.yml/badge.svg)](https://github.com/jedlamartin/ReSinc/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Language](https://img.shields.io/badge/language-C%2B%2B17-blue.svg)

`ReSinc` is a lightweight, real-time safe, header-only C++ library for audio oversampling.

It is designed for real-time audio applications (like VST/AU plugins) where running non-linear processes (distortion, saturation, compression) at a higher sample rate is necessary to prevent aliasing artifacts. The library uses a pre-calculated, windowed Sinc filter for high-fidelity interpolation and decimation.

## Features

* **Real-Time Safe:** All memory is pre-allocated during configuration. The real-time processing functions (`interpolate`, `decimate`, `process`) are guaranteed not to allocate heap memory, lock, or block.
* **Flexible I/O:** Supports multiple data formats directly:
    * **Standard Containers:** `std::vector<float>`, `std::vector<std::vector<float>>`, `std::array`.
    * **JUCE Framework:** Directly accepts `juce::AudioBuffer<float>` / `juce::AudioBuffer<double>`.
    * **Raw Pointers:** Standard `float**` arrays.
* **High-Quality:** Uses a Kaiser-windowed Sinc filter for near-perfect reconstruction.
* **Header-Only:** Single file `ReSinc.hpp`. No linking required.
* **Configurable:**
    * `TYPE`: Sample type (`float` or `double`).
    * `OVERSAMPLE_FACTOR`: e.g., `2`, `4`, `8`, `16`.
    * `SINC_RADIUS`: Controls filter steepness and latency.

---

## Quick Start

Here is a minimal example using `std::vector` to apply 4x oversampling to a simple distortion effect.

```cpp
#include "ReSinc.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

// 1. Define your oversampler instance
//    <float, 4x Oversampling, Sinc Radius of 32>
Oversampler<float, 4, 32> oversampler;

// 2. Call configure() once before processing begins
void setupAudio() {
    float sampleRate = 44100.0f;
    int maxChannels = 2;
    int maxBlockSize = 512;
    
    oversampler.configure(sampleRate, maxChannels, maxBlockSize);
}

// 3. Run in your real-time audio loop
void processBlock(std::vector<std::vector<float>>& input, 
                  std::vector<std::vector<float>>& output) {
    
    // 3a. Upsample: Input -> Internal Buffer
    //     (Supports std::vector, juce::AudioBuffer, or float**)
    oversampler.interpolate(input);

    // 3b. Process: Operate on the high-resolution buffer
    //     The callback receives the upsampled data (4x larger size)
    oversampler.process([&](std::vector<std::vector<float>>& upsampledData) {
        
        for (auto& channel : upsampledData) {
            for (auto& sample : channel) {
                // Apply non-linear process (e.g., Hard Clip)
                // Safe from aliasing because we are at 4x rate
                sample = std::max(-1.0f, std::min(1.0f, sample * 1.5f));
            }
        }
    });

    // 3c. Downsample: Internal Buffer -> Output
    //     Automatically applies anti-aliasing filter
    oversampler.decimate(output);
}
```
## API

### Template Parameters
`template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>`
* **`TYPE`**: The sample type (`float` or `double`).
* **`OVERSAMPLE_FACTOR`**: Integer factor (e.g., `2`, `4`, `8`).
* **`SINC_RADIUS`**: Half-length of the Sinc filter. 
    * Example: `32` taps per side.
    * Latency = `2 * SINC_RADIUS` samples (at input rate).

### Configuration

#### `void configure(TYPE sampleRate, int maxChannels, int maxBlockSize)`
Pre-allocates all internal memory. **Must** be called once (from a non-real-time thread) before processing.

### Real-Time Processing

The `interpolate`, `process`, and `decimate` methods provide overloads for three categories of data:

1.  **JUCE Types:** `juce::AudioBuffer<T>`
2.  **Standard Containers:** `std::vector<T>`, `std::vector<std::vector<T>>`
3.  **Raw Pointers:** `T**` / `T* const*`

#### 1. Interpolate (Input -> Internal)
Upsamples input data and fills the internal buffer.

```cpp
// JUCE Support
void interpolate(const juce::AudioBuffer<TYPE>& buffer);

// STL Support (Multi-channel & Single-channel)
void interpolate(const std::vector<std::vector<TYPE>>& buffer);
void interpolate(const std::vector<TYPE>& buffer);

// Raw Pointer Support
void interpolate(const TYPE* const* ptrToBuffers, int numChannels, int numSamples);
```
#### 2. Process (Callback)
Provides direct access to the upsampled data via a callback/lambda.

```cpp
// 1. Vector Access (Most Common)
//    Provides the internal buffer as a vector of vectors
oversampler.process([](std::vector<std::vector<TYPE>>& data) { ... });

// 2. JUCE Wrapper Access
//    Wraps the internal pointers in a temporary AudioBuffer for convenience.
//    (Only enables if passed a JUCE type)
oversampler.process([](juce::AudioBuffer<TYPE> data) { ... });

// 3. Per-Channel Access
//    Calls your lambda once for each channel vector
oversampler.process([](std::vector<TYPE>& channelData) { ... });
```
#### 3. Decimate (Internal -> Output)
Downsamples the internal buffer and writes to the output.

```cpp
// JUCE Support
void decimate(juce::AudioBuffer<TYPE>& buffer);

// STL Support
void decimate(std::vector<std::vector<TYPE>>& buffer);
void decimate(std::vector<TYPE>& buffer);

// Raw Pointer Support
void decimate(TYPE* const* ptrToBuffers, int numChannels, int numSamples);
```
## Latency

This library uses a linear-phase (symmetric) Sinc filter. By design, this introduces a known, fixed processing latency.

The total round-trip latency (from `interpolate` to `decimate`) is:

**`2 * SINC_RADIUS`** samples.

* **Example:** If you use `Oversampler<float, 4, 32>`, the latency is `2 * 32 = 64` samples.
* **Example:** If you use `Oversampler<float, 8, 64>`, the latency is `2 * 64 = 128` samples.

You must report this latency to the host (e.g., via `setLatencySamples()` in JUCE/VST) for automatic latency compensation.

---

## How to Use (Installation)

This is a header-only library.

1.  Copy `ReSinc.hpp` into your project's `include` directory.
2.  `#include "ReSinc.hpp"` in your C++ files.
3.  If you are compiling on Windows, you may need to define `_USE_MATH_DEFINES` before including the header.

---

## Building the Tests

The repository includes a `tests/tests.cpp` file for validation. You can compile it with:

```bash
g++ tests/tests.cpp -o run_tests -std=c++17 -O3
./run_tests
```
## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
