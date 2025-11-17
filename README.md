# Oversampler
[![CI](https://github.com/jedlamartin/ReSinc/actions/workflows/ci.yml/badge.svg)](https://github.com/jedlamartin/ReSinc/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Language](https://img.shields.io/badge/language-C%2B%2B-blue.svg)

`Oversampler` is a lightweight, real-time safe, header-only C++ library for high-quality, anti-aliased audio oversampling.

It's designed for use in real-time audio applications (like VST/AU plugins) where running a non-linear process (distortion, saturation, etc.) at a higher sample rate is necessary to prevent aliasing artifacts. The library uses a pre-calculated, windowed Sinc filter for interpolation and decimation, ensuring high-fidelity results.

## Features

* **Real-Time Safe:** All memory is pre-allocated on initialization. The real-time processing functions (`interpolate`, `decimate`, `process`) are guaranteed not to allocate heap memory, lock, or block, making them safe for any high-priority audio thread.
* **High-Quality:** Uses a Kaiser-windowed Sinc filter for "perfect" reconstruction and anti-aliasing.
* **Simple Workflow:** Provides a clear `interpolate()` -> `process()` -> `decimate()` workflow.
* **Header-Only:** Just drop `Oversampler.hpp` into your project and include it.
* **Flexible & Template-Based:** Fully configurable via template parameters:
    * `TYPE`: Sample type (`float` or `double`).
    * `OVERSAMPLE_FACTOR`: The factor to oversample by (e.g., `2`, `4`, `8`).
    * `SINC_RADIUS`: The "quality" of the filter, which also determines latency.

---

## Quick Start

Here is a minimal example of how to use `Oversampler` to apply 4x oversampling to a simple distortion effect.

```cpp
#include "ReSinc.hpp"
#include <vector>
#include <cmath>

// 1. Define your oversampler instance
//    <float, 4x Oversampling, Sinc Radius of 32>
Oversampler<float, 4, 32> oversampler;

// 2. Call configure() once before processing begins
//    (e.g., in your plugin's prepareToPlay())
void setupAudio() {
    float sampleRate = 44100.0f;
    int maxChannels = 2;
    int maxBlockSize = 1024;
    
    oversampler.configure(sampleRate, maxChannels, maxBlockSize);
}

// 3. Run in your real-time audio callback (e.g., processBlock())
void processAudio(const float* const* input, float* const* output, int numChannels, int numSamples) {
    
    // 3a. Upsample the audio into the internal buffer
    oversampler.interpolate(input, numChannels, numSamples);

    // 3b. Process the high-resolution internal buffer
    //     This lambda is called immediately.
    oversampler.process([&](std::vector<std::vector<float>>& internalBuffer) {
        
        // 'internalBuffer' is now 4x larger
        int oversampledSamples = numSamples * 4;

        for (int ch = 0; ch < numChannels; ++ch) {
            for (int s = 0; s < oversampledSamples; ++s) {
                // Apply a non-linear process (e.g., distortion)
                // This would alias badly at 1x, but is safe at 4x
                internalBuffer[ch][s] = std::tanh(internalBuffer[ch][s] * 2.0f);
            }
        }
    });

    // 3c. Downsample back to the original rate.
    //     The Sinc filter automatically anti-aliases the signal.
    oversampler.decimate(output, numChannels, numSamples);
}
```
## API

### Template Parameters
`template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>`
* **`TYPE`**: The sample type to use, e.g., `float` or `double`.
* **`OVERSAMPLE_FACTOR`**: The interpolation factor. `4` means 4x oversampling.
* **`SINC_RADIUS`**: The half-length of the Sinc filter. This controls the filter's quality. A good starting value is `32`. Higher values mean better anti-aliasing but higher CPU cost and latency.

### Public Methods

* **`void configure(TYPE sampleRate, int maxChannels, int maxBlockSize)`**
    Pre-allocates all internal memory. This **must** be called once from a non-real-time thread before any processing begins.

* **`void interpolate(const TYPE* const* ptrToBuffers, int numChannels, int numSamples)`**
    Upsamples a block of audio from `ptrToBuffers` into the internal high-sample-rate buffer.

* **`void process(std::function<void(std::vector<std::vector<TYPE>>&)> processBlock)`**
    Takes a lambda (or other `std::function`) and executes it, giving it direct mutable access to the internal high-sample-rate buffer. This is where you should apply your oversampled processing.

* **`void decimate(TYPE* const* ptrToBuffers, int numChannels, int numSamples)`**
    Downsamples the internal high-sample-rate buffer into the provided `ptrToBuffers`, automatically applying the anti-aliasing filter.

  ---

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
---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
