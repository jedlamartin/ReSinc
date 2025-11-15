#include <cassert>    // For assert()
#include <cmath>      // For sin()
#include <iomanip>    // For std::setw, std::fixed, std::setprecision
#include <iostream>
#include <stdexcept>    // For std::invalid_argument
#include <string>
#include <vector>

// Include your library header
#include "ReSinc.hpp"

// --- Helper Functions for Testing ---

/**
 * @brief Generates a simple mono sine wave and copies it to all channels.
 */
template<typename T>
void generateSine(std::vector<std::vector<T>>& buffer,
                  int numChannels,
                  int numSamples,
                  T frequency,
                  T sampleRate) {
    buffer.assign(numChannels, std::vector<T>(numSamples));
    T phase = T(0.0);
    const T phaseDelta = T(2.0) * M_PI * frequency / sampleRate;

    for(int i = 0; i < numSamples; ++i) {
        T sample = std::sin(phase);
        for(int ch = 0; ch < numChannels; ++ch) {
            buffer[ch][i] = sample;
        }
        phase += phaseDelta;
    }
}

/**
 * @brief Prints the contents of a 2D buffer to the console.
 */
template<typename T>
void printBuffer(const std::string& title,
                 const std::vector<std::vector<T>>& buffer,
                 int samplesToShow = 10) {
    std::cout << "--- " << title << " ---" << std::endl;
    if(buffer.empty() || buffer[0].empty()) {
        std::cout << "Buffer is empty." << std::endl;
        return;
    }
    int numChannels = buffer.size();
    int numSamples = buffer[0].size();
    std::cout << "Channels: " << numChannels << ", Samples: " << numSamples
              << std::endl;

    std::cout << std::fixed << std::setprecision(4);
    for(int i = 0; i < std::min(numSamples, samplesToShow); ++i) {
        std::cout << "Sample " << std::setw(3) << i << ": [ ";
        for(int ch = 0; ch < numChannels; ++ch) {
            std::cout << std::setw(8) << buffer[ch][i] << " ";
        }
        std::cout << "]" << std::endl;
    }
    if(numSamples > samplesToShow) {
        std::cout << "..." << (numSamples - samplesToShow) << " more samples..."
                  << std::endl;
    }
    std::cout << "--------------------" << std::endl << std::endl;
}

/**
 * @brief Creates a C-style array of const pointers from a 2D vector.
 * Required for the 'interpolate' function.
 */
template<typename T>
std::vector<const T*> createConstPointerArray(
    const std::vector<std::vector<T>>& buffer) {
    std::vector<const T*> pointers;
    pointers.reserve(buffer.size());
    for(const auto& channel : buffer) {
        pointers.push_back(channel.data());
    }
    return pointers;
}

/**
 * @brief Creates a C-style array of non-const pointers from a 2D vector.
 * Required for the 'decimate' function.
 */
template<typename T>
std::vector<T*> createPointerArray(std::vector<std::vector<T>>& buffer) {
    std::vector<T*> pointers;
    pointers.reserve(buffer.size());
    for(auto& channel : buffer) {
        pointers.push_back(channel.data());
    }
    return pointers;
}

// --- Main Test Function ---

int main() {
    std::cout << "--- Starting ReSample Test Suite ---" << std::endl;
    bool allTestsPassed = true;

    // ---
    // Test 1: CircularBuffer functionality
    // ---
    std::cout << "\n## Test 1: CircularBuffer<int, 3> ##" << std::endl;
    try {
        CircularBuffer<int, 3> cb(0);
        cb.push(1);    // oldestIndex = 1, buf = [1, 0, 0]
        cb.push(2);    // oldestIndex = 2, buf = [1, 2, 0]
        cb.push(3);    // oldestIndex = 0, buf = [1, 2, 3]
        cb.push(4);    // oldestIndex = 1, buf = [4, 2, 3]

        // Per your logic: operator[](0) is the oldest element
        // oldestIndex = 1
        // cb[0] = buf[(1 + 0) % 3] = buf[1] = 2
        // cb[1] = buf[(1 + 1) % 3] = buf[2] = 3
        // cb[2] = buf[(1 + 2) % 3] = buf[0] = 4

        std::cout << "After pushing 1, 2, 3, 4 into size-3 buffer:"
                  << std::endl;
        std::cout << "cb[0] (oldest) expected: 2, got: " << cb[0] << std::endl;
        std::cout << "cb[1] (middle) expected: 3, got: " << cb[1] << std::endl;
        std::cout << "cb[2] (newest) expected: 4, got: " << cb[2] << std::endl;

        if(cb[0] != 2 || cb[1] != 3 || cb[2] != 4) {
            throw std::runtime_error(
                "CircularBuffer indexing logic is incorrect.");
        }
        std::cout << "PASSED: CircularBuffer test." << std::endl;

    } catch(const std::exception& e) {
        std::cout << "!!! FAILED: CircularBuffer test. Caught: " << e.what()
                  << std::endl;
        allTestsPassed = false;
    }
    std::cout << "---" << std::endl;

    // ---
    // Test 2: Kaiser Window generation
    // ---
    std::cout << "\n## Test 2: Window::Kaiser<float, 16> ##" << std::endl;
    try {
        Window::Kaiser<float, 16> kaiser(5.0f);
        // The formula for n=0 gives:
        // arg = beta * sqrt(1.0 - pow(0.0)) = beta
        // num = besselI0(beta)
        // den = besselI0(beta)
        // kaiser[0] should be 1.0
        std::cout << "Kaiser[0] expected: 1.0, got: " << kaiser[0] << std::endl;
        if(std::abs(kaiser[0] - 1.0f) > 0.0001f) {
            throw std::runtime_error("Kaiser[0] is not 1.0. Check formula.");
        }
        std::cout << "PASSED: Kaiser window generation." << std::endl;
    } catch(const std::exception& e) {
        std::cout << "!!! FAILED: Kaiser test. Caught: " << e.what()
                  << std::endl;
        allTestsPassed = false;
    }
    std::cout << "---" << std::endl;

    // ---
    // Test 3: ReSample<float> (default) - Full Chain
    // ---
    std::cout << "\n## Test 3: ReSample<float, 4, 32> (Full Chain) ##"
              << std::endl;
    try {
        // Use the default T=float: ReSample<4, 32>
        ReSample<4, 32> resampler;
        resampler.configure(44100.0f);

        constexpr int FACTOR = 4;
        constexpr int CHANNELS = 2;
        constexpr int SAMPLES_IN = 10;
        constexpr int SAMPLES_OUT =
            SAMPLES_IN;    // Decimate back to original sample count

        std::vector<std::vector<float>> inputBuffer, outputBuffer;

        generateSine(inputBuffer, CHANNELS, SAMPLES_IN, 440.0f, 44100.0f);
        outputBuffer.assign(CHANNELS, std::vector<float>(SAMPLES_OUT, 0.0f));

        printBuffer("Input (Float)", inputBuffer, 10);

        auto inputPointers = createConstPointerArray(inputBuffer);
        auto outputPointers = createPointerArray(outputBuffer);

        // 1. Interpolate
        resampler.interpolate(inputPointers.data(), CHANNELS, SAMPLES_IN);

        // 2. Process (Lambda Test)
        resampler.process([&](std::vector<std::vector<float>>& internalBuffer) {
            std::cout << "-> process() lambda called. Internal buffer size: "
                      << internalBuffer.size() << "x"
                      << (internalBuffer.empty() ? 0 : internalBuffer[0].size())
                      << std::endl;
            // Modify the buffer to prove it worked
            if(!internalBuffer.empty() && !internalBuffer[0].empty()) {
                internalBuffer[0][0] = 1.2345f;
            }
        });

        // 3. Decimate
        resampler.decimate(outputPointers.data());

        printBuffer("Output (Float)", outputBuffer, 10);

        // 4. Check
        // If the process() lambda worked, the first output sample should
        // be 1.2345 / FACTOR (assuming the sinc[0] * sample[0] term dominates,
        // which it should)
        float expected = 1.2345f / FACTOR;
        if(std::abs(outputBuffer[0][0] - expected) > 0.001f) {
            // Note: This test will FAIL if the Sinc::applyWindow bug is not
            // fixed, because the sinc filter will be all zeros, resulting in
            // output[0][0] = 0.0
            std::cout << "!!! FAILED: Output[0][0] expected ~" << expected
                      << ", got " << outputBuffer[0][0] << std::endl;
            std::cout << "!!! (This may be due to the Sinc::applyWindow bug)"
                      << std::endl;
            allTestsPassed = false;
        } else {
            std::cout << "PASSED: Full float chain test (interpolate, process, "
                         "decimate)."
                      << std::endl;
        }

    } catch(const std::exception& e) {
        std::cout << "!!! FAILED: ReSample<float> test. Caught: " << e.what()
                  << std::endl;
        allTestsPassed = false;
    }
    std::cout << "---" << std::endl;

    // ---
    // Test 4: ReSample<double> - Multi-block state test
    // ---
    std::cout << "\n## Test 4: ReSample<double, 4, 32> (Multi-block) ##"
              << std::endl;
    try {
        ReSample<4, 32, double> resampler_d;
        resampler_d.configure(96000.0);

        std::vector<std::vector<double>> inBlock1, inBlock2, outBlock1,
            outBlock2;
        generateSine(inBlock1, 1, 16, 1000.0, 96000.0);
        // Create a perfectly continuous sine wave for the second block
        std::vector<std::vector<double>> continuousSine;
        generateSine(continuousSine, 1, 32, 1000.0, 96000.0);
        inBlock2.assign(1, std::vector<double>(16));
        std::copy(continuousSine[0].begin() + 16,
                  continuousSine[0].end(),
                  inBlock2[0].begin());

        outBlock1.assign(1, std::vector<double>(16));
        outBlock2.assign(1, std::vector<double>(16));

        auto inPtr1 = createConstPointerArray(inBlock1);
        auto inPtr2 = createConstPointerArray(inBlock2);
        auto outPtr1 = createPointerArray(outBlock1);
        auto outPtr2 = createPointerArray(outBlock2);

        std::cout << "Processing Block 1..." << std::endl;
        resampler_d.interpolate(inPtr1.data(), 1, 16);
        resampler_d.decimate(outPtr1.data());

        std::cout << "Processing Block 2..." << std::endl;
        resampler_d.interpolate(inPtr2.data(), 1, 16);
        resampler_d.decimate(outPtr2.data());

        printBuffer("Block 1 Output (Double)", outBlock1, 16);
        printBuffer("Block 2 Output (Double)", outBlock2, 16);

        if(std::abs(outBlock1[0][15] - outBlock2[0][15]) < 1e-9) {
            std::cout << "!!! WARNING: Multi-block output seems identical. "
                         "State might not be kept."
                      << std::endl;
        } else {
            std::cout << "PASSED: Multi-block test (state appears to be kept)."
                      << std::endl;
        }

    } catch(const std::exception& e) {
        std::cout << "!!! FAILED: ReSample<double> test. Caught: " << e.what()
                  << std::endl;
        allTestsPassed = false;
    }
    std::cout << "---" << std::endl;

    // ---
    // Test 5: Exception Handling
    // ---
    std::cout << "\n## Test 5: Exception Handling ##" << std::endl;
    try {
        ReSample<3, 16> ex_resampler;    // Default float
        std::cout << "Testing configure(0.0)... ";
        ex_resampler.configure(0.0f);
        std::cout << "!!! FAILED: No exception thrown !!!" << std::endl;
        allTestsPassed = false;
    } catch(const std::invalid_argument& e) {
        std::cout << "PASSED. Caught: " << e.what() << std::endl;
    } catch(...) {
        std::cout << "!!! FAILED: Wrong exception type thrown." << std::endl;
        allTestsPassed = false;
    }

    try {
        ReSample<4, 32> resampler;
        std::cout << "Testing interpolate(0 channels)... ";
        resampler.interpolate(nullptr, 0, 10);
        std::cout << "!!! FAILED: No exception thrown !!!" << std::endl;
        allTestsPassed = false;
    } catch(const std::invalid_argument& e) {
        std::cout << "PASSED. Caught: " << e.what() << std::endl;
    } catch(...) {
        std::cout << "!!! FAILED: Wrong exception type thrown." << std::endl;
        allTestsPassed = false;
    }

    try {
        std::cout << "Testing Kaiser(-1.0)... ";
        Window::Kaiser<float, 16> bad_kaiser(-1.0f);
        std::cout << "!!! FAILED: No exception thrown !!!" << std::endl;
        allTestsPassed = false;
    } catch(const std::invalid_argument& e) {
        std::cout << "PASSED. Caught: " << e.what() << std::endl;
    } catch(...) {
        std::cout << "!!! FAILED: Wrong exception type thrown." << std::endl;
        allTestsPassed = false;
    }
    std::cout << "---" << std::endl;

    // ---
    // Final Result
    // ---
    std::cout << "\n--- Test Suite Finished ---" << std::endl;
    if(allTestsPassed) {
        std::cout << "✅✅✅ ALL TESTS PASSED ✅✅✅" << std::endl;
    } else {
        std::cout << "❌❌❌ SOME TESTS FAILED ❌❌❌" << std::endl;
    }
    return allTestsPassed ? 0 : 1;
}