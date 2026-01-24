#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "ReSinc.hpp"

// --- Helper Functions ---

template<typename TYPE>
void generateSine(std::vector<std::vector<TYPE>>& buffer,
                  int numChannels,
                  int numSamples,
                  TYPE frequency,
                  TYPE sampleRate) {
    buffer.assign(numChannels, std::vector<TYPE>(numSamples));
    TYPE phase = TYPE(0.0);
    const TYPE phaseDelta = TYPE(2.0) * M_PI * frequency / sampleRate;

    for(int i = 0; i < numSamples; ++i) {
        TYPE sample = std::sin(phase);
        for(int ch = 0; ch < numChannels; ++ch) {
            buffer[ch][i] = sample;
        }
        phase += phaseDelta;
    }
}

template<typename TYPE>
void printBuffer(const std::string& title,
                 const std::vector<std::vector<TYPE>>& buffer,
                 int samplesToShow = 10) {
    std::cout << "--- " << title << " ---" << std::endl;
    if(buffer.empty() || buffer[0].empty()) {
        std::cout << "Buffer is empty." << std::endl;
        return;
    }
    int numChannels = (int) buffer.size();
    int numSamples = (int) buffer[0].size();
    std::cout << "Channels: " << numChannels << ", Samples: " << numSamples
              << std::endl;

    std::cout << std::fixed << std::setprecision(6);
    for(int i = 0; i < std::min(numSamples, samplesToShow); ++i) {
        std::cout << "Sample " << std::setw(3) << i << ": [ ";
        for(int ch = 0; ch < numChannels; ++ch) {
            std::cout << std::setw(10) << buffer[ch][i] << " ";
        }
        std::cout << "]" << std::endl;
    }
    if(numSamples > samplesToShow) {
        std::cout << "..." << (numSamples - samplesToShow) << " more samples..."
                  << std::endl;
    }
    std::cout << "--------------------" << std::endl << std::endl;
}

template<typename TYPE>
TYPE calculateMaxError(const std::vector<std::vector<TYPE>>& bufA,
                       const std::vector<std::vector<TYPE>>& bufB) {
    TYPE maxError = TYPE(0.0);
    for(size_t ch = 0; ch < bufA.size(); ++ch) {
        for(size_t i = 0; i < bufA[ch].size(); ++i) {
            TYPE error = std::abs(bufA[ch][i] - bufB[ch][i]);
            if(error > maxError) {
                maxError = error;
            }
        }
    }
    return maxError;
}

// --- Main Test Function ---

int main() {
    std::cout << "--- Starting Oversampler Test Suite ---" << std::endl;
    bool allTestsPassed = true;

    // ---
    // Test 1: CircularBuffer functionality
    // ---
    std::cout << "\n## Test 1: CircularBuffer<int, 3> ##" << std::endl;
    try {
        CircularBuffer<int, 3> cb(0);
        cb.push(1);
        cb.push(2);
        cb.push(3);
        cb.push(4);
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
        constexpr size_t N_KAISER = 16;
        Window::Kaiser<float, N_KAISER> kaiser {5.0f};

        size_t centerIndex = (N_KAISER - 1) / 2;    // = 7

        std::cout << "Kaiser[edge] (kaiser[0]): " << kaiser[0] << std::endl;
        std::cout << "Kaiser[center] (kaiser[" << centerIndex
                  << "]) expected: ~0.99, got: " << kaiser[centerIndex]
                  << std::endl;

        if(kaiser[centerIndex] < 0.99f) {
            throw std::runtime_error(
                "Kaiser[center] is not close to 1.0. Check formula.");
        }
        std::cout << "PASSED: Kaiser window generation." << std::endl;
    } catch(const std::exception& e) {
        std::cout << "!!! FAILED: Kaiser test. Caught: " << e.what()
                  << std::endl;
        allTestsPassed = false;
    }
    std::cout << "---" << std::endl;

    // ---
    // Test 3: Oversampler<float> - Signal Round-Trip
    // ---
    std::cout << "\n## Test 3: Oversampler<float, 4, 32> (Signal Round-Trip) ##"
              << std::endl;
    try {
        Oversampler<float, 4, 32> oversampler;

        constexpr int CHANNELS = 1;
        constexpr int SAMPLES_IN = 512;
        constexpr int SINC_RADIUS = 32;
        constexpr int TOTAL_DELAY = 2 * SINC_RADIUS;
        constexpr int WARMUP = SINC_RADIUS;    // Let filter warm up
        constexpr int COMPARE_SAMPLES = SAMPLES_IN - WARMUP - TOTAL_DELAY;

        // Configure with the max sizes for this test
        oversampler.configure(44100.0f, CHANNELS, SAMPLES_IN);

        if(COMPARE_SAMPLES <= 0) {
            throw std::runtime_error("SAMPLES_IN is too small for this test.");
        }

        std::vector<std::vector<float>> inputBuffer, outputBuffer;
        generateSine(inputBuffer, CHANNELS, SAMPLES_IN, 440.0f, 44100.0f);
        outputBuffer.assign(CHANNELS, std::vector<float>(SAMPLES_IN, 0.0f));

        // NEW API: Pass vectors directly!
        oversampler.interpolate(inputBuffer);
        oversampler.decimate(outputBuffer);

        printBuffer("Input (Float)", inputBuffer, 10);
        printBuffer("Output (Float)", outputBuffer, 10);

        std::vector<std::vector<float>> inputView(
            CHANNELS, std::vector<float>(COMPARE_SAMPLES));
        std::vector<std::vector<float>> outputView(
            CHANNELS, std::vector<float>(COMPARE_SAMPLES));

        for(int ch = 0; ch < CHANNELS; ++ch) {
            std::copy(inputBuffer[ch].begin() + WARMUP,
                      inputBuffer[ch].begin() + WARMUP + COMPARE_SAMPLES,
                      inputView[ch].begin());
            std::copy(outputBuffer[ch].begin() + WARMUP + TOTAL_DELAY,
                      outputBuffer[ch].begin() + WARMUP + TOTAL_DELAY +
                          COMPARE_SAMPLES,
                      outputView[ch].begin());
        }

        float maxError = calculateMaxError(inputView, outputView);
        std::cout << "Max error (ignoring first " << WARMUP << " samples and "
                  << TOTAL_DELAY << " sample delay): " << maxError << std::endl;

        if(maxError > 0.001f) {
            std::cout << "!!! FAILED: Signal round-trip error is too high."
                      << std::endl;
            allTestsPassed = false;
        } else {
            std::cout << "PASSED: Signal round-trip test." << std::endl;
        }

    } catch(const std::exception& e) {
        std::cout << "!!! FAILED: Oversampler<float> round-trip test. Caught: "
                  << e.what() << std::endl;
        allTestsPassed = false;
    }
    std::cout << "---" << std::endl;

    // ---
    // Test 4: Oversampler<float> - process() Lambda
    // ---
    std::cout << "\n## Test 4: Oversampler<float, 4, 32> (process() Lambda) ##"
              << std::endl;
    try {
        Oversampler<float, 4, 32> oversampler;

        constexpr int OVERSAMPLE_FACTOR = 4;
        constexpr int SINC_RADIUS = 32;
        constexpr int CHANNELS = 1;
        constexpr int SAMPLES_IN = 128;    // Must be >= SINC_RADIUS

        // Configure with the max sizes for this test
        oversampler.configure(44100.0f, CHANNELS, SAMPLES_IN);

        std::vector<std::vector<float>> inputBuffer, outputBuffer;
        inputBuffer.assign(CHANNELS, std::vector<float>(SAMPLES_IN, 0.0f));
        inputBuffer[0][0] = 1.0f;    // Simple impulse
        outputBuffer.assign(CHANNELS, std::vector<float>(SAMPLES_IN, 0.0f));

        // NEW API: Vectors
        oversampler.interpolate(inputBuffer);

        // NEW API: generic process callback
        oversampler.process(
            [&](std::vector<std::vector<float>>& internalBuffer) {
                std::cout << "-> process() lambda called. Overwriting [0][0]."
                          << std::endl;
                internalBuffer[0][0] = 1.2345f;
            });

        oversampler.decimate(outputBuffer);

        float expected = 1.2345f / OVERSAMPLE_FACTOR;
        float actual = outputBuffer[0][SINC_RADIUS];
        std::cout << "Output[0][" << SINC_RADIUS
                  << "] (due to delay) expected ~" << expected << ", got "
                  << actual << std::endl;

        if(std::abs(actual - expected) > 0.01f) {
            std::cout << "!!! FAILED: process() lambda modification not "
                         "reflected in output."
                      << std::endl;
            allTestsPassed = false;
        } else {
            std::cout << "PASSED: process() lambda test." << std::endl;
        }
    } catch(const std::exception& e) {
        std::cout << "!!! FAILED: Oversampler<float> process test. Caught: "
                  << e.what() << std::endl;
        allTestsPassed = false;
    }
    std::cout << "---" << std::endl;

    // ---
    // Test 5: Oversampler<double> - Multi-block state test
    // ---
    std::cout << "\n## Test 5: Oversampler<double, 4, 32> (Multi-block) ##"
              << std::endl;
    try {
        Oversampler<double, 4, 32> oversampler_block;
        Oversampler<double, 4, 32> oversampler_truth;

        constexpr int CHANNELS_D = 1;
        constexpr int SAMPLES_PER_BLOCK = 128;
        constexpr int NUM_BLOCKS = 2;
        constexpr int TOTAL_SAMPLES = SAMPLES_PER_BLOCK * NUM_BLOCKS;

        // Configure both with the max sizes for this test
        oversampler_block.configure(96000.0, CHANNELS_D, SAMPLES_PER_BLOCK);
        oversampler_truth.configure(96000.0, CHANNELS_D, TOTAL_SAMPLES);

        std::vector<std::vector<double>> inBlock1(
            CHANNELS_D, std::vector<double>(SAMPLES_PER_BLOCK));
        std::vector<std::vector<double>> inBlock2(
            CHANNELS_D, std::vector<double>(SAMPLES_PER_BLOCK));
        std::vector<std::vector<double>> inContinuous(
            CHANNELS_D, std::vector<double>(TOTAL_SAMPLES));

        std::vector<std::vector<double>> outBlock1(
            CHANNELS_D, std::vector<double>(SAMPLES_PER_BLOCK));
        std::vector<std::vector<double>> outBlock2(
            CHANNELS_D, std::vector<double>(SAMPLES_PER_BLOCK));
        std::vector<std::vector<double>> outTruth(
            CHANNELS_D, std::vector<double>(TOTAL_SAMPLES));

        generateSine(inContinuous, CHANNELS_D, TOTAL_SAMPLES, 1000.0, 96000.0);

        std::copy(inContinuous[0].begin(),
                  inContinuous[0].begin() + SAMPLES_PER_BLOCK,
                  inBlock1[0].begin());
        std::copy(inContinuous[0].begin() + SAMPLES_PER_BLOCK,
                  inContinuous[0].end(),
                  inBlock2[0].begin());

        // NEW API: Pass vectors directly
        std::cout << "Processing Block 1..." << std::endl;
        oversampler_block.interpolate(inBlock1);
        oversampler_block.decimate(outBlock1);

        std::cout << "Processing Block 2..." << std::endl;
        oversampler_block.interpolate(inBlock2);
        oversampler_block.decimate(outBlock2);

        std::cout << "Processing Ground Truth..." << std::endl;
        oversampler_truth.interpolate(inContinuous);
        oversampler_truth.decimate(outTruth);

        std::vector<std::vector<double>> outTruth1(
            CHANNELS_D, std::vector<double>(SAMPLES_PER_BLOCK));
        std::vector<std::vector<double>> outTruth2(
            CHANNELS_D, std::vector<double>(SAMPLES_PER_BLOCK));
        std::copy(outTruth[0].begin(),
                  outTruth[0].begin() + SAMPLES_PER_BLOCK,
                  outTruth1[0].begin());
        std::copy(outTruth[0].begin() + SAMPLES_PER_BLOCK,
                  outTruth[0].end(),
                  outTruth2[0].begin());

        float maxError1 = calculateMaxError(outBlock1, outTruth1);
        float maxError2 = calculateMaxError(outBlock2, outTruth2);

        std::cout << "Block 1 vs Ground Truth Max Error: " << maxError1
                  << std::endl;
        std::cout << "Block 2 vs Ground Truth Max Error: " << maxError2
                  << std::endl;

        if(maxError1 > 1e-9 || maxError2 > 1e-9) {
            std::cout << "!!! FAILED: Block processing does not match "
                         "continuous processing."
                      << std::endl;
            allTestsPassed = false;
        } else {
            std::cout << "PASSED: Multi-block test (state is kept correctly)."
                      << std::endl;
        }
    } catch(const std::exception& e) {
        std::cout << "!!! FAILED: Oversampler<double> test. Caught: "
                  << e.what() << std::endl;
        allTestsPassed = false;
    }
    std::cout << "---" << std::endl;

    // ---
    // Test 6: Exception Handling
    // ---
    std::cout << "\n## Test 6: Exception Handling ##" << std::endl;
    try {
        Oversampler<float, 3, 16> ex_oversampler;
        std::cout << "Testing configure(0.0)... ";
        ex_oversampler.configure(0.0f, 1, 128);
        std::cout << "!!! FAILED: No exception thrown !!!" << std::endl;
        allTestsPassed = false;
    } catch(const std::invalid_argument& e) {
        std::cout << "PASSED. Caught: " << e.what() << std::endl;
    } catch(...) {
        std::cout << "!!! FAILED: Wrong exception type thrown." << std::endl;
        allTestsPassed = false;
    }

    try {
        Oversampler<float, 3, 16> ex_oversampler;
        std::cout << "Testing configure(0 channels)... ";
        ex_oversampler.configure(44100.0f, 0, 128);
        std::cout << "!!! FAILED: No exception thrown !!!" << std::endl;
        allTestsPassed = false;
    } catch(const std::invalid_argument& e) {
        std::cout << "PASSED. Caught: " << e.what() << std::endl;
    } catch(...) {
        std::cout << "!!! FAILED: Wrong exception type thrown." << std::endl;
        allTestsPassed = false;
    }

    try {
        Oversampler<float, 4, 32> oversampler;
        oversampler.configure(44100.0f, 1, 128);
        std::cout << "Testing interpolate(nullptr, 0 channels)... ";
        // Explicitly calling raw pointer overload to test bounds checking
        const float* const* nullPtr = nullptr;
        oversampler.interpolate(nullPtr, 0, 10);
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