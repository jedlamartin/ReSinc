#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define _USE_MATH_DEFINES

// =============================================================================
//  SFINAE TRAITS
// =============================================================================

/**
 * @brief Internal traits to detect buffer capabilities at compile-time.
 * * These structures are used by the Oversampler to automatically select the
 * correct overload for interpolate/decimate based on the input type.
 */
namespace resinc_traits {
    // Helper: Detects if T has .data() and .size() methods
    template<typename T, typename = void>
    struct has_size_and_data : std::false_type {};
    template<typename T>
    struct has_size_and_data<T,
                             std::void_t<decltype(std::declval<T>().data()),
                                         decltype(std::declval<T>().size())>> :
        std::true_type {};

    /** @brief Detects single-channel containers (e.g. std::vector<float>). */
    template<typename T, typename = void>
    struct is_single_channel : std::false_type {};
    template<typename T>
    struct is_single_channel<
        T,
        std::enable_if_t<has_size_and_data<T>::value &&
                         std::is_arithmetic_v<
                             std::decay_t<decltype(std::declval<T>()[0])>>>> :
        std::true_type {};

    /** @brief Detects multi-channel containers (e.g.
     * std::vector<std::vector<float>>). */
    template<typename T, typename = void>
    struct is_multi_channel : std::false_type {};
    template<typename T>
    struct is_multi_channel<
        T,
        std::enable_if_t<has_size_and_data<T>::value &&
                         std::is_arithmetic_v<std::decay_t<
                             decltype(std::declval<T>()[0][0])>>>> :
        std::true_type {};

    /** @brief Detects JUCE-compatible AudioBuffer classes (has
     * getReadPointer/getWritePointer). */
    template<typename T, typename = void>
    struct is_juce_type : std::false_type {};
    template<typename T>
    struct is_juce_type<
        T,
        std::void_t<decltype(std::declval<T>().getNumChannels()),
                    decltype(std::declval<T>().getNumSamples()),
                    decltype(std::declval<T>().getReadPointer(0)),
                    decltype(std::declval<T>().getWritePointer(0))>> :
        std::true_type {};
}    // namespace resinc_traits

// =============================================================================
//  COMPONENTS
// =============================================================================

/**
 * @brief Fixed-size circular (ring) buffer.
 *
 * @tparam TYPE    Element type stored in the buffer.
 * @tparam size    Number of elements the buffer holds.
 *
 * Behaviour:
 * - push(element) inserts a new element and overwrites the oldest when full.
 * - operator[](index) accesses elements relative to the oldest element.
 * * Complexity: O(1) for all operations.
 */
template<class TYPE, size_t size>
class CircularBuffer {
private:
    std::array<TYPE, size> buf;
    size_t oldestIndex;

public:
    CircularBuffer(TYPE initValue);
    void push(TYPE element);
    TYPE& operator[](size_t index);
    void clear();
};

namespace Window {
    /**
     * @brief Generic window function base class.
     * Provides accessors and an applyOn() helper to multiply an array by the
     * window.
     */
    template<typename TYPE, size_t N>
    class Window {
    public:
        virtual ~Window() = default;
        virtual TYPE& operator[](size_t i) = 0;
        virtual const TYPE& operator[](size_t i) const = 0;
        void applyOn(std::array<TYPE, N>& data) const;
        size_t size() const;

    protected:
        Window() = default;
    };

    /**
     * @brief Kaiser window implementation.
     * parameterised by beta to control side-lobe attenuation.
     */
    template<typename TYPE, size_t N>
    class Kaiser : public Window<TYPE, N> {
    public:
        Kaiser(TYPE beta);
        TYPE& operator[](size_t i) override;
        const TYPE& operator[](size_t i) const override;

    private:
        TYPE besselI0(TYPE x);
        const TYPE beta;
        std::array<TYPE, (N + 1) / 2> window;
    };
}    // namespace Window

// =============================================================================
//  OVERSAMPLER CLASS
// =============================================================================

/**
 * @brief Multi-channel resampler (real-time / streaming-safe).
 *
 * Resample converts between sample rates by interpolating and decimating
 * buffers using a precomputed Sinc table. The class maintains per-channel
 * circular history buffers and is designed for continuous (real-time)
 * processing.
 *
 * Important notes:
 * - Internal buffering: The class maintains history across calls. Do not
 * assume independence; feed blocks consecutively.
 * - Latency: Effective latency is 2*N input samples (2 * SINC_RADIUS).
 *
 * Template parameters:
 * @tparam TYPE     Sample type (float, double).
 * @tparam OVERSAMPLE_FACTOR Interpolation factor (e.g., 2x, 4x).
 * @tparam N     Sinc kernel radius (taps per side).
 */
template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
class Oversampler {
public:
    Oversampler();
    ~Oversampler() = default;
    Oversampler(const Oversampler&) = delete;
    Oversampler& operator=(const Oversampler&) = delete;
    Oversampler(Oversampler&&) noexcept = default;
    Oversampler& operator=(Oversampler&&) noexcept = default;

    /**
     * @brief Configures internal buffers for a specific session format.
     * Must be called before processing.
     * @param sampleRate The base sample rate of the input signal.
     * @param maxChannels Maximum number of channels expected.
     * @param maxBlockSize Maximum number of samples per input block.
     */
    void configure(TYPE sampleRate, int maxChannels, int maxBlockSize);

    // =========================================================================
    // INTERPOLATE (Upsample Input -> Internal Buffer)
    // =========================================================================

    /**
     * @brief Interpolates a JUCE-style AudioBuffer.
     * Accepts any class with getNumChannels(), getNumSamples(), and
     * getArrayOfReadPointers().
     */
    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_juce_type<std::decay_t<T>>::value>
        interpolate(T&& buffer);

    /**
     * @brief Interpolates a single-channel container.
     * Accepts std::vector, std::array, or custom containers with .data() and
     * .size().
     */
    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_single_channel<std::decay_t<T>>::value>
        interpolate(T&& buffer);

    /**
     * @brief Interpolates a multi-channel container (vector of vectors).
     * Accepts std::vector<std::vector<float>>.
     */
    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_multi_channel<std::decay_t<T>>::value>
        interpolate(T&& buffer);

    /**
     * @brief Interpolates from raw pointers (Base Implementation).
     * @param ptrToBuffers Array of pointers to channel data [channel][sample].
     * @param numChannels Number of channels to read.
     * @param numSamples Number of input samples to read per channel.
     */
    void interpolate(const TYPE* const* ptrToBuffers,
                     int numChannels,
                     int numSamples);

    // =========================================================================
    // DECIMATE (Downsample Internal Buffer -> Output)
    // =========================================================================

    /**
     * @brief Decimates into a JUCE-style AudioBuffer.
     * Writes result to the provided buffer using getArrayOfWritePointers().
     */
    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_juce_type<std::decay_t<T>>::value>
        decimate(T& buffer);

    /**
     * @brief Decimates into a single-channel container.
     * Writes result directly to .data().
     */
    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_single_channel<std::decay_t<T>>::value>
        decimate(T& buffer);

    /**
     * @brief Decimates into a multi-channel container (vector of vectors).
     */
    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_multi_channel<std::decay_t<T>>::value>
        decimate(T& buffer);

    /**
     * @brief Decimates into raw pointers (Base Implementation).
     * @param ptrToBuffers Destination array of pointers [channel][sample].
     * @param numChannels Number of channels to write.
     * @param numSamples Number of samples to produce (target size).
     */
    void decimate(TYPE* const* ptrToBuffers, int numChannels, int numSamples);

    // =========================================================================
    // PROCESS CALLBACKS
    // =========================================================================

    /**
     * @brief Processes upsampled data using a JUCE-style wrapper.
     * Creates a temporary wrapper around the internal buffer and passes it to
     * the callback.
     * @param processBlock Function/Lambda receiving (T buffer).
     */
    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_juce_type<std::decay_t<T>>::value>
        process(std::function<void(T)> processBlock);

    /**
     * @brief Processes upsampled data via vector of vectors.
     * Direct access to the internal `vector<vector<TYPE>>`.
     */
    void process(
        std::function<void(std::vector<std::vector<TYPE>>&)> processBlock);

    /**
     * @brief Processes upsampled data channel-by-channel.
     * Iterates the internal buffer and calls the callback for each channel
     * vector.
     */
    void process(std::function<void(std::vector<TYPE>&)> processBlock);

private:
    /**
     * @brief Internal Sinc lookup table.
     * Handles symmetric storage and fractional indexing.
     */
    class Sinc {
    private:
        std::array<TYPE, (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR> sinc;

    public:
        Sinc();
        TYPE& operator[](int i);
        TYPE& operator()(int i, int delta);
        void configure(TYPE sampleRate);
        size_t size() const;
        void applyWindow(
            const Window::Window<TYPE,
                                 (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR * 2>&
                window);
    };

    Sinc sinc;
    std::vector<std::vector<TYPE>> interpolatedBuf;
    std::vector<CircularBuffer<TYPE, SINC_RADIUS>> beginBuf, endBuf;
    std::vector<CircularBuffer<TYPE, SINC_RADIUS * OVERSAMPLE_FACTOR>>
        decBeginBuf, decEndBuf;
    std::vector<std::vector<TYPE>> x_interp, x_decim;

    void interpolate_helper(const TYPE* const* ptrToBuffers,
                            int numChannels,
                            int numSamples);
    void decimate_helper(TYPE* const* ptrToBuffers,
                         int numChannels,
                         int numSamples);
};

template<typename TYPE, int SINC_RADIUS, int RESOLUTION = 256>
class Resampler {
public:
    Resampler();
    ~Resampler() = default;
    Resampler(const Resampler&) = delete;
    Resampler& operator=(const Resampler&) = delete;
    Resampler(Resampler&&) noexcept = default;
    Resampler& operator=(Resampler&&) noexcept = default;

    void configure(TYPE sampleRate,
                   int maxChannels,
                   int maxBlockSize,
                   TYPE targetSampleRate);

    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_single_channel<std::decay_t<T>>::value>
        resample(T&& input, T& output);

    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_multi_channel<std::decay_t<T>>::value>
        resample(T&& input, T& output);

    void resample(const TYPE* const* ptrToInBuffers,
                  TYPE* const* ptrToBuffers,
                  int numChannels,
                  int numSamples);

private:
    Oversampler<TYPE, RESOLUTION, SINC_RADIUS>::Sinc sinc;
    std::vector<CircularBuffer<TYPE, SINC_RADIUS>> inputBuffer;
    void resample_helper(const TYPE* const* ptrToInBuffers,
                         TYPE* const* ptrToOutBuffers,
                         int numChannels,
                         int numSamples);
};

// =============================================================================
//  IMPLEMENTATIONS: CircularBuffer
// =============================================================================

template<class TYPE, size_t size>
CircularBuffer<TYPE, size>::CircularBuffer(TYPE initValue) : oldestIndex {0} {
    buf.fill(initValue);
}

template<class TYPE, size_t size>
void CircularBuffer<TYPE, size>::push(TYPE element) {
    this->buf[this->oldestIndex] = std::move(element);
    this->oldestIndex = this->oldestIndex == size - 1 ? 0 : oldestIndex + 1;
}

template<class TYPE, size_t size>
TYPE& CircularBuffer<TYPE, size>::operator[](size_t index) {
    return this->buf[(this->oldestIndex + index) % size];
}

template<class TYPE, size_t size>
void CircularBuffer<TYPE, size>::clear() {
    buf.fill(TYPE {0});
    this->oldestIndex = 0;
}

// =============================================================================
//  IMPLEMENTATIONS: Window
// =============================================================================

template<typename TYPE, size_t N>
size_t Window::Window<TYPE, N>::size() const {
    return N;
}

template<typename TYPE, size_t N>
void Window::Window<TYPE, N>::applyOn(std::array<TYPE, N>& data) const {
    for(size_t i = 0; i < N; i++) data[i] *= (*this)[i];
}

template<typename TYPE, size_t N>
Window::Kaiser<TYPE, N>::Kaiser(TYPE beta) : Window<TYPE, N> {}, beta {beta} {
    if(beta < TYPE(0.0))
        throw std::invalid_argument("Kaiser beta value must be non-negative.");
    const TYPE M_denominator = static_cast<TYPE>(N - 1);
    const TYPE den = besselI0(beta);
    const size_t storageSize = (N + 1) / 2;
    const TYPE center = static_cast<TYPE>(N - 1) / TYPE(2.0);
    for(size_t i = 0; i < storageSize; i++) {
        TYPE n = static_cast<TYPE>(i) - center;
        TYPE arg =
            beta * std::sqrt(TYPE(1.0) - std::pow(TYPE(2.0) * n / M_denominator,
                                                  TYPE(2.0)));
        window[i] = besselI0(arg) / den;
    }
}

template<typename TYPE, size_t N>
TYPE& Window::Kaiser<TYPE, N>::operator[](size_t i) {
    const size_t pivot = N / 2;
    return window[(i < pivot) ? i : (N - 1 - i)];
}

template<typename TYPE, size_t N>
const TYPE& Window::Kaiser<TYPE, N>::operator[](size_t i) const {
    const size_t pivot = N / 2;
    return window[(i < pivot) ? i : (N - 1 - i)];
}

template<typename TYPE, size_t N>
TYPE Window::Kaiser<TYPE, N>::besselI0(TYPE x) {
    const TYPE epsilon = TYPE(1e-6);
    TYPE sum = TYPE(1.0), term = TYPE(1.0), k = TYPE(1.0),
         factorial = TYPE(1.0);
    while(term > epsilon * sum) {
        factorial *= k;
        term = std::pow(x / TYPE(2.0), TYPE(2) * k) / (factorial * factorial);
        sum += term;
        k += TYPE(1.0);
    }
    return sum;
}

// =============================================================================
//  IMPLEMENTATIONS: Sinc
// =============================================================================

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::Sinc::Sinc() : sinc {0} {}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
TYPE& Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::Sinc::operator[](
    int i) {
    return sinc[i < 0 ? -i : i];
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
TYPE& Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::Sinc::operator()(
    int i,
    int delta) {
    if(i < 0) return (*this)[(-i) * OVERSAMPLE_FACTOR - delta];
    return (*this)[i * OVERSAMPLE_FACTOR + delta];
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::Sinc::configure(
    TYPE sampleRate) {
    TYPE fc = sampleRate / TYPE(2);
    TYPE T_ = TYPE(1) / sampleRate;
    for(int i = 0; i <= SINC_RADIUS; i++) {
        for(int delta = 0; delta < OVERSAMPLE_FACTOR; delta++) {
            TYPE index = static_cast<TYPE>(i) +
                         (TYPE(1) / static_cast<TYPE>(OVERSAMPLE_FACTOR)) *
                             static_cast<TYPE>(delta);
            if(index == TYPE(0.0)) (*this)(i, delta) = TYPE(1.0);
            else
                (*this)(i, delta) = std::sin(TYPE(2) * M_PI * fc * index * T_) /
                                    (TYPE(2) * M_PI * fc * index * T_);
        }
    }
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
size_t Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::Sinc::size() const {
    return this->sinc.size();
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::Sinc::applyWindow(
    const Window::Window<TYPE, (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR * 2>&
        window) {
    for(int i = 0; i < (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR; ++i)
        sinc[i] *= window[i + (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR];
}

// =============================================================================
//  IMPLEMENTATIONS: Oversampler Lifecycle
// =============================================================================

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::Oversampler() {}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::configure(
    TYPE sampleRate,
    int maxChannels,
    int maxBlockSize) {
    if(sampleRate <= TYPE(0.0) || maxChannels <= 0 || maxBlockSize <= 0)
        throw std::invalid_argument("Invalid configuration.");

    sinc.configure(static_cast<TYPE>(sampleRate));
    sinc.applyWindow(
        Window::Kaiser<TYPE, (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR * 2> {5.0});

    int maxInterpolatedSize = maxBlockSize * OVERSAMPLE_FACTOR;
    beginBuf.clear();
    endBuf.clear();
    decBeginBuf.clear();
    decEndBuf.clear();
    beginBuf.resize(maxChannels, CircularBuffer<TYPE, SINC_RADIUS>(TYPE(0.0)));
    endBuf.resize(maxChannels, CircularBuffer<TYPE, SINC_RADIUS>(TYPE(0.0)));
    decBeginBuf.resize(
        maxChannels,
        CircularBuffer<TYPE, SINC_RADIUS * OVERSAMPLE_FACTOR>(TYPE(0.0)));
    decEndBuf.resize(
        maxChannels,
        CircularBuffer<TYPE, SINC_RADIUS * OVERSAMPLE_FACTOR>(TYPE(0.0)));

    x_interp.resize(maxChannels, std::vector<TYPE>(maxBlockSize + SINC_RADIUS));
    x_decim.resize(maxChannels,
                   std::vector<TYPE>(maxInterpolatedSize +
                                     SINC_RADIUS * OVERSAMPLE_FACTOR));
    interpolatedBuf.resize(maxChannels, std::vector<TYPE>(maxInterpolatedSize));
}

// =============================================================================
//  IMPLEMENTATIONS: Interpolate
// =============================================================================

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename T>
typename std::enable_if_t<resinc_traits::is_juce_type<std::decay_t<T>>::value>
    Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::interpolate(T&& buffer) {
    interpolate_helper(buffer.getArrayOfReadPointers(),
                       buffer.getNumChannels(),
                       buffer.getNumSamples());
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename T>
typename std::enable_if_t<
    resinc_traits::is_single_channel<std::decay_t<T>>::value>
    Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::interpolate(T&& buffer) {
    const TYPE* ptr = buffer.data();
    interpolate_helper(&ptr, 1, static_cast<int>(buffer.size()));
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename T>
typename std::enable_if_t<
    resinc_traits::is_multi_channel<std::decay_t<T>>::value>
    Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::interpolate(T&& buffer) {
    int channels = static_cast<int>(buffer.size());
    if(channels == 0) return;
    std::vector<const TYPE*> ptrs(channels);
    for(int i = 0; i < channels; ++i) ptrs[i] = buffer[i].data();
    interpolate_helper(
        ptrs.data(), channels, static_cast<int>(buffer[0].size()));
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::interpolate(
    const TYPE* const* ptrToBuffers,
    int numChannels,
    int numSamples) {
    interpolate_helper(ptrToBuffers, numChannels, numSamples);
}

// =============================================================================
//  IMPLEMENTATIONS: Decimate
// =============================================================================

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename T>
typename std::enable_if_t<resinc_traits::is_juce_type<std::decay_t<T>>::value>
    Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::decimate(T& buffer) {
    decimate_helper(buffer.getArrayOfWritePointers(),
                    buffer.getNumChannels(),
                    buffer.getNumSamples());
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename T>
typename std::enable_if_t<
    resinc_traits::is_single_channel<std::decay_t<T>>::value>
    Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::decimate(T& buffer) {
    TYPE* ptr = buffer.data();
    decimate_helper(&ptr, 1, static_cast<int>(buffer.size()));
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename T>
typename std::enable_if_t<
    resinc_traits::is_multi_channel<std::decay_t<T>>::value>
    Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::decimate(T& buffer) {
    int channels = static_cast<int>(buffer.size());
    if(channels == 0) return;
    std::vector<TYPE*> ptrs(channels);
    for(int i = 0; i < channels; ++i) ptrs[i] = buffer[i].data();
    decimate_helper(ptrs.data(), channels, static_cast<int>(buffer[0].size()));
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::decimate(
    TYPE* const* ptrToBuffers,
    int numChannels,
    int numSamples) {
    decimate_helper(ptrToBuffers, numChannels, numSamples);
}

// =============================================================================
//  IMPLEMENTATIONS: Process
// =============================================================================

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename T>
typename std::enable_if_t<resinc_traits::is_juce_type<std::decay_t<T>>::value>
    Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::process(
        std::function<void(T)> processBlock) {
    int ch = static_cast<int>(interpolatedBuf.size());
    std::vector<TYPE*> ptrs(ch);
    for(int i = 0; i < ch; ++i) ptrs[i] = interpolatedBuf[i].data();
    T buffer(ptrs.data(), ch, static_cast<int>(interpolatedBuf[0].size()));
    processBlock(buffer);
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::process(
    std::function<void(std::vector<std::vector<TYPE>>&)> processBlock) {
    processBlock(interpolatedBuf);
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::process(
    std::function<void(std::vector<TYPE>&)> processBlock) {
    for(auto& i : interpolatedBuf) {
        processBlock(i);
    }
}

// =============================================================================
//  IMPLEMENTATIONS: Helpers
// =============================================================================

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::interpolate_helper(
    const TYPE* const* ptrToBuffers,
    int numChannels,
    int numSamples) {
    if(numChannels <= 0 || numSamples <= 0) {
        throw std::invalid_argument(
            "Number of channels and samples must be positive.");
    }

    if(static_cast<size_t>(numChannels) > x_interp.size() ||
       static_cast<size_t>(numSamples + SINC_RADIUS) > x_interp[0].size())
        throw std::runtime_error(
            "Oversampler: Input size exceeds configured maxBlockSize.");

    int interpolatedBufSize = numSamples * OVERSAMPLE_FACTOR;
    for(int channel = 0; channel < numChannels; channel++) {
        for(int i = 0; i < numSamples + SINC_RADIUS; ++i) {
            x_interp[channel][i] =
                (i >= SINC_RADIUS ? ptrToBuffers[channel][i - SINC_RADIUS] :
                                    endBuf[channel][i]);
        }
    }

    for(int channel = 0; channel < numChannels; channel++) {
        interpolatedBuf[channel][0] = x_interp[channel][0];
        for(int k = 1; k < interpolatedBufSize; k++) {
            int delta = k % OVERSAMPLE_FACTOR;
            int index = k / OVERSAMPLE_FACTOR;
            if(delta != 0) {
                interpolatedBuf[channel][k] = 0;
                for(int n = -SINC_RADIUS; n <= 0; n++)
                    interpolatedBuf[channel][k] +=
                        this->sinc(n, delta) * x_interp[channel][index - n];
                for(int n = 1; n <= SINC_RADIUS; n++)
                    interpolatedBuf[channel][k] +=
                        this->sinc(n, delta) *
                        beginBuf[channel][SINC_RADIUS - n];
            } else {
                interpolatedBuf[channel][k] = x_interp[channel][index];
                beginBuf[channel].push(x_interp[channel][index - 1]);
            }
        }
        beginBuf[channel].push(x_interp[channel][numSamples - 1]);
        for(int i = 0; i < SINC_RADIUS; i++)
            endBuf[channel].push(x_interp[channel][numSamples + i]);
    }
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::decimate_helper(
    TYPE* const* ptrToBuffers,
    int numChannels,
    int numSamples) {
    if(static_cast<size_t>(numChannels) > x_decim.size() ||
       static_cast<size_t>(numSamples) >
           (interpolatedBuf[0].size() / OVERSAMPLE_FACTOR))
        throw std::runtime_error(
            "Oversampler: Output size exceeds configured maxBlockSize.");

    int interpolatedBufSize = static_cast<int>(interpolatedBuf[0].size());
    for(int channel = 0; channel < numChannels; channel++) {
        for(int i = 0;
            i < interpolatedBufSize + SINC_RADIUS * OVERSAMPLE_FACTOR;
            ++i) {
            x_decim[channel][i] =
                (i >= SINC_RADIUS * OVERSAMPLE_FACTOR ?
                     interpolatedBuf[channel]
                                    [i - SINC_RADIUS * OVERSAMPLE_FACTOR] :
                     decEndBuf[channel][i]);
        }
    }

    for(int channel = 0; channel < numChannels; channel++) {
        for(int k = 0; k < numSamples; k++) {
            int index = OVERSAMPLE_FACTOR * k;
            ptrToBuffers[channel][k] = 0;
            for(int n = 1; n <= SINC_RADIUS * OVERSAMPLE_FACTOR; n++)
                ptrToBuffers[channel][k] +=
                    sinc[n] *
                    decBeginBuf[channel][SINC_RADIUS * OVERSAMPLE_FACTOR - n];
            for(int n = 0; n >= -(SINC_RADIUS * OVERSAMPLE_FACTOR); n--)
                ptrToBuffers[channel][k] +=
                    sinc[n] * x_decim[channel][index - n];
            for(int i = 0; i < OVERSAMPLE_FACTOR; i++)
                decBeginBuf[channel].push(x_decim[channel][index + i]);
            ptrToBuffers[channel][k] /= OVERSAMPLE_FACTOR;
        }
        for(int i = 0; i < SINC_RADIUS * OVERSAMPLE_FACTOR; i++)
            decEndBuf[channel].push(x_decim[channel][interpolatedBufSize + i]);
    }
}