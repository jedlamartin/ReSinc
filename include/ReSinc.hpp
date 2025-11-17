#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#define _USE_MATH_DEFINES

/**
 * @brief Fixed-size circular (ring) buffer.
 *
 * Template parameters:
 * @tparam TYPE    Element type stored in the buffer.
 * @tparam size Number of elements the buffer holds.
 *
 * Behaviour:
 * - Construct with an initial value used to initialize stored elements.
 * - push(element) inserts a new element and overwrites the oldest when full.
 * - operator[](index) accesses elements relative to the oldest element:
 *   index 0 == oldest element, index increases towards more recently pushed
 *   elements.
 * - clear() zeroes the storage and resets the internal index.
 *
 * Complexity:
 * - push and operator[] are O(1).
 */
template<class TYPE, size_t size>
class CircularBuffer {
private:
    std::array<TYPE, size> buf;
    size_t oldestIndex;

public:
    CircularBuffer(TYPE initValue);
    CircularBuffer(const CircularBuffer&) = default;
    CircularBuffer(CircularBuffer&&) noexcept = default;
    CircularBuffer& operator=(const CircularBuffer&) = default;
    CircularBuffer& operator=(CircularBuffer&&) noexcept = default;
    void push(TYPE element);
    TYPE& operator[](size_t index);
    void clear();
};

namespace Window {
    /**
     * @brief Generic window function container.
     *
     * Window holds a precomputed window of length N and provides accessors
     * and an applyOn() helper to multiply an array by the window.
     *
     * @tparam N Length of the window (compile-time constant).
     *
     * Methods:
     * - operator[](i) / operator[](i) const : access window sample i.
     * - applyOn(data) : multiplies the provided data array in-place by the
     *   window values.
     * - size() : returns N.
     *
     * Complexity: operator[] and size() are O(1); applyOn() is O(N).
     */
    template<typename TYPE, size_t N>
    class Window {
    public:
        virtual ~Window() = default;
        Window(const Window&) = default;
        Window(Window&&) = default;
        Window& operator=(const Window&) = delete;
        Window& operator=(Window&&) = delete;

        virtual TYPE& operator[](size_t i) = 0;
        virtual const TYPE& operator[](size_t i) const = 0;
        void applyOn(std::array<TYPE, N>& data) const;
        size_t size() const;

    protected:
        Window() = default;

    private:
    };

    /**
     * @brief Kaiser window implementation.
     *
     * Uses the Kaiser window parameterised by beta. Provides a private
     * helper besselI0() for the window calculation.
     *
     * @tparam N Length of the window.
     * @param beta Shape parameter for the Kaiser window.
     */
    template<typename TYPE, size_t N>
    class Kaiser : public Window<TYPE, N> {
    public:
        Kaiser(TYPE beta);
        ~Kaiser() = default;
        Kaiser(const Kaiser&) = default;
        Kaiser(Kaiser&&) = default;
        Kaiser& operator=(const Kaiser&) = delete;
        Kaiser& operator=(Kaiser&&) = delete;

        TYPE& operator[](size_t i) override;
        const TYPE& operator[](size_t i) const override;

    private:
        TYPE besselI0(TYPE x);

        const TYPE beta;
        std::array<TYPE, (N + 1) / 2> window;
    };
}    // namespace Window

/**
 * @brief Multi-channel resampler (real-time / streaming-safe).
 *
 * Resample converts between sample rates by interpolating and decimating
 * buffers using a precomputed Sinc table. The class maintains per-channel
 * circular history buffers and is designed for continuous (real-time)
 * processing: call the processing methods repeatedly with consecutive
 * audio blocks (a "continuous buffer").
 *
 * Important notes:
 * - Internal buffering: The class keeps internal per-channel history
 *   (circular) buffers to perform interpolation/decimation across block
 *   boundaries. Do not assume each call is independent — feed consecutive
 *   blocks in order.
 * - Latency: Because of the symmetric Sinc kernel, the effective processing
 *   latency is 2*N input samples (two times the Sinc kernel radius). If
 *   you need sample-accurate alignment, compensate for this latency when
 *   consuming or presenting output.
 *
 * Template parameters:
 * @tparam TYPE     Sample type.
 * @tparam OVERSAMPLE_FACTOR Interpolation factor (number of sub-samples per
 * input sample).
 * @tparam N     Sinc kernel radius (number of taps on each side). The
 *               internal buffer latency is 2*N input samples.
 *
 * Main methods:
 * - configure(sampleRate): prepares internal tables for the given rate.
 * - interpolate(ptrToBuffers, numChannels, numSamples): performs upsampling
 *   on the provided continuous input blocks.
 * - decimate(ptrToBuffers): performs downsampling, reading from the
 *   internal interpolated buffer and history.
 * - process(processBlock): calls the provided function with the
 *   interpolated buffer for in-place processing or forwarding.
 *
 * Complexity: per-sample cost depends on N and OVERSAMPLE_FACTOR (roughly
 * O(N*OVERSAMPLE_FACTOR) per output sample in the straightforward
 * implementation).
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

    void configure(TYPE sampleRate, int maxChannels, int maxBlockSize);
    void interpolate(const TYPE* const* ptrToBuffers,
                     int numChannels,
                     int numSamples);
    void decimate(TYPE* const* ptrToBuffers, int numChannels, int numSamples);

    void process(
        std::function<void(std::vector<std::vector<TYPE>>&)> processBlock);

private:
    /**
     * @brief Precomputed Sinc lookup table used for interpolation/decimation.
     *
     * The Sinc class stores a symmetric table of sinc samples and provides
     * fast indexed access for building interpolated samples. It supports
     * accessing by integer sample index or by (index, delta) for sub-sample
     * fractional positions.
     *
     * Methods:
     * - operator[](i) : returns sinc at integer index (symmetric access).
     * - operator()(i, delta) : returns sinc value for sample index i and
     *   fractional offset delta (0..OVERSAMPLE_FACTOR-1).
     * - configure(sampleRate) : fills the table for the provided sample rate.
     * - applyWindow(window) : multiplies the sinc table by a window function
     *   (e.g. Kaiser) to reduce ringing.
     */
    class Sinc {
    private:
        std::array<TYPE, (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR> sinc;

    public:
        Sinc();
        ~Sinc() = default;
        Sinc(const Sinc&) = default;
        Sinc(Sinc&&) noexcept = default;
        Sinc& operator=(const Sinc&) = default;
        Sinc& operator=(Sinc&&) noexcept = default;

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
};

// CircularBuffer definitions
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

// Definitions for Window
template<typename TYPE, size_t N>
size_t Window::Window<TYPE, N>::size() const {
    return N;
}

template<typename TYPE, size_t N>
void Window::Window<TYPE, N>::applyOn(std::array<TYPE, N>& data) const {
    for(size_t i = 0; i < N; i++) {
        data[i] *= (*this)[i];
    }
}

// Definitions for Kaiser
template<typename TYPE, size_t N>
Window::Kaiser<TYPE, N>::Kaiser(TYPE beta) : Window<TYPE, N> {}, beta {beta} {
    if(beta < TYPE(0.0)) {
        throw std::invalid_argument("Kaiser beta value must be non-negative.");
    }

    const TYPE M_denominator = static_cast<TYPE>(N - 1);
    const TYPE den = besselI0(beta);
    const size_t storageSize = (N + 1) / 2;
    const TYPE center = static_cast<TYPE>(N - 1) / TYPE(2.0);

    for(size_t i = 0; i < storageSize; i++) {
        TYPE n = static_cast<TYPE>(i) - center;
        TYPE arg =
            beta * std::sqrt(TYPE(1.0) - std::pow(TYPE(2.0) * n / M_denominator,
                                                  TYPE(2.0)));
        TYPE num = besselI0(arg);
        window[i] = num / den;
    }
}

template<typename TYPE, size_t N>
TYPE& Window::Kaiser<TYPE, N>::operator[](size_t i) {
    const size_t pivot = N / 2;
    const size_t storageIndex = (i < pivot) ? i : (N - 1 - i);
    return window[storageIndex];
}

template<typename TYPE, size_t N>
const TYPE& Window::Kaiser<TYPE, N>::operator[](size_t i) const {
    const size_t pivot = N / 2;
    const size_t storageIndex = (i < pivot) ? i : (N - 1 - i);
    return window[storageIndex];
}

template<typename TYPE, size_t N>
TYPE Window::Kaiser<TYPE, N>::besselI0(TYPE x) {
    const TYPE epsilon = TYPE(1e-6);
    TYPE sum = TYPE(1.0);
    TYPE term = TYPE(1.0);
    TYPE k = TYPE(1.0);
    TYPE factorial = TYPE(1.0);
    while(term > epsilon * sum) {
        factorial *= k;
        term = std::pow(x / TYPE(2.0), TYPE(2) * k) / (factorial * factorial);
        sum += term;
        k += TYPE(1.0);
    }

    return sum;
}

// Definitions for sincArray
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
            if(index == TYPE(0.0)) {
                (*this)(i, delta) = TYPE(1.0);
            } else {
                (*this)(i, delta) = std::sin(TYPE(2) * M_PI * fc * index * T_) /
                                    (TYPE(2) * M_PI * fc * index * T_);
            }
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
    for(int i = 0; i < (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR; ++i) {
        sinc[i] = sinc[i] * window[i + (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR];
    }
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::Oversampler() {}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::configure(
    TYPE sampleRate,
    int maxChannels,
    int maxBlockSize) {
    if(sampleRate <= TYPE(0.0)) {
        throw std::invalid_argument("Sample rate must be a positive number.");
    }
    if(maxChannels <= 0 || maxBlockSize <= 0) {
        throw std::invalid_argument(
            "Max channels and block size must be positive.");
    }

    sinc.configure(static_cast<TYPE>(sampleRate));
    constexpr size_t windowSize = (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR * 2;
    sinc.applyWindow(Window::Kaiser<TYPE, windowSize> {5.0});

    int maxInterpolatedSize = maxBlockSize * OVERSAMPLE_FACTOR;

    // History Buffers
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

    // Processing Buffers (pre-allocated to max size)
    x_interp.resize(maxChannels, std::vector<TYPE>(maxBlockSize + SINC_RADIUS));
    x_decim.resize(maxChannels,
                   std::vector<TYPE>(maxInterpolatedSize +
                                     SINC_RADIUS * OVERSAMPLE_FACTOR));

    interpolatedBuf.resize(maxChannels, std::vector<TYPE>(maxInterpolatedSize));
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::interpolate(
    const TYPE* const* ptrToBuffers,
    int numChannels,
    int numSamples) {
    if(numChannels <= 0 || numSamples <= 0) {
        throw std::invalid_argument(
            "Number of channels and samples must be positive.");
    }

    if(static_cast<size_t>(numChannels) > x_interp.size() ||
       static_cast<size_t>(numSamples + SINC_RADIUS) > x_interp[0].size()) {
        throw std::runtime_error(
            "Oversampler: Input size exceeds configured maxBlockSize.");
    }

    int interpolatedBufSize = numSamples * OVERSAMPLE_FACTOR;
    for(int channel = 0; channel < numChannels; channel++) {
        for(int i = 0; i < numSamples + SINC_RADIUS; ++i) {
            x_interp[channel][i] =
                (i >= SINC_RADIUS ? ptrToBuffers[channel][i - SINC_RADIUS] :
                                    endBuf[channel][i]);
        }
    }

    for(int channel = 0; channel < numChannels; channel++) {
        // bevart mintak
        // az elsot automatikusan kitoltjuk, hogy a buffereles jo legyen
        interpolatedBuf[channel][0] = x_interp[channel][0];
        for(int k = 1; k < interpolatedBufSize; k++) {
            int delta = k % OVERSAMPLE_FACTOR;
            int index = k / OVERSAMPLE_FACTOR;
            if(delta != 0) {
                interpolatedBuf[channel][k] = 0;
                // mintakbol
                for(int n = -SINC_RADIUS; n <= 0; n++) {
                    interpolatedBuf[channel][k] +=
                        this->sinc(n, delta) * x_interp[channel][index - n];
                }
                // bufferbol
                for(int n = 1; n <= SINC_RADIUS; n++) {
                    interpolatedBuf[channel][k] +=
                        this->sinc(n, delta) *
                        beginBuf[channel][SINC_RADIUS - n];
                }
            } else {
                interpolatedBuf[channel][k] = x_interp[channel][index];
                beginBuf[channel].push(x_interp[channel][index - 1]);
            }
        }
        // az utolso is belekeruljon a bufferbe
        beginBuf[channel].push(x_interp[channel][numSamples - 1]);
        // endbuf feltoltese ha kesz minden
        for(int i = 0; i < SINC_RADIUS; i++) {
            endBuf[channel].push(x_interp[channel][numSamples + i]);
        }
    }
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::decimate(
    TYPE* const* ptrToBuffers,
    int numChannels,
    int numSamples) {
    if(static_cast<size_t>(numChannels) > x_decim.size() ||
       static_cast<size_t>(numSamples) >
           (interpolatedBuf[0].size() / OVERSAMPLE_FACTOR)) {
        throw std::runtime_error(
            "Oversampler: Output size exceeds configured maxBlockSize.");
    }
    int interpolatedBufSize = interpolatedBuf[0].size();
    // bepakolni az elozo veget az elejere
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

            // bufferbol
            for(int n = 1; n <= SINC_RADIUS * OVERSAMPLE_FACTOR; n++) {
                ptrToBuffers[channel][k] +=
                    sinc[n] *
                    decBeginBuf[channel][SINC_RADIUS * OVERSAMPLE_FACTOR - n];
            }

            // mintakbol
            for(int n = 0; n >= -(SINC_RADIUS * OVERSAMPLE_FACTOR); n--) {
                ptrToBuffers[channel][k] +=
                    sinc[n] * x_decim[channel][index - n];
            }

            // az elemek visszapusholasa 0-3
            for(int i = 0; i < OVERSAMPLE_FACTOR; i++) {
                decBeginBuf[channel].push(x_decim[channel][index + i]);
            }

            ptrToBuffers[channel][k] /= OVERSAMPLE_FACTOR;
        }

        // endbuf feltoltese ha kesz minden
        for(int i = 0; i < SINC_RADIUS * OVERSAMPLE_FACTOR; i++) {
            decEndBuf[channel].push(x_decim[channel][interpolatedBufSize + i]);
        }
    }
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::process(
    std::function<void(std::vector<std::vector<TYPE>>&)> processBlock) {
    processBlock(interpolatedBuf);
}
