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
 * @tparam T    Element type stored in the buffer.
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
template<class T, size_t size>
class CircularBuffer {
private:
    std::array<T, size> buf;
    size_t oldestIndex;

public:
    CircularBuffer(T initValue);
    CircularBuffer(const CircularBuffer&) = default;
    CircularBuffer(CircularBuffer&&) noexcept = default;
    CircularBuffer& operator=(const CircularBuffer&) = default;
    CircularBuffer& operator=(CircularBuffer&&) noexcept = default;
    void push(T element);
    T& operator[](size_t index);
    void clear();
};

namespace Window {
    template<typename T, size_t N>
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
    class Window {
    public:
        virtual ~Window() = default;
        Window(const Window&) = default;
        Window(Window&&) = default;
        Window& operator=(const Window&) = delete;
        Window& operator=(Window&&) = delete;

        virtual T& operator[](size_t i);
        virtual const T& operator[](size_t i) const;
        void applyOn(std::array<T, N>& data) const;
        size_t size() const;

    protected:
        Window() = default;

    private:
        std::array<T, N> window;
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
    template<typename T, size_t N>
    class Kaiser : public Window<T, N> {
    public:
        Kaiser(T beta);
        ~Kaiser() = default;
        Kaiser(const Kaiser&) = default;
        Kaiser(Kaiser&&) = default;
        Kaiser& operator=(const Kaiser&) = delete;
        Kaiser& operator=(Kaiser&&) = delete;

    private:
        const T beta;
        T besselI0(T x);
    };
}    // namespace Window

/**
 * @brief Multi-channel resampler.
 *
 * Resample converts between sample rates by interpolating and decimating
 * buffers using a precomputed Sinc table. The class manages per-channel
 * circular history buffers for block-based processing.
 *
 * Template parameters:
 * @tparam T     Sample type (defaults to float).
 * @tparam iSize Interpolation factor (number of sub-samples per input
 *               sample).
 * @tparam N     Sinc kernel radius (number of taps on each side).
 *
 * Main methods:
 * - configure(sampleRate): prepares internal tables for the given rate.
 * - interpolate(ptrToBuffers, numChannels, numSamples): performs upsampling.
 * - decimate(ptrToBuffers): performs downsampling.
 * - process(processBlock): calls the provided function with the
 *   interpolated buffer.
 *
 * Complexity: per-sample cost depends on N and iSize (O(N*iSize) per
 * output sample in the naive implementation).
 */
template<int iSize, int N, typename T = float>
class ReSample {
public:
    ReSample();
    ~ReSample() = default;
    ReSample(const ReSample&) = delete;
    ReSample& operator=(const ReSample&) = delete;
    ReSample(ReSample&&) noexcept = default;
    ReSample& operator=(ReSample&&) noexcept = default;

    void configure(T sampleRate);
    void interpolate(const T* const* ptrToBuffers,
                     int numChannels,
                     int numSamples);
    void decimate(T* const* ptrToBuffers);

    void process(
        std::function<void(std::vector<std::vector<T>>&)> processBlock);

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
     *   fractional offset delta (0..iSize-1).
     * - configure(sampleRate) : fills the table for the provided sample rate.
     * - applyWindow(window) : multiplies the sinc table by a window function
     *   (e.g. Kaiser) to reduce ringing.
     */
    class Sinc {
    private:
        std::array<T, (N + 1) * iSize> sinc;

    public:
        Sinc();
        ~Sinc() = default;
        Sinc(const Sinc&) = default;
        Sinc(Sinc&&) noexcept = default;
        Sinc& operator=(const Sinc&) = default;
        Sinc& operator=(Sinc&&) noexcept = default;

        T& operator[](int i);
        T& operator()(int i, int delta);
        void configure(T sampleRate);
        size_t size() const;
        void applyWindow(Window::Window<T, (N + 1) * iSize * 2> window);
    };
    Sinc sinc;
    std::vector<std::vector<T>> interpolatedBuf;
    std::vector<CircularBuffer<T, N>> beginBuf, endBuf;
    std::vector<CircularBuffer<T, N * iSize>> decBeginBuf, decEndBuf;
    size_t numChannels;
    size_t numSamples;
};

// CircularBuffer definitions
template<class T, size_t size>
inline CircularBuffer<T, size>::CircularBuffer(T initValue) : oldestIndex {0} {
    buf.fill(initValue);
}

template<class T, size_t size>
inline void CircularBuffer<T, size>::push(T element) {
    this->buf[this->oldestIndex] = std::move(element);
    this->oldestIndex = this->oldestIndex == size - 1 ? 0 : oldestIndex + 1;
}

template<class T, size_t size>
inline T& CircularBuffer<T, size>::operator[](size_t index) {
    return this->buf[(this->oldestIndex + index) % size];
}

template<class T, size_t size>
inline void CircularBuffer<T, size>::clear() {
    buf.fill(T {0});
    this->oldestIndex = 0;
}

// Definitions for Window
template<typename T, size_t N>
inline size_t Window::Window<T, N>::size() const {
    return N;
}

template<typename T, size_t N>
inline T& Window::Window<T, N>::operator[](size_t i) {
    return window[i];
}

template<typename T, size_t N>
inline const T& Window::Window<T, N>::operator[](size_t i) const {
    return window[i];
}

template<typename T, size_t N>
inline void Window::Window<T, N>::applyOn(std::array<T, N>& data) const {
    for(size_t i = 0; i < N; i++) {
        data[i] *= window[i];
    }
}

// Definitions for Kaiser
template<typename T, size_t N>
inline Window::Kaiser<T, N>::Kaiser(T beta) : Window<T, N> {}, beta {beta} {
    if(beta < T(0.0)) {
        throw std::invalid_argument("Kaiser beta value must be non-negative.");
    }
    const T M = static_cast<T>(N * 2);
    const T den = besselI0(beta);
    for(int n = 0; n < N; n++) {
        T arg = beta * std::sqrt(1.0 - std::pow(2.0 * n / M, 2.0));
        T num = besselI0(arg);
        (*this)[n] = num / den;
    }
}

template<typename T, size_t N>
inline T Window::Kaiser<T, N>::besselI0(T x) {
    const T epsilon = T(1e-6);
    T sum = T(1.0);
    T term = T(1.0);
    T k = T(1.0);
    T factorial = T(1.0);
    while(term > epsilon * sum) {
        factorial *= k;
        term = std::pow(x / T(2.0), T(2) * k) / (factorial * factorial);
        sum += term;
        k += T(1.0);
    }

    return sum;
}

// Definitions for sincArray
template<int iSize, int N, typename T>
inline ReSample<iSize, N, T>::Sinc::Sinc() : sinc {0} {}

template<int iSize, int N, typename T>
inline T& ReSample<iSize, N, T>::Sinc::operator[](int i) {
    return sinc[i < 0 ? -i : i];
}

template<int iSize, int N, typename T>
inline T& ReSample<iSize, N, T>::Sinc::operator()(int i, int delta) {
    if(i < 0) return (*this)[(-i) * iSize - delta];
    return (*this)[i * iSize + delta];
}

template<int iSize, int N, typename T>
inline void ReSample<iSize, N, T>::Sinc::configure(T sampleRate) {
    T fc = sampleRate / T(2);
    T T_ = T(1) / sampleRate;
    for(int i = 0; i <= N; i++) {
        for(int delta = 0; delta < iSize; delta++) {
            T index = static_cast<T>(i) +
                      (T(1) / static_cast<T>(iSize)) * static_cast<T>(delta);
            (*this)(i, delta) = std::sin(T(2) * M_PI * fc * index * T_) /
                                (T(2) * M_PI * fc * index * T_);
        }
    }
    (*this)[0] = T(1);
}

template<int iSize, int N, typename T>
inline size_t ReSample<iSize, N, T>::Sinc::size() const {
    return this->sinc.size();
}

template<int iSize, int N, typename T>
inline void ReSample<iSize, N, T>::Sinc::applyWindow(
    Window::Window<T, (N + 1) * iSize * 2> window) {
    for(int i = 0; i < (N + 1) * iSize; ++i) {
        sinc[i] = sinc[i] * window[i + (N + 1) * iSize];
    }
}

template<int iSize, int N, typename T>
ReSample<iSize, N, T>::ReSample() : numChannels(0), numSamples(0) {}

template<int iSize, int N, typename T>
inline void ReSample<iSize, N, T>::configure(T sampleRate) {
    if(sampleRate <= T(0.0)) {
        throw std::invalid_argument("Sample rate must be a positive number.");
    }

    sinc.configure(static_cast<T>(sampleRate));
    sinc.applyWindow(Window::Kaiser<T, (N + 1) * iSize * 2> {5.0});
    beginBuf.clear();
    endBuf.clear();
    decBeginBuf.clear();
    decEndBuf.clear();
}

template<int iSize, int N, typename T>
inline void ReSample<iSize, N, T>::interpolate(const T* const* ptrToBuffers,
                                               int numChannels,
                                               int numSamples) {
    if(numChannels <= 0 || numSamples <= 0) {
        throw std::invalid_argument(
            "Number of channels and samples must be positive.");
    }
    this->numChannels = numChannels;
    this->numSamples = numSamples;
    int interpolatedBufSize = numSamples * iSize;
    beginBuf.resize(numChannels, CircularBuffer<T, N>(T(0.0)));
    endBuf.resize(numChannels, CircularBuffer<T, N>(T(0.0)));
    std::vector<std::vector<T>> x(numChannels, std::vector<T>(numSamples + N));
    for(int channel = 0; channel < numChannels; channel++) {
        for(int i = 0; i < numSamples + N; ++i) {
            x[channel][i] =
                i >= N ? ptrToBuffers[channel][i - N] : endBuf[channel][i];
        }
    }

    interpolatedBuf.resize(numChannels, std::vector<T>(interpolatedBufSize));
    for(int channel = 0; channel < numChannels; channel++) {
        // bevart mintak
        // az elsot automatikusan kitoltjuk, hogy a buffereles jo legyen
        interpolatedBuf[channel][0] = x[channel][0];
        for(int k = 1; k < interpolatedBufSize; k++) {
            int delta = k % iSize;
            int index = k / iSize;
            if(delta != 0) {
                interpolatedBuf[channel][k] = 0;
                // mintakbol
                for(int n = -N; n <= 0; n++) {
                    interpolatedBuf[channel][k] +=
                        this->sinc(n, delta) * x[channel][index - n];
                }
                // bufferbol
                for(int n = 1; n <= N; n++) {
                    interpolatedBuf[channel][k] +=
                        this->sinc(n, delta) * beginBuf[channel][N - n];
                }
            } else {
                interpolatedBuf[channel][k] = x[channel][index];
                beginBuf[channel].push(x[channel][index - 1]);
            }
        }
        // az utolso is belekeruljon a bufferbe
        beginBuf[channel].push(x[channel][numSamples - 1]);
        // endbuf feltoltese ha kesz minden
        for(int i = 0; i < N; i++) {
            endBuf[channel].push(x[channel][numSamples + i]);
        }
    }
}

template<int iSize, int N, typename T>
inline void ReSample<iSize, N, T>::decimate(T* const* ptrToBuffers) {
    int interpolatedBufSize = numSamples * iSize;
    decBeginBuf.resize(numChannels, CircularBuffer<T, N * iSize>(T(0.0)));
    decEndBuf.resize(numChannels, CircularBuffer<T, N * iSize>(T(0.0)));

    // bepakolni az elozo veget az elejere
    std::vector<std::vector<T>> x(
        numChannels, std::vector<T>(interpolatedBufSize + N * iSize));
    for(int channel = 0; channel < numChannels; channel++) {
        for(int i = 0; i < interpolatedBufSize + N * iSize; ++i) {
            x[channel][i] = i >= N * iSize ?
                                interpolatedBuf[channel][i - N * iSize] :
                                decEndBuf[channel][i];
        }
    }

    for(int channel = 0; channel < numChannels; channel++) {
        for(int k = 0; k < numSamples; k++) {
            int index = iSize * k;
            ptrToBuffers[channel][k] = 0;

            // bufferbol
            for(int n = 1; n <= N * iSize; n++) {
                ptrToBuffers[channel][k] +=
                    sinc[n] * decBeginBuf[channel][N * iSize - n];
            }

            // mintakbol
            for(int n = 0; n >= -(N * iSize); n--) {
                ptrToBuffers[channel][k] += sinc[n] * x[channel][index - n];
            }

            // az elemek visszapusholasa 0-3
            for(int i = 0; i < iSize; i++) {
                decBeginBuf[channel].push(x[channel][index + i]);
            }

            ptrToBuffers[channel][k] /= iSize;
        }

        // endbuf feltoltese ha kesz minden
        for(int i = 0; i < N * iSize; i++) {
            decEndBuf[channel].push(x[channel][interpolatedBufSize + i]);
        }
    }
}

template<int iSize, int N, typename T>
inline void ReSample<iSize, N, T>::process(
    std::function<void(std::vector<std::vector<T>>&)> processBlock) {
    processBlock(interpolatedBuf);
}
