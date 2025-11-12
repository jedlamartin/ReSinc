#pragma once

#include <array>
#include <cmath>
#include <functional>
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
 * - operator[](index) accesses elements relative to the newest element:
 *   index 0 == most recently pushed element, index increases towards older
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
    void push(T element);
    T& operator[](size_t index);
    void clear();
};

namespace Window {
    template<size_t N>
    class Window {
    public:
        virtual ~Window() = default;
        virtual float& operator[](size_t i);
        virtual const float& operator[](size_t i) const;
        void applyOn(std::array<float, N>& data) const;
        size_t size() const;

    protected:
        Window() = default;

    private:
        std::array<float, N> window;
    };

    template<size_t N>
    class Kaiser : public Window<N> {
    public:
        Kaiser(float beta);
        ~Kaiser() = default;

    private:
        const float beta;
        float besselI0(float x);
    };
}    // namespace Window

// TODO: all resample functions
template<int iSize, int N>
class ReSample {
public:
    void configure(double sampleRate);
    void interpolate(juce::AudioBuffer<float>& buffer);
    void decimate(juce::AudioBuffer<float>& buffer);

    void process(std::function<void(juce::AudioBuffer<float>&)> processBlock);

private:
    Sinc<iSize, N> sinc;
    juce::AudioBuffer<float> interpolatedBuf;
    std::vector<CircularBuffer<float, N>> beginBuf, endBuf;
    std::vector<CircularBuffer<float, N * iSize>> decBeginBuf, decEndBuf;

    class Sinc {
    private:
        std::array<float, (N + 1) * iSize> sinc;

    public:
        Sinc();

        float& operator[](int i);

        float& operator()(int i, int delta);

        void configure(float sampleRate);

        size_t size() const;

        void applyKaiser(float beta);
    };
};

// Definitions for Window
template<size_t N>
inline size_t Window::Window<N>::size() const {
    return N;
}

template<size_t N>
inline float& Window::Window<N>::operator[](size_t i) {
    return window[i];
}

template<size_t N>
inline const float& Window::Window<N>::operator[](size_t i) const {
    return window[i];
}

template<size_t N>
inline void Window::Window<N>::applyOn(std::array<float, N>& data) const {
    for(size_t i = 0; i < N; i++) {
        data[i] *= window[i];
    }
}

// Definitions for Kaiser
template<size_t N>
inline Window::Kaiser<N>::Kaiser(float beta) : Window {}, beta {beta} {
    for(int n = 0; n < (N + 1) / 2; n++) {
        float arg = beta * std::sqrt(1 - (2.0f * n / N) * (2.0f * n / N));
        float num = besselI0(arg);
        float den = besselI0(beta);
        this[n] = num / den;
    }
}

template<size_t N>
inline float Window::Kaiser<N>::besselI0(float x) {
    const float epsilon = 1e-6f;
    float sum = 1.0f;
    float term = 1.0f;
    float k = 1.0f;
    float factorial = 1.0f;

    while(term > epsilon * sum) {
        factorial *= k;
        term = std::pow(x / 2.0f, 2 * k) / (factorial * factorial);
        sum += term;
        k += 1.0f;
    }

    return sum;
}

// Definitions for sincArray
template<int iSize, int N>
inline ReSample<iSize, N>::Sinc::Sinc() : sinc {0} {}

template<int iSize, int N>
inline float& ReSample<iSize, N>::Sinc::operator[](int i) {
    return sinc[i < 0 ? -i : i];
}

template<int iSize, int N>
inline float& ReSample<iSize, N>::Sinc::operator()(int i, int delta) {
    if(i < 0) return (*this)[(-i) * iSize - delta];
    return (*this)[i * iSize + delta];
}

template<int iSize, int N>
inline void ReSample<iSize, N>::Sinc::configure(float sampleRate) {
    float fc = sampleRate / 2;
    float T = 1 / sampleRate;
    for(int i = 0; i <= N; i++) {
        for(int delta = 0; delta < iSize; delta++) {
            float index =
                static_cast<float>(i) +
                (1.0f / static_cast<float>(iSize)) * static_cast<float>(delta);
            (*this)(i, delta) = std::sin(2.0f * M_PI * fc * index * T) /
                                (2.0f * M_PI * fc * index * T);
        }
    }
    (*this)[0] = 1;
}

template<int iSize, int N>
inline size_t ReSample<iSize, N>::Sinc::size() const {
    return this->sinc.size();
}

template<int iSize, int N>
inline void ReSample<iSize, N>::Sinc::applyKaiser(float beta) {
    Kaiser<iSize, (N + 1) * iSize * 2> kaiser(beta);
    kaiser.applyOn(sinc);
}

// Definitions for ReSample
template<int iSize, int N>
inline void ReSample<iSize, N>::configure(double sampleRate) {
    sinc.configure(static_cast<float>(sampleRate));
    sinc.applyKaiser(5.0f);
    beginBuf.clear();
    endBuf.clear();
    decBeginBuf.clear();
    decEndBuf.clear();
}

template<int iSize, int N>
inline void ReSample<iSize, N>::interpolate(juce::AudioBuffer<float>& buffer) {
    int channelSize = buffer.getNumChannels();
    int originalBufSize = buffer.getNumSamples();
    int interpolatedBufSize = originalBufSize * iSize;
    beginBuf.resize(channelSize, CircularBuffer<float, N>(0.0f));
    endBuf.resize(channelSize, CircularBuffer<float, N>(0.0f));
    juce::AudioBuffer<float> x(channelSize, originalBufSize + N);
    for(int channel = 0; channel < channelSize; channel++) {
        x.copyFrom(channel, N, buffer, channel, 0, originalBufSize);
        float* currentSample = x.getWritePointer(channel);
        for(int i = 0; i < N; i++) {
            // forditva masolom bele
            currentSample[N - i - 1] = endBuf[channel][i];
        }
    }

    interpolatedBuf.setSize(channelSize, interpolatedBufSize, true, true, true);
    for(int channel = 0; channel < channelSize; channel++) {
        float const* channelSamples = x.getReadPointer(channel);
        float* iSamples = interpolatedBuf.getWritePointer(channel);
        // bevart mintak
        // az elsot automatikusan kitoltjuk, hogy a buffereles jo legyen
        *iSamples = *channelSamples;
        for(int k = 1; k < interpolatedBufSize; k++) {
            int delta = k % iSize;
            int index = k / iSize;
            if(delta != 0) {
                iSamples[k] = 0;
                // mintakbol
                for(int n = -N; n <= 0; n++) {
                    iSamples[k] +=
                        this->sinc(n, delta) * channelSamples[index - n];
                }
                // bufferbol
                for(int n = 1; n <= N; n++) {
                    iSamples[k] +=
                        this->sinc(n, delta) * beginBuf[channel][n - 1];
                }
            } else {
                iSamples[k] = channelSamples[index];
                beginBuf[channel].push(channelSamples[index - 1]);
            }
        }
        // az utolso is belekeruljon a bufferbe
        beginBuf[channel].push(channelSamples[originalBufSize - 1]);
        // endbuf feltoltese ha kesz minden
        for(int i = 0; i < N; i++) {
            endBuf[channel].push(channelSamples[originalBufSize + i]);
        }
    }
}

template<int iSize, int N>
inline void ReSample<iSize, N>::decimate(juce::AudioBuffer<float>& buffer) {
    int channelSize = buffer.getNumChannels();
    int originalBufSize = buffer.getNumSamples();
    int interpolatedBufSize = this->interpolatedBuf.getNumSamples();
    decBeginBuf.resize(channelSize, CircularBuffer<float, N * iSize>(0.0f));
    decEndBuf.resize(channelSize, CircularBuffer<float, N * iSize>(0.0f));

    // bepakolni az elozo veget az elejere
    juce::AudioBuffer<float> x(channelSize, interpolatedBufSize + N * iSize);
    for(int channel = 0; channel < channelSize; channel++) {
        x.copyFrom(channel,
                   N * iSize,
                   interpolatedBuf,
                   channel,
                   0,
                   interpolatedBufSize);
        float* currentSample = x.getWritePointer(channel);
        for(int i = 0; i < N * iSize; i++) {
            currentSample[N * iSize - i - 1] = decEndBuf[channel][i];
        }
    }

    for(int channel = 0; channel < channelSize; channel++) {
        float const* iSamples = x.getReadPointer(channel);
        float* samples = buffer.getWritePointer(channel);
        for(int k = 0; k < originalBufSize; k++) {
            int index = iSize * k;
            samples[k] = 0;

            // bufferbol
            for(int n = 1; n <= N * iSize; n++) {
                samples[k] += sinc[n] * decBeginBuf[channel][n - 1];
            }

            // mintakbol
            for(int n = 0; n >= -(N * iSize); n--) {
                samples[k] += sinc[n] * iSamples[index - n];
            }

            // az elemek visszapusholasa 0-3
            for(int i = 0; i < iSize; i++) {
                decBeginBuf[channel].push(iSamples[index + i]);
            }

            samples[k] /= iSize;
        }

        // endbuf feltoltese ha kesz minden
        for(int i = 0; i < N * iSize; i++) {
            decEndBuf[channel].push(iSamples[interpolatedBufSize + i]);
        }
    }
}

template<int iSize, int N>
inline void ReSample<iSize, N>::process(
    std::function<void(juce::AudioBuffer<float>&)> processBlock) {
    processBlock(this->interpolatedBuf);
}

template<class T, size_t size>
inline CircularBuffer<T, size>::CircularBuffer(T initValue) :
    oldestIndex {0}, buf {std::move(initValue)} {}

template<class T, size_t size>
inline void CircularBuffer<T, size>::push(T element) {
    this->buf[this->oldestIndex] = std::move(element);
    this->oldestIndex = this->oldestIndex == size ? 0 : oldestIndex + 1;
}

template<class T, size_t size>
inline T& CircularBuffer<T, size>::operator[](size_t index) {
    return this->buf[(this->oldestIndex - index - 1 + size) % size];
}

template<class T, size_t size>
inline void CircularBuffer<T, size>::clear() {
    memset(this->buf.data(), T {0}, size * sizeof(T));
    this->oldestIndex = 0;
}
