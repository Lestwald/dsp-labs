import librosa
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fft, fftshift


def convolve(a, b):
    result = np.zeros(len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            result[i + j] += a[i] * b[j]
    return result


def karplus_strong(freq, duration=1, sr=44100):
    noise = np.random.uniform(-1, 1, int(sr / freq))
    samples = np.zeros(int(sr * duration))
    for i in range(len(noise)):
        samples[i] = noise[i]
    for i in range(len(noise), len(samples)):
        samples[i] = (samples[i - len(noise)] + samples[i - len(noise) - 1]) / 2
    return samples


def main():
    # convolution of square signals
    t = np.linspace(0, 1, 100)
    square1 = np.concatenate((signal.square(np.pi * t) / 2 + 0.5, np.zeros(100)))
    square2 = np.concatenate((np.zeros(100), signal.square(np.pi * t) / 2 + 0.5))
    result1 = np.convolve(square1, square2)
    result2 = convolve(square1, square2)
    t = np.linspace(0, 2, len(square1))

    fig = plt.figure()
    plt.subplot(3, 2, 1)
    plt.title('Signal 1')
    plt.plot(t, square1)
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.title('Signal 2')
    plt.plot(t, square2)
    plt.subplot(3, 1, 2)
    plt.grid(True)

    plt.title('Convolution (np.convolve)')
    plt.plot(np.linspace(0, 4, len(result1)), result1)
    plt.subplot(3, 1, 3)
    plt.grid(True)

    plt.title('Convolution (my function)')
    plt.plot(np.linspace(0, 4, len(result2)), result2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig('lab1/fig/convolution.png')

    # load speech signal
    sig, sr = librosa.core.load('lab1/wav/voice.wav')
    # sig, sr = librosa.core.load('lab1/wav/voice.wav', sr=11000)
    # sig, sr = librosa.core.load('lab1/wav/voice.wav', sr=8000)
    # sd.play(sig, sr)
    # sd.wait()

    # low-pass filter
    cutoff_freq = 1200
    t = np.arange(sr) - (sr - 1) / 2
    sinc_filter = np.sinc(2 * cutoff_freq / sr * t)
    sinc_filter /= np.sum(sinc_filter)
    sinc_filter_spectrum = abs(fftshift(fft(sinc_filter)))

    fig = plt.figure()
    plt.suptitle('Low-pass filter', fontsize=14)
    plt.subplot(2, 1, 1)
    plt.title('Impulse response')
    plt.plot(t, sinc_filter)
    plt.xlim(-100, 100)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.title('Frequency response')
    plt.plot(t, sinc_filter_spectrum)
    plt.xlim(-3000, 3000)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig('lab1/fig/filter.png')

    sig_spectrum = abs(fftshift(fft(sig)))
    sig_filtered = signal.fftconvolve(sig, sinc_filter)
    sig_filtered_spectrum = abs(fftshift(np.fft.fft(sig_filtered)))
    # sd.play(sig_filtered, sr)
    # sd.wait()

    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('Raw signal')
    plt.plot(np.linspace(-sr / 2, sr / 2, len(sig)), sig)
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.title('Raw signal spectrum')
    plt.plot(np.linspace(-sr / 2, sr / 2, len(sig_spectrum)), sig_spectrum)
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.title('Filtered signal')
    plt.plot(np.linspace(-sr / 2, sr / 2, len(sig_filtered)), sig_filtered)
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.title('Filtered signal spectrum')
    plt.plot(np.linspace(-sr / 2, sr / 2, len(sig_filtered_spectrum)), sig_filtered_spectrum)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig('lab1/fig/filtered_signal.png')

    # cathedral reverberation
    cath_sig, sr = librosa.core.load('lab1/wav/cath_IR.wav')
    sig_rev = signal.fftconvolve(sig, cath_sig)
    # sd.play(sig_rev, sr)
    # sd.wait()

    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Signal')
    plt.plot(sig)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.title('Signal with reverberation')
    plt.plot(sig_rev)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig('lab1/fig/signal_with_reverberation.png')

    # karplus-strong
    sr = 44100
    duration = 3
    d = karplus_strong(293.66, duration, sr)
    f = karplus_strong(349.23, duration, sr)
    a = karplus_strong(440.0, duration, sr)
    dm = d + f + a
    sd.play(dm, sr)
    sd.wait()


if __name__ == '__main__':
    main()
