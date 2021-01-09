from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift


def main():
    # sine wave with noise
    f = 10
    fs = 200
    t = np.arange(0, 20, 1 / fs)
    sig = np.sin(2 * np.pi * f * t)
    noise = np.random.normal(1, 0.5, len(t))
    sig = sig + noise
    sig_spectrum = abs(fftshift(fft(sig)))
    x_sig_spectrum = np.linspace(-fs / 2, fs / 2, len(sig_spectrum))

    # finite impulse response filter
    nyq_rate = fs / 2.0
    cutoff_freq = 11
    beta = 5
    N = 35
    taps = signal.firwin(N, cutoff_freq / nyq_rate, window=('kaiser', beta))

    fig = plt.figure(figsize=(6, 6))
    plt.suptitle('FIR-filter with Kaiser window', fontsize=14)
    plt.subplot(3, 1, 1)
    plt.plot(taps)
    plt.title('Impulse response')
    plt.grid(True)

    w, h = signal.freqz(taps)
    plt.subplot(3, 1, 2)
    plt.title('Amplitude response')
    plt.plot((w / np.pi) * nyq_rate, abs(h))
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.title('Phase response')
    plt.plot((w / np.pi) * nyq_rate, np.arctan2(np.imag(h), np.real(h)))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig('lab3/fig/fir_filter.png')

    # signal filtered with fir-filter
    filtered_sig = signal.lfilter(taps, 1, sig)
    filtered_sig_spectrum = abs(fftshift(fft(filtered_sig)))

    fig = plt.figure()
    plt.suptitle('Signal filtered with FIR-filter', fontsize=14)
    plt.subplot(2, 2, 1)
    plt.title('Source signal')
    plt.plot(t, sig)
    plt.xlim(10, 11)
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.title('Source signal spectrum')
    plt.plot(x_sig_spectrum, sig_spectrum)
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.title('Filtered signal')
    plt.plot(t, filtered_sig)
    plt.xlim(10, 11)
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.title('Filtered signal spectrum')
    plt.plot(x_sig_spectrum, filtered_sig_spectrum)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig('lab3/fig/fir_filtered_signal.png')

    # infinite impulse response filter
    cutoff_freq = 11
    b, a = signal.butter(3, cutoff_freq / nyq_rate)

    fig = plt.figure(figsize=(6, 6))
    plt.suptitle('Butterworth filter', fontsize=14)
    plt.subplot(3, 1, 1)
    plt.plot(b)
    plt.title('Impulse response')
    plt.grid(True)

    w, h = signal.freqz(b, a)
    plt.subplot(3, 1, 2)
    plt.title('Amplitude response')
    plt.plot((w / np.pi) * nyq_rate, abs(h))
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.title('Phase response')
    plt.plot((w / np.pi) * nyq_rate, np.arctan2(np.imag(h), np.real(h)))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig('lab3/fig/iir_filter.png')

    # signal filtered with iir-filter
    filtered_sig = signal.lfilter(b, a, sig)
    filtered_sig_spectrum = abs(fftshift(fft(filtered_sig)))

    fig = plt.figure()
    plt.suptitle('Signal filtered with IIR-filter', fontsize=14)
    plt.subplot(2, 2, 1)
    plt.title('Source signal')
    plt.plot(t, sig)
    plt.xlim(10, 11)
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.title('Source signal spectrum')
    plt.plot(x_sig_spectrum, sig_spectrum)
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.title('Filtered signal')
    plt.plot(t, filtered_sig)
    plt.xlim(10, 11)
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.title('Filtered signal spectrum')
    plt.plot(x_sig_spectrum, filtered_sig_spectrum)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig('lab3/fig/iir_filtered_signal.png')


if __name__ == '__main__':
    main()
