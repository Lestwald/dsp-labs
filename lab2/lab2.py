import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from numpy.fft import fft, fftshift, ifft
from scipy import signal
from scipy.fftpack import dct


def dft(x):
    N = len(x)
    spectrum = np.zeros(N, dtype=np.complex_)
    for n in range(N):
        for k in range(N):
            spectrum[n] += x[k] * np.exp(-2j * np.pi * n * k / N)
    return spectrum


def phase_spectrum(spectrum):
    return np.arctan2(np.imag(spectrum), np.real(spectrum))


def mfcc(x, fs, n_filters):
    spectrum = fft(x)
    f_min = 0
    f_max = 1125 * np.log(1 + (fs / 2) / 700)
    mels = np.linspace(f_min, f_max, num=n_filters + 1)
    freqs = 700 * (np.exp(mels / 1125) - 1)
    filter_points = np.floor((len(x) + 1) / fs * freqs).astype(int)
    filters = np.zeros((len(filter_points) - 2, int((len(x)))))
    for n in range(len(filter_points) - 2):
        filters[n, filter_points[n]: filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1]: filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[
            n + 1])
    coeff = np.zeros(len(filters))
    for i in range(len(filters)):
        filters[i] = spectrum * filters[i]
        coeff[i] = sum(abs(filters[i]) ** 2) / len(filters[i])
    coeff = dct(np.log(coeff))
    return coeff


def main():
    # sine wave
    f = 4
    fs = 100
    t = np.arange(0, 20, 1 / fs)
    sine_wave = np.sin(2 * np.pi * f * t)

    sine_wave_spectrum1 = fftshift(dft(sine_wave))
    sine_wave_spectrum2 = fftshift(fft(sine_wave))
    x_sine_wave_spectrum = np.linspace(-fs / 2, fs / 2, len(sine_wave_spectrum1))

    fig = plt.figure(figsize=(6, 9))
    plt.suptitle('Sine wave', fontsize=14)
    plt.subplot(5, 1, 1)
    plt.title('Signal')
    plt.plot(t, sine_wave)
    plt.xlim(0, 1)
    plt.grid(True)

    plt.subplot(5, 1, 2)
    plt.title('Amplitude spectrum (dft)')
    plt.plot(x_sine_wave_spectrum, abs(sine_wave_spectrum1))
    plt.xlim(-6, 6)
    plt.grid(True)

    plt.subplot(5, 1, 3)
    plt.title('Phase spectrum (dft)')
    plt.plot(x_sine_wave_spectrum, phase_spectrum(sine_wave_spectrum1))
    plt.xlim(-6, 6)
    plt.grid(True)

    plt.subplot(5, 1, 4)
    plt.title('Amplitude spectrum (numpy.fft)')
    plt.plot(x_sine_wave_spectrum, abs(sine_wave_spectrum2))
    plt.xlim(-6, 6)
    plt.grid(True)

    plt.subplot(5, 1, 5)
    plt.title('Phase spectrum (numpy.fft)')
    plt.plot(x_sine_wave_spectrum, phase_spectrum(sine_wave_spectrum2))
    plt.xlim(-6, 6)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig('lab2/fig/sine_wave.png')

    # unit impulse
    f = 4
    fs = 100
    unit_impulse = signal.unit_impulse(fs, f)

    unit_impulse_spectrum1 = fftshift(dft(unit_impulse))
    unit_impulse_spectrum2 = fftshift(fft(unit_impulse))
    x_unit_impulse_spectrum = np.linspace(-fs / 2, fs / 2, len(unit_impulse_spectrum1))

    fig = plt.figure(figsize=(6, 9))
    plt.suptitle('Unit impulse', fontsize=14)
    plt.subplot(5, 1, 1)
    plt.title('Signal')
    plt.plot(unit_impulse)
    plt.grid(True)

    plt.subplot(5, 1, 2)
    plt.title('Amplitude spectrum (dft)')
    plt.plot(x_unit_impulse_spectrum, abs(unit_impulse_spectrum1))
    plt.grid(True)

    plt.subplot(5, 1, 3)
    plt.title('Phase spectrum (dft)')
    plt.plot(x_unit_impulse_spectrum, phase_spectrum(unit_impulse_spectrum1))
    plt.grid(True)

    plt.subplot(5, 1, 4)
    plt.title('Amplitude spectrum (numpy.fft)')
    plt.plot(x_unit_impulse_spectrum, abs(unit_impulse_spectrum2))
    plt.grid(True)

    plt.subplot(5, 1, 5)
    plt.title('Phase spectrum (numpy.fft)')
    plt.plot(x_unit_impulse_spectrum, phase_spectrum(unit_impulse_spectrum2))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig('lab2/fig/unit_impulse.png')

    # unit step
    fs = 100
    t = np.arange(-10, 10, 1 / fs)
    unit_step = 1 * (t >= 0)

    unit_step_spectrum1 = fftshift(dft(unit_step))
    unit_step_spectrum2 = fftshift(fft(unit_step))
    x_unit_step_spectrum = np.linspace(-fs / 2, fs / 2, len(unit_step_spectrum1))

    fig = plt.figure(figsize=(6, 9))
    plt.suptitle('Unit step', fontsize=14)
    plt.subplot(5, 1, 1)
    plt.title('Signal')
    plt.plot(unit_step)
    plt.grid(True)

    plt.subplot(5, 1, 2)
    plt.title('Amplitude spectrum (dft)')
    plt.plot(x_unit_step_spectrum, abs(unit_step_spectrum1))
    plt.xlim(-5, 5)
    plt.grid(True)

    plt.subplot(5, 1, 3)
    plt.title('Phase spectrum (dft)')
    plt.plot(x_unit_step_spectrum, phase_spectrum(unit_step_spectrum1))
    plt.xlim(-5, 5)
    plt.grid(True)

    plt.subplot(5, 1, 4)
    plt.title('Amplitude spectrum (numpy.fft)')
    plt.plot(x_unit_step_spectrum, abs(unit_step_spectrum2))
    plt.xlim(-5, 5)
    plt.grid(True)

    plt.subplot(5, 1, 5)
    plt.title('Phase spectrum (numpy.fft)')
    plt.plot(x_unit_step_spectrum, phase_spectrum(unit_step_spectrum2))
    plt.xlim(-5, 5)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig('lab2/fig/unit_step.png')

    # train whistle
    train_whistle, fs = librosa.core.load('lab2/wav/train_whistle.wav')
    train_whistle_spectrum = fftshift(fft(train_whistle))
    x_train_whistle_spectrum = np.linspace(-fs / 2, fs / 2, len(train_whistle_spectrum))

    fig = plt.figure()
    plt.suptitle('Train whistle signal', fontsize=14)
    plt.subplot(2, 1, 1)
    plt.title('Signal')
    plt.plot(train_whistle)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.title('Amplitude spectrum')
    plt.plot(x_train_whistle_spectrum, abs(train_whistle_spectrum))
    plt.xlim(-1000, 1000)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig('lab2/fig/train_whistle.png')

    # sputnik
    sputnik_signal, fs = librosa.core.load('lab2/wav/sputnik_1.wav')
    t = np.linspace(0, len(sputnik_signal) / fs, len(sputnik_signal))
    sputnik_spectrum = fftshift(fft(sputnik_signal))
    f_sputnik_spectrum = np.linspace(-fs / 2, fs / 2, len(sputnik_spectrum))

    fig = plt.figure(figsize=(6, 6))
    plt.suptitle('Sputnik signal', fontsize=14)
    plt.subplot(3, 1, 1)
    plt.title('Signal')
    plt.plot(t, sputnik_signal)
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.title('Amplitude spectrum')
    plt.plot(f_sputnik_spectrum, abs(sputnik_spectrum))
    plt.xlim(-2000, 2000)
    plt.grid(True)

    f, t, Sxx = signal.spectrogram(sputnik_signal, fs)
    plt.subplot(3, 1, 3)
    plt.title('Spectrogram')
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylim(0, 2000)
    plt.tight_layout()
    plt.show()
    fig.savefig('lab2/fig/sputnik.png')

    # dtmf
    dtmf_signal, fs = librosa.core.load('lab2/wav/dtmf.wav')
    f, t, Sxx = signal.spectrogram(dtmf_signal, fs)

    fig = plt.figure()
    plt.title('DTMF signal spectrogram')
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylim(0, 1700)
    plt.show()
    fig.savefig('lab2/fig/dtmf.png')

    # ifft
    speech_signal, fs = librosa.core.load('lab2/wav/human_speech.wav')
    # sd.play(speech_signal, fs)
    # sd.wait()
    speech_spectrum = fft(speech_signal)
    speech_signal_rebuilt_as = np.real(ifft(abs(speech_spectrum)))
    # sd.play(speech_signal_rebuilt_as, fs)
    # sd.wait()
    speech_signal_rebuilt_phs = np.imag(ifft(phase_spectrum(speech_spectrum)))
    # sd.play(speech_signal_rebuilt_phs, fs)
    # sd.wait()

    speech_signal, fs = librosa.core.load('lab2/wav/human_speech.wav')
    print(mfcc(speech_signal, fs, 10))


if __name__ == '__main__':
    main()
