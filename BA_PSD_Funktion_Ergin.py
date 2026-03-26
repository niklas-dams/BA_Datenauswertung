import dwdatareader as dw
import numpy as np
from scipy.signal import get_window

def compute_psd_1d(
    signal: np.ndarray,                 #Zeitsignal
    nFFT: int,                          #Segment- und FFT-Länge
    fs: float,                          #Abtastrate in Hz
    overlap: float = 0.5,               #Überlappung
    window_type: str = 'hann',          #Fensterfunktion
    show_progress: bool = False         #Hilfs-Tracking
):
    """
    Berechnet das gemittelte Leistungsdichtespektrum (PSD) eines 1D-Signals
    mit überlappenden Segmenten.

    Parameter
    ----------
    signal : np.ndarray
        1D-Zeitreihe (z. B. Druck- oder Spannungssignal).
    nFFT : int
        Länge der FFT und des Analysefensters.
    fs : float
        Abtastrate in Hz.
    overlap : float, optional
        Anteil der Überlappung zwischen Fenstern (0–1). Standard = 0.5.
    window_type : str, optional
        Fensterart (z. B. 'hann', 'hamming'). Standard = 'hann'.
    show_progress : bool, optional
        Zeigt Fortschrittsbalken für Segmente. Standard = False.

    Rückgabe
    --------
    freqs : np.ndarray
        Frequenzachse (bis Nyquist).
    psd : np.ndarray
        Gemitteltes PSD (realer Anteil).
    psd_per_bin : np.ndarray
        PSD multipliziert mit Δf (Leistung pro Frequenz-Bin).
    """

    if not 0 <= overlap < 1:
        raise ValueError("overlap muss zwischen 0 und kleiner als 1 liegen.")

    if nFFT <= 0:
        raise ValueError("nFFT muss > 0 sein.")

    if fs <= 0:
        raise ValueError("fs muss > 0 sein.")

    signal = np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError("signal muss ein 1D-Array sein.")

    # Fenster und Normierung
    window = get_window(window_type, nFFT)  #Erzeugt Fenster mit Länge nFFT
    U = np.sum(window ** 2)                 #Energienormierung des Fensters

    # Schrittweite aus Überlappung
    step = nFFT - int(nFFT * overlap)       # mit overlap =0.5 => step=nFFT/2
    if step <= 0:
        raise ValueError("Die Schrittweite ist <= 0. overlap ist zu groß.")

    # Frequenzachse bis Nyquist
    freqs = np.fft.rfftfreq(nFFT, d=1.0 / fs)                   #f(a,b), a=länge, b=abstand/abtastrate => Rückgabe: Frequenzwerte (0Hz - f_Ny)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else fs / nFFT   #Breite eines Bins = Frequenzauflösung

    #Signal in Float umwandeln
    sig = signal.astype(float)

    # Anzahl Segmente
    nSeg = max(1, 1 + (len(sig) - nFFT) // step)

    # Speicher für aufsummierte PSD
    nFreqBins = len(freqs)                      #Anzahl Frequenzpunkte von 0 bis Nyquist
    Pxx = np.zeros(nFreqBins, dtype=complex)    #hier wird die aufsummierte PSD gespeichert

    iterator = range(nSeg)                      #Wegen error abgewandelt...

    for k in iterator:
        start = k * step
        seg = sig[start:start + nFFT]

        # falls letztes Segment zu kurz ist
        if len(seg) < nFFT:
            seg = np.pad(seg, (0, nFFT - len(seg)), mode='constant')    #Zero Padding

        # Mittelwert entfernen und fenstern
        seg = (seg - np.mean(seg)) * window

        # FFT
        X = np.fft.fft(seg, nFFT)

        # Nur positive Frequenzen bis Nyquist verwenden
        psd_seg = 2 * X[:nFreqBins] * np.conj(X[:nFreqBins]) / (nFFT * U)

        # Aufsummieren
        Pxx += psd_seg

    # Mittelung über alle Segmente
    Pxx /= nSeg

    # Realteil nehmen
    psd = np.real(Pxx)

    # Leistung pro Frequenz-Bin
    psd_per_bin = psd * df

    return freqs, psd, psd_per_bin



def read_d7d_info(filepath: str):
    """
    Infos aus .d7d-Datei mit dwdatareader.
    """
    with dw.DWFile(filepath) as f:
        info = f.info
    return info


def list_channels(filepath: str):
    """
    verfügbare Kanäle
    """
    with dw.DWFile(filepath) as f:
        print(f"Datei: {filepath}")
        print("Datei-Info:")
        print(f.info)
        print("\nVerfügbare Kanäle:\n")

        for i, ch in enumerate(f.values()):
            try:
                print(f"{i:3d} | {ch.name}")
            except Exception:
                print(f"{i:3d} | <unbekannter Kanalname>")


def load_d7d_channel(filepath: str, channel_name: str):
    with dw.DWFile(filepath) as f:
        available_channels = list(f.keys())

        if channel_name not in available_channels:
            matches = [ch for ch in available_channels if channel_name.lower() in ch.lower()]

            if len(matches) == 0:
                print("\nVerfügbare Kanäle:")
                for ch_name in available_channels:
                    print(f"  {ch_name}")
                raise KeyError(
                    f"Kanal '{channel_name}' wurde nicht gefunden."
                )
            elif len(matches) == 1:
                real_channel_name = matches[0]
                print(f"Kanal '{channel_name}' nicht exakt gefunden.")
                print(f"Verwende stattdessen: '{real_channel_name}'")
            else:
                print(f"Kanal '{channel_name}' nicht exakt gefunden.")
                print("Ähnliche Treffer:")
                for m in matches:
                    print(f"  {m}")
                raise KeyError(
                    f"Mehrere mögliche Kanäle für '{channel_name}' gefunden."
                )
        else:
            real_channel_name = channel_name

        ch = f[real_channel_name]
        series = ch.series()
        signal = series.to_numpy(dtype=float)

        fs = None

        # Versuch 1: direkt aus Kanalattributen
        for attr in ["sample_rate", "sampling_rate", "fs", "rate"]:
            if hasattr(ch, attr):
                value = getattr(ch, attr)
                if value is not None:
                    try:
                        fs = float(value)
                        break
                    except Exception:
                        pass

        # Versuch 2: aus Zeitindex berechnen
        if fs is None and len(series.index) > 1:
            try:
                dt = series.index[1] - series.index[0]

                if isinstance(dt, (int, float, np.integer, np.floating)):
                    if dt > 0:
                        fs = 1.0 / float(dt)

                elif hasattr(dt, "total_seconds"):
                    dt_sec = dt.total_seconds()
                    if dt_sec > 0:
                        fs = 1.0 / dt_sec

            except Exception:
                pass

        if fs is None:
            raise ValueError(
                f"Abtastrate für Kanal '{real_channel_name}' konnte nicht bestimmt werden."
            )

        print(f"Verwendeter Kanal: {real_channel_name}")
        print(f"Ermittelte Abtastrate fs: {fs:.3f} Hz")
        #print(f"Erste 5 Indexwerte: {series.index[:5]}")
        #print(f"Erste 5 Signalwerte: {signal[:5]}")

    return signal, fs, series
