import dwdatareader as dw
import numpy as np
from scipy.signal import get_window
import re #regular expressions ((Regex) für filename search)
import os #Operating System (für arbeiten im Dateipfard, beides a bit dirty...)

def compute_psd_1d(
    signal: np.ndarray,                 #Zeitsignal
    nFFT: int,                          #Segment- und FFT-Länge
    fs: float,                          #Abtastrate in Hz
    overlap: float = 0.5,               #Überlappung
    window_type: str = 'hann',          #Fensterfunktion
    #show_progress: bool = False         #Hilfs-Tracking
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
        ch = f[channel_name]
        series = ch.series()
        signal = series.to_numpy(dtype=float)

        fs = None

        #aus Kanalattributen
        for attr in ["sample_rate", "sampling_rate", "fs", "rate"]:
            if hasattr(ch, attr):
                value = getattr(ch, attr)
                if value is not None:
                    try:
                        fs = float(value)
                        break
                    except Exception:
                        pass

        #aus Zeitindex (backup)
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
                f"Abtastrate für Kanal '{channel_name}' konnte nicht bestimmt werden."
            )

        #print(f"Verwendeter Kanal: {channel_name}")
        #print(f"Ermittelte Abtastrate fs: {fs:.3f} Hz")
        #print(f"Erste 5 Indexwerte: {series.index[:5]}")
        #print(f"Erste 5 Signalwerte: {signal[:5]}")

    return signal, fs, series


def get_psi(
    ps1: np.ndarray,            # statischer Druck Einlass [mbar]
    ps2: np.ndarray,            # statischer Druck Auslass [mbar]
    n: np.ndarray,              # Drehzahl [rpm]
    r: float,                   # Radius [m]
    p_halle: np.ndarray,        # Umgebungsdruck [mbar]
    T_halle: np.ndarray         # Temperatur [K]
):
    """
    Berechnet den Druckbeiwert psi eines Verdichters.

    Parameter
    ----------
    ps1 : np.ndarray
        statischer Druck vor dem Verdichter [mbar]
    ps2 : np.ndarray
        statischer Druck nach dem Verdichter [mbar]
    n : np.ndarray
        Drehzahl [rpm]
    r : float
        Radius (z. B. Schaufelspitze) [m]
    p_amb : float
        Umgebungsdruck [Pa]
    T_amb : float
        Umgebungstemperatur [K]

    Rückgabe
    --------
    psi : float
        Druckbeiwert
    """

    # Mittelwerte bilden (stationärer Betrieb angenommen)
    ps1_mean = np.mean(ps1)
    ps2_mean = np.mean(ps2)
    n_mean = np.mean(n)
    p_mean = np.mean(p_halle)
    T_mean = np.mean(T_halle)

    # Einheit: mbar → Pa
    ps1_mean *= 100
    ps2_mean *= 100
    p_mean   *= 100

    # Druckdifferenz
    dp = ps2_mean - ps1_mean

    # Luftdichte (ideales Gas)
    R = 287.0                   # J/(kg K)
    rho = p_mean / (R * T_mean)

    # Drehzahl → Umfangsgeschwindigkeit
    omega = 2 * np.pi * n_mean / 60.0
    U = omega * r

    # Sicherheitscheck
    if U == 0:
        raise ValueError("Umfangsgeschwindigkeit error")

    # psi berechnen
    psi = dp / (rho * U**2)

    return psi


def get_drosselwert_from_filename(filepath: str):
    """
    Extrahiert den Drosselwert aus dem Dateinamen.
    Beispiel:
        ..._d122_... -> 122

    Rückgabe:
        int oder None
    """
    filename = os.path.basename(filepath)
    match = re.search(r"_d(\d+)_", filename)

    if match:
        return int(match.group(1))
    return None
