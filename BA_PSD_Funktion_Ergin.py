import dwdatareader as dw
import numpy as np
from scipy.signal import get_window, welch
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

        # FFT (Fourier-Transformation)
        X = np.fft.fft(seg, nFFT)

        # Nur positive Frequenzen bis Nyquist verwenden
        psd_seg = 2 * X[:nFreqBins] * np.conj(X[:nFreqBins]) / (nFFT * U) #Grund für unterschiedliche y-Skalierung zu Scipy! Hier pro Bin nicht pro Hz
        # Im Grunde (|X(f)|^2)/(nFFT * U) ohne nomierung auf Faktor somit pro Bin, zudem werden alle Freqs verdoppelt (0 Hz (DC) und Ny müssten *1 bleiben aber egal...)


        # Aufsummieren
        Pxx += psd_seg

    # Mittelung über alle Segmente
    Pxx /= nSeg

    # Realteil nehmen
    psd = np.real(Pxx)

    # Leistung pro Frequenz-Bin
    psd_per_bin = psd * df

    return freqs, psd, psd_per_bin


def compute_psd_1d_scipy(
    signal: np.ndarray,
    nFFT: int,
    fs: float,
    overlap: float = 0.5,
    window_type: str = "hann"
):
    if not 0 <= overlap < 1:
        raise ValueError("overlap muss zwischen 0 und kleiner als 1 liegen.")
    if nFFT <= 0:
        raise ValueError("nFFT muss > 0 sein.")
    if fs <= 0:
        raise ValueError("fs muss > 0 sein.")

    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal muss ein 1D-Array sein.")

    noverlap = int(nFFT * overlap)

    freqs, psd = welch(
        signal,
        fs=fs,
        window=window_type,
        nperseg=nFFT,
        noverlap=noverlap,
        nfft=nFFT,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average="mean",
    )

    df = freqs[1] - freqs[0] if len(freqs) > 1 else fs / nFFT
    psd_per_bin = psd * df

    return freqs, psd, psd_per_bin


def read_d7d_info(filepath: str):
    """
    Infos aus .d7d-Datei mit dwdatareader.
    """
    with dw.DWFile(filepath) as f:
        info = f.info
    return info

def find_pressure_psi_channels(filepath: str):
    keywords = [
        "ps", "pt", "p0", "total", "tot", "stau",
        "druck", "pressure", "psi"
    ]

    with dw.DWFile(filepath) as f:
        for i, ch in enumerate(f.values()):
            try:
                name = ch.name
                unit = getattr(ch, "unit", "")
                text = f"{name} {unit}".lower()

                if any(k in text for k in keywords):
                    series = ch.series()
                    values = series.to_numpy(dtype=float)

                    print("\n" + "-" * 80)
                    print(f"{i:3d} | {name}")
                    print(f"    Einheit: {unit}")
                    print(f"    Samples: {len(values)}")
                    print(f"    Mittelwert: {np.nanmean(values)}")
                    print(f"    Min:        {np.nanmin(values)}")
                    print(f"    Max:        {np.nanmax(values)}")
                    print(f"    Erste 5:    {values[:5]}")

            except Exception as e:
                print(f"{i:3d} | Fehler: {e}")

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

def get_rho(filepath):
    pt1_signal, _, _ = load_d7d_channel(filepath, "pt")
    pHalle_signal, _, _= load_d7d_channel(filepath, "pHalle")
    T_signal, _, _ = load_d7d_channel(filepath, "THalle")
    
    pHalle_mean = np.mean(pHalle_signal) *100
    pt1_mean = np.mean(pt1_signal) * 100
    T_mean = np.mean(T_signal)
    rho_mean = (pt1_mean + pHalle_mean) / (287.0 * T_mean)

    return rho_mean

def get_psi(filepath):
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

    # Kanäle laden
    ps1,_,_ = load_d7d_channel(filepath, "ps1")
    ps2,_,_ = load_d7d_channel(filepath, "ps2")
    pt1,_,_ = load_d7d_channel(filepath, "pt")
    n,_,_ = load_d7d_channel(filepath, "Drehzahl")
    pHalle,_,_ = load_d7d_channel(filepath, "pHalle")
    THalle,_,_ = load_d7d_channel(filepath, "THalle")

    # Mittelwerte bilden (stationärer Betrieb angenommen)
    ps1_diff_mean = np.mean(ps1)
    ps2_diff_mean = np.mean(ps2)
    pt1_diff_mean = np.mean(pt1)
    n_mean = np.mean(n)
    p_mean = np.mean(pHalle)
    T_mean = np.mean(THalle)
    print(f"drehzahl: {n_mean}")

    # Druckdifferenz in Absolutdruck
    ps1_mean = ps1_diff_mean + p_mean
    ps2_mean = ps2_diff_mean + p_mean
    pt1_mean = pt1_diff_mean + p_mean

    # Einheit: mbar → Pa
    ps1_mean *= 100
    ps2_mean *= 100
    p_mean   *= 100
    pt1_mean *= 100

    # Luftdichte (ideales Gas)
    R = 287.0                   # J/(kg K)
    rho = p_mean / (R * T_mean)
    #print(f"rho berechnet: {rho}")

    # Druckdifferenz
    dp = ps2_mean - pt1_mean #total to static!

    # psi berechnen
    psi = dp / (rho * n_mean**2)
    #print(f"Psi berechnet = {psi}")

    return psi

def get_psi_from_d7d(filepath):
    psi_signal, fs, _ = load_d7d_channel(filepath, "psi")
    psi = np.mean(psi_signal)

    return psi

def get_m_dot(filepath, area):
    rho = get_rho(filepath)
    v = get_v1(filepath)
    m_dot = rho * area * v
    print(f"m_dot berechet: {m_dot}")
    return m_dot

def get_m_dot_from_d7d(filepath):
    m_dot_signal, fs, _ =load_d7d_channel(filepath, "mDot")
    m_dot = np.mean(m_dot_signal)

    return m_dot

def get_m_dot_red(filepath, area, radius):
    m_dot = get_m_dot(filepath, area, radius)
    p_ein, _, _ = load_d7d_channel(filepath, "pHalle")
    T_ein, _, _ = load_d7d_channel(filepath, "THalle")
    return (m_dot / np.sqrt(T_ein.mean())) / p_ein.mean()

def get_phi(filepath, area, radius):
    #m_dot = get_m_dot(filepath, area, radius)
    m_dot = get_m_dot_from_d7d(filepath)
    p_ein, _, _ = load_d7d_channel(filepath, "pHalle")
    T_ein, _, _ = load_d7d_channel(filepath, "THalle")

    uTip_signal, _, _ = load_d7d_channel(filepath, "uTip")
    uTip = np.mean(uTip_signal)

    rho = p_ein.mean() * 100 / (287.0 * T_ein.mean())
    U = 2 * np.pi * 10000 / 60 * radius #für Vergleich
    return (m_dot / (rho * area * uTip))

def get_area(radius):
    return np.pi * radius**2

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

def get_v1(filepath):
    pt1_signal, _, _ = load_d7d_channel(filepath, "pt")
    ps_signal, _, _ = load_d7d_channel(filepath, "ps1")
    pHalle_signal, _, _= load_d7d_channel(filepath, "pHalle")
    T_signal, _, _ = load_d7d_channel(filepath, "THalle")
    
    pHalle_mean = np.mean(pHalle_signal) *100
    pt1_mean = np.mean(pt1_signal) * 100
    T_mean = np.mean(T_signal)
    rho_mean = (pt1_mean + pHalle_mean) / (287.0 * T_mean)
    ps_mean = np.mean(ps_signal) * 100

    return np.sqrt((2 * ((pt1_mean + pHalle_mean) - (ps_mean + pHalle_mean)))/(rho_mean))
    pt2_signal, _, _ = load_d7d_channel(filepath, "pt")
    ps_signal, _, _ = load_d7d_channel(filepath, "ps2")
    T_signal, _, _ = load_d7d_channel(filepath, "THalle")
    
    pt2_mean = np.mean(pt2_signal) * 100000
    T_mean = np.mean(T_signal)
    rho_mean = pt2_mean / (287.0 * T_mean)
    ps_mean = np.mean(ps_signal)

    return np.sqrt((2 * (pt2_mean))/(rho_mean))
