import matplotlib.pyplot as plt
import re
from BA_PSD_Funktion_Ergin import list_channels, load_d7d_channel, compute_psd_1d

def main():
    filepath = r"C:\Users\Nikla\OneDrive\Dokumente\A_Studium\A_Verkehrswesen\A_Bachelor\MA_NG_Base_n10k_stationary\MA_NG_Base_n10k_stationary\UmTrieb_MA_NG_Stall_d140_pUequi_0000.d7d"

    # d-Wert aus Dateinamen extrahieren
    match = re.search(r"d\d+", filepath)

    if match:
        d_value = match.group()   # z.B. "d122"
    else:
        d_value = "unbekannt"

    # alle Kanäle anzeigen
    print("Verfügbare Kanäle:")
    list_channels(filepath)

    # Drucksensor-Kanäle
    channels = [f"pU{i:02d}" for i in range(1,21)]
    #channels = ["pU04"]

    # FFT-/PSD-Einstellungen
    nFFT = 2**13
    overlap = 0.5
    window_type = "hann"

    # Plot
    plt.figure(figsize=(10, 5))

    for channel_name in channels:
        try:
            # Kanal laden + fs aus der d7d holen
            signal, fs, df = load_d7d_channel(filepath, channel_name)

            print()
            print(f"Gewählter Kanal: {channel_name}")
            print(f"Signal-Länge: {len(signal)}")
            print(f"Abtastrate fs: {fs} Hz")

            # PSD berechnen
            freqs, psd, psd_per_bin = compute_psd_1d(
                signal=signal,
                nFFT=nFFT,
                fs=fs,
                overlap=overlap,
                window_type=window_type,
                show_progress=False
            )

            # In denselben Plot
            plt.semilogy(freqs, psd, label=channel_name)

        except Exception as e:
            print(f"Fehler bei Kanal {channel_name}: {e}")

    # Plot formatieren
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("PSD")
    plt.xlim(0, 2500)
    plt.title(f"PSD von pU01 bis pU20 bei {d_value}")
    plt.grid(True)
    # plt.legend(ncol=2, fontsize=8) #kann man eh nicht erkennen
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()