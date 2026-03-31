import matplotlib.pyplot as plt
import re
import os
import glob
from BA_PSD_Funktion_Ergin import list_channels, load_d7d_channel, compute_psd_1d, get_psi, get_drosselwert_from_filename

def main():

    #-----------------------------------------------------------------------------------------------------------
    #--------------------------PSD berechnen und plotten--------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------

    filepath = r"C:\Users\Nikla\OneDrive\Dokumente\A_Studium\A_Verkehrswesen\A_Bachelor\MA_NG_Base_n10k_stationary\MA_NG_Base_n10k_stationary\UmTrieb_MA_NG_Stall_d200_pUequi_0000.d7d"

    # d-Wert aus Dateinamen extrahieren
    match = re.search(r"d\d+", filepath)

    d_value_PSD = get_drosselwert_from_filename(filepath)

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
            #print(f"Signal-Länge: {len(signal)}")
            #print(f"Abtastrate fs: {fs} Hz")

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
    plt.title(f"PSD von pU01 bis pU20 bei d{d_value_PSD}")
    plt.grid(True)
    # plt.legend(ncol=2, fontsize=8) #kann man eh nicht erkennen
    plt.tight_layout()
    #plt.show()



    #-----------------------------------------------------------------------------------------------------------
    #--------------------------Pseudo Kennfeld------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------
    # Ordner mit deinen d7d-Dateien
    folderpath = r"C:\Users\Nikla\OneDrive\Dokumente\A_Studium\A_Verkehrswesen\A_Bachelor\MA_NG_Base_n10k_stationary\MA_NG_Base_n10k_stationary"

    # Alle d7d-Dateien im Ordner holen
    filepaths = glob.glob(os.path.join(folderpath, "*.d7d"))

    # Radius anpassen!
    r = 0.05  # [m]

    d_values = []
    psi_values = []

    for filepath in filepaths:
        try:
            d_value = get_drosselwert_from_filename(filepath)

            if d_value is None:
                print(f"Kein Drosselwert im Dateinamen gefunden: {filepath}")
                continue

            # Kanäle laden
            ps1, fs, _ = load_d7d_channel(filepath, "ps1")
            ps2, fs, _ = load_d7d_channel(filepath, "ps2")
            n, fs, _ = load_d7d_channel(filepath, "Drehzahl")
            pHalle, fs, _ = load_d7d_channel(filepath, "pHalle")
            THalle, fs, _ = load_d7d_channel(filepath, "THalle")

            # psi berechnen
            psi = get_psi(
                ps1=ps1,
                ps2=ps2,
                n=n,
                r=r,
                p_halle=pHalle,
                T_halle=THalle
            )

            d_values.append(d_value)
            psi_values.append(psi)

            print(f"{os.path.basename(filepath)} -> d = {d_value}, psi = {psi:.5f}")

        except Exception as e:
            print(f"Fehler bei Datei {os.path.basename(filepath)}: {e}")

    # Nach Drosselwert sortieren
    if len(d_values) == 0:
        print("Keine gültigen Daten zum Plotten gefunden.")
        return

    data_sorted = sorted(zip(d_values, psi_values), key=lambda x: x[0])
    d_values_sorted, psi_values_sorted = zip(*data_sorted)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(d_values_sorted, psi_values_sorted, 'o-')
    plt.xlabel("Drosselwert d")
    plt.ylabel(r"$\psi$")
    plt.title(r"$\psi$ über Drosselwert")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()