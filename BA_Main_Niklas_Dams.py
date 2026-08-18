import matplotlib.pyplot as plt
import re
import os
import glob
import numpy as np
from BA_Funktionen_Niklas_Dams import get_mean_operating_point, compute_mean_psd, get_m_dot, list_channels, get_v1, read_d7d_info, get_psi_from_d7d, get_area, get_phi, find_pressure_psi_channels, load_d7d_channel, compute_psd_1d, get_psi, get_drosselwert_from_filename, compute_psd_1d_scipy, compute_mean_coherence

def main():
    # ! -------------------------------------------------------------------------------------------------------
    # ! Notizen
    # ! -------------------------------------------------------------------------------------------------------
        #* Für die Referenzmessung ist der RI Bereich zwischen d20.0 - 14.5, d14.5 ist der Peak / letzter stabiler OP!
        #* LG_Ref_3D_Einlauf -> LG_IGV00 Renamed!
        #* 
        #* B sensoren
        #* Sensoren pU08 und pU13: beides B-Sensoren mit 90° Verschiebung

    # ! ===========================================================================================
    # ! ====================================== KONFIGURATION ======================================
    # ! ===========================================================================================

    # ! -------------------------------------------------------------------------------------------------------
    # ! Auswertung auswählen
    # ! -------------------------------------------------------------------------------------------------------

    RUN_PSD = True                                                                                                 #TODO
    RUN_KENNFELD = True                                                                                           #TODO
    RUN_COHERENCE = True

    print("\n" + "=" * 60)
    print("Ausgewählte Auswertung")
    print("=" * 60)
    print(f"PSD-Auswertung:         {'AN' if RUN_PSD else 'AUS'}")
    print(f"Kennfeld-Auswertung:    {'AN' if RUN_KENNFELD else 'AUS'}")
    print(f"Kohärenzanalyse:        {'AN' if RUN_COHERENCE else 'AUS'}")
    print("=" * 60)


    # ! -------------------------------------------------------------------------------------------------------
    # ! Geometrie
    # ! -------------------------------------------------------------------------------------------------------

    r = 0.038                      # [m] mittlerer Laufradradius
    area = get_area(r)             # [m²]


    # ! -------------------------------------------------------------------------------------------------------
    # ! FFT-Einstellungen
    # ! -------------------------------------------------------------------------------------------------------

    nFFT = 2**13                            #! 2**13 -> 2**15                                                       #TODO
    overlap = 0.5                           #! 0.5 -> 0,25 oder no                                                  #TODO
    window_type = "hann"                                                                                            #TODO

    f_min = 0                                                                                                       #TODO
    f_max = 2500                                                                                                    #TODO


    # ! -------------------------------------------------------------------------------------------------------
    # ! Sensoren auswählen
    # ! -------------------------------------------------------------------------------------------------------

    #channels = ["pU03"]                                                                                             #TODO

    # Beispiele:
    channels = [
            "pU08",
            "pU13",
        ]
    #channels = [f"pU{i:02d}" for i in range(5, 14)]


    # ! -------------------------------------------------------------------------------------------------------
    # ! Drosselbereich auswählen
    # ! -------------------------------------------------------------------------------------------------------

    d_start = 22.0                                   #RI: 22.0 - 14.5                                               #TODO
    d_end = 14.5                                                                                                    #TODO

    # * Umrechnung auf Dateinamen-Skalierung
    d_start_int = int(round(d_start * 10))
    d_end_int = int(round(d_end * 10))

    d_min = min(d_start_int, d_end_int)
    d_max = max(d_start_int, d_end_int)

    # ! -------------------------------------------------------------------------------------------------------
    # ! Plot-Ausgabe
    # ! -------------------------------------------------------------------------------------------------------

    SAVE_PLOTS = False                                                                                             #TODO

    save_folder = (r"C:\Users\Niklas\OneDrive\Dokumente\A_Studium\A_Verkehrswesen\A_Bachelor\Plots\PSD")


    # ! -------------------------------------------------------------------------------------------------------
    # ! Verfügbare Messungen
    # ! -------------------------------------------------------------------------------------------------------

    measurement_folders = {

        "LG_IGV00": r"C:\Users\Niklas\OneDrive\Dokumente\A_Studium\A_Verkehrswesen\A_Bachelor\Messungen\LG_IGV00",

        "LG_IGV02": r"C:\Users\Niklas\OneDrive\Dokumente\A_Studium\A_Verkehrswesen\A_Bachelor\Messungen\LG_IGV02",

        "LG_IGV06": r"C:\Users\Niklas\OneDrive\Dokumente\A_Studium\A_Verkehrswesen\A_Bachelor\Messungen\LG_IGV06",

        "LG_IGV07": r"C:\Users\Niklas\OneDrive\Dokumente\A_Studium\A_Verkehrswesen\A_Bachelor\Messungen\LG_IGV07",

        "LG_IGV12": r"C:\Users\Niklas\OneDrive\Dokumente\A_Studium\A_Verkehrswesen\A_Bachelor\Messungen\LG_IGV12",
    }


    # ! -------------------------------------------------------------------------------------------------------
    # ! Messungen auswählen
    # ! -------------------------------------------------------------------------------------------------------

    # selected_measurements = [                                                                                       #TODO
    #     "LG_IGV00",
    #     "LG_IGV02",
    #     "LG_IGV06",
    #     "LG_IGV07",
    #     "LG_IGV12"
    # ]

    selected_measurements = [
            "LG_IGV12"
        ]


    # ! ========================================================================================================
    # ! Kohärenzanalyse konfigurieren
    # ! ========================================================================================================

    # * Messreihe auswählen
    coherence_measurements = [                                                                                      #TODO
    "LG_IGV12",
    #"LG_IGV12",
    ] 

    # * Drosselstellung auswählen
    coherence_d = 18.0                                                                                              #TODO

    # * Zwei Sensoren für die Kohärenzanalyse
    coherence_channels = [                                                                                          #TODO
        "pU08",
        "pU13"
    ]

    # * Geometrischer Umfangsabstand der Sensoren
    coherence_sensor_angle = 90.0      # [°]                                                                        #TODO

    # * Stationäre Rotordrehzahl
    coherence_rpm = 10000              # [rpm]

    # * Bis zu welcher Engine Order Linien eingezeichnet werden
    max_engine_order = 14                                                                                           #TODO


    # ! ==========================================================================================================
    # ! ======================================  DEBUG / PLAUSIBILITÄT ============================================
    # ! ==========================================================================================================

    # TODO: Bei Bedarf einkommentieren

    # rho_signal, _, _ = load_d7d_channel(filepath, "Dichte")
    # print(f"Dichte aus d7d: {np.mean(rho_signal):.5f} kg/m³")

    # velo_signal, _, _ = load_d7d_channel(filepath, "velo")
    # print(f"Geschwindigkeit aus d7d: {np.mean(velo_signal):.3f} m/s")

    # psi_signal, _, _ = load_d7d_channel(filepath, "psi")
    # print(f"psi aus d7d: {np.mean(psi_signal):.5f}")

    # mDot_signal, _, _ = load_d7d_channel(filepath, "mDot")
    # print(f"Massenstrom aus d7d: {np.mean(mDot_signal):.5f} kg/s")

    # print(f"v1 berechnet: {get_v1(filepath):.3f} m/s")
    # print(f"mDot berechnet: {get_m_dot(filepath, area):.5f} kg/s")
    # print(f"phi berechnet: {get_phi(filepath, r):.5f}")
    # print(f"psi berechnet: {get_psi(filepath):.5f}")







    # ! =======================================================================================================
    # ! ======================================= ▼ MAIN CODE ▼ ================================================
    # ! =======================================================================================================

    if RUN_PSD:

        # ! -------------------------------------------------------------------------------------------------------
        # ! -------------------------------------- PSD-AUSWERTUNG -------------------------------------------------
        # ! -------------------------------------------------------------------------------------------------------

        # ! -------------------------------------------------------------------------------------------------------
        # ! Gemeinsame Drosselstellungen der ausgewählten Messreihen bestimmen
        # ! -------------------------------------------------------------------------------------------------------

        # * Für jede Messreihe werden zunächst die vorhandenen Drosselstellungen gespeichert
        d_values_per_measurement = {}

        for measurement in selected_measurements:

            folderpath = measurement_folders[measurement]

            filepaths = glob.glob(
                os.path.join(folderpath, "*.d7d")
            )

            # ? Messordner prüfen
            if len(filepaths) == 0:
                print(
                    f"Warnung: Keine d7d-Dateien für "
                    f"{measurement} gefunden:\n{folderpath}"
                )

                d_values_per_measurement[measurement] = set()
                continue

            available_d_values = set()

            for filepath_temp in filepaths:

                d_value = get_drosselwert_from_filename(filepath_temp)

                if d_value is not None:
                    available_d_values.add(d_value)

            # * Nur Drosselstellungen innerhalb des gewählten Bereichs behalten
            available_d_values = {
                d for d in available_d_values
                if d_min <= d <= d_max
            }

            d_values_per_measurement[measurement] = available_d_values


        # ! -------------------------------------------------------------------------------------------------------
        # ! Gemeinsame Drosselstellungen bestimmen
        # ! -------------------------------------------------------------------------------------------------------

        # * Nur Drosselstellungen verwenden, die in ALLEN ausgewählten Messreihen vorhanden sind
        valid_d_sets = [
            d_values_per_measurement[measurement]
            for measurement in selected_measurements
            if len(d_values_per_measurement[measurement]) > 0
        ]

        if len(valid_d_sets) == 0:
            print("Keine gültigen Drosselstellungen für die PSD-Auswertung gefunden.")

        else:

            common_d_values = set.intersection(*valid_d_sets)

            selected_d_values = sorted(
                common_d_values,
                reverse=True
            )

            print()
            print("=" * 90)
            print("PSD-Auswertung")
            print("=" * 90)
            print(f"Messreihen: {selected_measurements}")
            print(f"Sensoren:   {channels}")
            print(f"Drosseln:   {selected_d_values}")
            print(f"nFFT:       {nFFT}")
            print(f"Overlap:    {overlap}")
            print(f"Window:     {window_type}")
            print("=" * 90)


            # ! ---------------------------------------------------------------------------------------------------
            # ! Für jede Drosselstellung einen eigenen Vergleichsplot erstellen
            # ! ---------------------------------------------------------------------------------------------------

            for d_value in selected_d_values:

                fig, ax = plt.subplots(figsize=(10, 5))


                # ! -----------------------------------------------------------------------------------------------
                # ! Alle ausgewählten Messreihen auswerten
                # ! -----------------------------------------------------------------------------------------------

                for measurement in selected_measurements:

                    folderpath = measurement_folders[measurement]


                    # * Referenzdatei 0000 für diese Drosselstellung suchen
                    # * compute_mean_psd() lädt anschließend automatisch 0000 bis 0004
                    reference_files = sorted(
                        glob.glob(
                            os.path.join(
                                folderpath,
                                f"*_d{d_value}_0000.d7d"
                            )
                        )
                    )

                    if len(reference_files) == 0:
                        print(
                            f"Keine Referenzdatei gefunden: "
                            f"{measurement}, d{d_value}"
                        )
                        continue

                    filepath_psd = reference_files[0]


                    # ! -------------------------------------------------------------------------------------------
                    # ! Gewählte Sensoren dieser Messreihe auswerten
                    # ! -------------------------------------------------------------------------------------------

                    for channel_name in channels:

                        try:

                            # * PSD der fünf Wiederholungsmessungen mitteln
                            freqs, psd = compute_mean_psd(
                                filepath=filepath_psd,
                                channel_name=channel_name,
                                nFFT=nFFT,
                                overlap=overlap,
                                window_type=window_type
                            )

                            # * Jede Kombination aus Messreihe und Sensor erhält einen Legendeneintrag
                            ax.semilogy(
                                freqs,
                                psd,
                                label=f"{measurement} – {channel_name}"
                            )

                        except Exception as e:

                            print(
                                f"Fehler bei "
                                f"{measurement}, "
                                f"d{d_value}, "
                                f"{channel_name}: {e}"
                            )


                # ! -----------------------------------------------------------------------------------------------
                # ! Plot dieser Drosselstellung formatieren
                # ! -----------------------------------------------------------------------------------------------

                ax.set_xlabel("Frequenz [Hz]")
                ax.set_ylabel("PSD")

                ax.set_xlim(
                    f_min,
                    f_max
                )

                ax.set_title(
                    f"PSD-Vergleich bei d{d_value}"
                )

                ax.grid(True)
                ax.legend(fontsize=8)

                fig.tight_layout()

                #! Plot optional als PDF speichern
                if SAVE_PLOTS:

                    os.makedirs(save_folder, exist_ok=True)

                    # * IGV-Bezeichnungen für Titel erzeugen
                    igv_labels = []

                    # * Sensornamen für Dateinamen zusammenbauen
                    sensor_text_file = "+".join(channels)

                    nFFT_exp = int(np.log2(nFFT))
                    overlap_text = str(overlap).replace(".", ",")

                    for measurement in selected_measurements:

                        if "LG_IGV00" in measurement:
                            igv_labels.append("00")
                        else:
                            match = re.search(r"IGV(\d+)", measurement)
                            if match:
                                igv_labels.append(match.group(1))

                    igv_text = "+".join(igv_labels)

                    filename = f"PSD_D{d_value}_IGV{igv_text}_{sensor_text_file}_nFFT{nFFT_exp}_overlap{overlap_text}.pdf"

                    fig.savefig(
                        os.path.join(save_folder, filename),
                        format="pdf",
                        bbox_inches="tight"
                    )

                    print(f"PDF gespeichert: {filename}")


            # ! ---------------------------------------------------------------------------------------------------
            # ! PSD-Plots anzeigen
            # ! ---------------------------------------------------------------------------------------------------

            # * Wenn kein Kennfeld berechnet wird, PSD-Plots sofort anzeigen
            if not RUN_KENNFELD:
                plt.show()



    # ! ===========================================================================================
    # ! ================================ ▼ KOHÄRENZANALYSE ▼ =====================================
    # ! ===========================================================================================

    if RUN_COHERENCE:

        print("\n" + "=" * 90)
        print("KOHÄRENZANALYSE")
        print("=" * 90)

        # * Sensoren
        channel_1 = coherence_channels[0]
        channel_2 = coherence_channels[1]

        # * Drosselstellung: 16.0 -> d160
        d_value_coherence = int(round(coherence_d * 10))

        # * Rotationsfrequenz
        f_rot = coherence_rpm / 60.0

        # * Engine-Order-Frequenzen erzeugen
        engine_orders = []

        for eo in range(1, max_engine_order + 1):

            f_eo = eo * f_rot

            # * Nur EOs innerhalb des dargestellten Frequenzbereichs
            if f_min <= f_eo <= f_max:
                engine_orders.append((eo, f_eo))


        # ! ---------------------------------------------------------------------------------------------------
        # ! Einstellungen ausgeben
        # ! ---------------------------------------------------------------------------------------------------

        print(f"Messreihen:       {coherence_measurements}")
        print(f"Drosselstellung:  d{d_value_coherence}")
        print(f"Sensorpaar:       {channel_1} - {channel_2}")
        print(f"Sensorabstand:    {coherence_sensor_angle:.1f}°")
        print(f"nFFT:             {nFFT}")
        print(f"Overlap:          {overlap}")
        print(f"Window:           {window_type}")
        print("=" * 90)


        # ! ---------------------------------------------------------------------------------------------------
        # ! Gemeinsame Figure erstellen
        # ! ---------------------------------------------------------------------------------------------------

        fig, (ax_psd, ax_coh, ax_phase) = plt.subplots(
            3,
            1,
            figsize=(11, 10),
            sharex=True,
            gridspec_kw={
                "height_ratios": [1.3, 1, 1]
            }
        )


        # ! ---------------------------------------------------------------------------------------------------
        # ! Ausgewählte Messreihen nacheinander auswerten
        # ! ---------------------------------------------------------------------------------------------------

        for measurement in coherence_measurements:

            folderpath_coherence = measurement_folders[measurement]


            # ! -----------------------------------------------------------------------------------------------
            # ! Referenzdatei 0000 suchen
            # ! -----------------------------------------------------------------------------------------------

            reference_files = sorted(
                glob.glob(
                    os.path.join(
                        folderpath_coherence,
                        f"*_d{d_value_coherence}_0000.d7d"
                    )
                )
            )

            if len(reference_files) == 0:

                print(
                    f"Keine Datei für "
                    f"{measurement}, "
                    f"d{d_value_coherence} gefunden."
                )

                continue


            filepath_coherence = reference_files[0]


            try:

                # ! -------------------------------------------------------------------------------------------
                # ! PSD Sensor 1 berechnen
                # ! -------------------------------------------------------------------------------------------

                freqs_psd_1, psd_1 = compute_mean_psd(
                    filepath=filepath_coherence,
                    channel_name=channel_1,
                    nFFT=nFFT,
                    overlap=overlap,
                    window_type=window_type
                )


                # ! -------------------------------------------------------------------------------------------
                # ! PSD Sensor 2 berechnen
                # ! -------------------------------------------------------------------------------------------

                freqs_psd_2, psd_2 = compute_mean_psd(
                    filepath=filepath_coherence,
                    channel_name=channel_2,
                    nFFT=nFFT,
                    overlap=overlap,
                    window_type=window_type
                )


                # ! -------------------------------------------------------------------------------------------
                # ! Kohärenz und Phase berechnen
                # ! -------------------------------------------------------------------------------------------

                freqs_coh, coherence_mean, phase_mean = compute_mean_coherence(
                    filepath=filepath_coherence,
                    channel_1=channel_1,
                    channel_2=channel_2,
                    nFFT=nFFT,
                    overlap=overlap,
                    window_type=window_type
                )


                # ! -------------------------------------------------------------------------------------------
                # ! PSD plotten
                # ! -------------------------------------------------------------------------------------------

                # * Sensor 1 plotten und automatisch eine neue Farbe für die Messreihe wählen
                line_1, = ax_psd.semilogy(
                    freqs_psd_1,
                    psd_1,
                    linestyle="-",
                    label=f"{measurement} - {channel_1}"
                )

                # * Gewählte Farbe von Sensor 1 übernehmen
                color = line_1.get_color()

                # * Sensor 2 mit derselben Farbe, aber gestrichelt plotten
                ax_psd.semilogy(
                    freqs_psd_2,
                    psd_2,
                    color=color,
                    linestyle="--",
                    label=f"{measurement} - {channel_2}"
                )

                # ! -------------------------------------------------------------------------------------------
                # ! Kohärenz plotten
                # ! -------------------------------------------------------------------------------------------

                ax_coh.plot(
                    freqs_coh,
                    coherence_mean,
                    label=measurement
                )


                # ! -------------------------------------------------------------------------------------------
                # ! Phase plotten
                # ! -------------------------------------------------------------------------------------------

                # * Phase nur dort anzeigen, wo Kohärenz >= 0.6 ist
                # * Dadurch wird die physikalisch wenig belastbare Phase
                # * in unkorrelierten Frequenzbereichen ausgeblendet.
                phase_filtered = np.where(
                    coherence_mean >= 0.0,                                                                              #TODO
                    phase_mean,
                    np.nan
                )

                ax_phase.plot(
                    freqs_coh,
                    phase_filtered,
                    label=measurement
                )


            except Exception as e:

                print(
                    f"Fehler bei Kohärenzanalyse "
                    f"{measurement}, d{d_value_coherence}: {e}"
                )


        # ! ---------------------------------------------------------------------------------------------------
        # ! Engine Orders in ALLE drei Plots einzeichnen
        # ! ---------------------------------------------------------------------------------------------------

        for eo, f_eo in engine_orders:

            for ax in (ax_psd, ax_coh, ax_phase):

                ax.axvline(
                    f_eo,
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.7
                )

            # * EO-Beschriftung nur im oberen PSD-Plot
            ax_psd.text(
                f_eo,
                ax_psd.get_ylim()[1],
                f"{eo}.EO",
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=7
            )


        # ! ---------------------------------------------------------------------------------------------------
        # ! PSD formatieren
        # ! ---------------------------------------------------------------------------------------------------

        ax_psd.set_ylabel("PSD")

        ax_psd.set_title(
            f"PSD / Kohärenz / Phase | "
            f"{channel_1}-{channel_2} | "
            f"d{d_value_coherence}"
        )

        ax_psd.grid(True)

        ax_psd.legend(
            fontsize=8
        )


        # ! ---------------------------------------------------------------------------------------------------
        # ! Kohärenz formatieren
        # ! ---------------------------------------------------------------------------------------------------

        ax_coh.set_ylabel(
            "Kohärenz [-]"
        )

        ax_coh.set_ylim(
            0,
            1
        )

        # # * Schwelle sichtbar machen
        # ax_coh.axhline(
        #     0.6,
        #     linestyle="--",
        #     linewidth=1,
        #     label="C = 0.6"
        # )

        ax_coh.grid(True)

        ax_coh.legend(
            fontsize=8
        )


        # ! ---------------------------------------------------------------------------------------------------
        # ! Phase formatieren
        # ! ---------------------------------------------------------------------------------------------------

        ax_phase.set_xlabel(
            "Frequenz [Hz]"
        )

        ax_phase.set_ylabel(
            "Phasenwinkel [°]"
        )

        ax_phase.set_ylim(
            -180,
            180
        )

        ax_phase.grid(True)

        ax_phase.legend(
            fontsize=8
        )


        # ! ---------------------------------------------------------------------------------------------------
        # ! Gemeinsame Frequenzachse
        # ! ---------------------------------------------------------------------------------------------------

        ax_phase.set_xlim(
            f_min,
            f_max
        )

        fig.tight_layout()


        # ! ---------------------------------------------------------------------------------------------------
        # ! Plot anzeigen
        # ! ---------------------------------------------------------------------------------------------------

        if not RUN_KENNFELD:
            plt.show()




    if RUN_KENNFELD:
        # ! -----------------------------------------------------------------------------------------------------------
        # ! ----------------------------------------- KENNFELDER -------------------------------------------------------
        # ! -----------------------------------------------------------------------------------------------------------
        print("\n" + "=" * 90)
        print("Kennfeld-Auswertung")
        print("=" * 90)

        # ! -------------------------------------------------------------------------------------------------------
        # ! Gemeinsame Figuren vorbereiten
        # ! -------------------------------------------------------------------------------------------------------

        # * Plot 1: psi über Drosselstellung
        fig_d, ax_d = plt.subplots(figsize=(8, 5))

        # * Plot 2: Verdichterkennfeld psi über phi
        fig_phi, ax_phi = plt.subplots(figsize=(8, 5))


        # ! -------------------------------------------------------------------------------------------------------
        # ! Ausgewählte Messungen nacheinander auswerten
        # ! -------------------------------------------------------------------------------------------------------

        for measurement in selected_measurements:

            print("\n")
            print(f"Auswertung: {measurement}")
            print()

            folderpath = measurement_folders[measurement]

            # * Alle d7d-Dateien dieser Messreihe suchen
            filepaths = glob.glob(os.path.join(folderpath, "*.d7d"))

            # ? Messung überspringen, wenn keine Dateien gefunden wurden
            if len(filepaths) == 0:
                print(f"Keine d7d-Dateien gefunden in:\n{folderpath}")
                continue


            # ! ---------------------------------------------------------------------------------------------------
            # ! Speicher für Kennfeldpunkte dieser Messreihe
            # ! ---------------------------------------------------------------------------------------------------

            d_values = []
            psi_values = []
            phi_values = []

            psi_errors = []
            phi_errors = []


            # ! ---------------------------------------------------------------------------------------------------
            # ! Dateien nach Drosselstellung gruppieren
            # ! ---------------------------------------------------------------------------------------------------

            groups = {}

            for filepath in filepaths:

                d_value = get_drosselwert_from_filename(filepath)

                # ? Dateien ohne gültige Drosselstellung ignorieren
                if d_value is None:
                    print(
                        f"Kein Drosselwert gefunden: "
                        f"{os.path.basename(filepath)}"
                    )
                    continue

                # * Alle Wiederholungsmessungen derselben Drosselstellung sammeln
                groups.setdefault(d_value, []).append(filepath)


            # ! ---------------------------------------------------------------------------------------------------
            # ! Betriebspunkte berechnen
            # ! ---------------------------------------------------------------------------------------------------

            for d_value, files_for_d in groups.items():

                try:

                    # * Dateien sortieren: 0000, 0001, ..., 0004
                    files_for_d = sorted(files_for_d)

                    # ? Kontrolle der Anzahl an Wiederholungsmessungen
                    if len(files_for_d) != 5:
                        print(
                            f"Warnung: {measurement}, d={d_value}: "
                            f"{len(files_for_d)} Dateien statt erwarteten 5."
                        )

                    # * Für jede Einzelmessung psi und phi berechnen
                    # * Danach Mittelwert und Standardabweichung über die Wiederholungen bilden
                    psi_mean, phi_mean, psi_std, phi_std = get_mean_operating_point(
                        files_for_d,
                        r
                    )

                    # * Gemittelten Betriebspunkt speichern
                    d_values.append(d_value)
                    psi_values.append(psi_mean)
                    phi_values.append(phi_mean)

                    # * Streuung der Wiederholungsmessungen speichern
                    psi_errors.append(psi_std)
                    phi_errors.append(phi_std)

                    print(
                        f"d = {d_value:3d} | "
                        f"Messungen = {len(files_for_d)} | "
                        f"psi = {psi_mean:.5f} ± {psi_std:.5f} | "
                        f"phi = {phi_mean:.5f} ± {phi_std:.5f}"
                    )

                except Exception as e:
                    print(
                        f"Fehler bei {measurement}, "
                        f"d={d_value}: {e}"
                    )


            # ! ---------------------------------------------------------------------------------------------------
            # ! Prüfen, ob gültige Betriebspunkte vorhanden sind
            # ! ---------------------------------------------------------------------------------------------------

            if len(d_values) == 0:
                print(f"Keine gültigen Kennfelddaten für {measurement}.")
                continue


            # ! ---------------------------------------------------------------------------------------------------
            # ! Kennfeldpunkte nach Drosselstellung sortieren
            # ! ---------------------------------------------------------------------------------------------------

            data_sorted = sorted(
                zip(
                    d_values,
                    psi_values,
                    phi_values,
                    psi_errors,
                    phi_errors
                ),
                key=lambda x: x[0]
            )

            (
                d_values_sorted,
                psi_values_sorted,
                phi_values_sorted,
                psi_errors_sorted,
                phi_errors_sorted
            ) = zip(*data_sorted)


            # ! ---------------------------------------------------------------------------------------------------
            # ! Messreihe in Plot 1 eintragen: psi über d
            # ! ---------------------------------------------------------------------------------------------------

            ax_d.errorbar(
                d_values_sorted,
                psi_values_sorted,
                yerr=psi_errors_sorted,
                fmt="o-",
                capsize=3,
                label=measurement
            )


            # ! ---------------------------------------------------------------------------------------------------
            # ! Messreihe in Plot 2 eintragen: psi über phi
            # ! ---------------------------------------------------------------------------------------------------

            ax_phi.errorbar(
                phi_values_sorted,
                psi_values_sorted,
                xerr=phi_errors_sorted,
                yerr=psi_errors_sorted,
                fmt="o-",
                capsize=3,
                label=measurement
            )


        # ! -------------------------------------------------------------------------------------------------------
        # ! Plot 1 formatieren: psi über Drosselstellung
        # ! -------------------------------------------------------------------------------------------------------

        ax_d.set_xlabel("Drosselwert d")
        ax_d.set_ylabel(r"$\psi$")
        ax_d.set_title(r"$\psi$ über Drosselstellung")

        ax_d.grid(True)
        ax_d.legend()

        fig_d.tight_layout()


        # ! -------------------------------------------------------------------------------------------------------
        # ! Plot 2 formatieren: Verdichterkennfeld psi über phi
        # ! -------------------------------------------------------------------------------------------------------

        ax_phi.set_xlabel(r"$\phi$")
        ax_phi.set_ylabel(r"$\psi$")
        ax_phi.set_title(r"Vergleich der Verdichterkennfelder")

        ax_phi.grid(True)
        ax_phi.legend()

        fig_phi.tight_layout()


        # ! -------------------------------------------------------------------------------------------------------
        # ! Alle Figuren anzeigen
        # ! -------------------------------------------------------------------------------------------------------

        plt.show()




if __name__ == "__main__":
    main()