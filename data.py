import io
import requests
import pandas as pd
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import simpledialog, messagebox

# =============================
#  Paramètres Infoclimat
# =============================

API_TOKEN = "TmPCPmBIuVyEKVfK1v4iWkdlKcAAGAm4Tkmot21lToNPlEl65hEGA"  # Clé API Infoclimat
STATION_ID = "07510"  # Identifiant de la station Bordeaux-Mérignac sur InfoClimat

BASE_URL = "https://www.infoclimat.fr/opendata/"


def build_url(station_id: str, start_date: str, end_date: str) -> str:
    """
    Construit l'URL d'appel à l'API OpenData Infoclimat v2 en CSV.

    Exemple :
    https://www.infoclimat.fr/opendata/?version=2&method=get&format=csv&stations[]=07510&start=2026-01-12&end=2026-01-14&token=...
    """
    return (
        f"{BASE_URL}"
        f"?version=2"
        f"&method=get"
        f"&format=csv"
        f"&stations[]={station_id}"
        f"&start={start_date}"
        f"&end={end_date}"
        f"&token={API_TOKEN}"
    )


def fetch_csv_from_infoclimat(url: str) -> str:
    """Télécharge le CSV brut depuis l'API Infoclimat et retourne le texte."""
    print("Requête :", url)
    resp = requests.get(url)
    print("Status HTTP:", resp.status_code)

    txt = resp.text

    # Cas fréquent en OpenData Infoclimat : IP différente de celle déclarée
    if "wrong ip adress" in txt.lower():
        raise RuntimeError(
            "Infoclimat renvoie 'wrong ip adress'. "
            "Vérifie que l'IP publique utilisée pour lancer le script "
            "correspond bien à celle déclarée lors de la génération du token."
        )

    resp.raise_for_status()
    return txt

def parse_infoclimat_csv(csv_text: str) -> pd.DataFrame:
    """
    Parse le CSV Infoclimat en DataFrame et nettoie les lignes parasites.

    - Ignore les lignes commençant par '#'
    - Supprime la ligne d’unités (dh_utc = 'YYYY-MM-DD hh:mm:ss')
    - Supprime les éventuelles lignes où 'station_id' == 'station_id' (header dupliqué)
    """
    buffer = io.StringIO(csv_text)

    df_raw = pd.read_csv(
        buffer,
        sep=";",
        comment="#",
        engine="python",
        on_bad_lines="skip",  # saute les lignes mal formées -> entraîne un crash du script
    )

    # Si le CSV est vide / bizarre
    if df_raw.empty or "dh_utc" not in df_raw.columns:
        return pd.DataFrame()

    # Ligne d’unités = dh_utc = "YYYY-MM-DD hh:mm:ss" → on la vire
    mask_units = df_raw["dh_utc"].astype(str).str.contains("YYYY-MM-DD", na=False)
    df = df_raw.loc[~mask_units].copy()

    # Par sécurité : on enlève aussi d'éventuelles lignes où dh_utc == 'dh_utc'
    df = df[df["dh_utc"] != "dh_utc"]

    # dh_utc → datetime
    df["dh_utc"] = pd.to_datetime(df["dh_utc"], errors="coerce")

    # Colonnes numériques courantes (présentes selon la station)
    numeric_cols = [
        "temperature",
        "pression",
        "humidite",
        "pluie_1h",
        "pluie_3h",
        "pluie_6h",
        "pluie_12h",
        "pluie_24h",
        # Vent
        "vent_moyen",
        "vent_direction",
        # Ensoleillement / rayonnement (selon stations)
        "ensoleillement",
        "ensoleillement_1h",
        "rayonnement",
        "rayonnement_solaire",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Choix de la colonne pluie
    rain_col = None
    for candidate in ["pluie_1h", "pluie_3h", "pluie_24h"]:
        if candidate in df.columns:
            rain_col = candidate
            break

    def first_existing(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # Choix des colonnes vent / ensoleillement (selon la station)
    wind_avg_col = first_existing(["vent_moyen", "vent_moyen_10m", "vent_moyen_kmh"])
    wind_dir_col = first_existing(["vent_direction", "direction_vent", "vent_dir"])
    sun_col = first_existing([
        "ensoleillement",
        "ensoleillement_1h",
        "rayonnement",
        "rayonnement_solaire",
    ])

    out = pd.DataFrame()
    if "station_id" in df.columns:
        out["station_id"] = df["station_id"].astype(str)
    else:
        out["station_id"] = STATION_ID

    out["datetime_utc"] = df["dh_utc"]
    out["temp_C"] = df.get("temperature")
    out["pressure_hPa"] = df.get("pression")
    out["humidity_pct"] = df.get("humidite")

    # Vent / ensoleillement (colonnes optionnelles selon station)
    out["wind_avg"] = df[wind_avg_col] if wind_avg_col is not None else pd.NA
    out["wind_dir_deg"] = df[wind_dir_col] if wind_dir_col is not None else pd.NA
    out["sunshine"] = df[sun_col] if sun_col is not None else pd.NA

    if rain_col is not None:
        out["rain_mm"] = df[rain_col]
    else:
        out["rain_mm"] = 0.0

    out["hour_utc"] = out["datetime_utc"].dt.hour

    # On garde un ordre de colonnes fixe
    out = out[
        [
            "station_id",
            "datetime_utc",
            "hour_utc",
            "temp_C",
            "pressure_hPa",
            "humidity_pct",
            "wind_avg",
            "wind_dir_deg",
            "sunshine",
            "rain_mm",
        ]
    ]

    # On enlève les lignes sans datetime
    out = out[out["datetime_utc"].notna()]

    return out


class DateFileDialog(simpledialog.Dialog):
    """Boîte de dialogue Tkinter pour saisir date début, date fin et nom de fichier."""

    def body(self, master):
        tk.Label(master, text="Date de début (YYYY-MM-DD) :").grid(row=0, column=0, sticky="w")
        tk.Label(master, text="Date de fin (YYYY-MM-DD) :").grid(row=1, column=0, sticky="w")
        tk.Label(master, text="Nom du fichier CSV de sortie :").grid(row=2, column=0, sticky="w")

        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        self.file_var = tk.StringVar(value="observations_infoclimat_full.csv")

        self.start_entry = tk.Entry(master, textvariable=self.start_var)
        self.end_entry = tk.Entry(master, textvariable=self.end_var)
        self.file_entry = tk.Entry(master, textvariable=self.file_var)

        self.start_entry.grid(row=0, column=1, padx=5, pady=2)
        self.end_entry.grid(row=1, column=1, padx=5, pady=2)
        self.file_entry.grid(row=2, column=1, padx=5, pady=2)

        return self.start_entry  # focus initial

    def apply(self):
        start_str = self.start_var.get().strip()
        end_str = self.end_var.get().strip()
        filename = self.file_var.get().strip()
        self.result = (start_str, end_str, filename)


def ask_date_and_filename_via_popup():
    """
    Demande date début / fin et nom de fichier via une seule pop-up Tkinter,
    retourne (date_debut, date_fin, nom_fichier).
    """
    root = tk.Tk()
    root.withdraw()  # pas de fenêtre principale visible

    dialog = DateFileDialog(root, "Paramètres de téléchargement")
    if dialog.result is None:
        raise SystemExit("Saisie annulée")

    start_str, end_str, filename = dialog.result

    if not filename:
        messagebox.showerror("Erreur", "Le nom de fichier ne peut pas être vide.")
        raise SystemExit(1)

    # Ajoute une extension .csv si l'utilisateur n'en a pas mis
    if "." not in filename:
        filename = filename + ".csv"

    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError:
        messagebox.showerror("Erreur", "Format de date invalide. Utilise YYYY-MM-DD.")
        raise SystemExit(1)

    if end_date < start_date:
        messagebox.showerror("Erreur", "La date de fin est avant la date de début.")
        raise SystemExit(1)

    return start_date, end_date, filename


def generate_chunks(start_date, end_date, max_days=7):
    """
    Génère des paires (start, end) pour respecter la limite de 7 jours max par requête.
    Les dates sont inclusives.
    """
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=max_days - 1), end_date)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def main():
    # Sélection des dates et nom de fichier par une seule pop-up
    start_date, end_date, output_file = ask_date_and_filename_via_popup()
    print(f"Période demandée : {start_date} -> {end_date}")
    print(f"Fichier de sortie : {output_file}")

    # Découpage en tranches de 7 jours max (réglementation Infoclimat)
    all_dfs = []

    for chunk_start, chunk_end in generate_chunks(start_date, end_date, max_days=7):
        start_str = chunk_start.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")
        print(f"\nTéléchargement du bloc {start_str} -> {end_str}")

        url = build_url(STATION_ID, start_str, end_str)
        csv_text = fetch_csv_from_infoclimat(url)
        df_chunk = parse_infoclimat_csv(csv_text)

        if df_chunk.empty:
            print("⚠️  Aucune donnée dans ce bloc.")
        else:
            print(f"  -> {len(df_chunk)} lignes")
            all_dfs.append(df_chunk)

    if not all_dfs:
        print("Aucune donnée sur toute la période demandée.")
        return

    # Concaténation de tous les blocs
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.sort_values("datetime_utc")
    df_all = df_all.drop_duplicates(subset=["datetime_utc", "station_id"])

    # Sauvegarde CSV final
    df_all.to_csv(output_file, index=False)
    print(f"\nDonnées consolidées sauvegardées dans {output_file}")
    print(f"Nombre total de lignes : {len(df_all)}")


if __name__ == "__main__":
    main()