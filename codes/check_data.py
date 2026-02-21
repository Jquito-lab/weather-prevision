

import os
import sys
from datetime import datetime
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText


PREFERRED_DT_COLS = [
    "datetime_utc",
    "dh_utc",
    "date_utc",
    "timestamp_utc",
    "timestamp",
    "datetime",
]


def _detect_datetime_column(df: pd.DataFrame) -> str:
    # 1) Prefer known column names
    for col in PREFERRED_DT_COLS:
        if col in df.columns:
            return col

    # 2) Otherwise pick the first column that looks like a datetime column
    for col in df.columns:
        name = str(col).lower()
        if "date" in name or "time" in name or "dh" in name:
            return col

    raise ValueError(
        "Impossible de trouver une colonne datetime. Colonnes disponibles: "
        + ", ".join(map(str, df.columns))
    )


def analyze_csv(csv_path: str) -> str:
    df = pd.read_csv(csv_path)

    dt_col = _detect_datetime_column(df)

    # Parse datetime
    dt = pd.to_datetime(df[dt_col], errors="coerce", utc=False)
    invalid_dt = int(dt.isna().sum())
    df = df.copy()
    df["_dt"] = dt
    df = df.dropna(subset=["_dt"]).copy()

    # Build date/hour
    df["_date"] = df["_dt"].dt.date
    df["_hour"] = df["_dt"].dt.hour

    # Per-day checks
    expected_hours = set(range(24))

    per_day = []
    for day, g in df.groupby("_date"):
        hours = g["_hour"].astype(int).tolist()
        hours_set = set(hours)
        missing = sorted(expected_hours - hours_set)

        # duplicates: any hour appearing more than once
        counts = g["_hour"].value_counts().sort_index()
        dup_hours = [int(h) for h, c in counts.items() if int(c) > 1]

        per_day.append(
            {
                "date": day,
                "n_rows": int(len(g)),
                "n_hours": int(len(hours_set)),
                "missing_hours": missing,
                "duplicate_hours": sorted(dup_hours),
            }
        )

    if not per_day:
        raise ValueError("Aucune date valide trouvée après parsing de la colonne datetime.")

    per_day_sorted = sorted(per_day, key=lambda x: x["date"])

    # Summary
    total_days = len(per_day_sorted)
    ok_days = sum(1 for d in per_day_sorted if not d["missing_hours"] and not d["duplicate_hours"] and d["n_hours"] == 24)
    missing_days = sum(1 for d in per_day_sorted if d["missing_hours"])
    dup_days = sum(1 for d in per_day_sorted if d["duplicate_hours"])

    date_min = per_day_sorted[0]["date"]
    date_max = per_day_sorted[-1]["date"]

    worst_missing = max(per_day_sorted, key=lambda x: len(x["missing_hours"]))

    lines = []
    lines.append("=== Rapport de vérification des données horaires ===")
    lines.append(f"Fichier : {os.path.abspath(csv_path)}")
    lines.append(f"Colonne datetime détectée : {dt_col}")
    lines.append(f"Période : {date_min} -> {date_max}")
    lines.append("")
    lines.append(f"Lignes totales lues : {int(len(pd.read_csv(csv_path)))}")
    lines.append(f"Lignes conservées (datetime valide) : {int(len(df))}")
    lines.append(f"Lignes avec datetime invalide : {invalid_dt}")
    lines.append("")
    lines.append(f"Nombre total de jours : {total_days}")
    lines.append(f"Jours complets (24h, sans doublons) : {ok_days}")
    lines.append(f"Jours avec heures manquantes : {missing_days}")
    lines.append(f"Jours avec doublons d'heures : {dup_days}")
    lines.append("")
    lines.append(
        f"Pire jour (heures manquantes) : {worst_missing['date']} "
        f"({len(worst_missing['missing_hours'])} heure(s) manquante(s))"
    )
    lines.append("")

    # Detailed section
    lines.append("--- Détail par jour (uniquement anomalies) ---")
    any_anomaly = False
    for d in per_day_sorted:
        missing = d["missing_hours"]
        dup = d["duplicate_hours"]
        # Consider anomaly if missing hours or duplicates or not exactly 24 unique hours
        if missing or dup or d["n_hours"] != 24:
            any_anomaly = True
            parts = [
                f"{d['date']}  |  lignes: {d['n_rows']}  |  heures uniques: {d['n_hours']}/24"
            ]
            if missing:
                parts.append(f"  -> Manquantes: {', '.join(f'{h:02d}h' for h in missing)}")
            if dup:
                parts.append(f"  -> Doublons: {', '.join(f'{h:02d}h' for h in dup)}")
            lines.extend(parts)

    if not any_anomaly:
        lines.append("Aucune anomalie détectée : tous les jours ont 24 heures et pas de doublons.")

    lines.append("")
    lines.append("Note: ce contrôle vérifie la présence d'au moins une mesure par heure (UTC).")

    return "\n".join(lines)


def show_report_window(report_text: str, title: str = "Rapport de vérification") -> None:
    root = tk.Tk()
    root.title(title)
    root.geometry("900x650")

    txt = ScrolledText(root, wrap=tk.WORD)
    txt.pack(fill=tk.BOTH, expand=True)
    txt.insert("1.0", report_text)
    txt.configure(state=tk.DISABLED)

    # Simple close shortcut
    root.bind("<Escape>", lambda e: root.destroy())

    root.mainloop()


def main() -> None:
    # If a path is provided, use it. Otherwise, ask via file dialog.
    csv_path = None
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]

    if not csv_path:
        root = tk.Tk()
        root.withdraw()
        csv_path = filedialog.askopenfilename(
            title="Choisir un fichier CSV",
            filetypes=[("CSV", "*.csv"), ("Tous les fichiers", "*.*")],
        )
        root.destroy()

    if not csv_path:
        return

    try:
        report = analyze_csv(csv_path)
    except Exception as e:
        # Show error in a Tk dialog
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Erreur", f"Impossible d'analyser le fichier.\n\n{e}")
        root.destroy()
        raise

    show_report_window(report, title=f"Rapport - {os.path.basename(csv_path)}")


if __name__ == "__main__":
    main()