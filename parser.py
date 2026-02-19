import os
import sys
import pandas as pd
from tkinter import Tk, filedialog, messagebox


PREFERRED_DT_COLS = [
    "datetime_utc",
    "dh_utc",
    "date_utc",
    "timestamp_utc",
    "timestamp",
    "datetime",
]


def detect_datetime_column(df: pd.DataFrame) -> str:
    for c in PREFERRED_DT_COLS:
        if c in df.columns:
            return c
    for c in df.columns:
        name = str(c).lower()
        if "date" in name or "time" in name or "dh" in name:
            return c
    raise ValueError("Impossible de trouver une colonne datetime dans le CSV.")


def keep_only_complete_days(df: pd.DataFrame) -> pd.DataFrame:
    # Garde uniquement les jours où toutes les heures 0..23 sont présentes (au moins une mesure par heure)
    expected = set(range(24))
    grp = df.groupby("_date")["_hour"].apply(lambda s: expected.issubset(set(s.astype(int).tolist())))
    valid_dates = set(grp[grp].index.tolist())
    return df[df["_date"].isin(valid_dates)].copy()


def build_consecutive_segments(df: pd.DataFrame, dt_col: str) -> list[pd.DataFrame]:
    # df doit être trié par _dt
    # Un segment = timestamps consécutifs à 1h pile
    diffs = df["_dt"].diff()
    cut = diffs.ne(pd.Timedelta(hours=1))  # True si rupture (y compris première ligne)
    seg_id = cut.cumsum()
    segments = [g.copy() for _, g in df.groupby(seg_id)]
    return segments


def insert_separator_row(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    # Crée une ligne avec "*" dans dt_col, le reste vide
    sep = {c: "" for c in df.columns}
    sep[dt_col] = "*"
    return pd.DataFrame([sep])


def clean_csv_72h(csv_path: str, min_hours: int = 72) -> tuple[str, str]:
    raw = pd.read_csv(csv_path)
    dt_col = detect_datetime_column(raw)

    df = raw.copy()
    df["_dt"] = pd.to_datetime(df[dt_col], errors="coerce", utc=False)
    invalid_dt = int(df["_dt"].isna().sum())
    df = df.dropna(subset=["_dt"]).copy()

    # Normalisation : tri + suppression doublons de timestamps (optionnel mais très utile)
    df = df.sort_values("_dt").copy()
    before_dups = len(df)
    df = df.drop_duplicates(subset=["_dt"], keep="first").copy()
    removed_dups = before_dups - len(df)

    df["_date"] = df["_dt"].dt.date
    df["_hour"] = df["_dt"].dt.hour

    # 1) Supprimer les jours incomplets
    before_days = df["_date"].nunique()
    df = keep_only_complete_days(df)
    after_days = df["_date"].nunique()
    removed_days = before_days - after_days

    if df.empty:
        report = (
            "Après suppression des dates invalides et des jours incomplets, il ne reste aucune donnée.\n"
            f"- Datetime invalides: {invalid_dt}\n"
            f"- Doublons timestamp supprimés: {removed_dups}\n"
            f"- Jours supprimés car incomplets: {removed_days}\n"
        )
        return "", report

    # 2) Segments horaires consécutifs
    df = df.sort_values("_dt").copy()
    segments = build_consecutive_segments(df, dt_col)

    # 3) Garder uniquement les segments >= 72h
    kept = []
    for seg in segments:
        # longueur en heures = nombre de lignes (si 1 ligne/heure après drop_duplicates)
        if len(seg) >= min_hours:
            kept.append(seg)

    if not kept:
        # Rien n'a au moins 72h consécutives
        max_len = max(len(s) for s in segments) if segments else 0
        report = (
            "Aucun segment ne contient au moins 72 heures consécutives.\n"
            f"- Datetime invalides: {invalid_dt}\n"
            f"- Doublons timestamp supprimés: {removed_dups}\n"
            f"- Jours supprimés car incomplets: {removed_days}\n"
            f"- Plus long segment consécutif trouvé: {max_len} heure(s)\n"
        )
        return "", report

    # 4) Recomposer le CSV final + séparateurs "*"
    out_parts = []
    for i, seg in enumerate(kept):
        seg_out = seg.drop(columns=["_dt", "_date", "_hour"], errors="ignore")
        if i > 0:
            out_parts.append(insert_separator_row(seg_out, dt_col))
        out_parts.append(seg_out)

    out_df = pd.concat(out_parts, ignore_index=True)

    out_path = os.path.splitext(csv_path)[0] + "_clean72h.csv"
    out_df.to_csv(out_path, index=False)

    report_lines = []
    report_lines.append("=== Nettoyage 72h consécutives ===")
    report_lines.append(f"Fichier entrée : {os.path.abspath(csv_path)}")
    report_lines.append(f"Fichier sortie : {os.path.abspath(out_path)}")
    report_lines.append(f"Colonne datetime : {dt_col}")
    report_lines.append("")
    report_lines.append(f"Datetime invalides supprimées : {invalid_dt}")
    report_lines.append(f"Doublons timestamp supprimés : {removed_dups}")
    report_lines.append(f"Jours supprimés (incomplets) : {removed_days}")
    report_lines.append(f"Segments conservés (>= {min_hours}h) : {len(kept)}")
    report_lines.append("Longueurs des segments conservés : " + ", ".join(str(len(s)) for s in kept))
    report_lines.append("")
    report_lines.append("Note: une ligne contenant '*' est insérée entre deux segments conservés.")
    return out_path, "\n".join(report_lines)


def pick_file() -> str:
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choisir un fichier CSV",
        filetypes=[("CSV", "*.csv"), ("Tous les fichiers", "*.*")],
    )
    root.destroy()
    return path


def main():
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    else:
        csv_path = pick_file()

    if not csv_path:
        return

    try:
        out_path, report = clean_csv_72h(csv_path, min_hours=72)
        root = Tk()
        root.withdraw()
        if out_path:
            messagebox.showinfo("OK", report)
        else:
            messagebox.showwarning("Aucun résultat", report)
        root.destroy()
    except Exception as e:
        root = Tk()
        root.withdraw()
        messagebox.showerror("Erreur", str(e))
        root.destroy()
        raise


if __name__ == "__main__":
    main()