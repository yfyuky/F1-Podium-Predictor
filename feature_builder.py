import pandas as pd

AVG_FINISH_FALLBACK = 10.489663
RATE_FALLBACK = 0.0
COUNT_FALLBACK = 0.0

def previous_rows(df, season, rnd):
    return df[(df["season"] < season) | ((df["season"] == season) & (df["round"] < rnd))]

def latest_row(df):
    if df.empty:
        return None
    return df.sort_values(["season", "round"]).iloc[-1]

def build_2025_lineup_map(df_feat):
    season_df = (
        df_feat[df_feat["season"] == 2025][["driverId", "constructorId"]]
        .drop_duplicates()
    )
    return dict(zip(season_df["driverId"], season_df["constructorId"]))

def infer_latest_round_2025(df_feat):
    rows_2025 = df_feat[df_feat["season"] == 2025]
    if rows_2025.empty:
        raise ValueError("No 2025 rows found in dataset.")
    return int(rows_2025["round"].max())

def build_feature_row(df_feat, lineup_map, circuit_id, driver_id, qual_position, grid):
    season = 2025
    rnd = infer_latest_round_2025(df_feat)

    if driver_id not in lineup_map:
        raise ValueError(f"Driver '{driver_id}' not found in 2025 lineup map.")
    if circuit_id not in set(df_feat["circuitId"].dropna().unique()):
        raise ValueError(f"Circuit '{circuit_id}' not found in dataset.")
    if float(qual_position) <= 0:
        raise ValueError("Qualifying position must be greater than 0.")
    if int(grid) <= 0:
        raise ValueError("Grid must be greater than 0.")

    constructor_id = lineup_map[driver_id]
    hist = previous_rows(df_feat, season, rnd)

    driver_hist = hist[hist["driverId"] == driver_id]
    constructor_hist = hist[hist["constructorId"] == constructor_id]
    driver_track_hist = hist[(hist["driverId"] == driver_id) & (hist["circuitId"] == circuit_id)]
    constructor_track_hist = hist[(hist["constructorId"] == constructor_id) & (hist["circuitId"] == circuit_id)]
    driver_constructor_hist = hist[(hist["driverId"] == driver_id) & (hist["constructorId"] == constructor_id)]

    latest_driver = latest_row(driver_hist)
    latest_constructor = latest_row(constructor_hist)
    latest_driver_track = latest_row(driver_track_hist)
    latest_constructor_track = latest_row(constructor_track_hist)
    latest_driver_constructor = latest_row(driver_constructor_hist)

    row = {
        "season": season,
        "round": rnd,
        "circuitId": circuit_id,
        "driverId": driver_id,
        "constructorId": constructor_id,
        "grid": int(grid),
        "qual_position": float(qual_position),
        "driver_points_last3": float(latest_driver["driver_points_last3"]) if latest_driver is not None else COUNT_FALLBACK,
        "constructor_points_last3": float(latest_constructor["constructor_points_last3"]) if latest_constructor is not None else COUNT_FALLBACK,
        "driver_podiums_last3": float(latest_driver["driver_podiums_last3"]) if latest_driver is not None else COUNT_FALLBACK,
        "driver_finishpos_last3": float(latest_driver["driver_finishpos_last3"]) if latest_driver is not None else AVG_FINISH_FALLBACK,
        "constructor_podiums_last3": float(latest_constructor["constructor_podiums_last3"]) if latest_constructor is not None else COUNT_FALLBACK,
        "driver_track_avg_finish": float(latest_driver_track["driver_track_avg_finish"]) if latest_driver_track is not None else AVG_FINISH_FALLBACK,
        "driver_track_podium_rate": float(latest_driver_track["driver_track_podium_rate"]) if latest_driver_track is not None else RATE_FALLBACK,
        "constructor_track_avg_finish": float(latest_constructor_track["constructor_track_avg_finish"]) if latest_constructor_track is not None else AVG_FINISH_FALLBACK,
        "constructor_track_podium_rate": float(latest_constructor_track["constructor_track_podium_rate"]) if latest_constructor_track is not None else RATE_FALLBACK,
        "driver_constructor_avg_finish": float(latest_driver_constructor["driver_constructor_avg_finish"]) if latest_driver_constructor is not None else AVG_FINISH_FALLBACK,
        "driver_constructor_podium_rate": float(latest_driver_constructor["driver_constructor_podium_rate"]) if latest_driver_constructor is not None else RATE_FALLBACK,
        "grid_inverse": 1.0 / float(grid),
    }

    return row
