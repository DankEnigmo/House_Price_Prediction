import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from zenml import step


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "99Acres data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "99acres_clean.csv"
FACETS_DIR_NAME = "facets"
PRICE_SENTINEL = 2_147_483_647

OUTPUT_COLUMNS = [
    "Price_in_Lakhs",
    "Area_SqFt",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Bathrooms",
    "Balcony_Num",
    "Floor_No",
    "Floor_Label",
    "Total_Floors",
    "Age_Label",
    "Furnish",
    "Facing",
    "Ownership_Type",
    "Society_Name",
    "Building_Name",
    "Landmark_Count",
    "Amenities",
    "Source_File",
    "Prop_ID",
]


def _normalize_code(value: object) -> Optional[str]:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n"}:
        return None

    if re.fullmatch(r"\d+\.0+", text):
        text = str(int(float(text)))

    if re.fullmatch(r"\d+", text):
        stripped = text.lstrip("0")
        normalized = stripped or "0"
        return None if normalized == "0" else normalized

    if re.fullmatch(r"\d+\+", text):
        return text.lstrip("0")

    return text.upper() if len(text) == 1 else text


def _is_code_like(value: object) -> bool:
    code = _normalize_code(value)
    if code is None:
        return True
    return bool(re.fullmatch(r"\d+\+?", code) or re.fullmatch(r"[A-Z]", code))


def _facet_lookup_keys(value: object) -> set[str]:
    text = str(value).strip()
    normalized = _normalize_code(text)
    keys = {text}
    if normalized:
        keys.add(normalized)
    if re.fullmatch(r"\d+", text):
        keys.add(text.zfill(2))
        keys.add(text.zfill(3))
    return keys


def _load_facet_maps(data_path: Path) -> dict[str, dict[str, str]]:
    facets_dir = data_path.parent / FACETS_DIR_NAME if data_path.is_file() else data_path / FACETS_DIR_NAME
    facet_maps: dict[str, dict[str, str]] = {}

    if not facets_dir.exists():
        print(f"Facet directory not found at {facets_dir}; continuing with raw labels.")
        return facet_maps

    for facet_file in sorted(facets_dir.glob("*.csv")):
        frame = pd.read_csv(facet_file, dtype=str).fillna("")
        if "id" not in frame.columns or "label" not in frame.columns:
            continue

        lookup: dict[str, str] = {}
        for _, row in frame.iterrows():
            label = str(row["label"]).strip()
            if not label:
                continue
            for key in _facet_lookup_keys(row["id"]):
                lookup[key] = label

        facet_maps[facet_file.stem] = lookup

    return facet_maps


def _read_99acres_files(data_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if data_path.is_file():
        frame = pd.read_csv(data_path, low_memory=False)
        frame["SOURCE_FILE"] = data_path.name
        return frame, {data_path.name: len(frame)}

    csv_files = sorted(
        file for file in data_path.glob("*.csv") if file.parent.name != FACETS_DIR_NAME
    )
    if not csv_files:
        raise FileNotFoundError(f"No listing CSV files found in {data_path}")

    frames = []
    source_counts = {}
    for csv_file in csv_files:
        frame = pd.read_csv(csv_file, low_memory=False)
        frame["SOURCE_FILE"] = csv_file.name
        frames.append(frame)
        source_counts[csv_file.name] = len(frame)

    return pd.concat(frames, ignore_index=True, sort=False), source_counts


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="object")

    values = df[column].astype("string").str.strip()
    return values.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "N": pd.NA})


def _first_non_empty(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    for column in columns:
        result = result.fillna(_text_series(df, column))
    return result


def _first_available_numeric(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=df.index)
    for column in columns:
        result = result.fillna(_numeric_series(df, column))
    return result


def _parse_price_to_lakhs(value: object) -> Optional[float]:
    if pd.isna(value):
        return None

    text = str(value).replace(",", "").strip().lower()
    if not text:
        return None

    values = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
    if not values:
        return None

    amount = sum(values) / len(values)
    if "cr" in text or "crore" in text:
        return amount * 100
    if "lac" in text or "lakh" in text or re.search(r"\bl\b", text):
        return amount

    return amount / 100_000


def _derive_price_lakhs(df: pd.DataFrame) -> pd.Series:
    min_price = _numeric_series(df, "MIN_PRICE").replace(
        {0: np.nan, PRICE_SENTINEL: np.nan}
    )
    max_price = _numeric_series(df, "MAX_PRICE").replace(
        {0: np.nan, PRICE_SENTINEL: np.nan}
    )
    numeric_price_lakhs = pd.concat([min_price, max_price], axis=1).mean(axis=1) / 100_000

    parsed_price_lakhs = (
        df["PRICE"].apply(_parse_price_to_lakhs)
        if "PRICE" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    return numeric_price_lakhs.fillna(parsed_price_lakhs)


def _parse_area_sqft(value: object) -> Optional[float]:
    if pd.isna(value):
        return None

    text = str(value).replace(",", "").lower()
    values = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
    if not values:
        return None

    area = sum(values) / len(values)
    if "sq.m" in text or "sqm" in text:
        area *= 10.7639
    return area


def _derive_area_sqft(df: pd.DataFrame) -> pd.Series:
    min_area = _numeric_series(df, "MIN_AREA_SQFT")
    max_area = _numeric_series(df, "MAX_AREA_SQFT")
    area = pd.concat([min_area, max_area], axis=1).mean(axis=1)

    area = area.fillna(
        _first_available_numeric(
            df,
            [
                "SUPERBUILTUP_SQFT",
                "BUILTUP_SQFT",
                "CARPET_SQFT",
                "SUPER_SQFT",
                "SUPER_AREA",
            ],
        )
    )

    if "AREA" in df.columns:
        area = area.fillna(df["AREA"].apply(_parse_area_sqft))

    return area


def _extract_dict_text_field(series: pd.Series, field: str) -> pd.Series:
    pattern = rf"['\"]{re.escape(field)}['\"]\s*:\s*['\"]([^'\"]+)['\"]"
    return series.fillna("").astype(str).str.extract(pattern, expand=False)


def _decode_code_series(
    series: pd.Series,
    lookup: dict[str, str],
    field_name: str,
    unmapped_counts: dict[str, dict[str, int]],
    raw_fallback: Optional[pd.Series] = None,
) -> pd.Series:
    decoded = []
    misses: dict[str, int] = {}

    for idx, value in series.items():
        code = _normalize_code(value)
        label = lookup.get(code) if code is not None else None

        if label is None and raw_fallback is not None:
            fallback_value = raw_fallback.loc[idx]
            if pd.notna(fallback_value):
                fallback_text = str(fallback_value).strip()
                if fallback_text and not _is_code_like(fallback_text):
                    label = fallback_text

        if label is None and code is not None:
            misses[code] = misses.get(code, 0) + 1

        decoded.append(label if label is not None else pd.NA)

    if misses:
        unmapped_counts[field_name] = misses

    return pd.Series(decoded, index=series.index, dtype="object")


def _decode_multi_code_series(
    series: pd.Series,
    lookup: dict[str, str],
    field_name: str,
    unmapped_counts: dict[str, dict[str, int]],
) -> pd.Series:
    decoded_rows = []
    misses: dict[str, int] = {}

    for value in series:
        if pd.isna(value):
            decoded_rows.append(pd.NA)
            continue

        labels = []
        for part in re.split(r"[,|]", str(value)):
            code = _normalize_code(part)
            if code is None:
                continue
            label = lookup.get(code)
            if label:
                labels.append(label)
            else:
                misses[code] = misses.get(code, 0) + 1

        unique_labels = sorted(set(labels))
        decoded_rows.append(", ".join(unique_labels) if unique_labels else pd.NA)

    if misses:
        unmapped_counts[field_name] = misses

    return pd.Series(decoded_rows, index=series.index, dtype="object")


def _numeric_from_label(series: pd.Series) -> pd.Series:
    def parse(value: object) -> Optional[float]:
        if pd.isna(value):
            return None

        text = str(value).strip().lower()
        if text in {"ground"}:
            return 0
        if text in {"basement", "lower ground"}:
            return -1
        if text in {"multi-storied"}:
            return None

        match = re.search(r"\d+(?:\.\d+)?", text)
        return float(match.group()) if match else None

    return series.apply(parse)


def _normalize_99acres(
    df: pd.DataFrame,
    facet_maps: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_row_count = len(df)
    if "PREFERENCE" in df.columns:
        df = df[df["PREFERENCE"].fillna("").astype(str).str.upper().eq("S")].copy()

    sale_row_count = len(df)
    unmapped_counts: dict[str, dict[str, int]] = {}
    location_text = _text_series(df, "location")

    normalized = pd.DataFrame(index=df.index)
    normalized["Price_in_Lakhs"] = _derive_price_lakhs(df)
    normalized["Area_SqFt"] = _derive_area_sqft(df)
    normalized["City"] = _text_series(df, "CITY").fillna(
        _extract_dict_text_field(location_text, "CITY_NAME")
    )

    normalized["Locality"] = _first_non_empty(df, ["LOCALITY", "LOCALITY_WO_CITY"])
    normalized["Locality"] = normalized["Locality"].fillna(
        _extract_dict_text_field(location_text, "LOCALITY_NAME")
    )
    normalized["Locality"] = normalized["Locality"].fillna(
        _decode_code_series(
            _text_series(df, "LOCALITY_ID"),
            facet_maps.get("LOCALITY_ID", {}),
            "LOCALITY_ID",
            unmapped_counts,
        )
    )

    normalized["Property_Type"] = _decode_code_series(
        _text_series(df, "PROPERTY_TYPE"),
        facet_maps.get("PROPERTY_TYPE", {}),
        "PROPERTY_TYPE",
        unmapped_counts,
        raw_fallback=_text_series(df, "PROPERTY_TYPE"),
    )

    bedroom_label = _decode_code_series(
        _text_series(df, "BEDROOM_NUM"),
        facet_maps.get("BEDROOM_NUM", {}),
        "BEDROOM_NUM",
        unmapped_counts,
    )
    bathroom_label = _decode_code_series(
        _text_series(df, "BATHROOM_NUM"),
        facet_maps.get("BATHROOM_NUM", {}),
        "BATHROOM_NUM",
        unmapped_counts,
    )
    normalized["BHK"] = _numeric_series(df, "BEDROOM_NUM").fillna(
        _numeric_from_label(bedroom_label)
    )
    normalized["Bathrooms"] = _numeric_series(df, "BATHROOM_NUM").fillna(
        _numeric_from_label(bathroom_label)
    )
    normalized["Balcony_Num"] = _numeric_series(df, "BALCONY_NUM")

    normalized["Floor_Label"] = _decode_code_series(
        _text_series(df, "FLOOR_NUM"),
        facet_maps.get("FLOOR_NUM", {}),
        "FLOOR_NUM",
        unmapped_counts,
    )
    normalized["Floor_No"] = _numeric_from_label(normalized["Floor_Label"]).fillna(
        _numeric_series(df, "FLOOR_NUM")
    )

    total_floor_label = _decode_code_series(
        _text_series(df, "TOTAL_FLOOR"),
        facet_maps.get("TOTAL_FLOOR", {}),
        "TOTAL_FLOOR",
        unmapped_counts,
    )
    normalized["Total_Floors"] = _numeric_from_label(total_floor_label).fillna(
        _numeric_series(df, "TOTAL_FLOOR")
    )

    normalized["Age_Label"] = _decode_code_series(
        _text_series(df, "AGE"),
        facet_maps.get("AGE", {}),
        "AGE",
        unmapped_counts,
    )
    normalized["Furnish"] = _decode_code_series(
        _text_series(df, "FURNISH"),
        facet_maps.get("FURNISH", {}),
        "FURNISH",
        unmapped_counts,
    )
    normalized["Facing"] = _decode_code_series(
        _text_series(df, "FACING"),
        facet_maps.get("FACING_DIRECTION", {}),
        "FACING",
        unmapped_counts,
    )
    normalized["Ownership_Type"] = _decode_code_series(
        _text_series(df, "OWNTYPE"),
        facet_maps.get("OWNERSHIP_TYPE", {}),
        "OWNTYPE",
        unmapped_counts,
    )

    normalized["Society_Name"] = _first_non_empty(df, ["SOCIETY_NAME", "PROP_NAME"])
    normalized["Building_Name"] = _decode_code_series(
        _text_series(df, "BUILDING_ID"),
        facet_maps.get("BUILDING_ID", {}),
        "BUILDING_ID",
        unmapped_counts,
        raw_fallback=_first_non_empty(df, ["BUILDING_NAME", "SOCIETY_NAME", "PROP_NAME"]),
    )
    normalized["Landmark_Count"] = _numeric_series(df, "TOTAL_LANDMARK_COUNT")

    amenity_lookup = {
        **facet_maps.get("FEATURES", {}),
        **facet_maps.get("AMENITIES", {}),
    }
    amenities = _decode_multi_code_series(
        _text_series(df, "AMENITIES"),
        amenity_lookup,
        "AMENITIES",
        unmapped_counts,
    )
    features = _decode_multi_code_series(
        _text_series(df, "FEATURES"),
        amenity_lookup,
        "FEATURES",
        unmapped_counts,
    )
    normalized["Amenities"] = (
        amenities.fillna("").astype(str) + ", " + features.fillna("").astype(str)
    ).str.strip(" ,")

    normalized["Source_File"] = _text_series(df, "SOURCE_FILE")
    normalized["Prop_ID"] = _first_non_empty(df, ["PROP_ID", "SPID"])

    normalized = normalized.replace({"": pd.NA})
    duplicate_count = int(normalized.duplicated().sum())
    normalized = normalized.drop_duplicates().reset_index(drop=True)

    quality_report = {
        "raw_rows": raw_row_count,
        "sale_rows": sale_row_count,
        "price_rows": int(normalized["Price_in_Lakhs"].notna().sum()),
        "area_rows": int(normalized["Area_SqFt"].notna().sum()),
        "duplicates_removed": duplicate_count,
        "unmapped_counts": unmapped_counts,
    }

    return normalized[OUTPUT_COLUMNS], quality_report


def _write_processed_dataset(df: pd.DataFrame) -> None:
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Wrote processed dataset to {PROCESSED_DATA_PATH}")


def _print_quality_report(
    source_counts: dict[str, int],
    quality_report: dict[str, object],
) -> None:
    print("Raw rows per source file:")
    for source_file, count in source_counts.items():
        print(f"  - {source_file}: {count}")

    print(f"Rows kept after sale filter: {quality_report['sale_rows']}")
    print(f"Rows with parsed price: {quality_report['price_rows']}")
    print(f"Rows with parsed area: {quality_report['area_rows']}")
    print(f"Duplicate rows removed: {quality_report['duplicates_removed']}")

    unmapped_counts = quality_report.get("unmapped_counts", {})
    if not unmapped_counts:
        print("Unmapped facet codes: none")
        return

    print("Unmapped facet code counts:")
    for field_name, counts in sorted(unmapped_counts.items()):
        top_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10]
        formatted = ", ".join(f"{code}={count}" for code, count in top_counts)
        print(f"  - {field_name}: {formatted}")


@step
def data_ingestion() -> pd.DataFrame:
    """
    Load, decode, normalize, and persist 99Acres sale listing data.

    Returns:
        A model-ready Pandas DataFrame containing normalized sale listings.
    """
    data_path = Path(os.getenv("HOUSE_PRICE_DATA_PATH", DEFAULT_DATA_PATH))

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found at {data_path}")

    print(f"Loading 99Acres data from {data_path}...")
    raw_df, source_counts = _read_99acres_files(data_path)
    facet_maps = _load_facet_maps(data_path)
    df, quality_report = _normalize_99acres(raw_df, facet_maps)

    _write_processed_dataset(df)
    _print_quality_report(source_counts, quality_report)

    print(f"Data ingested. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df
