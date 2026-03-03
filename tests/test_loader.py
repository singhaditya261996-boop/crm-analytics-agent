"""
tests/test_loader.py — Comprehensive tests for data/loader.py (Module 2).

All test data is generated synthetically via Faker — no real CRM data
is hardcoded anywhere in this file.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pytest
from faker import Faker

from data.loader import (
    ColumnSchema,
    DataLoader,
    FolderLoadResult,
    LoadingSummary,
    TableSchema,
    TypeInferrer,
)

fake = Faker()
Faker.seed(42)
random.seed(42)

# ── Config fixture ─────────────────────────────────────────────────────────────

@pytest.fixture()
def config(tmp_path: Path) -> dict:
    return {
        "data": {
            "upload_folder": str(tmp_path / "uploads"),
            "cache_folder": str(tmp_path / ".cache"),
        }
    }


@pytest.fixture()
def loader(config) -> DataLoader:
    return DataLoader(config)


@pytest.fixture()
def inferrer() -> TypeInferrer:
    return TypeInferrer()


# ── Synthetic data builders ────────────────────────────────────────────────────

def make_crm_df(n: int = 50) -> pd.DataFrame:
    """Generate a realistic CRM accounts DataFrame with mixed column types."""
    stages = ["Lead", "Qualified", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
    industries = ["Tech", "Finance", "Healthcare", "Retail", "Energy", "Manufacturing"]
    return pd.DataFrame({
        "account_id":    [fake.uuid4() for _ in range(n)],
        "account_name":  [fake.company() for _ in range(n)],
        "industry":      [random.choice(industries) for _ in range(n)],
        "stage":         [random.choice(stages) for _ in range(n)],
        "revenue":       [round(random.uniform(10_000, 2_000_000), 2) for _ in range(n)],
        "deal_value":    [f"£{random.randint(5_000, 500_000):,}" for _ in range(n)],
        "win_rate":      [f"{random.randint(5, 95)}%" for _ in range(n)],
        "email":         [fake.company_email() for _ in range(n)],
        "phone":         [fake.phone_number() for _ in range(n)],
        "created_date":  [fake.date_between(start_date="-3y", end_date="today").isoformat()
                          for _ in range(n)],
        "notes":         [fake.sentence(nb_words=15) for _ in range(n)],
    })


def make_currency_series(n: int = 40, symbol: str = "£") -> pd.Series:
    vals = [f"{symbol}{random.randint(1_000, 999_000):,}" for _ in range(n)]
    return pd.Series(vals, name="deal_value")


def make_percentage_series(n: int = 40) -> pd.Series:
    vals = [f"{random.randint(1, 100)}%" for _ in range(n)]
    return pd.Series(vals, name="win_rate")


def make_date_series(n: int = 40) -> pd.Series:
    vals = [fake.date_between(start_date="-5y", end_date="today").isoformat()
            for _ in range(n)]
    return pd.Series(vals, name="created_date")


def make_id_series(n: int = 40) -> pd.Series:
    return pd.Series([fake.uuid4() for _ in range(n)], name="account_id")


def make_categorical_series(n: int = 40) -> pd.Series:
    stages = ["Lead", "Qualified", "Proposal", "Closed Won", "Closed Lost"]
    return pd.Series([random.choice(stages) for _ in range(n)], name="stage")


def save_csv(df: pd.DataFrame, folder: Path, name: str = "accounts.csv") -> Path:
    p = folder / name
    df.to_csv(p, index=False)
    return p


def save_excel(df: pd.DataFrame, folder: Path, name: str = "accounts.xlsx") -> Path:
    p = folder / name
    df.to_excel(p, index=False, engine="openpyxl")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# TypeInferrer tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTypeInferrer:

    # ── Currency detection ────────────────────────────────────────────────────

    def test_detects_gbp_symbol_column(self, inferrer):
        df = pd.DataFrame({"deal_value": make_currency_series(symbol="£")})
        cast_df, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "currency"
        assert pd.api.types.is_numeric_dtype(cast_df["deal_value"])

    def test_detects_usd_symbol_column(self, inferrer):
        df = pd.DataFrame({"deal_value": make_currency_series(symbol="$")})
        cast_df, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "currency"

    def test_detects_eur_symbol_column(self, inferrer):
        df = pd.DataFrame({"deal_value": make_currency_series(symbol="€")})
        cast_df, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "currency"

    def test_currency_strips_symbols_and_commas(self, inferrer):
        df = pd.DataFrame({"revenue": ["£1,234,567", "£890,000", "£45,000"]})
        cast_df, _ = inferrer.infer_and_cast(df)
        assert cast_df["revenue"].tolist() == [1_234_567.0, 890_000.0, 45_000.0]

    def test_currency_name_keyword_triggers_detection(self, inferrer):
        # No symbol — detected by column name alone
        df = pd.DataFrame({"revenue": [100000, 250000, 75000]})
        cast_df, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "currency"

    def test_currency_name_keywords(self, inferrer):
        for name in ["revenue", "amount", "price", "cost", "budget", "spend", "income"]:
            df = pd.DataFrame({name: [1000, 2000, 3000]})
            _, schemas = inferrer.infer_and_cast(df)
            assert schemas[0].inferred_type == "currency", f"Failed for column name: {name}"

    def test_currency_result_is_float(self, inferrer):
        df = pd.DataFrame({"deal_value": ["$10,000", "$20,000", "$30,000"]})
        cast_df, _ = inferrer.infer_and_cast(df)
        assert cast_df["deal_value"].dtype in (float, "float64")

    # ── Percentage detection ──────────────────────────────────────────────────

    def test_detects_percentage_symbol_column(self, inferrer):
        df = pd.DataFrame({"win_rate": make_percentage_series()})
        cast_df, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "percentage"
        assert pd.api.types.is_numeric_dtype(cast_df["win_rate"])

    def test_percentage_strips_symbol(self, inferrer):
        df = pd.DataFrame({"win_rate": ["75%", "40%", "90%"]})
        cast_df, _ = inferrer.infer_and_cast(df)
        assert cast_df["win_rate"].tolist() == [75.0, 40.0, 90.0]

    def test_percentage_name_keyword_triggers_detection(self, inferrer):
        for name in ["win_rate", "churn", "conversion", "pct", "ratio", "discount"]:
            df = pd.DataFrame({name: [10.0, 20.0, 30.0]})
            _, schemas = inferrer.infer_and_cast(df)
            assert schemas[0].inferred_type == "percentage", f"Failed for: {name}"

    def test_percentage_result_is_float(self, inferrer):
        df = pd.DataFrame({"completion": ["50%", "75%", "100%"]})
        cast_df, _ = inferrer.infer_and_cast(df)
        assert cast_df["completion"].dtype in (float, "float64")

    # ── Date detection ────────────────────────────────────────────────────────

    def test_detects_iso_date_column(self, inferrer):
        df = pd.DataFrame({"created_date": make_date_series()})
        cast_df, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "date"
        assert pd.api.types.is_datetime64_any_dtype(cast_df["created_date"])

    def test_detects_various_date_formats(self, inferrer):
        for dates in [
            ["2024-01-15", "2024-02-20", "2024-03-10"],
            ["15/01/2024", "20/02/2024", "10/03/2024"],
            ["January 15 2024", "February 20 2024", "March 10 2024"],
        ]:
            df = pd.DataFrame({"close_date": dates})
            cast_df, schemas = inferrer.infer_and_cast(df)
            assert schemas[0].inferred_type == "date", f"Failed for format: {dates[0]}"

    def test_does_not_convert_non_date_strings(self, inferrer):
        df = pd.DataFrame({"name": [fake.name() for _ in range(20)]})
        _, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type != "date"

    def test_already_datetime_column_stays_date(self, inferrer):
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-02-01"])})
        _, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "date"

    # ── Identifier detection ──────────────────────────────────────────────────

    def test_detects_uuid_id_column(self, inferrer):
        df = pd.DataFrame({"account_id": make_id_series(40)})
        _, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "identifier"

    def test_detects_sequential_integer_id(self, inferrer):
        df = pd.DataFrame({"contact_id": list(range(1, 51))})
        _, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "identifier"

    def test_id_suffixes_recognized(self, inferrer):
        for suffix_name in ["record_id", "account_key", "lead_ref", "deal_code", "invoice_no"]:
            df = pd.DataFrame({suffix_name: [fake.uuid4() for _ in range(40)]})
            _, schemas = inferrer.infer_and_cast(df)
            assert schemas[0].inferred_type == "identifier", f"Failed for: {suffix_name}"

    # ── Categorical detection ─────────────────────────────────────────────────

    def test_detects_low_cardinality_categorical(self, inferrer):
        df = pd.DataFrame({"stage": make_categorical_series(50)})
        _, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "categorical"

    def test_boolean_column_is_categorical(self, inferrer):
        df = pd.DataFrame({"is_active": [True, False, True, True, False]})
        _, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "categorical"

    # ── Email / Phone ─────────────────────────────────────────────────────────

    def test_detects_email_column(self, inferrer):
        df = pd.DataFrame({"email": [fake.email() for _ in range(20)]})
        _, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "email"

    def test_detects_phone_column(self, inferrer):
        df = pd.DataFrame({"phone": [fake.phone_number() for _ in range(20)]})
        _, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "phone"

    # ── Numeric ───────────────────────────────────────────────────────────────

    def test_plain_numeric_column(self, inferrer):
        df = pd.DataFrame({"headcount": [10, 25, 50, 100, 200]})
        _, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "numeric"

    def test_numeric_string_column_is_cast(self, inferrer):
        df = pd.DataFrame({"employees": ["10", "25", "50", "100"]})
        cast_df, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "numeric"
        assert pd.api.types.is_numeric_dtype(cast_df["employees"])

    # ── Text ──────────────────────────────────────────────────────────────────

    def test_high_cardinality_string_is_text(self, inferrer):
        df = pd.DataFrame({"notes": [fake.sentence(nb_words=10) for _ in range(100)]})
        _, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].inferred_type == "text"

    # ── Null handling ─────────────────────────────────────────────────────────

    def test_null_count_in_schema(self, inferrer):
        vals = ["£1,000", None, "£2,000", None, "£3,000"]
        df = pd.DataFrame({"deal_value": vals})
        _, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].null_count == 2
        assert schemas[0].null_pct == 40.0

    def test_all_null_column_does_not_crash(self, inferrer):
        df = pd.DataFrame({"empty_col": [None, None, None]})
        cast_df, schemas = inferrer.infer_and_cast(df)
        assert schemas[0].null_count == 3

    # ── Sample values ─────────────────────────────────────────────────────────

    def test_sample_values_max_three(self, inferrer):
        df = pd.DataFrame({"stage": ["Lead", "Won", "Lost", "Prospect", "Lead"]})
        _, schemas = inferrer.infer_and_cast(df)
        assert len(schemas[0].sample_values) <= 3

    def test_sample_values_excludes_nulls(self, inferrer):
        df = pd.DataFrame({"stage": [None, None, "Lead", "Won", "Lost"]})
        _, schemas = inferrer.infer_and_cast(df)
        assert None not in schemas[0].sample_values

    # ── Full DataFrame ────────────────────────────────────────────────────────

    def test_full_crm_dataframe_all_columns_inferred(self, inferrer):
        df = make_crm_df(80)
        cast_df, schemas = inferrer.infer_and_cast(df)
        schema_map = {s.name: s.inferred_type for s in schemas}

        assert schema_map["revenue"] == "currency"
        assert schema_map["deal_value"] == "currency"
        assert schema_map["win_rate"] == "percentage"
        assert schema_map["created_date"] == "date"
        assert schema_map["stage"] == "categorical"
        assert schema_map["industry"] == "categorical"
        assert schema_map["email"] == "email"
        assert schema_map["phone"] == "phone"
        assert schema_map["account_id"] == "identifier"

    def test_infer_returns_same_row_count(self, inferrer):
        df = make_crm_df(100)
        cast_df, _ = inferrer.infer_and_cast(df)
        assert len(cast_df) == 100


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader — scan_folder
# ═══════════════════════════════════════════════════════════════════════════════

class TestScanFolder:

    def test_finds_csv_files(self, loader, tmp_path):
        (tmp_path / "a.csv").write_text("col\n1\n2")
        (tmp_path / "b.csv").write_text("col\n3\n4")
        found = loader.scan_folder(tmp_path)
        assert len(found) == 2

    def test_finds_xlsx_files(self, loader, tmp_path):
        df = pd.DataFrame({"x": [1, 2]})
        df.to_excel(tmp_path / "accounts.xlsx", index=False)
        found = loader.scan_folder(tmp_path)
        assert len(found) == 1

    def test_ignores_non_tabular_files(self, loader, tmp_path):
        (tmp_path / "report.pdf").write_bytes(b"%PDF")
        (tmp_path / "notes.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("col\n1")
        found = loader.scan_folder(tmp_path)
        assert len(found) == 1
        assert found[0].suffix == ".csv"

    def test_empty_folder_returns_empty_list(self, loader, tmp_path):
        found = loader.scan_folder(tmp_path)
        assert found == []

    def test_missing_folder_raises(self, loader, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader.scan_folder(tmp_path / "nonexistent")

    def test_returns_sorted_paths(self, loader, tmp_path):
        for name in ["c.csv", "a.csv", "b.csv"]:
            (tmp_path / name).write_text("x\n1")
        found = loader.scan_folder(tmp_path)
        names = [p.name for p in found]
        assert names == sorted(names)


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader — load_folder
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadFolder:

    def test_loads_single_csv(self, loader, tmp_path):
        save_csv(make_crm_df(30), tmp_path, "accounts.csv")
        result = loader.load_folder(tmp_path)
        assert isinstance(result, FolderLoadResult)
        assert "accounts" in result.dataframes
        assert len(result.dataframes["accounts"]) == 30

    def test_loads_multiple_files(self, loader, tmp_path):
        save_csv(make_crm_df(20), tmp_path, "accounts.csv")
        save_csv(make_crm_df(15), tmp_path, "contacts.csv")
        result = loader.load_folder(tmp_path)
        assert len(result.dataframes) == 2

    def test_summary_counts_correct(self, loader, tmp_path):
        save_csv(make_crm_df(25), tmp_path, "a.csv")
        save_csv(make_crm_df(10), tmp_path, "b.csv")
        result = loader.load_folder(tmp_path)
        assert result.summary.files_found == 2
        assert result.summary.files_loaded == 2
        assert result.summary.files_failed == 0
        assert result.summary.total_rows == 35

    def test_schemas_present_for_each_file(self, loader, tmp_path):
        save_csv(make_crm_df(20), tmp_path, "accounts.csv")
        result = loader.load_folder(tmp_path)
        assert "accounts" in result.schemas
        schema = result.schemas["accounts"]
        assert isinstance(schema, TableSchema)
        assert schema.row_count == 20

    def test_schema_has_correct_column_count(self, loader, tmp_path):
        df = make_crm_df(20)
        save_csv(df, tmp_path, "accounts.csv")
        result = loader.load_folder(tmp_path)
        assert result.schemas["accounts"].col_count == len(df.columns)

    def test_types_correctly_inferred_in_folder_load(self, loader, tmp_path):
        save_csv(make_crm_df(50), tmp_path, "accounts.csv")
        result = loader.load_folder(tmp_path)
        schema = result.schemas["accounts"]
        type_map = {c.name: c.inferred_type for c in schema.columns}
        assert type_map["revenue"] == "currency"
        assert type_map["win_rate"] == "percentage"
        assert type_map["created_date"] == "date"

    def test_excel_file_loads(self, loader, tmp_path):
        save_excel(make_crm_df(20), tmp_path, "accounts.xlsx")
        result = loader.load_folder(tmp_path)
        assert "accounts" in result.dataframes

    def test_multi_sheet_excel_loads_both_sheets(self, loader, tmp_path):
        p = tmp_path / "crm.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            make_crm_df(15).to_excel(w, sheet_name="Accounts", index=False)
            make_crm_df(10).to_excel(w, sheet_name="Contacts", index=False)
        result = loader.load_folder(tmp_path)
        assert len(result.dataframes) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader — Parquet caching
# ═══════════════════════════════════════════════════════════════════════════════

class TestCaching:

    def test_first_load_is_cache_miss(self, loader, tmp_path):
        save_csv(make_crm_df(20), tmp_path, "accounts.csv")
        result = loader.load_folder(tmp_path)
        assert result.summary.cache_misses == 1
        assert result.summary.cache_hits == 0

    def test_second_load_is_cache_hit(self, loader, tmp_path):
        save_csv(make_crm_df(20), tmp_path, "accounts.csv")
        loader.load_folder(tmp_path)               # first load → cache miss
        result = loader.load_folder(tmp_path)       # second load → cache hit
        assert result.summary.cache_hits == 1
        assert result.summary.cache_misses == 0

    def test_cache_hit_returns_same_data(self, loader, tmp_path):
        df = make_crm_df(20)
        save_csv(df, tmp_path, "accounts.csv")
        r1 = loader.load_folder(tmp_path)
        r2 = loader.load_folder(tmp_path)
        pd.testing.assert_frame_equal(
            r1.dataframes["accounts"].reset_index(drop=True),
            r2.dataframes["accounts"].reset_index(drop=True),
            check_like=True,
        )

    def test_cache_miss_after_clear(self, loader, tmp_path):
        save_csv(make_crm_df(20), tmp_path, "accounts.csv")
        loader.load_folder(tmp_path)
        loader.cache.clear_all()
        result = loader.load_folder(tmp_path)
        assert result.summary.cache_misses == 1

    def test_cache_schema_hit_flag(self, loader, tmp_path):
        save_csv(make_crm_df(20), tmp_path, "accounts.csv")
        loader.load_folder(tmp_path)
        result = loader.load_folder(tmp_path)
        assert result.schemas["accounts"].cache_hit is True

    def test_parquet_files_created_in_cache_folder(self, loader, config, tmp_path):
        save_csv(make_crm_df(20), tmp_path, "accounts.csv")
        loader.load_folder(tmp_path)
        cache_folder = Path(config["data"]["cache_folder"])
        parquet_files = list(cache_folder.glob("*.parquet"))
        assert len(parquet_files) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader — legacy API (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLegacyAPI:

    def test_load_csv_returns_dict(self, loader, tmp_path):
        p = save_csv(make_crm_df(10), tmp_path, "accounts.csv")
        result = loader.load(p)
        assert isinstance(result, dict)
        assert "accounts" in result

    def test_load_returns_dataframe(self, loader, tmp_path):
        p = save_csv(make_crm_df(10), tmp_path, "accounts.csv")
        df = loader.load(p)["accounts"]
        assert isinstance(df, pd.DataFrame)

    def test_load_excel_single_sheet(self, loader, tmp_path):
        p = save_excel(make_crm_df(10), tmp_path, "contacts.xlsx")
        result = loader.load(p)
        assert "contacts" in result

    def test_load_excel_multi_sheet(self, loader, tmp_path):
        p = tmp_path / "multi.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            make_crm_df(5).to_excel(w, sheet_name="Sheet1", index=False)
            make_crm_df(5).to_excel(w, sheet_name="Sheet2", index=False)
        result = loader.load(p)
        assert "multi__Sheet1" in result
        assert "multi__Sheet2" in result

    def test_load_missing_file_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load("/no/such/file.csv")

    def test_load_unsupported_type_raises(self, loader, tmp_path):
        p = tmp_path / "data.parquet"
        pd.DataFrame({"x": [1]}).to_parquet(p)
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(p)

    def test_load_many_combines_files(self, loader, tmp_path):
        p1 = save_csv(make_crm_df(10), tmp_path, "accounts.csv")
        p2 = save_csv(make_crm_df(8), tmp_path, "contacts.csv")
        result = loader.load_many([p1, p2])
        assert len(result) == 2

    def test_load_many_skips_bad_file(self, loader, tmp_path):
        good = save_csv(make_crm_df(10), tmp_path, "good.csv")
        bad = tmp_path / "missing.csv"   # doesn't exist
        result = loader.load_many([good, bad])
        assert "good" in result


# ═══════════════════════════════════════════════════════════════════════════════
# TableSchema and ColumnSchema
# ═══════════════════════════════════════════════════════════════════════════════

class TestTableSchema:

    def test_detect_schema_returns_table_schema(self, loader, tmp_path):
        df = make_crm_df(30)
        schema = loader.detect_schema(df, name="accounts.csv")
        assert isinstance(schema, TableSchema)

    def test_schema_row_count(self, loader):
        df = make_crm_df(42)
        schema = loader.detect_schema(df)
        assert schema.row_count == 42

    def test_schema_col_count(self, loader):
        df = make_crm_df(10)
        schema = loader.detect_schema(df)
        assert schema.col_count == len(df.columns)

    def test_schema_columns_list_length(self, loader):
        df = make_crm_df(10)
        schema = loader.detect_schema(df)
        assert len(schema.columns) == len(df.columns)

    def test_column_schema_has_all_fields(self, loader):
        df = make_crm_df(10)
        schema = loader.detect_schema(df)
        for col_schema in schema.columns:
            assert isinstance(col_schema, ColumnSchema)
            assert col_schema.name
            assert col_schema.inferred_type
            assert isinstance(col_schema.null_count, int)
            assert isinstance(col_schema.null_pct, float)
            assert isinstance(col_schema.unique_count, int)
            assert isinstance(col_schema.sample_values, list)

    def test_schema_null_counts_accurate(self, loader):
        df = pd.DataFrame({
            "name":    [fake.name(), None, fake.name(), fake.name()],
            "revenue": [1000.0, 2000.0, None, 4000.0],
        })
        schema = loader.detect_schema(df)
        name_schema = schema.column("name")
        rev_schema = schema.column("revenue")
        assert name_schema.null_count == 1
        assert rev_schema.null_count == 1

    def test_schema_columns_of_type(self, loader):
        df = make_crm_df(30)
        schema = loader.detect_schema(df)
        currency_cols = schema.columns_of_type("currency")
        assert len(currency_cols) >= 1
        assert all(c.inferred_type == "currency" for c in currency_cols)

    def test_schema_column_lookup_by_name(self, loader):
        df = make_crm_df(10)
        schema = loader.detect_schema(df)
        col = schema.column("revenue")
        assert col is not None
        assert col.inferred_type == "currency"

    def test_schema_column_lookup_missing_returns_none(self, loader):
        df = make_crm_df(10)
        schema = loader.detect_schema(df)
        assert schema.column("nonexistent_column") is None


# ═══════════════════════════════════════════════════════════════════════════════
# LoadingSummary
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadingSummary:

    def test_summary_elapsed_ms_positive(self, loader, tmp_path):
        save_csv(make_crm_df(20), tmp_path, "a.csv")
        result = loader.load_folder(tmp_path)
        assert result.summary.elapsed_ms > 0

    def test_summary_print_runs_without_error(self, loader, tmp_path, capsys):
        save_csv(make_crm_df(10), tmp_path, "a.csv")
        result = loader.load_folder(tmp_path)
        result.summary.print()
        captured = capsys.readouterr()
        assert "CRM DATA LOADER" in captured.out
        assert "Files found" in captured.out

    def test_summary_error_recorded_for_bad_file(self, loader, tmp_path):
        # Create a file whose name has .xlsx but is binary garbage
        bad = tmp_path / "corrupt.xlsx"
        bad.write_bytes(b"\x00\x01\x02\x03garbage data")
        result = loader.load_folder(tmp_path)
        assert result.summary.files_failed == 1
        assert len(result.summary.errors) == 1
        assert "corrupt.xlsx" in result.summary.errors[0][0]

    def test_summary_total_rows_is_sum_of_all_files(self, loader, tmp_path):
        save_csv(make_crm_df(20), tmp_path, "a.csv")
        save_csv(make_crm_df(30), tmp_path, "b.csv")
        result = loader.load_folder(tmp_path)
        assert result.summary.total_rows == 50
