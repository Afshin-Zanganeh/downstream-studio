"""Export database rows matching IDs from an embedding CSV.

The implementation streams IDs into a temporary PostgreSQL table and streams
the query result back to CSV. It does not interpolate table or column names
into SQL strings and does not load the result table into Python memory.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

try:
    import psycopg
    from psycopg import sql
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise SystemExit(
        "PostgreSQL support is not installed. Run: "
        "python -m pip install -e '.[postgres]'"
    ) from exc


TEMP_TABLE = "interface_downstream_requested_ids"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all columns from PostgreSQL rows whose IDs occur in an embedding CSV."
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("PDRUG_DSN"),
        help="PostgreSQL DSN. Defaults to environment variable PDRUG_DSN.",
    )
    parser.add_argument("--input-csv", required=True, help="Embedding CSV containing the requested IDs.")
    parser.add_argument("--output-csv", required=True, help="Destination CSV containing all database columns.")
    parser.add_argument("--table", required=True, help="Database table, for example interface_af.")
    parser.add_argument("--schema", default="public", help="Database schema. Default: public.")
    parser.add_argument("--input-id-column", default="id", help="ID header in the embedding CSV.")
    parser.add_argument("--db-id-column", default="id", help="Matching ID column in the database table.")
    parser.add_argument(
        "--missing-ids-csv",
        default=None,
        help="Optional missing-ID output. Default: <output-csv>.missing_ids.csv",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Return a failure status if one or more input IDs are absent from the database table.",
    )
    return parser.parse_args()


def read_unique_ids(path: Path, id_column: str) -> List[str]:
    """Read IDs while rejecting empty or duplicate values."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or id_column not in reader.fieldnames:
            raise ValueError(
                f"Input CSV has no {id_column!r} column. Available columns: {reader.fieldnames}"
            )
        ids: List[str] = []
        seen = set()
        duplicates: List[str] = []
        for row_number, row in enumerate(reader, start=2):
            value = (row.get(id_column) or "").strip()
            if not value:
                raise ValueError(f"Empty ID in {path} at CSV row {row_number}.")
            if value in seen:
                if len(duplicates) < 5:
                    duplicates.append(value)
                continue
            seen.add(value)
            ids.append(value)
    if duplicates:
        raise ValueError(f"Input CSV contains duplicate IDs, for example: {duplicates}")
    if not ids:
        raise ValueError(f"Input CSV {path} contains no IDs.")
    return ids


def create_requested_ids_table(
    connection: psycopg.Connection, ids: Sequence[str], database_id_type: str
) -> None:
    """Copy IDs plus their CSV order into a transaction-local table."""
    # database_id_type comes from PostgreSQL's own format_type() catalog
    # function, not from user input. Matching the real type keeps the database
    # ID index usable during the join.
    create_query = sql.SQL(
        "CREATE TEMP TABLE {} (requested_id {} PRIMARY KEY, input_order bigint NOT NULL) ON COMMIT DROP"
    ).format(sql.Identifier(TEMP_TABLE), sql.SQL(database_id_type))
    copy_query = sql.SQL("COPY {} (requested_id, input_order) FROM STDIN").format(
        sql.Identifier(TEMP_TABLE)
    )
    with connection.cursor() as cursor:
        cursor.execute(create_query)
        with cursor.copy(copy_query) as copy:
            for input_order, interface_id in enumerate(ids):
                copy.write_row((interface_id, input_order))
        cursor.execute(
            sql.SQL("ANALYZE {}").format(sql.Identifier(TEMP_TABLE))
        )


def database_id_type(
    connection: psycopg.Connection, schema: str, table: str, id_column: str
) -> str:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT pg_catalog.format_type(attribute.atttypid, attribute.atttypmod)
            FROM pg_catalog.pg_attribute AS attribute
            JOIN pg_catalog.pg_class AS relation
              ON relation.oid = attribute.attrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = %s
              AND relation.relname = %s
              AND attribute.attname = %s
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
            """,
            (schema, table, id_column),
        )
        row = cursor.fetchone()
    if row is None:
        raise ValueError(
            f"Database table/column {schema}.{table}.{id_column} does not exist."
        )
    return str(row[0])


def matching_queries(schema: str, table: str, db_id_column: str):
    qualified_table = sql.Identifier(schema, table)
    database_id = sql.Identifier(db_id_column)
    requested = sql.Identifier(TEMP_TABLE)
    join_condition = sql.SQL("db.{} = requested.requested_id").format(database_id)

    count_query = sql.SQL(
        "SELECT COUNT(*) AS row_count, COUNT(DISTINCT requested.requested_id) AS matched_ids "
        "FROM {} AS db JOIN {} AS requested ON {}"
    ).format(qualified_table, requested, join_condition)
    missing_query = sql.SQL(
        "SELECT requested.requested_id "
        "FROM {} AS requested LEFT JOIN {} AS db ON {} "
        "WHERE db.{} IS NULL ORDER BY requested.input_order"
    ).format(requested, qualified_table, join_condition, database_id)
    export_select = sql.SQL(
        "SELECT db.* FROM {} AS db JOIN {} AS requested ON {} ORDER BY requested.input_order"
    ).format(qualified_table, requested, join_condition)
    export_copy = sql.SQL("COPY ({}) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)").format(export_select)
    return count_query, missing_query, export_copy


def write_missing_ids(path: Path, missing_ids: Sequence[str], id_column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([id_column])
        writer.writerows((value,) for value in missing_ids)


def stream_export(connection: psycopg.Connection, copy_query, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".partial")
    try:
        with connection.cursor() as cursor, temporary_path.open("wb") as output:
            with cursor.copy(copy_query) as copy:
                for block in copy:
                    output.write(bytes(block))
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def export_rows(
    dsn: str,
    input_csv: Path,
    output_csv: Path,
    table: str,
    schema: str = "public",
    input_id_column: str = "id",
    db_id_column: str = "id",
    missing_ids_csv: Path | None = None,
) -> Tuple[int, int, List[str]]:
    """Export rows and return ``(exported_rows, matched_ids, missing_ids)``."""
    ids = read_unique_ids(input_csv, input_id_column)
    missing_path = missing_ids_csv or output_csv.with_name(output_csv.name + ".missing_ids.csv")
    with psycopg.connect(dsn) as connection:
        id_type = database_id_type(connection, schema, table, db_id_column)
        create_requested_ids_table(connection, ids, id_type)
        count_query, missing_query, copy_query = matching_queries(schema, table, db_id_column)
        with connection.cursor() as cursor:
            cursor.execute(count_query)
            exported_rows, matched_ids = (int(value) for value in cursor.fetchone())
            cursor.execute(missing_query)
            missing_ids = [str(row[0]) for row in cursor.fetchall()]
        stream_export(connection, copy_query, output_csv)
        write_missing_ids(missing_path, missing_ids, input_id_column)
    return exported_rows, matched_ids, missing_ids


def main() -> None:
    args = parse_args()
    if not args.dsn:
        raise SystemExit("Provide --dsn or set the PDRUG_DSN environment variable.")
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    missing_csv = (
        Path(args.missing_ids_csv).expanduser().resolve() if args.missing_ids_csv else None
    )
    if not input_csv.is_file():
        raise SystemExit(f"Input CSV does not exist: {input_csv}")

    try:
        exported_rows, matched_ids, missing_ids = export_rows(
            dsn=args.dsn,
            input_csv=input_csv,
            output_csv=output_csv,
            table=args.table,
            schema=args.schema,
            input_id_column=args.input_id_column,
            db_id_column=args.db_id_column,
            missing_ids_csv=missing_csv,
        )
    except (psycopg.Error, OSError, ValueError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    requested_count = matched_ids + len(missing_ids)
    print(f"Requested IDs: {requested_count}")
    print(f"Matched IDs:   {matched_ids}")
    print(f"Exported rows: {exported_rows}")
    print(f"Missing IDs:   {len(missing_ids)}")
    print(f"Output CSV:    {output_csv}")
    if exported_rows > matched_ids:
        print(
            "Warning: the database returned multiple rows for at least one requested ID.",
            file=sys.stderr,
        )
    if args.fail_on_missing and missing_ids:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
