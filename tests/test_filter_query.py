from __future__ import annotations

import sqlite3
import unittest

from pipeline import db
from pipeline.filter_query import FilterQueryError, parse_filter_query


class FilterQueryTests(unittest.TestCase):
    def test_example_query(self) -> None:
        query = 'Movie/Show = "Porco Rosso" OR "Slam Dunk" AND People >= 1 AND MINSEC = 5'
        clauses = parse_filter_query(query)

        self.assertEqual(clauses[0].field, "title")
        self.assertEqual(clauses[0].values, ("Porco Rosso", "Slam Dunk"))
        self.assertEqual((clauses[1].field, clauses[1].operator), ("people", ">="))
        self.assertEqual((clauses[2].field, clauses[2].operator), ("minsec", "="))

    def test_example_compiles_to_parameterized_sql(self) -> None:
        where, values = db._clip_where(  # noqa: SLF001 - focused SQL compiler test
            {"filter_query": 'Movie/Show = "Porco Rosso" OR "Slam Dunk" AND People >= 1 AND MINSEC = 5'}
        )

        sql = " AND ".join(where)
        self.assertIn("LOWER(movies.collection_title) IN (?, ?)", sql)
        self.assertIn("clips.duration >= ?", sql)
        self.assertEqual(values, ["porco rosso", "slam dunk", 1, 5.0])

    def test_example_filters_rows_in_sqlite(self) -> None:
        where, values = db._clip_where(  # noqa: SLF001 - focused SQL compiler test
            {"filter_query": 'Movie/Show = "Porco Rosso" OR "Slam Dunk" AND People >= 1 AND MINSEC = 5'}
        )
        connection = sqlite3.connect(":memory:")
        connection.executescript(
            """
            CREATE TABLE movies (id INTEGER PRIMARY KEY, collection_title TEXT);
            CREATE TABLE clips (
                id INTEGER PRIMARY KEY,
                movie_id INTEGER,
                people_count TEXT,
                duration REAL
            );
            INSERT INTO movies VALUES (1, 'Porco Rosso'), (2, 'Slam Dunk'), (3, 'Other');
            INSERT INTO clips VALUES
                (10, 1, 'one', 5.0),
                (11, 2, 'none', 8.0),
                (12, 2, 'group', 4.5),
                (13, 3, 'group', 8.0);
            """
        )
        rows = connection.execute(
            "SELECT clips.id FROM clips LEFT JOIN movies ON movies.id = clips.movie_id WHERE "
            + " AND ".join(where),
            values,
        ).fetchall()

        self.assertEqual(rows, [(10,)])

    def test_rejects_unknown_fields(self) -> None:
        with self.assertRaisesRegex(FilterQueryError, "Unknown field"):
            parse_filter_query("Director = Miyazaki")

    def test_rejects_unclosed_quotes(self) -> None:
        with self.assertRaisesRegex(FilterQueryError, "Unclosed quoted value"):
            parse_filter_query('Movie/Show = "Porco Rosso')

    def test_rejects_numeric_or(self) -> None:
        with self.assertRaisesRegex(FilterQueryError, "one numeric value"):
            parse_filter_query("People = 1 OR 2")


if __name__ == "__main__":
    unittest.main()
