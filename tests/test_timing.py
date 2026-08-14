from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from pipeline.timing import timing_event


class TimingEventTests(unittest.TestCase):
    def test_emits_timestamped_json_line(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            timing_event("stage_end", movie_id=7, elapsed_s=1.25)

        line = output.getvalue().strip()
        self.assertTrue(line.startswith("TIMING "))
        payload = json.loads(line.removeprefix("TIMING "))
        self.assertEqual(payload["event"], "stage_end")
        self.assertEqual(payload["movie_id"], 7)
        self.assertEqual(payload["elapsed_s"], 1.25)
        self.assertRegex(payload["ts"], r"^\d{4}-\d{2}-\d{2}T")


if __name__ == "__main__":
    unittest.main()
