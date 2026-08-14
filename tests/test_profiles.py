from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from pipeline import db
from pipeline.profiles import (
    BUILTIN_PROFILES,
    get_profile,
    normalize_embeddings_per_clip,
    adaptive_embeddings_per_clip,
)


class EmbeddingProfileTests(unittest.TestCase):
    def test_adaptive_vector_count_scales_and_clamps(self) -> None:
        self.assertEqual(adaptive_embeddings_per_clip(2.0), 3)
        self.assertEqual(adaptive_embeddings_per_clip(10.0), 7)
        self.assertEqual(adaptive_embeddings_per_clip(30.0), 16)

    def test_required_models_are_registered(self) -> None:
        models = {profile.model_name for profile in BUILTIN_PROFILES}
        self.assertIn("google/siglip2-base-patch16-224", models)
        self.assertIn("google/siglip2-base-patch16-256", models)
        self.assertIn("LanguageBind/LanguageBind_Video_V1.5_FT", models)
        self.assertIn("LanguageBind/LanguageBind_Video_Huge_V1.5_FT", models)

    def test_languagebind_uses_fixed_frames_per_vector(self) -> None:
        profile = get_profile("languagebind-video-huge-1.5")
        self.assertEqual(profile.model_type, "languagebind")
        self.assertEqual(profile.frames_per_embedding, 8)

    def test_vector_count_is_bounded(self) -> None:
        profile = get_profile("siglip2-base-224")
        self.assertEqual(normalize_embeddings_per_clip(None, profile), 5)
        with self.assertRaisesRegex(ValueError, "between 1 and 64"):
            normalize_embeddings_per_clip(65, profile)

    def test_profile_completeness_is_scoped_to_movie(self) -> None:
        original_path = db.DB_PATH
        with tempfile.TemporaryDirectory() as tmp:
            db.DB_PATH = Path(tmp) / "test.sqlite3"
            try:
                db.init_db()
                movie_id = db.create_movie(
                    original_name="Movie",
                    filename="movie.mp4",
                    path=Path(tmp) / "movie.mp4",
                    checksum="abc",
                    duration=10,
                    fps=25,
                    width=1920,
                    height=1080,
                )
                first = db.upsert_clip(
                    movie_id,
                    1,
                    start_frame=0,
                    end_frame=100,
                    start_time=0,
                    end_time=4,
                    duration=4,
                    status="indexed",
                )
                db.upsert_clip(
                    movie_id,
                    2,
                    start_frame=101,
                    end_frame=200,
                    start_time=4,
                    end_time=8,
                    duration=4,
                    status="indexed",
                )
                db.upsert_clip_embedding(
                    first,
                    "siglip2-base-224",
                    artifact_path="one.npz",
                    frame_count=5,
                    dimension=768,
                    status="complete",
                )
                profiles = {item["profile_id"]: item for item in db.list_embedding_profiles(movie_id)}
                self.assertEqual(profiles["siglip2-base-224"]["complete_count"], 1)
                self.assertEqual(profiles["siglip2-base-224"]["target_count"], 2)
                self.assertEqual(profiles["siglip2-base-224"]["missing_count"], 1)
            finally:
                db.DB_PATH = original_path


if __name__ == "__main__":
    unittest.main()
