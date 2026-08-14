from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EmbeddingProfile:
    id: str
    label: str
    model_name: str
    model_type: str
    input_size: int
    default_embeddings_per_clip: int
    frames_per_embedding: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


BUILTIN_PROFILES = (
    EmbeddingProfile(
        "siglip2-base-224",
        "SigLIP2 Base 224",
        "google/siglip2-base-patch16-224",
        "siglip2",
        224,
        5,
    ),
    EmbeddingProfile(
        "siglip2-base-256",
        "SigLIP2 Base 256",
        "google/siglip2-base-patch16-256",
        "siglip2",
        256,
        8,
    ),
    EmbeddingProfile(
        "languagebind-video-1.5",
        "LanguageBind Video 1.5",
        "LanguageBind/LanguageBind_Video_V1.5_FT",
        "languagebind",
        224,
        1,
        8,
    ),
    EmbeddingProfile(
        "languagebind-video-huge-1.5",
        "LanguageBind Video Huge 1.5",
        "LanguageBind/LanguageBind_Video_Huge_V1.5_FT",
        "languagebind",
        224,
        1,
        8,
    ),
)

PROFILE_BY_ID = {profile.id: profile for profile in BUILTIN_PROFILES}
DEFAULT_PROFILE_ID = "siglip2-base-224"


def get_profile(profile_id: str) -> EmbeddingProfile:
    try:
        return PROFILE_BY_ID[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown embedding profile: {profile_id}") from exc


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def normalize_embeddings_per_clip(value: int | None, profile: EmbeddingProfile) -> int:
    count = profile.default_embeddings_per_clip if value is None else int(value)
    if count < 1 or count > 64:
        raise ValueError("embeddings_per_clip must be between 1 and 64")
    return count
