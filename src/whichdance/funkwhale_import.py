"""Import tunes from a Funkwhale instance's playlists.

Each playlist becomes a dance label (its name, lowercased and trimmed).
Tracks are streamed from the instance straight into feature extraction;
by default the raw audio is discarded immediately afterwards and only the
much smaller log-mel spectrogram is cached, so a large library doesn't
blow through local disk quota.

Auth: reads FUNKWHALE_URL and FUNKWHALE_TOKEN from the environment (see
.env.example, gitignored). FUNKWHALE_TOKEN is the OAuth access token for
an Application with read access to your library/playlists, sent as a
Bearer token.

Usage:
    export FUNKWHALE_URL=https://music.example.org
    export FUNKWHALE_TOKEN=xxxxx
    python -m whichdance.funkwhale_import --out data/labels.csv

    # only import specific playlists, keep the audio, cap tracks per
    # playlist for a quick test run:
    python -m whichdance.funkwhale_import --include waltz polka \
        --keep-audio --limit-per-playlist 5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urljoin

import requests
import torch
from tqdm import tqdm

from whichdance import config
from whichdance.features import audio_file_to_features

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MIME_TO_EXT = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/opus": ".opus",
    "audio/webm": ".webm",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}


def _session() -> requests.Session:
    token = os.environ.get("FUNKWHALE_TOKEN")
    if not token:
        raise RuntimeError("Set FUNKWHALE_TOKEN in the environment (see .env.example)")
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def _base_url() -> str:
    url = os.environ.get("FUNKWHALE_URL")
    if not url:
        raise RuntimeError("Set FUNKWHALE_URL in the environment (see .env.example)")
    return url.rstrip("/")


def _paginated(session: requests.Session, url: str) -> Iterator[dict]:
    while url:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        yield from data["results"]
        url = data.get("next")


def sanitize_label(name: str) -> str:
    return name.strip().lower()


def list_playlists(
    session: requests.Session,
    base_url: str,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> list[dict]:
    url = f"{base_url}/api/v2/playlists/?scope=me"
    playlists = list(_paginated(session, url))
    playlists = [p for p in playlists if p.get("tracks_count", 0) > 0]
    if include:
        wanted = {n.lower() for n in include}
        playlists = [p for p in playlists if p["name"].lower() in wanted]
    if exclude:
        skip = {n.lower() for n in exclude}
        playlists = [p for p in playlists if p["name"].lower() not in skip]
    return playlists


def list_playlist_tracks(
    session: requests.Session, base_url: str, playlist_uuid: str
) -> Iterator[dict]:
    url = f"{base_url}/api/v2/playlists/{playlist_uuid}/tracks/"
    for entry in _paginated(session, url):
        yield entry["track"]


def download_track(
    session: requests.Session,
    base_url: str,
    track: dict,
    dest_dir: Path,
    basename: str,
) -> Path | None:
    """Download the best available upload for `track` into `dest_dir`.

    Returns the path written, or None if the track has no downloadable
    upload (e.g. it failed to transcode on the server).
    """
    uploads = track.get("uploads") or []
    upload = uploads[0] if uploads else None
    listen_url = upload.get("listen_url") if upload else track.get("listen_url")
    if not listen_url:
        return None

    full_url = urljoin(base_url + "/", listen_url.lstrip("/"))
    resp = session.get(full_url, stream=True, timeout=60)
    resp.raise_for_status()

    # Prefer the upload's own extension/mimetype fields: the download
    # response's Content-Type often comes back as a generic
    # application/octet-stream regardless of the actual audio format.
    ext = None
    if upload and upload.get("extension"):
        ext = f".{upload['extension']}"
    elif upload and upload.get("mimetype"):
        ext = MIME_TO_EXT.get(upload["mimetype"])
    if not ext:
        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
        ext = MIME_TO_EXT.get(content_type, ".mp3")
    dest = dest_dir / f"{basename}{ext}"
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return dest


def cache_path_for(track_id: int) -> Path:
    digest = hashlib.sha1(f"funkwhale:{track_id}".encode()).hexdigest()[:16]
    return config.PROCESSED_DIR / f"{digest}.pt"


def import_all(
    out_csv: str,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    keep_audio: bool = False,
    limit_per_playlist: int | None = None,
) -> None:
    base_url = _base_url()
    session = _session()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = config.RAW_DIR / "funkwhale"
    if keep_audio:
        raw_dir.mkdir(parents=True, exist_ok=True)

    playlists = list_playlists(session, base_url, include, exclude)
    print(f"found {len(playlists)} playlist(s): {[p['name'] for p in playlists]}")

    rows = []
    for playlist in playlists:
        label = sanitize_label(playlist["name"])
        tracks = list(list_playlist_tracks(session, base_url, playlist["uuid"]))
        if limit_per_playlist:
            tracks = tracks[:limit_per_playlist]
        print(f"[{label}] {len(tracks)} track(s)")

        for track in tqdm(tracks, desc=label, leave=False):
            track_id = track["id"]
            cache_path = cache_path_for(track_id)
            filepath_field = f"funkwhale:{track_id}"

            if not cache_path.exists():
                with tempfile.TemporaryDirectory() as tmp_dir:
                    audio_path = download_track(
                        session, base_url, track, Path(tmp_dir), str(track_id)
                    )
                    if audio_path is None:
                        print(f"  skip (no upload): {track.get('title')!r}")
                        continue
                    features = audio_file_to_features(str(audio_path))
                    torch.save(features, cache_path)
                    if keep_audio:
                        audio_path.rename(raw_dir / audio_path.name)

            rows.append({"filepath": filepath_field, "label": label})

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(config.DATA_DIR / "labels.csv"))
    parser.add_argument("--include", nargs="*", help="only import these playlist names")
    parser.add_argument("--exclude", nargs="*", help="skip these playlist names")
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="keep downloaded audio under data/raw/funkwhale/ instead of discarding it",
    )
    parser.add_argument(
        "--limit-per-playlist",
        type=int,
        default=None,
        help="cap tracks per playlist, useful for a quick test run",
    )
    args = parser.parse_args()
    import_all(
        args.out, args.include, args.exclude, args.keep_audio, args.limit_per_playlist
    )


if __name__ == "__main__":
    main()
