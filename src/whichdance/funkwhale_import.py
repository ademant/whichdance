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
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urljoin

import requests
import torch
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry

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

AUDIO_EXTENSIONS = {".ogg", ".mp3", ".flac", ".opus", ".wav", ".m4a", ".aac", ".wma"}


_LEADING_TRACK_NUMBER = re.compile(r"^\s*\d{1,3}[\s.\-_)]+")


def _normalize(s: str) -> str:
    """Fold to lowercase ascii alnum-only, for fuzzy filename matching that
    tolerates accents/punctuation/spacing differences between the API's
    track title and however Funkwhale (or the original uploader) named the
    file on disk."""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _strip_track_number(stem: str) -> str:
    """Drop a leading track-number prefix, e.g. "05 - Title" -> "Title"."""
    return _LEADING_TRACK_NUMBER.sub("", stem)


class MediaIndex:
    """Maps a normalized track title to local file path(s), built by
    walking a local directory (e.g. an sshfs mount of Funkwhale's
    media/music folder laid out as artist/album/track). Lets the importer
    read audio straight off disk instead of downloading over HTTP.
    """

    def __init__(self, root: Path):
        self.root = root
        self._by_title: dict[str, list[Path]] = {}
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                key = _normalize(_strip_track_number(path.stem))
                self._by_title.setdefault(key, []).append(path)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_title.values())

    def find(self, title: str, artist: str | None = None) -> Path | None:
        title_key = _normalize(title)
        candidates = self._by_title.get(title_key)

        if not candidates:
            # Fall back to substring matching (handles subtitle/edition
            # differences the exact key missed), only scanning on a miss
            # since it's O(n) over the index.
            candidates = [
                paths[0]
                for key, paths in self._by_title.items()
                if title_key and (title_key in key or key in title_key)
            ]

        if not candidates:
            return None
        if len(candidates) == 1 or not artist:
            return candidates[0]

        artist_key = _normalize(artist)
        for c in candidates:
            if artist_key in _normalize(str(c.parent)):
                return c
        return candidates[0]  # ambiguous: title matched, artist didn't disambiguate


def _session() -> requests.Session:
    token = os.environ.get("FUNKWHALE_TOKEN")
    if not token:
        raise RuntimeError("Set FUNKWHALE_TOKEN in the environment (see .env.example)")
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"

    # The network here has shown itself to be flaky (connection resets
    # mid-download); retry transient failures automatically rather than
    # aborting the whole import over one bad connection.
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
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


DOWNLOAD_RETRIES = 4


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

    # The adapter-level Retry only covers connect/response-setup failures.
    # A dropped connection *mid-stream* (seen in practice on this network)
    # surfaces while iterating chunks below, which urllib3's Retry does not
    # catch — so retry the whole streamed download explicitly.
    last_exc: Exception | None = None
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            resp = session.get(full_url, stream=True, timeout=60)
            resp.raise_for_status()

            # Prefer the upload's own extension/mimetype fields: the
            # download response's Content-Type often comes back as a
            # generic application/octet-stream regardless of the actual
            # audio format.
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
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            time.sleep(2**attempt)  # 1s, 2s, 4s, 8s
    raise last_exc


def cache_path_for(track_id: int) -> Path:
    digest = hashlib.sha1(f"funkwhale:{track_id}".encode()).hexdigest()[:16]
    return config.PROCESSED_DIR / f"{digest}.pt"


def _track_artist_name(track: dict) -> str | None:
    credits = track.get("artist_credit") or []
    if not credits:
        return None
    return credits[0].get("credit") or (credits[0].get("artist") or {}).get("name")


def import_all(
    out_csv: str,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    keep_audio: bool = False,
    limit_per_playlist: int | None = None,
    media_root: str | None = None,
) -> None:
    base_url = _base_url()
    session = _session()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = config.RAW_DIR / "funkwhale"
    if keep_audio:
        raw_dir.mkdir(parents=True, exist_ok=True)

    media_index = None
    if media_root:
        print(f"indexing local media root {media_root} ...")
        media_index = MediaIndex(Path(media_root))
        print(f"indexed {len(media_index)} audio file(s)")

    playlists = list_playlists(session, base_url, include, exclude)
    print(f"found {len(playlists)} playlist(s): {[p['name'] for p in playlists]}")

    rows = []
    local_hits, downloads, failed = 0, 0, 0
    try:
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
                    try:
                        local_path = (
                            media_index.find(track["title"], _track_artist_name(track))
                            if media_index
                            else None
                        )
                        if local_path is not None:
                            local_hits += 1
                            features = audio_file_to_features(str(local_path))
                            torch.save(features, cache_path)
                        else:
                            downloads += 1
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
                    except Exception as exc:  # noqa: BLE001
                        # One bad track (network hiccup that outlasted the
                        # retries, a corrupt file, ...) shouldn't sink the
                        # whole import. Log it and move on; labels.csv is
                        # still written for everything that did succeed.
                        failed += 1
                        print(f"  FAILED {track.get('title')!r}: {exc}")
                        continue

                rows.append({"filepath": filepath_field, "label": label})
    finally:
        # Written even on an unexpected/unhandled crash, so a mid-run
        # failure doesn't discard the labels for tracks already cached.
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "label"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {out_path}")
        if media_index:
            print(f"read locally: {local_hits}, downloaded: {downloads}")
        if failed:
            print(f"failed (skipped): {failed}")


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
    parser.add_argument(
        "--media-root",
        default=os.environ.get("FUNKWHALE_MEDIA_ROOT"),
        help="local directory (e.g. an sshfs mount of Funkwhale's media/music "
        "folder, laid out as artist/album/track) to read audio from directly "
        "instead of downloading over HTTP. Falls back to download per-track "
        "if a match isn't found. Also settable via FUNKWHALE_MEDIA_ROOT.",
    )
    args = parser.parse_args()
    import_all(
        args.out,
        args.include,
        args.exclude,
        args.keep_audio,
        args.limit_per_playlist,
        args.media_root,
    )


if __name__ == "__main__":
    main()
