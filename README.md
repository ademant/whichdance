# whichdance

Predict which traditional folk dance a tune belongs to (e.g. waltz, polka,
mazurka, reel, jig, schottische, ...) from audio, using a neural network
trained on a labeled set of tunes.

## Idea

Traditional dance tunes are strongly characterized by tempo, meter, and
rhythmic accent pattern. The model consumes a fixed-length log-mel
spectrogram of each tune (or tune excerpt) and predicts a dance-type label
via a small CNN. This is a solid baseline; swap in a CRNN or pretrained
audio backbone later if accuracy plateaus.

## Project layout

```
data/
  raw/            # your original audio files, organized however you like
  processed/      # cached extracted features (.pt), generated
  splits/         # train.csv / val.csv / test.csv (filepath,label)
  labels.csv      # template: filepath,label for the whole dataset
src/whichdance/
  config.py       # central hyperparameters / paths
  features.py     # audio -> log-mel spectrogram
  dataset.py      # PyTorch Dataset + label encoding
  model.py        # CNN classifier
  train.py        # training loop, CLI entrypoint
  predict.py      # run inference on a single audio file
  split.py        # build train/val/test splits from labels.csv
tests/
  test_features.py
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Add your data

### Option A: manually

Put audio files under `data/raw/` (any layout) and fill in `data/labels.csv`:

```csv
filepath,label
raw/waltz/tune001.mp3,waltz
raw/polka/tune002.mp3,polka
...
```

Labels are free-text dance names; they get encoded automatically. Keep the
label vocabulary consistent (e.g. always "mazurka", not sometimes
"Mazurka").

### Option B: import from Funkwhale

If your tunes live on a Funkwhale instance, sorted into playlists named
after the dance, `funkwhale_import.py` pulls them directly: each playlist
becomes a label, tracks are streamed and turned into cached feature
tensors, and (by default) the raw audio is discarded right after so a
large library doesn't blow through disk quota.

```bash
cp .env.example .env   # fill in FUNKWHALE_URL and FUNKWHALE_TOKEN
python -m whichdance.funkwhale_import --out data/labels.csv
```

`FUNKWHALE_TOKEN` is the OAuth access token for a Funkwhale Application
(Settings > Applications) with read access to your library/playlists.

Useful flags: `--include waltz polka` / `--exclude misc` to restrict which
playlists are imported, `--limit-per-playlist 5` for a quick test run
first, `--keep-audio` to retain the downloaded files under
`data/raw/funkwhale/` instead of discarding them.

**Faster: read audio directly instead of downloading it.** If you have
filesystem access to the Funkwhale server's media directory (e.g. via
`sshfs user@host:/var/www/funkwhale/data/media/music /mnt/funkwhale-media`),
pass `--media-root /mnt/funkwhale-media` (or set `FUNKWHALE_MEDIA_ROOT` in
`.env`). The importer builds a normalized filename index of that directory
(tolerating accents, punctuation, and track-number prefixes like `05 - `)
and reads matched tracks straight off disk — skipping the HTTP download
entirely. Any track it can't match locally falls back to the normal
download automatically, so this is safe to use even if the mount doesn't
cover everything.

This writes `data/labels.csv` directly — skip to step 2 (split).

## 2. Split

```bash
python -m whichdance.split --labels data/labels.csv --out data/splits
```

Labels with fewer than 4 examples can't be divided into train/val/test
without a stratified split failing outright, so they're put entirely into
`train.csv` (a warning is printed listing them). They'll train but won't be
evaluated on, and the model won't be validated against them either — expect
this for any small "misc" playlists.

## 3. Train

```bash
python -m whichdance.train --splits data/splits --epochs 30
```

**Memory:** the default `--batch-size 32` needs roughly **2GB of free RAM**
and will get OOM-killed on anything with less — measured directly on a
constrained VPS (3.6GB total RAM), where the default batch size crashed
immediately while `--batch-size 4` ran stably at ~800MB RSS. The model
itself is tiny (~1MB); the memory cost comes from activations of the CNN's
early conv layers over the full `128 x ~1292` mel spectrogram, and it
scales roughly linearly with batch size. If training gets killed with no
error message (or you see `Killed` / exit code 137), drop the batch size:

```bash
python -m whichdance.train --splits data/splits --epochs 30 --batch-size 8
```

Smaller batches are slower per epoch on CPU but are the only way to avoid
OOM on a small machine.

Checkpoints and the label encoder are written to `checkpoints/`:

```
checkpoints/
  best.pt              # weights from the epoch with the highest val accuracy
  last.pt              # weights from the final epoch
  label_encoder.json   # index <-> dance-label mapping, needed to interpret predictions
```

**`checkpoints/` is gitignored — trained weights are not part of this repo.**
They're a local, regenerable artifact: re-run steps 1-3 above (data import →
split → train) to reproduce them. If you retrain on a machine other than
where you'll run predictions/the web app, copy the whole `checkpoints/`
directory over (or point `--checkpoint` at wherever you keep it) — both
`best.pt`/`last.pt` and `label_encoder.json` are required together.

## 4. Predict

```bash
python -m whichdance.predict --audio path/to/tune.mp3 --checkpoint checkpoints/best.pt
```

Prints a JSON result to stdout:

```json
{
  "filename": "tune.mp3",
  "fingerprint": "AQAAE0mUaEkSZSoAAAAAAAAA...",
  "guessed_dances": [
    {"label": "mazurka", "probability": 0.71},
    {"label": "polska", "probability": 0.19}
  ],
  "bpm": 128.4,
  "duration_seconds": 187.0
}
```

`fingerprint` is a Chromaprint acoustic fingerprint (via `pyacoustid` +
the `fpcalc` binary — install `chromaprint`/`libchromaprint-tools` if
it's missing) identifying this exact recording, independent of file
format/bitrate; `null` if `fpcalc` isn't available. The FastAPI service
(below) returns this same JSON shape from `POST /predict`.

## 5. Web app

Once you have a trained checkpoint, `src/whichdance/app.py` exposes it over
HTTP:

```bash
uvicorn whichdance.app:app --host 127.0.0.1 --port 8000
```

- `GET /health` — reports whether a model is loaded
- `POST /predict` (multipart `file=`) — returns the same JSON shape as the
  CLI above (filename, fingerprint, guessed dances, BPM, duration)
- `GET /` — serves `static/index.html`, a bare-bones test page (upload a
  file, see the result) for exercising the API directly without WordPress

This service is meant to run internally (e.g. `127.0.0.1`, not exposed to
the internet) — the WordPress plugin proxies to it server-side.

### WordPress plugin

`wordpress-plugin/whichdance-wp/` is a thin client: no ML runs in PHP. It
registers a `[whichdance]` shortcode (upload widget) and a WP REST route
(`/wp-json/whichdance/v1/predict`) that forwards uploads to the FastAPI
service via `wp_remote_post()` — keeps the browser same-origin (no CORS)
and the inference service off the public internet.

To install: copy `wordpress-plugin/whichdance-wp/` into your WordPress
`wp-content/plugins/` directory, activate it, then set the inference
service URL under **Settings > whichdance**.

## Notes / next steps

- Start with whole-tune audio; if tunes vary a lot in length, the dataset
  clips/pads to `config.CLIP_SECONDS` (default 30s) taken from the middle.
- If you have thousands of tunes, this baseline (log-mel + small CNN)
  should get you a reasonable first accuracy number quickly. Tempo/beat
  tracking features (e.g. via `librosa.beat`) are an obvious enrichment
  once the pipeline works end to end, since meter/tempo is highly
  discriminative for dance type.
- Class imbalance is likely (some dances have far more tunes than others);
  `train.py` uses a weighted loss by default.
