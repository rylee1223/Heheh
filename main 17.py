from __future__ import annotations

import logging
import sys
from typing import Dict, Optional, Tuple
import re
from urllib.parse import urlparse
import os
import time
import asyncio
import datetime
from pathlib import Path
from collections import defaultdict
import uuid
from dataclasses import dataclass
import signal

import yt_dlp
from dotenv import load_dotenv
from telegram import (
    Bot,
    Update,
    Message,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    BotCommand,
)
from telegram.error import TelegramError
from telegram.ext import (
    ContextTypes,
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

# ============================================================
# utils/logger.py
# ============================================================

"""Logging configuration for the TikTok bot."""

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with a consistent format.

    Args:
        name: Logger name (usually __name__).
        level: Optional log level override. Defaults to INFO.

    Returns:
        Configured logger instance.
    """
    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Mirror all logs to logs/bot.log so the web dashboard's live
        # terminal can tail them. Failure to create the file is non-fatal.
        try:
            log_dir = Path(__file__).resolve().parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "bot.log", encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            pass

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)

    return logger

# Root application logger
log = setup_logger("tiktok_bot")

# ============================================================
# utils/validator.py
# ============================================================

"""URL validation utilities for the TikTok bot."""

# Accepted TikTok hostnames
_TIKTOK_HOSTS = {
    "tiktok.com",
    "www.tiktok.com",
    "m.tiktok.com",
    "vm.tiktok.com",
    "vt.tiktok.com",
}

# Minimal pattern: tiktok.com/â¦ or vm.tiktok.com/â¦ (short links)
_TIKTOK_RE = re.compile(
    r"https?://"
    r"(?:(?:www\.|m\.|vm\.|vt\.)?tiktok\.com)"
    r"(?:/[^\s]*)?",
    re.IGNORECASE,
)

def is_valid_tiktok_url(url: str) -> bool:
    """
    Return True if *url* is a syntactically valid TikTok URL.

    Accepts both full video URLs and short links (vm.tiktok.com/â¦).
    Does NOT make a network request.
    """
    url = url.strip()
    if not url:
        return False

    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    bare = parsed.netloc.lower()
    return bare in _TIKTOK_HOSTS

def extract_url(text: str) -> str | None:
    """
    Extract the first TikTok URL found in *text*.

    Returns the URL string or None if none is found.
    """
    match = _TIKTOK_RE.search(text)
    return match.group(0) if match else None

# ============================================================
# utils/helpers.py
# ============================================================

"""Miscellaneous helper utilities."""

log = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting  (concurrency-safe via per-user asyncio locks)
# ---------------------------------------------------------------------------

# Per-user state: (last_monotonic_ts, count_in_window, window_start_wall_time)
# window_start_wall_time is a time.time() float recorded when the window opened,
# used to compute the human-readable reset time.
_user_state: Dict[int, Tuple[float, int, float]] = defaultdict(lambda: (0.0, 0, 0.0))
_user_locks: Dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

COOLDOWN_SECONDS: int = 20
WINDOW_SECONDS: int = 3600
MAX_REQUESTS_PER_WINDOW: int = 15

# ---------------------------------------------------------------------------
# Usage tracking â lets /status report "active users in the past hour" and
# a running total of completed downloads. This is separate from the rate
# limiter above: it records *every* interaction (even ones that get rate
# limited or fail validation), so it reflects real bot usage rather than
# only successful downloads.
# ---------------------------------------------------------------------------
_activity_log: Dict[int, float] = {}   # user_id -> last-seen monotonic ts
_total_downloads: int = 0              # lifetime count of successfully sent videos/audio

def record_user_activity(user_id: int) -> None:
    """Mark *user_id* as active right now. Call on every incoming update."""
    _activity_log[user_id] = time.monotonic()

    # Periodically prune old entries so this dict doesn't grow forever.
    if len(_activity_log) > 5000:
        cutoff = time.monotonic() - WINDOW_SECONDS
        stale = [uid for uid, ts in _activity_log.items() if ts < cutoff]
        for uid in stale:
            _activity_log.pop(uid, None)

def get_active_user_count(window_seconds: int = WINDOW_SECONDS) -> int:
    """Return the number of distinct users seen within *window_seconds*."""
    cutoff = time.monotonic() - window_seconds
    return sum(1 for ts in _activity_log.values() if ts >= cutoff)

def record_download_completed() -> None:
    """Increment the lifetime successful-download counter."""
    global _total_downloads
    _total_downloads += 1

def get_total_downloads() -> int:
    return _total_downloads

async def check_and_record_request(user_id: int) -> Tuple[bool, str]:
    """
    Atomically check rate limits and, if allowed, record the request.

    Returns (limited, reason_message). When limited=False the request was
    accepted and state has already been updated.
    """
    async with _user_locks[user_id]:
        now_mono = time.monotonic()
        now_wall = time.time()
        last_mono, count, window_start = _user_state[user_id]

        # Reset window if it has expired
        window_expired = last_mono > 0 and (now_mono - last_mono) > WINDOW_SECONDS
        if window_expired or last_mono == 0:
            count = 0
            window_start = now_wall  # start a fresh window

        # Enforce cooldown
        if last_mono > 0 and (now_mono - last_mono) < COOLDOWN_SECONDS:
            remaining_secs = int(COOLDOWN_SECONDS - (now_mono - last_mono)) + 1
            return True, f"â³ Please wait {remaining_secs} seconds before sending another link."

        # Enforce hourly cap
        if count >= MAX_REQUESTS_PER_WINDOW:
            reset_str = _reset_time_str(window_start)
            return True, (
                f"ð« You've used all {MAX_REQUESTS_PER_WINDOW} downloads for this hour.\n"
                f"â° Resets at {reset_str}"
            )

        # Allowed â record
        _user_state[user_id] = (now_mono, count + 1, window_start)
        return False, ""

def get_download_status(user_id: int) -> Tuple[int, Optional[str]]:
    """
    Return (downloads_remaining, reset_time_string) for *user_id*.

    reset_time_string is None if no downloads have been made yet.
    """
    last_mono, count, window_start = _user_state[user_id]
    if last_mono == 0:
        return MAX_REQUESTS_PER_WINDOW, None

    # If window has already expired, the counter would reset on next request
    now_mono = time.monotonic()
    if (now_mono - last_mono) > WINDOW_SECONDS:
        return MAX_REQUESTS_PER_WINDOW, None

    remaining = max(0, MAX_REQUESTS_PER_WINDOW - count)
    reset_str = _reset_time_str(window_start)
    return remaining, reset_str

def _reset_time_str(window_start_wall: float) -> str:
    """Human-readable clock time when the current window expires."""
    reset_ts = window_start_wall + WINDOW_SECONDS
    reset_dt = datetime.datetime.fromtimestamp(reset_ts)
    # Format: 3:45 PM
    return reset_dt.strftime("%-I:%M %p")

# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

DOWNLOADS_DIR = Path(__file__).resolve().parent / "downloads"
TELEGRAM_SIZE_LIMIT = 50 * 1024 * 1024  # 50 MB

def ensure_downloads_dir() -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

def file_size(path) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

async def delete_file(path) -> None:
    def _rm():
        try:
            os.remove(path)
            log.debug("Deleted temp file: %s", path)
        except OSError:
            pass
    await asyncio.to_thread(_rm)

def human_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} TB"

# ============================================================
# services/downloader.py
# ============================================================

"""TikTok video downloader powered by yt-dlp.

Tuned for low CPU usage:
  - Format selectors always prefer pre-muxed H.264 + AAC streams, so yt-dlp
    never has to invoke ffmpeg to merge separate video/audio tracks.
  - Oversized videos are detected from metadata *before* downloading the
    actual file, so we never spend bandwidth/CPU pulling down bytes that
    will just be discarded in favor of a CDN link.
  - Audio extraction prefers sources that are already AAC/M4A so ffmpeg can
    stream-copy instead of transcoding.
Tuned for better quality:
  - No artificial resolution cap on the primary format selector â since we
    never re-encode, higher resolution costs bandwidth, not CPU.
"""

log = setup_logger(__name__)

# Fallback resolution cap, only used if no uncapped H.264/AAC stream exists.
_MAX_HEIGHT = 1080

# Cap how many yt-dlp downloads run at once, regardless of how many users
# are requesting simultaneously. Keeps CPU/bandwidth/disk usage predictable
# under load. Lowered default from 3 -> 2 to reduce peak resource usage.
# Override with MAX_CONCURRENT_DOWNLOADS env var if needed.
_MAX_CONCURRENT_DOWNLOADS = int(os.environ.get("MAX_CONCURRENT_DOWNLOADS", "2"))
_download_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_DOWNLOADS)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

@dataclass
class DownloadResult:
    """Result returned by the downloader."""

    success: bool
    file_path: Optional[Path] = None
    direct_url: Optional[str] = None       # CDN URL for oversized files
    title: Optional[str] = None
    duration: Optional[int] = None         # seconds
    description: Optional[str] = None      # video caption written by creator
    tags: Optional[list[str]] = None       # hashtag strings (without #)
    error: Optional[str] = None            # user-friendly error message, or "OVERSIZE"
    raw_info: Optional[dict] = None        # yt-dlp info dict, reused to avoid re-fetching

def _build_ydl_opts(output_path: str) -> dict:
    """Build yt-dlp options that maximise quality with zero re-encode CPU cost."""
    return {
        # TikTok serves pre-muxed streams with the original AAC audio already embedded.
        # All formats have both video+audio â no merging/transcoding needed, so
        # picking the highest quality option here costs bandwidth, not CPU.
        # We must pick H.264 (vcodec^=h264) explicitly because yt-dlp's "best"
        # often selects H.265/HEVC (bytevc1) which Telegram cannot decode properly,
        # causing distorted audio and playback issues.
        "format": (
            # Best H.264 pre-muxed MP4 with original AAC audio â no height cap,
            # so we always get the highest quality TikTok actually offers.
            "best[ext=mp4][vcodec^=h264][acodec=aac]/"
            # Any H.264 MP4 with audio, no cap
            "best[ext=mp4][vcodec^=h264][acodec!=none]/"
            # Capped fallback, in case only very large/unusual formats exist uncapped
            f"best[ext=mp4][vcodec^=h264][acodec!=none][height<={_MAX_HEIGHT}]/"
            # Any pre-muxed MP4 with audio as fallback
            "best[ext=mp4][acodec!=none]/"
            "best"
        ),
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        "prefer_free_formats": False,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        # Parallel fragment fetching speeds up I/O-bound downloads without
        # adding CPU load â it's just concurrent HTTP requests.
        "concurrent_fragment_downloads": 4,
        "http_headers": {"User-Agent": _USER_AGENT},
        "verbose": False,
    }

def _estimate_filesize(info: dict) -> Optional[int]:
    """
    Best-effort filesize estimate from a yt-dlp info dict, checked BEFORE
    downloading so we can skip pulling oversized videos entirely.
    """
    for key in ("filesize", "filesize_approx"):
        val = info.get(key)
        if val:
            return int(val)

    formats = info.get("formats") or []
    for f in reversed(formats):  # yt-dlp orders formats worst -> best
        for key in ("filesize", "filesize_approx"):
            val = f.get(key)
            if val:
                return int(val)
    return None

async def download_audio(url: str) -> DownloadResult:
    """
    Download just the audio track from a TikTok video as an m4a file.

    Prefers sources already in AAC/M4A so yt-dlp's FFmpegExtractAudio
    postprocessor can stream-copy the audio instead of transcoding it â
    same original quality, a fraction of the CPU cost.
    """
    ensure_downloads_dir()
    file_id = uuid.uuid4().hex
    output_template = str(DOWNLOADS_DIR / f"{file_id}.%(ext)s")

    opts = {
        # Prefer already-AAC/M4A audio first so extraction is a stream copy.
        "format": "bestaudio[acodec^=mp4a]/bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": "0",  # keep original quality; no-op when copying
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "concurrent_fragment_downloads": 4,
        "http_headers": {"User-Agent": _USER_AGENT},
        "verbose": False,
    }

    def _run() -> DownloadResult:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get("title") or "TikTok Audio"
                duration = int(info.get("duration") or 0)
                artist = info.get("uploader") or info.get("creator") or ""

            # FFmpegExtractAudio always outputs .m4a
            out_file = DOWNLOADS_DIR / f"{file_id}.m4a"
            if not out_file.exists():
                # Glob fallback in case ext differs
                matches = list(DOWNLOADS_DIR.glob(f"{file_id}.*"))
                if not matches:
                    return DownloadResult(success=False, error="Audio file not found after download.")
                out_file = matches[0]

            log.info("Audio extracted: '%s' â %s", title, out_file.name)
            return DownloadResult(
                success=True,
                file_path=out_file,
                title=title,
                duration=duration,
                description=artist,  # reuse description field for artist name
            )
        except yt_dlp.utils.DownloadError as exc:
            msg = str(exc)
            log.warning("yt-dlp audio error: %s", msg)
            return DownloadResult(success=False, error=_friendly_error(msg))
        except Exception as exc:  # noqa: BLE001
            log.exception("Unexpected audio download error: %s", exc)
            return DownloadResult(success=False, error="An unexpected error occurred.")

    return await asyncio.to_thread(_run)

async def download_video(url: str) -> DownloadResult:
    """
    Download a TikTok video asynchronously.

    First probes metadata only (cheap â no video bytes transferred) to check
    the estimated file size. If it's already clear the video exceeds
    Telegram's upload limit, we skip the full download entirely and return
    the metadata so the caller can go straight to the CDN-link flow. This
    avoids wasting bandwidth/CPU downloading bytes that would just be
    discarded.

    On success, the local file path is returned in *DownloadResult.file_path*.
    On failure, a friendly error string is populated instead.
    """
    ensure_downloads_dir()
    token = uuid.uuid4().hex
    output_template = str(DOWNLOADS_DIR / f"{token}.%(ext)s")

    def _run() -> DownloadResult:
        opts = _build_ydl_opts(output_template)
        downloaded: Optional[Path] = None

        # ââ Cheap metadata-only probe âââââââââââââââââââââââââââââââââââ
        try:
            with yt_dlp.YoutubeDL({**opts, "skip_download": True}) as probe:
                probe_info = probe.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError:
            probe_info = None
        except Exception:
            probe_info = None

        if probe_info is not None:
            est_size = _estimate_filesize(probe_info)
            if est_size and est_size > TELEGRAM_SIZE_LIMIT:
                log.info(
                    "Skipping full download â estimated size %s exceeds Telegram limit.",
                    human_size(est_size),
                )
                return DownloadResult(
                    success=False,
                    error="OVERSIZE",
                    title=probe_info.get("title", "TikTok Video"),
                    duration=probe_info.get("duration", 0) or 0,
                    description=_extract_description(probe_info),
                    tags=_extract_tags(probe_info),
                    raw_info=probe_info,
                )

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)

                if info is None:
                    return DownloadResult(
                        success=False,
                        error="Could not retrieve video information. The video may be private or deleted.",
                    )

                title: str = info.get("title", "TikTok Video")
                duration: int = info.get("duration", 0) or 0
                description: str = _extract_description(info)
                tags: list[str] = _extract_tags(info)

                # Find the actual output file (extension may differ)
                for p in DOWNLOADS_DIR.iterdir():
                    if p.stem == token:
                        downloaded = p
                        break

                if downloaded is None or not downloaded.exists():
                    return DownloadResult(
                        success=False,
                        error="Download finished but the file could not be located.",
                    )

                log.info("Downloaded '%s' â %s (%d s)", title, downloaded.name, duration)
                return DownloadResult(
                    success=True,
                    file_path=downloaded,
                    title=title,
                    duration=duration,
                    description=description,
                    tags=tags,
                )

        except yt_dlp.utils.DownloadError as exc:
            msg = str(exc).lower()
            log.warning("yt-dlp DownloadError for %s: %s", url, exc)

            if "private" in msg:
                friendly = "This video is private and cannot be downloaded."
            elif "deleted" in msg or "no longer available" in msg:
                friendly = "This video has been deleted or is no longer available."
            elif "age" in msg:
                friendly = "This video is age-restricted and cannot be downloaded."
            elif "copyright" in msg:
                friendly = "This video is unavailable due to copyright restrictions."
            elif "timeout" in msg or "timed out" in msg:
                friendly = "The download timed out. Please try again."
            else:
                friendly = "Failed to download the video. It may be private, deleted, or unsupported."

            return DownloadResult(success=False, error=friendly)

        except Exception as exc:
            log.exception("Unexpected error downloading %s", url)
            return DownloadResult(
                success=False,
                error="An unexpected error occurred while downloading the video.",
            )

        finally:
            # Clean up any partial/intermediate yt-dlp artefacts (e.g. .part,
            # .ytdl, unmerged streams) that share the same token stem but are
            # NOT the successfully-located output file.
            try:
                for p in DOWNLOADS_DIR.iterdir():
                    if p.stem.startswith(token) and p != downloaded:
                        try:
                            p.unlink(missing_ok=True)
                            log.debug("Cleaned up intermediate file: %s", p.name)
                        except OSError:
                            pass
            except OSError:
                pass

    return await asyncio.to_thread(_run)

# Safety margin below Telegram's 50 MB hard cap so the upload never bounces.
_CAPPED_TARGET = TELEGRAM_SIZE_LIMIT - 2 * 1024 * 1024  # 48 MB

async def download_video_capped(url: str, info: Optional[dict] = None) -> DownloadResult:
    """
    Download the best-quality version of the video that still fits under
    Telegram's upload limit.

    This is the fallback for oversized videos. Raw TikTok CDN links are
    signed for the *server's* IP/User-Agent, so sending them to users
    results in "Access Denied" — instead we grab a smaller rendition and
    upload it through Telegram directly.
    """
    ensure_downloads_dir()
    token = uuid.uuid4().hex
    output_template = str(DOWNLOADS_DIR / f"{token}.%(ext)s")
    size_mb = _CAPPED_TARGET // (1024 * 1024)

    opts = _build_ydl_opts(output_template)
    # Prefer the largest rendition that fits; fall back through progressively
    # lower resolution caps for formats that don't report a filesize.
    opts["format"] = (
        f"best[ext=mp4][vcodec^=h264][acodec!=none][filesize<{size_mb}M]/"
        f"best[ext=mp4][vcodec^=h264][acodec!=none][filesize_approx<{size_mb}M]/"
        f"best[ext=mp4][acodec!=none][filesize<{size_mb}M]/"
        "best[ext=mp4][vcodec^=h264][acodec!=none][height<=720]/"
        "best[ext=mp4][vcodec^=h264][acodec!=none][height<=480]/"
        "best[ext=mp4][acodec!=none][height<=480]/"
        "worst[ext=mp4][acodec!=none]/"
        "worst"
    )

    def _run() -> DownloadResult:
        downloaded: Optional[Path] = None
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                dl_info = ydl.extract_info(url, download=True)

            if dl_info is None:
                return DownloadResult(
                    success=False,
                    error="Could not retrieve video information.",
                )

            for p in DOWNLOADS_DIR.iterdir():
                if p.stem == token:
                    downloaded = p
                    break

            if downloaded is None or not downloaded.exists():
                return DownloadResult(
                    success=False,
                    error="Download finished but the file could not be located.",
                )

            # Verify the fallback actually fits — delete it if it doesn't.
            actual = file_size(downloaded)
            if actual > TELEGRAM_SIZE_LIMIT:
                log.info(
                    "Capped download still too large (%s) — giving up on upload.",
                    human_size(actual),
                )
                try:
                    downloaded.unlink(missing_ok=True)
                except OSError:
                    pass
                return DownloadResult(success=False, error="OVERSIZE")

            meta = info or dl_info
            log.info(
                "Capped download OK: %s (%s)",
                downloaded.name, human_size(actual),
            )
            return DownloadResult(
                success=True,
                file_path=downloaded,
                title=meta.get("title", "TikTok Video"),
                duration=int(meta.get("duration") or 0),
                description=_extract_description(meta),
                tags=_extract_tags(meta),
            )

        except yt_dlp.utils.DownloadError as exc:
            log.warning("yt-dlp capped download error: %s", exc)
            return DownloadResult(success=False, error="Failed to download a smaller version of the video.")
        except Exception:  # noqa: BLE001
            log.exception("Unexpected capped download error for %s", url)
            return DownloadResult(success=False, error="An unexpected error occurred.")
        finally:
            # Clean up partial artefacts that aren't the located output file.
            try:
                for p in DOWNLOADS_DIR.iterdir():
                    if p.stem.startswith(token) and p != downloaded:
                        try:
                            p.unlink(missing_ok=True)
                        except OSError:
                            pass
            except OSError:
                pass

    return await asyncio.to_thread(_run)

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _extract_description(info: dict) -> str:
    """
    Return the creator's caption text with all inline #hashtags removed.

    TikTok often uses the full description (text + hashtags) as the video
    title, so we strip hashtags from BOTH the raw description AND the title
    before comparing â this prevents showing the same sentence twice.
    """
    import re

    def _strip_tags(s: str) -> str:
        s = re.sub(r"#\w+", "", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip(" \t\n,|Â·â¢ââ-")

    raw = info.get("description") or ""
    if not raw:
        return ""

    cleaned = _strip_tags(raw)
    if not cleaned:
        return ""

    # Compare against the title with hashtags also stripped so we don't
    # show identical text twice (title line vs description line).
    title_cleaned = _strip_tags(info.get("title") or "")
    if cleaned.lower() == title_cleaned.lower():
        return ""

    return cleaned  # no truncation â caller decides how to handle length

def _extract_tags(info: dict) -> list[str]:
    """
    Return a deduplicated list of hashtag strings (without the # prefix).

    Sources in priority order:
    1. info["tags"]  â explicit tag list from the extractor
    2. Hashtags parsed inline from info["description"]
    """
    import re

    tags: list[str] = []
    seen: set[str] = set()

    # Source 1: explicit tags
    for t in info.get("tags") or []:
        clean = t.lstrip("#").strip().lower()
        if clean and clean not in seen:
            tags.append(t.lstrip("#").strip())
            seen.add(clean)

    # Source 2: inline #hashtags from description
    desc = info.get("description") or ""
    for m in re.finditer(r"#(\w+)", desc):
        clean = m.group(1).lower()
        if clean not in seen:
            tags.append(m.group(1))
            seen.add(clean)

    return tags  # return all hashtags â caller decides how to display

async def extract_info_only(url: str) -> Optional[dict]:
    """
    Extract video metadata without downloading.

    Returns the raw yt-dlp info dict or None on failure.
    """
    def _run() -> Optional[dict]:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "socket_timeout": 20,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception:
            return None

    return await asyncio.to_thread(_run)

# ============================================================
# services/uploader.py
# ============================================================

"""Handles sending videos to Telegram and large-file fallback logic."""

log = setup_logger(__name__)

# Telegram hard limits
_VIDEO_CAPTION_LIMIT = 1024   # max chars for a video/photo caption
_TEXT_MSG_LIMIT      = 4096   # max chars for a plain text message

async def send_video(
    bot: Bot,
    chat_id: int,
    file_path: Path,
    title: str,
    duration: int,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
    reply_to_message_id: Optional[int] = None,
) -> bool:
    """
    Send a video file to *chat_id*.

    Caption strategy:
    - Short content  â everything fits in the video caption (â¤1024 chars).
    - Long content   â video gets a compact caption (title + duration),
                       then a separate text message with the full
                       description and all hashtags follows immediately.

    Returns True on success, False on failure.
    """
    size = file_size(file_path)
    log.info(
        "Sending video to chat %d | file=%s | size=%s",
        chat_id, file_path.name, human_size(size),
    )

    full_caption = _build_full_caption(title, duration, description, tags)
    short_caption = _build_short_caption(title, duration)
    overflow_text = _build_overflow_text(description, tags)

    use_short = len(full_caption) > _VIDEO_CAPTION_LIMIT

    try:
        with open(file_path, "rb") as video_fh:
            await bot.send_video(
                chat_id=chat_id,
                video=video_fh,
                caption=short_caption if use_short else full_caption,
                parse_mode="MarkdownV2",
                duration=duration or None,
                supports_streaming=True,
                reply_to_message_id=reply_to_message_id,
                read_timeout=120,
                write_timeout=120,
                connect_timeout=30,
            )

        # Send full description + hashtags as a follow-up if they didn't fit
        if use_short and overflow_text:
            for chunk in _split_text(overflow_text, _TEXT_MSG_LIMIT):
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode="MarkdownV2",
                    disable_web_page_preview=True,
                )

        return True

    except TelegramError as exc:
        log.error("Telegram send_video failed for chat %d: %s", chat_id, exc)
        return False

async def handle_large_video(
    bot: Bot,
    chat_id: int,
    url: str,
    title: str,
    duration: int,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
    reply_to_message_id: Optional[int] = None,
    info: Optional[dict] = None,
) -> bool:
    """
    For videos that exceed Telegram's upload limit, extract the best direct
    CDN URL and send it as a clickable link, followed by the full caption text.

    *info* can be a yt-dlp info dict already fetched by the caller (e.g. from
    the metadata probe in download_video), avoiding a redundant network call.

    Returns True if a link was found and sent, False otherwise.
    """
    if info is None:
        log.info("Attempting CDN URL extraction for large video: %s", url)
        info = await extract_info_only(url)
    if not info:
        return False

    direct_url: Optional[str] = None
    formats: list = info.get("formats") or []
    candidates = [
        f for f in formats
        if f.get("url") and f.get("ext") in ("mp4", "webm", "m4v", None)
        and not f.get("url", "").startswith("blob:")
    ]
    candidates.sort(key=lambda f: (f.get("height") or 0), reverse=True)
    if candidates:
        direct_url = candidates[0].get("url")
    elif info.get("url"):
        direct_url = info["url"]

    if not direct_url:
        return False

    link_msg = (
        f"ð¥ *{_escape_md(title)}*\n"
        f"â± {_format_duration(duration)}\n\n"
        f"_Video is \\>50 MB â too large for Telegram\\._ "
        f"[â¬ï¸ Download here]({direct_url})\n"
        f"_Link expires soon â download quickly\\!_"
    )

    try:
        await bot.send_message(
            chat_id=chat_id,
            text=link_msg,
            parse_mode="MarkdownV2",
            reply_to_message_id=reply_to_message_id,
            disable_web_page_preview=False,
        )

        # Send full description + hashtags as a follow-up
        overflow_text = _build_overflow_text(description, tags)
        if overflow_text:
            for chunk in _split_text(overflow_text, _TEXT_MSG_LIMIT):
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode="MarkdownV2",
                    disable_web_page_preview=True,
                )

        return True

    except TelegramError as exc:
        log.error("Failed to send CDN link to chat %d: %s", chat_id, exc)
        return False

def exceeds_telegram_limit(file_path: Path) -> bool:
    """Return True if *file_path* is larger than Telegram's bot upload limit."""
    return file_size(file_path) > TELEGRAM_SIZE_LIMIT

# ---------------------------------------------------------------------------
# Caption builders
# ---------------------------------------------------------------------------

def _build_short_caption(title: str, duration: int) -> str:
    """Compact caption used when full content overflows the video caption limit."""
    return f"*{_escape_md(title)}*"

def _build_full_caption(
    title: str,
    duration: int,
    description: Optional[str],
    tags: Optional[list[str]],
) -> str:
    """
    Caption shown with the video.

    Shows description + hashtags when a description exists.
    Falls back to the title when there is nothing else to show.
    The title line is intentionally omitted when a description is present
    because TikTok titles are usually just the description text + hashtags.
    """
    parts: list[str] = []

    if description:
        parts.append(f"ð {_escape_md(description)}")
    else:
        # No description â use the title as the only line of text
        parts.append(f"*{_escape_md(title)}*")

    hashtag_line = _format_hashtags(tags or [])
    if hashtag_line:
        parts.append(hashtag_line)

    return "\n".join(parts)

def _build_overflow_text(
    description: Optional[str],
    tags: Optional[list[str]],
) -> str:
    """
    Build the follow-up text message that carries the full description
    and all hashtags when they don't fit in the video caption.
    """
    parts: list[str] = []
    if description:
        parts.append(f"ð {_escape_md(description)}")
    hashtag_line = _format_hashtags(tags or [])
    if hashtag_line:
        parts.append(hashtag_line)
    return "\n\n".join(parts)

def _format_hashtags(tags: list[str]) -> str:
    """Return all hashtags as a space-separated MarkdownV2 string."""
    if not tags:
        return ""
    return " ".join(f"\\#{_escape_tag(t)}" for t in tags)

def _split_text(text: str, limit: int) -> list[str]:
    """Split *text* into chunks that each fit within *limit* characters."""
    chunks: list[str] = []
    while len(text) > limit:
        # Try to split on a newline near the boundary
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _escape_tag(tag: str) -> str:
    special = r"\_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special else c for c in tag)

def _escape_md(text: str) -> str:
    special = r"\_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special else c for c in text)

def _format_duration(seconds: int) -> str:
    if not seconds:
        return "unknown"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

# ============================================================
# bot/keyboards.py
# ============================================================

"""Telegram keyboards used by the bot."""

# ---------------------------------------------------------------------------
# Persistent reply keyboard (always visible at the bottom of the chat)
# ---------------------------------------------------------------------------

MENU_DOWNLOAD = "ð¥ Download"
MENU_HELP     = "ð Help"
MENU_ABOUT    = "â¹ï¸ About"
MENU_STATUS   = "ð Status"

MENU_BUTTONS = {
    MENU_DOWNLOAD, MENU_HELP, MENU_ABOUT, MENU_STATUS,
}

def main_menu_keyboard() -> ReplyKeyboardMarkup:
    """
    Persistent bottom keyboard shown to users after /start.
    Stays visible for the entire session.
    """
    return ReplyKeyboardMarkup(
        keyboard=[
            [MENU_DOWNLOAD, MENU_HELP],
            [MENU_ABOUT,    MENU_STATUS],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
        input_field_placeholder="Paste a TikTok link hereâ¦",
    )

# ---------------------------------------------------------------------------
# Inline keyboards (appear inside specific messages)
# ---------------------------------------------------------------------------

def help_inline_keyboard() -> InlineKeyboardMarkup:
    """Inline buttons shown with the /help response."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("ℹ️ About", callback_data="nav:about"),
                InlineKeyboardButton("📊 Status", callback_data="nav:status"),
            ],
            [InlineKeyboardButton("🏠 Home", callback_data="nav:start")],
        ]
    )

def about_inline_keyboard() -> InlineKeyboardMarkup:
    """Inline buttons shown with the /about response."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📖 Help", callback_data="nav:help"),
                InlineKeyboardButton("📊 Status", callback_data="nav:status"),
            ],
            [InlineKeyboardButton("🏠 Home", callback_data="nav:start")],
        ]
    )

def status_inline_keyboard() -> InlineKeyboardMarkup:
    """Inline buttons shown with the /status response."""
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔄 Refresh", callback_data="nav:status")],
            [
                InlineKeyboardButton("📖 Help", callback_data="nav:help"),
                InlineKeyboardButton("ℹ️ About", callback_data="nav:about"),
            ],
        ]
    )

def start_inline_keyboard() -> InlineKeyboardMarkup:
    """Inline buttons shown with the welcome message."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📖 Help", callback_data="nav:help"),
                InlineKeyboardButton("ℹ️ About", callback_data="nav:about"),
                InlineKeyboardButton("📊 Status", callback_data="nav:status"),
            ]
        ]
    )

def video_result_keyboard(audio_key: str, source_url: Optional[str] = None) -> InlineKeyboardMarkup:
    """Buttons shown under each delivered video."""
    rows = [[InlineKeyboardButton("🎵 Get Audio", callback_data=f"audio:{audio_key}")]]
    if source_url:
        rows.append([InlineKeyboardButton("🔗 Open on TikTok", url=source_url)])
    return InlineKeyboardMarkup(rows)

# Backwards-compatible alias (older call sites)
def audio_keyboard(audio_key: str) -> InlineKeyboardMarkup:
    return video_result_keyboard(audio_key)

def audio_keyboard(audio_key: str) -> InlineKeyboardMarkup:
    """Button shown after each video â lets the user grab the audio track."""
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("ðµ Get Audio", callback_data=f"audio:{audio_key}")]]
    )

# ============================================================
# bot/commands.py
# ============================================================

"""Bot command handlers: /start, /help, /about, /menu, /status."""

log = setup_logger(__name__)

# Track bot start time for uptime display
_START_TIME: float = time.monotonic()

_START_TEXT = (
    "ð *Welcome to TikTok Downloader Bot\\!*\n\n"
    "Send me any TikTok video link and I'll download it for you â "
    "*without a watermark* whenever possible\\.\n\n"
    "I'll include the original *caption* and *hashtags* with every video\\.\n\n"
    "ð Just paste the URL and I'll handle the rest\\!\n\n"
    "_Tip: Works with full links and short `vm\\.tiktok\\.com` links too\\._"
)

_HELP_TEXT = (
    "â¹ï¸ *How to use this bot*\n\n"
    "1\\. Copy a TikTok video URL\n"
    "2\\. Paste it here\n"
    "3\\. Wait a few seconds while I download it\n"
    "4\\. Receive your watermark\\-free video with caption \\& hashtags\\!\n\n"
    "*Supported link formats:*\n"
    "â¢ `https://www.tiktok.com/@user/video/123â¦`\n"
    "â¢ `https://vm.tiktok.com/XXXXX/`\n"
    "â¢ `https://vt.tiktok.com/XXXXX/`\n\n"
    "*What you get:*\n"
    "ð¬ Video \\(watermark\\-free, highest quality available\\)\n"
    "ð Original caption\n"
    "\\#ï¸â£ Hashtags\n"
    "â± Duration\n\n"
    "*File size:*\n"
    "â¢ â¤ 50 MB â sent directly as a video\n"
    "â¢ \\> 50 MB â sent in reduced quality that fits Telegram\n\n"
    "*Rate limits:*\n"
    "â¢ 20\\-second cooldown between requests\n"
    "â¢ 15 downloads per hour per user\n\n"
    "*Commands:*\n"
    "/start â Show welcome message\n"
    "/help â Show this help\n"
    "/about â About this bot\n"
    "/status â Bot status & uptime"
)

_ABOUT_TEXT = (
    "ð¤ *TikTok Downloader Bot*\n\n"
    "Downloads TikTok videos without watermarks, complete with the "
    "original caption and hashtags\\.\n\n"
    "Built with:\n"
    "â¢ Python 3\\.12\n"
    "â¢ python\\-telegram\\-bot v21\\+\n"
    "â¢ yt\\-dlp\n\n"
    "_No data stored\\. Temp files deleted immediately after sending\\._"
)

DOWNLOAD_HINT = (
    "ð¥ *Ready to download\\!*\n\n"
    "Just paste a TikTok link and I'll grab the video for you\\.\n\n"
    "Example:\n"
    "`https://www.tiktok.com/@user/video/123456789`\n"
    "or a short link:\n"
    "`https://vm.tiktok.com/XXXXX/`"
)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start â show welcome message and persistent menu keyboard."""
    user = update.effective_user
    record_user_activity(user.id)
    log.info("/start from user %d (%s)", user.id, user.username or "no-username")
    await update.message.reply_text(
        _START_TEXT,
        parse_mode="MarkdownV2",
        reply_markup=main_menu_keyboard(),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help."""
    user = update.effective_user
    record_user_activity(user.id)
    log.info("/help from user %d", user.id)
    await update.message.reply_text(
        _HELP_TEXT,
        parse_mode="MarkdownV2",
        reply_markup=help_inline_keyboard(),
    )

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /about."""
    user = update.effective_user
    record_user_activity(user.id)
    log.info("/about from user %d", user.id)
    await update.message.reply_text(
        _ABOUT_TEXT,
        parse_mode="MarkdownV2",
        reply_markup=about_inline_keyboard(),
    )

async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /menu â re-display the persistent keyboard."""
    record_user_activity(update.effective_user.id)
    await update.message.reply_text(
        "ð *Menu*\n\nUse the buttons below or paste a TikTok link to download\\.",
        parse_mode="MarkdownV2",
        reply_markup=main_menu_keyboard(),
    )

def _build_status_text() -> str:
    """Shared status text used by /status and the inline 'status' callback."""
    elapsed = int(time.monotonic() - _START_TIME)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    uptime = f"{h}h {m}m {s}s"
    active_users = get_active_user_count()
    total_downloads = get_total_downloads()

    return (
        "ð *Bot Status*\n\n"
        f"ð¢ Online\n"
        f"â± Uptime: `{uptime}`\n"
        f"ð Mode: Polling\n"
        f"ð¥ Active users \\(past hour\\): *{active_users}*\n"
        f"ð¥ Total downloads served: *{total_downloads}*\n"
        f"ð« Rate limit: 15 downloads/hour per user\n"
        f"ð¦ Max direct upload: 50 MB\n"
        f"âï¸ Max concurrent downloads: {_MAX_CONCURRENT_DOWNLOADS}"
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status â show bot uptime, usage stats, and basic info."""
    record_user_activity(update.effective_user.id)
    await update.message.reply_text(
        _build_status_text(),
        parse_mode="MarkdownV2",
        reply_markup=status_inline_keyboard(),
    )

# callback_query_handler has moved to bot/handlers.py so it can also
# handle audio:KEY callbacks without creating a circular import.

# ============================================================
# bot/handlers.py
# ============================================================

"""Message handler: processes TikTok URLs, menu button taps, and inline callbacks."""

# NOTE: MENU_BUTTONS, MENU_DOWNLOAD, MENU_HELP, MENU_ABOUT, MENU_STATUS,
# audio_keyboard, and main_menu_keyboard are all defined earlier in this
# same file (bot/keyboards.py section), so no import is needed here.

log = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Audio URL cache â maps short key â {"url": str, "title": str}
# Cleared FIFO-style when it exceeds _CACHE_MAX entries.
# ---------------------------------------------------------------------------
_audio_cache: dict[str, dict] = {}
_cache_keys: list[str] = []          # insertion order for eviction
_CACHE_MAX = 300

def _cache_audio(url: str, title: str) -> str:
    """Store *url* in the cache and return the lookup key."""
    global _audio_cache, _cache_keys
    if len(_cache_keys) >= _CACHE_MAX:
        evict = _cache_keys[:50]
        for k in evict:
            _audio_cache.pop(k, None)
        _cache_keys = _cache_keys[50:]
    key = uuid.uuid4().hex[:12]
    _audio_cache[key] = {"url": url, "title": title}
    _cache_keys.append(key)
    return key

# ---------------------------------------------------------------------------
# Inline callback router (was callback_query_handler in commands.py)
# ---------------------------------------------------------------------------

async def _nav_edit(query, text: str, keyboard: InlineKeyboardMarkup) -> None:
    """
    Edit the current message in place for smooth menu navigation.
    Falls back to sending a new message if the edit fails (e.g. the
    message is too old or the content is identical).
    """
    try:
        await query.message.edit_text(
            text, parse_mode="MarkdownV2", reply_markup=keyboard
        )
    except TelegramError:
        try:
            await query.message.reply_text(
                text, parse_mode="MarkdownV2", reply_markup=keyboard
            )
        except TelegramError:
            pass

async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route all inline keyboard button presses."""
    query = update.callback_query
    data: str = query.data or ""
    record_user_activity(update.effective_user.id)

    # Normalise: support both new "nav:x" and legacy "x" callback data.
    target = data[len("nav:"):] if data.startswith("nav:") else data

    if target == "about":
        await query.answer()
        await _nav_edit(query, _ABOUT_TEXT, about_inline_keyboard())
    elif target == "help":
        await query.answer()
        await _nav_edit(query, _HELP_TEXT, help_inline_keyboard())
    elif target == "status":
        await query.answer("Refreshing status…")
        await _nav_edit(query, _build_status_text(), status_inline_keyboard())
    elif target == "start":
        await query.answer()
        await _nav_edit(query, _START_TEXT, start_inline_keyboard())
    elif data.startswith("audio:"):
        await query.answer()
        audio_key = data[len("audio:"):]
        await _handle_audio_callback(query, audio_key, context)
    else:
        await query.answer()

# ---------------------------------------------------------------------------
# Audio callback
# ---------------------------------------------------------------------------

async def _handle_audio_callback(query, audio_key: str, context) -> None:
    """Download the audio track and send both the file and a direct CDN link."""

    entry = _audio_cache.get(audio_key)
    if not entry:
        await query.message.reply_text("â ï¸ This audio link has expired. Please re-send the TikTok URL.")
        return

    url   = entry["url"]
    title = entry["title"]
    chat_id = query.message.chat_id

    status = await context.bot.send_message(chat_id=chat_id, text="ðµ Extracting audioâ¦")

    # Run audio download and CDN info extraction in parallel
    import asyncio as _asyncio
    audio_task = _asyncio.create_task(download_audio(url))
    info_task  = _asyncio.create_task(extract_info_only(url))
    result, info = await _asyncio.gather(audio_task, info_task, return_exceptions=False)

    audio_path: Optional[Path] = result.file_path if result.success else None

    # ââ Send audio file âââââââââââââââââââââââââââââââââââââââââââââââââââ
    if result.success and audio_path:
        try:
            await status.edit_text("ð¤ Uploading audioâ¦")
            with open(audio_path, "rb") as af:
                await context.bot.send_audio(
                    chat_id=chat_id,
                    audio=af,
                    title=title,
                    performer=result.description or "",
                    duration=result.duration or None,
                    read_timeout=120,
                    write_timeout=120,
                )
            await status.delete()
        except TelegramError as exc:
            log.error("Failed to send audio file to chat %d: %s", chat_id, exc)
            await status.edit_text("â ï¸ Could not send the audio file.")
        finally:
            await delete_file(audio_path)
    else:
        await status.edit_text(f"â {result.error}")

    # ââ Send TikTok music page link âââââââââââââââââââââââââââââââââââââââ
    if info:
        music_url = await _get_tiktok_music_url(info)
        track     = info.get("track") or ""
        artist    = info.get("artist") or info.get("uploader") or ""
        if music_url:
            label = f"*{_escape_md_simple(track)}*" if track else f"*{_escape_md_simple(title)}*"
            by    = f" by {_escape_md_simple(artist)}" if artist else ""
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"ðµ *TikTok Sound*\n"
                        f"{label}{by}\n\n"
                        f"[ð Open sound on TikTok]({music_url})"
                    ),
                    parse_mode="MarkdownV2",
                    disable_web_page_preview=False,
                )
            except TelegramError as exc:
                log.warning("Failed to send TikTok music link: %s", exc)

async def _get_tiktok_music_url(info: dict) -> Optional[str]:
    """
    Fetch the TikTok video page and extract the music/sound page URL.

    yt-dlp doesn't expose music_id, so we request the canonical video page
    and pull it from the embedded JSON ('music':{'id':'...'}).
    """
    import re
    import urllib.request as _urllib

    page_url = info.get("webpage_url")
    if not page_url:
        return None

    def _fetch() -> Optional[str]:
        try:
            req = _urllib.Request(
                page_url,
                headers={
                    "User-Agent": _USER_AGENT,
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            with _urllib.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            ids = re.findall(r'"music":\{[^}]*"id":"(\d+)"', html)
            if not ids:
                return None

            music_id = ids[0]
            track    = info.get("track") or "original-sound"
            slug     = re.sub(r"[^a-z0-9]+", "-", track.lower()).strip("-")
            return f"https://www.tiktok.com/music/{slug}-{music_id}"
        except Exception as exc:
            log.debug("Could not fetch TikTok music URL: %s", exc)
            return None

    import asyncio as _aio
    return await _aio.to_thread(_fetch)

def _escape_md_simple(text: str) -> str:
    special = r"\_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special else c for c in (text or ""))

# ---------------------------------------------------------------------------
# Menu button handler
# ---------------------------------------------------------------------------

async def menu_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route persistent keyboard button taps to the right handler."""
    text = (update.effective_message.text or "").strip()

    if text == MENU_HELP:
        await help_command(update, context)
    elif text == MENU_ABOUT:
        await about_command(update, context)
    elif text == MENU_STATUS:
        await status_command(update, context)
    elif text == MENU_DOWNLOAD:
        record_user_activity(update.effective_user.id)
        await update.effective_message.reply_text(DOWNLOAD_HINT, parse_mode="MarkdownV2")

# ---------------------------------------------------------------------------
# Main URL message handler
# ---------------------------------------------------------------------------

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Entry point for every non-command text message that isn't a menu button.

    Flow:
    1. Validate TikTok URL.
    2. Atomic rate-limit check + record.
    3. Acknowledge.
    4. Download via yt-dlp (metadata probe skips full download if oversized).
    5. Send video (caption + hashtags) or CDN link.
    6. Send "ðµ Get Audio" button.
    7. Send quota reminder.
    8. Clean up temp files.
    """
    message: Message = update.effective_message
    user = update.effective_user
    text: str = message.text or ""

    # ââ 0. Track activity (counts toward "active users this hour") âââââââ
    record_user_activity(user.id)

    # ââ 1. Validate URL âââââââââââââââââââââââââââââââââââââââââââââââââââ
    url = extract_url(text)
    if not url or not is_valid_tiktok_url(url):
        await message.reply_text(
            "â That doesn't look like a valid TikTok URL.\n\n"
            "Send me a link like:\n"
            "`https://www.tiktok.com/@user/video/123â¦`\n"
            "or a short link: `https://vm.tiktok.com/XXXXX/`\n\n"
            "Tap ð¥ *Download* for instructions.",
            parse_mode="Markdown",
        )
        return

    log.info("URL received from user %d: %s", user.id, url)

    # ââ 2. Rate limit âââââââââââââââââââââââââââââââââââââââââââââââââââââ
    limited, reason = await check_and_record_request(user.id)
    if limited:
        await message.reply_text(reason)
        return

    # ââ 3. Acknowledge ââââââââââââââââââââââââââââââââââââââââââââââââââââ
    status_msg: Message = await message.reply_text("â¬ï¸ Downloading your videoâ¦")

    # ââ 4. Download (capped concurrency so a burst of requests can't
    #      saturate CPU/bandwidth/disk all at once) âââââââââââââââââââââââ
    async with _download_semaphore:
        result = await download_video(url)

    title: str = result.title or "TikTok Video"
    duration: int = result.duration or 0
    description: Optional[str] = result.description
    tags: Optional[list[str]] = result.tags
    file_path: Optional[Path] = result.file_path
    sent = False

    # ââ Oversize short-circuit: metadata already told us this won't fit,
    #    so no bytes were downloaded â go straight to the CDN-link flow. ââ
    if not result.success and result.error == "OVERSIZE":
        try:
            await _edit_or_reply(
                status_msg,
                "📦 Video is larger than 50 MB — downloading a smaller version…",
            )
            # Raw CDN links are signed for this server's IP/User-Agent and
            # give users "Access Denied" — so download a smaller rendition
            # that fits under Telegram's limit and upload it directly.
            async with _download_semaphore:
                capped = await download_video_capped(url, info=result.raw_info)

            if capped.success and capped.file_path:
                file_path = capped.file_path
                await _edit_or_reply(status_msg, "📤 Uploading video…")
                sent = await send_video(
                    bot=context.bot,
                    chat_id=message.chat_id,
                    file_path=file_path,
                    title=capped.title or title,
                    duration=capped.duration or duration,
                    description=capped.description or description,
                    tags=capped.tags or tags,
                    reply_to_message_id=message.message_id,
                )

            if sent:
                await status_msg.delete()
            else:
                await _edit_or_reply(
                    status_msg,
                    "⚠️ This video is too large for Telegram (over 50 MB) and no smaller "
                    "version could be downloaded. Please try a shorter video.",
                )
            sent = await handle_large_video(
                bot=context.bot,
                chat_id=message.chat_id,
                url=url,
                title=title,
                duration=duration,
                description=description,
                tags=tags,
                reply_to_message_id=message.message_id,
                info=result.raw_info,
            )
            if sent:
                await status_msg.delete()
            else:
                await _edit_or_reply(
                    status_msg,
                    "â ï¸ The video is too large to send and a direct link could not be generated. "
                    "Please try again or look for a shorter version.",
                )
        except TelegramError as exc:
            log.error("TelegramError for user %d: %s", user.id, exc)
            await _edit_or_reply(
                status_msg,
                "â ï¸ A Telegram error occurred while sending the video. Please try again.",
            )
        finally:
            if sent:
                record_download_completed()
                audio_key = _cache_audio(url, title)
                try:
                    await context.bot.send_message(
                        chat_id=message.chat_id,
                        text="⬇️ What next?",
                        reply_markup=video_result_keyboard(audio_key, source_url=url),
                    )
                except TelegramError:
                    pass
                await _send_quota_reminder(context, message.chat_id, user.id)
            if file_path:
                await delete_file(file_path)
        return

    if not result.success:
        await _edit_or_reply(status_msg, f"â {result.error}")
        return

    try:
        # ââ 5a. Small file ââââââââââââââââââââââââââââââââââââââââââââââââ
        if not exceeds_telegram_limit(file_path):
            await _edit_or_reply(status_msg, "ð¤ Uploading videoâ¦")
            sent = await send_video(
                bot=context.bot,
                chat_id=message.chat_id,
                file_path=file_path,
                title=title,
                duration=duration,
                description=description,
                tags=tags,
                reply_to_message_id=message.message_id,
            )
            if sent:
                await status_msg.delete()
            else:
                await _edit_or_reply(
                    status_msg,
                    "â ï¸ The video was downloaded but could not be sent to Telegram. Please try again.",
                )

        # ââ 5b. Large file (metadata estimate was wrong/missing): CDN link â
        else:
            await _edit_or_reply(
                status_msg,
                "📦 Video is larger than 50 MB — downloading a smaller version…",
            )
            # The full-quality file is oversized; delete it and grab a
            # smaller rendition instead of sending a CDN link (those are
            # IP-locked and give users "Access Denied").
            await delete_file(file_path)
            file_path = None

            async with _download_semaphore:
                capped = await download_video_capped(url)

            if capped.success and capped.file_path:
                file_path = capped.file_path
                await _edit_or_reply(status_msg, "📤 Uploading video…")
                sent = await send_video(
                    bot=context.bot,
                    chat_id=message.chat_id,
                    file_path=file_path,
                    title=capped.title or title,
                    duration=capped.duration or duration,
                    description=capped.description or description,
                    tags=capped.tags or tags,
                    reply_to_message_id=message.message_id,
                )

            if sent:
                await status_msg.delete()
            else:
                await _edit_or_reply(
                    status_msg,
                    "⚠️ This video is too large for Telegram (over 50 MB) and no smaller "
                    "version could be downloaded. Please try a shorter video.",
                )
            sent = await handle_large_video(
                bot=context.bot,
                chat_id=message.chat_id,
                url=url,
                title=title,
                duration=duration,
                description=description,
                tags=tags,
                reply_to_message_id=message.message_id,
            )
            if sent:
                await status_msg.delete()
            else:
                await _edit_or_reply(
                    status_msg,
                    "â ï¸ The video is too large to send and a direct link could not be generated. "
                    "Please try again or look for a shorter version.",
                )

    except TelegramError as exc:
        log.error("TelegramError for user %d: %s", user.id, exc)
        await _edit_or_reply(
            status_msg,
            "â ï¸ A Telegram error occurred while sending the video. Please try again.",
        )

    finally:
        if sent:
            record_download_completed()

            # ââ 6. "Get Audio" button âââââââââââââââââââââââââââââââââââââ
            audio_key = _cache_audio(url, title)
            try:
                await context.bot.send_message(
                    chat_id=message.chat_id,
                    text="⬇️ What next?",
                    reply_markup=video_result_keyboard(audio_key, source_url=url),
                )
            except TelegramError:
                pass

            # ââ 7. Quota reminder âââââââââââââââââââââââââââââââââââââââââ
            await _send_quota_reminder(context, message.chat_id, user.id)

        # ââ 8. Clean up âââââââââââââââââââââââââââââââââââââââââââââââââââ
        if file_path:
            await delete_file(file_path)

async def unknown_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "â Unknown command. Use /help to see what I can do."
    )

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _send_quota_reminder(context, chat_id: int, user_id: int) -> None:
    try:
        remaining, reset_str = get_download_status(user_id)
        if reset_str is None:
            return
        if remaining == 0:
            text = f"ð« That was your last download this hour\\. Resets at *{reset_str}*\\."
        elif remaining == 1:
            text = f"â ï¸ 1 download left this hour Â· Resets at *{reset_str}*"
        else:
            text = f"ð {remaining} of {MAX_REQUESTS_PER_WINDOW} downloads left this hour Â· Resets at *{reset_str}*"
        await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="MarkdownV2")
    except TelegramError:
        pass

async def _edit_or_reply(msg: Message, text: str) -> None:
    try:
        await msg.edit_text(text)
    except TelegramError:
        try:
            await msg.reply_text(text)
        except TelegramError:
            pass

# ============================================================
# app.py
# ============================================================

"""Entry point for the TikTok Downloader Telegram Bot."""

# NOTE: Application, ApplicationBuilder, CommandHandler, MessageHandler,
# CallbackQueryHandler, and filters are imported once at the top of this
# file. start_command, help_command, about_command, menu_command,
# status_command, message_handler, menu_button_handler,
# unknown_command_handler, and callback_query_handler are all defined
# earlier in this same file, so no further imports are needed here.

# Load .env (no-op on Replit where secrets are injected as env vars)
load_dotenv()

_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

def _validate_env() -> None:
    """Abort early if required environment variables are missing."""
    if not _BOT_TOKEN:
        log.critical(
            "TELEGRAM_BOT_TOKEN is not set. "
            "Add it as a Replit Secret or in your .env file."
        )
        sys.exit(1)

async def _set_bot_commands(app: Application) -> None:
    """Register bot commands so they appear in the Telegram command menu."""
    await app.bot.set_my_commands(
        [
            BotCommand("start",  "Welcome message & menu"),
            BotCommand("help",   "How to use this bot"),
            BotCommand("about",  "About this bot"),
            BotCommand("menu",   "Show the menu keyboard"),
            BotCommand("status", "Bot status & uptime"),
        ]
    )
    log.info("Bot commands registered.")

def _build_application() -> Application:
    """Construct and configure the Telegram Application."""
    app = (
        ApplicationBuilder()
        .token(_BOT_TOKEN)
        .concurrent_updates(True)   # handle multiple users simultaneously
        .build()
    )

    # ââ Command handlers âââââââââââââââââââââââââââââââââââââââââââââââââââ
    app.add_handler(CommandHandler("start",  start_command))
    app.add_handler(CommandHandler("help",   help_command))
    app.add_handler(CommandHandler("about",  about_command))
    app.add_handler(CommandHandler("menu",   menu_command))
    app.add_handler(CommandHandler("status", status_command))

    # ââ Inline keyboard callbacks ââââââââââââââââââââââââââââââââââââââââââ
    app.add_handler(CallbackQueryHandler(callback_query_handler))

    # ââ Persistent menu button taps (must be checked BEFORE url handler) ââ
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.Regex(
                "^(" + "|".join(map(lambda b: b.replace(".", r"\."), MENU_BUTTONS)) + ")$"
            ),
            menu_button_handler,
        )
    )

    # ââ TikTok URL messages ââââââââââââââââââââââââââââââââââââââââââââââââ
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler)
    )

    # ââ Unrecognised commands ââââââââââââââââââââââââââââââââââââââââââââââ
    app.add_handler(
        MessageHandler(filters.COMMAND, unknown_command_handler)
    )

    return app

async def main() -> None:
    """Start the bot using long-polling."""
    _validate_env()
    ensure_downloads_dir()

    app = _build_application()

    log.info("Starting TikTok Downloader Bot (polling mode)â¦")

    async with app:
        await _set_bot_commands(app)
        await app.start()
        await app.updater.start_polling(
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True,
        )

        log.info("Bot is running. Press Ctrl+C to stop.")

        stop_event = asyncio.Event()

        def _handle_signal(*_):
            log.info("Shutdown signal received.")
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_event_loop().add_signal_handler(sig, _handle_signal)
            except NotImplementedError:
                pass  # Windows

        await stop_event.wait()

        log.info("Shutting downâ¦")
        await app.updater.stop()
        await app.stop()

if __name__ == "__main__":
    asyncio.run(main())
