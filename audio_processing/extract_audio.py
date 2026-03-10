"""
audio_processing/extract_audio.py
──────────────────────────────────
Extracts a mono 16-kHz WAV track from any video file using FFmpeg.

Dependencies: ffmpeg-python, ffmpeg binary on PATH
"""

import sys
import tempfile
import subprocess
import shutil
from pathlib import Path
import soundfile as sf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SAMPLE_RATE


def _resolve_ffmpeg_executable() -> str:
    """Resolve ffmpeg binary from PATH, then fallback to imageio-ffmpeg."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "FFmpeg binary not found. Install ffmpeg on PATH or install imageio-ffmpeg."
        ) from exc


def extract_audio_ffmpeg(video_path: str, output_path: str | None = None) -> str:
    """
    Extract audio from *video_path* and write a mono 16-kHz WAV.

    Parameters
    ----------
    video_path  : Path to the input video (mp4, avi, mov, mkv …)
    output_path : Where to write the WAV. If None, a temp file is created.

    Returns
    -------
    str  path to the output WAV file.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    cmd = [
        _resolve_ffmpeg_executable(),
        "-y",                      # overwrite without asking
        "-i", str(video_path),
        "-vn",                     # no video
        "-acodec", "pcm_s16le",    # 16-bit PCM
        "-ar", str(SAMPLE_RATE),   # 16 kHz
        "-ac", "1",                # mono
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed:\n{result.stderr}"
        )

    return str(output_path)


def get_audio_duration(audio_path: str) -> float:
    """Return duration in seconds using soundfile metadata."""
    try:
        info = sf.info(str(audio_path))
        if info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
        return 0.0
    except Exception:
        return 0.0
