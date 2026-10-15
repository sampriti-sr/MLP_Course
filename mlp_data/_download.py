"""Download, cache and verify helpers. Standard library only."""
from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import urllib.error
import urllib.request
import warnings
import zipfile
from pathlib import Path

_UA = "Mozilla/5.0 (compatible; mlp-course-data/0.1; +https://github.com/sampriti-sr/MLP_Course)"


def _ssl_context() -> ssl.SSLContext:
    """Verified TLS context. Prefers certifi's CA bundle because python.org builds of Python
    on macOS ship without system certificates until the user runs 'Install Certificates.command'.
    Certificate verification is never disabled."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def data_dir() -> Path:
    """Cache directory. Override with the MLP_DATA_DIR environment variable."""
    root = os.environ.get("MLP_DATA_DIR") or (Path.home() / ".cache" / "mlp_course")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, expected: str | None, label: str) -> bool:
    """Warn (never fail) if a file differs from the checksum recorded in data/README.md.

    A mismatch usually means the publisher updated the file, which can change the
    numbers shown in the course pages, so the student deserves to know.
    """
    if expected is None:
        return True
    actual = sha256_of(path)
    if actual != expected:
        warnings.warn(
            f"{label}: checksum differs from the version this course was written against "
            f"(expected {expected[:12]}..., got {actual[:12]}...). The publisher may have "
            "updated the file; outputs in the course pages may not match exactly.",
            stacklevel=3,
        )
        return False
    return True


def fetch(url: str, dest: Path, *, sha256: str | None = None, label: str | None = None) -> Path:
    """Download `url` to `dest` once. Atomic: a partial download never masquerades as a cache hit."""
    label = label or dest.name
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp, open(tmp, "wb") as out:
            total = resp.headers.get("Content-Length")
            size = f" ({int(total) / 1e6:.1f} MB)" if total else ""
            print(f"Downloading {label}{size} ...", flush=True)
            shutil.copyfileobj(resp, out, length=1 << 20)
    except urllib.error.HTTPError as e:
        e.close()
        tmp.unlink(missing_ok=True)
        hint = (
            " The publisher may have removed or renamed this file; see data/README.md."
            if e.code in (403, 404, 410) else ""
        )
        raise RuntimeError(f"Could not download {label}: HTTP {e.code} from {url}.{hint}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        tmp.unlink(missing_ok=True)
        hint = (
            " Python cannot verify HTTPS certificates. Run `pip install -U certifi`, or on macOS "
            "python.org installs run 'Install Certificates.command' in /Applications/Python 3.x/."
            if "CERTIFICATE_VERIFY_FAILED" in str(e) else " Check your connection."
        )
        raise RuntimeError(f"Could not download {label} from {url}: {e}.{hint}") from e
    os.replace(tmp, dest)
    verify(dest, sha256, label)
    return dest


def extract_member(zip_path: Path, member: str, dest: Path) -> Path:
    """Copy exactly one named member out of a zip. Never uses extractall, so a
    hostile archive cannot write outside `dest` (zip-slip)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            raise RuntimeError(f"Expected '{member}' inside {zip_path.name}; found {zf.namelist()}.")
        with zf.open(member) as src, open(tmp, "wb") as out:
            shutil.copyfileobj(src, out, length=1 << 20)
    os.replace(tmp, dest)
    return dest
