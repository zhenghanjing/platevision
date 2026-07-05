"""Read a subset of entries out of a large remote ZIP file without downloading it whole.

UECFOOD-256's archive is ~4GB, but `download_uecfood256`'s subset mode only
needs a handful of category folders. The server serves it as a static file
with `Accept-Ranges: bytes`, so we can open it as a seekable stream of HTTP
Range requests and let the stdlib `zipfile` module read just the central
directory plus the specific entries we want.

Uses a pooled `requests.Session` (keep-alive) rather than one-off
`urllib.request.urlopen` calls -- extracting ~500 small files means ~500
Range requests, and reusing the TCP connection avoids paying a fresh
handshake for every one of them.
"""

import io
import zipfile

import requests


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable file-like object backed by HTTP Range requests."""

    def __init__(self, url: str, session: requests.Session | None = None) -> None:
        self._url = url
        self._pos = 0
        self._session = session or requests.Session()
        response = self._session.head(url, timeout=30)
        response.raise_for_status()
        self._length = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._length + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        return self._pos

    def readinto(self, buffer) -> int:
        size = len(buffer)
        if size == 0 or self._pos >= self._length:
            return 0
        end = min(self._pos + size, self._length) - 1
        response = self._session.get(
            self._url, headers={"Range": f"bytes={self._pos}-{end}"}, timeout=30
        )
        response.raise_for_status()
        data = response.content
        n = len(data)
        buffer[:n] = data
        self._pos += n
        return n


def open_remote_zip(url: str, buffer_size: int = 4 << 20) -> zipfile.ZipFile:
    """Open `url` as a `zipfile.ZipFile` backed by buffered HTTP Range reads.

    `buffer_size` (default 4MB) batches small zipfile reads into fewer, larger
    Range requests -- latency per request dominates over raw bandwidth here.
    """
    stream = io.BufferedReader(HTTPRangeFile(url), buffer_size=buffer_size)
    return zipfile.ZipFile(stream)
