"""Bounded-memory trace I/O shared by the independent replay tools."""
import hashlib


def iter_trace_lines(path):
    with path.open("r",encoding="utf-8") as stream:
        yield from stream


def file_sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()
