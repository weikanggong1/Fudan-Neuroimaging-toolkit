"""Verified, published runtime resources in FNIT's fixed GitHub Release.

This module only reads packaged metadata. It does not discover network assets or
accept an arbitrary mirror: a resource must match its frozen bytes and size.
"""

from functools import lru_cache
from importlib.resources import files
import json
import re

RELEASE_BASE = (
    "https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/"
    "releases/download/assets-v1/"
)


@lru_cache(maxsize=1)
def _catalog():
    document = json.loads(files("fnit").joinpath("_release_asset_catalog.json").read_text())
    if (document.get("schema_version") != 1
            or document.get("release") != "assets-v1"
            or document.get("repository") != "weikanggong1/Fudan-Neuroimaging-toolkit"):
        raise ValueError("Invalid FNIT Release catalog identity")
    records = {}
    for item in document["assets"]:
        if item.get("status") != "published":
            continue
        name, size, digest = item["name"], item["size"], item["sha256"]
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.+-]+", name)
                or type(size) is not int or size <= 0
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Invalid FNIT Release asset record")
        key = (digest, size)
        if key not in records:
            records[key] = dict(item)
    return records


def release_asset_metadata(sha256, size=None):
    """Return a copy of a published record matching exact SHA-256 and byte count."""
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        return None
    if size is not None:
        if type(size) is not int or size <= 0:
            return None
        record = _catalog().get((sha256, size))
    else:
        matches = [record for (digest, _), record in _catalog().items() if digest == sha256]
        record = matches[0] if len(matches) == 1 else None
    return dict(record) if record is not None else None


def release_url_for(sha256, size=None):
    """Return the fixed Release URL only for a catalogued, verified resource."""
    record = release_asset_metadata(sha256, size)
    return RELEASE_BASE + record["name"] if record else None


def published_release_assets(group=None):
    """Return copies of the verified published entries, optionally by group."""
    return tuple(dict(record) for record in _catalog().values()
                 if group is None or record.get("group") == group)
