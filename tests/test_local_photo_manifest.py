from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.build_local_photo_manifest import build_manifest


class LocalPhotoManifestTests(unittest.TestCase):
    def test_records_local_bytes_hash_and_missing_media_without_omission(self):
        catalogue = {
            "area": {"instance_id": "area-1", "name": "Test area"},
            "observations": [
                {"arborphy_id": "obs-ready", "photo_url": "/api/observations/media/obs-ready/photo.jpg"},
                {"arborphy_id": "obs-missing", "photo_url": "/api/observations/media/obs-missing/photo.jpg"},
                {"arborphy_id": "obs-none", "photo_url": None},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            media_root = Path(directory)
            image = media_root / "obs-ready" / "photo.jpg"
            image.parent.mkdir()
            image.write_bytes(b"real image bytes")
            manifest = build_manifest(catalogue, media_root)

        ready = next(item for item in manifest["frames"] if item["observation_id"] == "obs-ready")
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["image_sha256"], f"sha256:{hashlib.sha256(b'real image bytes').hexdigest()}")
        self.assertEqual(manifest["summary"], {"observations": 3, "ready": 1, "non_ready": 2})
        self.assertEqual({item["status"] for item in manifest["frames"]}, {"ready", "missing_media", "no_photo"})


if __name__ == "__main__":
    unittest.main()
