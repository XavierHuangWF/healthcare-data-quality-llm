"""A Git clone without intentionally untracked art still renders safely."""

from pathlib import Path

import exploration_banner


def test_missing_user_managed_asset_has_inline_fallback(tmp_path):
    url = exploration_banner._local_mascot_data_url(tmp_path / 'not-in-git.png')
    assert url.startswith('data:image/svg+xml;base64,')


def test_existing_asset_preserves_original_media_type(tmp_path):
    path = tmp_path / 'image.webp'
    path.write_bytes(b'RIFFTESTWEBP')
    url = exploration_banner._local_mascot_data_url(path)
    assert url.startswith('data:image/webp;base64,')
