# Release layout and maintenance

GitHub carries the code, note, figures, and derived data in one versioned repository.
Hugging Face carries the same tracked release files as an **analysis dataset** so the
layer-pair and offset CSVs can be inspected without installing the scanner. Neither
destination contains raw model weights, local caches, credentials, or working journals.

The current matching release is v0.1.2. Both destinations use the same `checksums.json` so the
scientific artifacts can be compared byte for byte. Hosting-generated metadata such as
the Hub's `.gitattributes` is outside that checksum manifest.

When updating evidence:

1. Keep original captures in version history. Document the substantive change in
   `CORRECTIONS.md` and update the version/citation when appropriate.
2. Change source revisions explicitly in `models.lock.json`; never silently relock.
3. Re-run capture, regenerate tables/figures, refresh checksums, and run verification.
4. Commit the complete release, tag it, and mirror the same tracked files to the Hub.

An online rerun uses `--out reproduction/data --provenance reproduction/metadata` so
the distributed capture remains available for comparison. Reproduction paths and
tensor caches are ignored by git.

Refresh the checksum manifest with `git add .` followed by
`python scripts/build_checksums.py`, then stage `checksums.json` too. The builder only
hashes tracked/staged files, so local caches and unrelated working artifacts cannot
enter the manifest.

The optional Actions configuration is stored as `docs/verify-workflow.yml`; it is not
installed as an active workflow. See `CORRECTIONS.md` for the v0.1.0 Hub snapshot and
the packaging-only v0.1.1 change.
