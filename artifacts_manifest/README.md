# Artifact manifest

This directory documents binary artifacts that are intentionally kept outside
the normal Git history. The server-exit transfer writes the machine-specific
`transfer_manifest.json` and `SHA256SUMS.csv` under the local archive root.

Each record contains the remote path, local path, byte size, SHA256, artifact
kind, timestamp and validation status. Do not add passwords, access tokens or
private keys to these files.
