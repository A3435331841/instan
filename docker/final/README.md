# Final CUDA image definitions

`Dockerfile.ort-cu128` is the primary v5 delivery.  It runs the exported
ONNX B224/T224/ODTrack graphs using `onnxruntime-gpu` and preserves the locked
causal geometry routing policy.

`Dockerfile.torch-cu128` is a PyTorch CUDA reference/fallback.  It runs the
upstream SUTrack B224 or T224 ERP three-tile model and is intended for backend
comparison, exporter debugging, and conservative fallback validation.  It is
not claimed to be identical to the multi-expert v5 ORT route.

Do not build directly from the repository root: use the package's
`scripts/build_image.ps1`, which materializes a checksum-verified build
context without adding weights to Git.
