# Third-party source provenance

These directories are vendored source snapshots used by the local
reproduction/continuation package.  They are source-only: checkpoints,
datasets and caches remain outside Git.

| Directory | Upstream | Snapshot |
|---|---|---|
| `sutrack/` | [chenxin-dlut/SUTrack](https://github.com/chenxin-dlut/SUTrack) | `d65052d1ba3fcf55010e1fb3665ee6616c139a2c` |
| `lorat/` | [LitingLin/LoRAT](https://github.com/LitingLin/LoRAT) | server-exit source snapshot; upstream commit was not present in the recovered copy |
| `uetrack/` | [kangben258/UETrack](https://github.com/kangben258/UETrack) | `fd13b0eaf16d51536008295f3b27807c69eaad50` |

SUTrack includes its original `LICENSE.txt`.  The recovered LoRAT and UETrack
copies did not contain a license file; retain their upstream notices and
check the upstream repositories before redistribution outside this team.
The source snapshots are included to make imports and historical experiments
reproducible; our final ORT image does not execute them.
