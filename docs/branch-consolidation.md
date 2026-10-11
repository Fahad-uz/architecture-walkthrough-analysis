# Branch consolidation, 11 October 2026

All local and GitHub branches were combined into `main`, then their merged
branch names were removed at the user's request. Only `main` remains locally
and on GitHub. Every recorded tip below is reachable from `main`.

- [PR #13](https://github.com/Fahad-uz/architecture-walkthrough-analysis/pull/13)
  merged at `bd56cdf59717df56966570a14cea4498ddf6054b`, preserving its five
  individual feature/checkpoint commits. All three checks passed before merging.
- The separate local `master` import was merged at
  `b3f51445da26654ee809d75e95f19c99d6b3997b`, preserving both parents. Its tree
  exactly matched historical upstream commit `4168810`; the merge added history
  while retaining the current project tree.
- All 17 local/remote branch references passed ancestry checks before cleanup.
  Remote deletions used the verified tip SHAs as leases and one atomic push.
  Local deletions used normal merged-branch checks. Default branch remains `main`.

## Preserved branch tips

| Former branch | Preserved tip |
| --- | --- |
| agent/fix-analysis-timeout | `1b2d3ee810a3f7d769fc2a72e4931d6c7033a50e` |
| agent/harden-analysis-sanity | `8274cd2a0e265aa3914a592f138b56efc2138a2c` |
| agent/harden-job-execution | `e7becb5d97564454e4796d7a1622faba041d87a8` |
| agent/honor-material-presets | `20ec975f7d91bd3d225e6e399fbd34e1ec822af9` |
| agent/improve-furniture-silhouettes | `f20f1ba20a935d01b72c4518609ba8f7a9b46a42` |
| agent/planstride-accuracy-review | `bc74552a7377a8c479a0fd3a31ab9806b897a26b` |
| agent/planstride-foundation (remote) | `bbd2b29c5637d45689f55038f5b917fb120eff74` |
| agent/planstride-foundation (local) | `1c7106d17c96c4ccc3d6800750f1624b1205996b` |
| agent/preserve-furniture-semantics | `dd4efa86993cb4b09188c9623757c47d292a01bd` |
| agent/reconcile-balcony-ownership | `f89f1160e3a5a9ebcd93e9c669f9108411b2aa33` |
| agent/stabilize-room-topology | `38bc5f07abaceb3775778639c07a44823826b584` |
| codex/walkthrough-quality-repair | `a0bb34b0c64537bb97c3c0d0d99953f96378716c` |
| master (local) | `16591a3ebbf3a0a7f3f1537bc7a396507055bacc` |

## Dependency repair and additional installed packages

The fresh `main` workflow found the [upstream Sharp advisory](https://github.com/advisories/GHSA-wq5f-xc86-pv6w)
published after the previous successful PR checks. Commit
`610602b4c819f1a435bf43e74a7fa7abac937346` pins Sharp and its existing transitive
override to the patched 0.35.5 release. The lockfile update is limited to Sharp
and its platform binaries; other dependency versions are unchanged.

The following three packages were updated on this Mac. Sizes are sums of
installed regular-file bytes, not compressed network transfer sizes. Licenses
are the published package declarations; bundled third-party notices also apply.

| Package / version | Source | License | Installed bytes | Purpose |
| --- | --- | --- | ---: | --- |
| sharp 0.35.5 | [npm archive](https://registry.npmjs.org/sharp/-/sharp-0.35.5.tgz) | Apache-2.0 | 962,168 | GLB texture conversion |
| @img/sharp-darwin-arm64 0.35.5 | [npm archive](https://registry.npmjs.org/@img/sharp-darwin-arm64/-/sharp-darwin-arm64-0.35.5.tgz) | Apache-2.0 | 292,231 | Native macOS ARM64 binding |
| @img/sharp-libvips-darwin-arm64 1.3.4 | [npm archive](https://registry.npmjs.org/@img/sharp-libvips-darwin-arm64/-/sharp-libvips-darwin-arm64-1.3.4.tgz) | LGPL-3.0-or-later | 18,243,502 | Image-processing runtime |

No new floor plans, models or asset packs were acquired. Other platform variants
are recorded in the lockfile for reproducible installs; they were not installed
on this Mac.

## Validation

- Python and frontend checks passed on the merged `main` code.
- Local GLB runtime dependency audit reports zero vulnerabilities. The dependency
  tree resolves both direct and transitive Sharp usage to 0.35.5.
- SVG to PNG to lossless WebP conversion passes with expected dimensions.
  Runtime versions: Sharp 0.35.5, libvips 8.18.7, librsvg 2.63.2.
- Furnished model optimization succeeds (1.18 MB to 1.13 MB). The result decodes
  successfully, preserving one scene, 274 nodes, 248 meshed nodes, all geometry
  bounds and 35,846 nondegenerate triangles. Draco removes 12 zero-area triangles.
  The smoke output/report remain ignored under `outputs/branch-consolidation/`.
- All three [GitHub checks](https://github.com/Fahad-uz/architecture-walkthrough-analysis/actions/runs/38137172904)
  pass on the dependency-fix commit `610602b`: Python, frontend and GLB tools.

Consolidation is complete; generalized reconstruction and the broader PlanStride
upgrade still have the limitations and next steps recorded in `CHECKPOINT.md`.
