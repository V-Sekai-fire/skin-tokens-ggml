The decoded SkinVAE weights live on 54K surface samples. Upstream gives each vertex the inverse-distance mix of its eight nearest samples, so a vertex can take weight from another finger, the other leg or a touching part. When the rig plays animation, those vertices tear away from their neighbours.

`integrate_robust_binding` replaces that step. It is on by default; `robust_transfer = false` (CLI `--no-robust-transfer`) keeps the upstream result.

1. Weld seam vertices so twins share one weight row.
2. Accept a sample only when it is close, not back-facing, and about as near along the surface as in a straight line (geodesic <= 2x euclidean).
3. Mark matched vertices that disagree with their ring as outliers.
4. Inpaint unmatched and outlier vertices: minimise tr(W^T Q W), Q = -L + L M^-1 L, mollified cotan Laplacian (Abdrashitov et al. 2023; Sharp and Crane 2020).
5. Smooth each row towards its one ring (15 half-steps).
6. Per separate mesh part, drop joints the skeleton reaches only through two joints in a row that carry no weight on that part (a right shoe drops the left foot).

New tools:
- `skintokens-skin-stress RIGGED.glb`: plays 2 minutes at 30 fps, every joint swinging on a random axis, and counts torn vertices (a vertex leaving its one ring by more than 3x its rest offset, floored at 0.5% of the mesh size).
- `skintokens-transfer-replay`: replays the transfer from a trace (`SKINTOKENS_DUMP_TRANSFER`) in milliseconds, without the model.

Torn vertices, raw -> robust, guest rig on Metal, 60 degrees seed 1: vrmgirl 1464 -> 0, sophia 334 -> 0, alicia 34 -> 0, godette 18 -> 0, MarocchinoLP 5 -> 0, six stage models 0 -> 0. All 11 models are 0 at seeds 1-3, 60 and 90 degrees. No joint loses more than half its vertex weight.

Tests: `skintokens-robust-transfer-test` pairs each check with a planted control that fails (legs, fingers, decoder spike, seam twins, shoes, fingertip on a whole-body part, smoothing). The raw path gives positions, faces and weights identical to the previous rigs.
