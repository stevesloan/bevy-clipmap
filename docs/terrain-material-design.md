# Terrain Material & Shadowing — Design Doc

Status: **Draft / agreed direction** · Last updated: 2026-07-07

This document records the decisions for adding an Unreal-style multi-material
landscape system (splat-blended textures) and a terrain shadowing pipeline to
`bevy-clipmap`. It captures *what* we're building, *why*, and the *order* to
build it. It is the reference for future sessions and contributors.

---

## 1. Goals

- **Unreal-style landscape multi-material**: blend multiple textures (grass,
  rock, dirt, snow, …) across the terrain from a control/weight map.
- **Two view distances, both beautiful**:
  - **Close range** convincing enough for **VR** (no visible tiling, high texel
    density, stable — no shimmer).
  - **Long range** for beautiful vistas.
- **VR is the primary target.** Flatscreen players get an **optional higher
  visual tier**.
- **Static objects cast shadows onto the terrain** (buildings, rocks, props).
- Stay a reusable, published crate — features should be opt-in.

### Non-goals (for now)
- Dynamic time-of-day sun (deferred — see §5, phase "Later").
- Streamed/virtual *source* data for planet-scale worlds (deferred — see §8).
- Virtual Shadow Maps / hardware ray-traced shadows (evaluated and rejected for
  a VR-first target — see §5).

---

## 2. Chosen architecture — Runtime Virtual Texture (RVT), "Tier 2 / AAA"

We use the **RVT approach**: rather than blending N material layers every frame
for every pixel, **bake the blended result into a texture** and have the main
terrain pass do a single fetch. This **decouples per-pixel shading cost from
material complexity** — the lever that lets one material serve both a tight VR
budget and a lavish flatscreen budget (see §7).

**The world is finite-ish, so the RVT is a static, full-terrain, bake-once
texture — NOT a camera-centered clipmap.** This drops the toroidal complexity
entirely. What gets baked:

- Albedo (sRGB) + **sun-visibility** in alpha (baked shadow — see §4)
- Octahedral world normal + roughness + **dominant material ID** in alpha (§3.5)

The bake runs via top-down orthographic cameras → `RenderTarget::Image`, then
stops. The main `terrain.wgsl` pass collapses from ~14 samples to ~2 fetches.

**Bake lifecycle — sentinel-gated, not frame-counted.** A camera that renders
before its pipeline is compiled and source textures are GPU-resident bakes an
empty (black) RVT → the terrain reads as a chrome mirror (`metallic=1`,
`roughness=0`). A fixed frame-count wait is a machine-dependent guess (and a slow
one — the shadow-march bake is expensive per frame). Instead each bake camera
starts **inactive behind a sentinel**: the same bake material drawn to a 4×4
readback target. The first **non-black readback** proves the pipeline is compiled
and textures are up, so the full-res bake fires for ~2 frames, then everything
deactivates and the sentinel entities despawn. (`init_rvt` / `drive_rvt_bake`.)

**Close-up detail does NOT come from RVT density.** The RVT caches at a fixed
texel density, so sub-cm rock/sand detail is impossible at any affordable
resolution. Close-range fidelity is a *separate* **near-range detail overlay**
(§3.5); RVT density only controls the base material and material-boundary
sharpness.

### 2.1 Future scaling — camera-centered toroidal RVT

Not needed for a finite world; kept as a triggered option. A static RVT costs
`(world_size / texel)² × bytes × targets` of VRAM (scales with area). Switch to a
**camera-centered clipmap with toroidal incremental updates** (reusing the
existing geometry-clipmap toroidal machinery in `update_grids`) only when:

- the world outgrows a static RVT at the target base density — roughly **>10–15 km
  at 1 m/texel** before VRAM gets unreasonable, or
- terrain becomes **streamed / procedural / effectively infinite**.

Even then, toroidal only keeps the *base* dense near the camera — close-up sub-cm
detail is still the detail overlay's job.

---

## 3. Material system — data-driven splat blending

### 3.1 `TerrainLayerSet` asset (data-driven)

Material definition lives in a **hot-reloadable asset**, not in scattered
uniforms, so the RVT bake pass and any forward fallback consume the same data.

```
TerrainLayerSet (Asset)
  layers: Vec<TerrainLayer {
      albedo_index, normal_index, orm_index,   // slices into the arrays
      tiling_scale,
      height_blend_sharpness,
      triplanar: bool,
      slope_rule: Option<SlopeRule>,           // e.g. auto-rock above N degrees
  }>
  albedo_array, normal_array, orm_array: Handle<Image>   // texture_2d_array
  control_maps: Vec<Handle<Image>>                        // RGBA weightmaps
```

### 3.2 Blending rules

- Sample the control map(s) by world-space UV (same UV the shader already
  computes). RGBA per map = 4 weights; add a second map for 8 layers
  (MicroSplat-style).
- **Sort weights, sample only the top ~4 contributing layers** to bound cost.
- **Height/depth blending** (not linear alpha) for crisp, natural transitions.
- **Slope-based auto-rock**: the surface normal is already available in the
  shader — mix in the rock layer where slope exceeds a threshold, so cliffs get
  rock with no authoring.
- **Macro variation map** (the existing `color` texture) multiplied over the
  blended result to add large-scale color variation and break up tiling — a
  standard AAA technique. Strength is adjustable via `macro_strength`.

**Guardrails (baked in):** arrays are `Handle<Image>`, so BC7/BC5 mipmapped KTX2
work unchanged; the tiling sampler uses repeat + anisotropic filtering; albedo is
sampled sRGB, control/height linear; weights are normalized in-shader; the
packing and blend loops extend to 8–16 layers (top-4 per pixel) by adding control
maps.

### 3.3 Close-range VR fidelity

The RVT gives a cheap base everywhere; close-up crispness comes from the detail
overlay (§3.5). The remaining near-range items:

- **Stochastic / hex tiling** to eliminate visible repetition in the *baked base*.
  ~3× samples + `textureSampleGrad`, so it runs **inside the RVT bake** (§2,
  `bake.wgsl`), not per-fragment — amortized to ~zero per-frame. ✅ done.
  - **Gotcha — hex cell size (`triangle_grid`).** Cells must span **~1 texture
    repeat**. Sub-repeat cells randomize the UV every few RVT texels, so under
    the bake's heavy minification the base collapses into per-texel speckle
    ("looks like noise"); multi-repeat cells let the in-cell repetition show.
    ~1 repeat = every cell is a distinct crop, no visible tiling, structure intact.
- **Distance→macro vista blend**: dissolve distant terrain toward the macro color
  map so vistas show art-directed color instead of tiling (cheap: one sample + a
  lerp, driven by camera distance). ✅ done.
- **Triplanar on steep slopes** — **deferred as a stretch goal** (performance
  first). It does *not* amortize into the RVT bake (RVT is XZ-parameterized, so
  cliffs are its inherent weak spot) and would stay a live per-frame cost. If
  revisited, slope-gated biplanar (flat stays 1× planar; only cliffs pay) on
  albedo + normal. Heightfields can't do overhangs anyway, so cliff stretch is a
  bounded artifact.

### 3.4 Precomputed normals ✅

The RVT bakes the reoriented world normal (octahedral), so the main pass samples
it instead of reconstructing from heightmap derivatives per-fragment. Removes the
wasted samples **and the shimmer source VR exaggerates**.

### 3.5 Near-range detail overlay — close-up VR fidelity

The RVT base can't represent sub-cm surface detail (fixed texel density), so
"real rocks and sand up close" comes from **high-frequency detail textures blended
onto the RVT sample, near the camera only, faded with distance** — the standard
AAA detail-mapping / macro-micro technique. Cheap: a couple of extra samples that
only matter on near pixels.

- **v1 — generic detail** ✅ done: a high-frequency detail *normal* + detail
  albedo grain tiled small (~1.5 m), blended onto the RVT normal / albedo in the
  main pass, faded out by camera distance (`detail_near`/`detail_far`). The
  detail normal is reoriented onto the RVT world normal.
- **v2 — per-material detail** ✅ done (the full "real rocks/sand"): per-material
  detail albedo + normal + ORM arrays, selected by the **dominant layer ID baked
  into the RVT's metallic slot** (terrain is never metallic). Rock gets rock
  detail, snow gets snow detail, etc. Real photographic detail textures drop
  straight into the same `Handle<Image>` array slots.

### 3.6 Loading real texture sets ✅

`load_terrain_array(images, paths, srgb)` (public API) decodes one image file per
layer and stacks them into the tiling `2d_array` the layer/detail slots expect.

- **Generates a full mip chain on load.** Runtime-decoded PNGs carry no mips; the
  RVT bake samples the arrays heavily minified, so without mips real textures
  alias into per-texel noise. sRGB layers are averaged in ~linear space.
- All layer files must share dimensions (decode-only, no resample); `srgb` picks
  color vs linear (normal / ORM) space; **ORM = occlusion, roughness, metallic in
  R, G, B** (metallic ~0 for terrain).
- `assets/fetch_textures.py` pulls CC0 sets from Poly Haven, converts to 8-bit,
  and packs ORM. Textures stay out of git (`assets/terrain/` is `.gitignore`d).
- Dev-profile note: `png`/`image`/`fdeflate`/`miniz_oxide` get `opt-level = 3` in
  `Cargo.toml` — debug-mode PNG inflate made example startup ~30 s.

---

## 4. Lighting & shadows — baked baseline (fixed sun)

Shadows are **tiered** (maps onto the quality presets, §7): a cheap **baked
baseline** for all VR, plus an optional **dynamic tier** (§5) for high-power rigs.

Two distinct shadow problems, two tools:
- **Terrain self-shadow** (mountains → valleys): **heightfield ray-march** — march
  the heightmap toward the sun and test occlusion. A shadow map is the *wrong* tool
  here (resolution / peter-panning, and it would need the displaced terrain to
  cast).
- **Objects → terrain** (buildings, rocks, props): **cascaded shadow maps (CSM)** —
  Bevy has it built-in, and the terrain already *receives* it (stock PBR lighting
  calls the shadow fetch).

**Baked baseline (cheapest, VR-ideal).** Because the RVT is static + finite, run
the heightfield ray-march **once, in the RVT bake**, against a **fixed sun**, and
store it in a **sun-visibility channel**. Terrain self-shadow then costs one
texture read per frame — zero per-frame shadow work, perfectly stable (no shimmer).
Static objects bake into the same channel (render them from the sun during the
bake).

**Terrain self-shadow ✅ done** (`bake.wgsl` `sun_visibility`, stored in the RVT
albedo target's alpha; `Clipmap::sun_direction`). Implementation notes:
- The march starts biased along the surface normal and uses **adaptive
  (geometric) step size** — fine near-field so grazing sun-facing slopes don't
  self-shadow (acne), growing steps for reach (`MAX_DIST` 6000 covers low-sun
  long casts). Soft penumbra via a clearance/distance ramp.
- **Applied through a compact lighting fork** (`terrain.wgsl`
  `terrain_apply_lighting`): base layer + directional + ambient + env map only,
  with `sun_vis` multiplied onto **only the direct sun term** so shadowed areas
  keep sky/ambient (not black). No stock hook injects a per-pixel shadow factor,
  hence the small fork (§6.1). Much smaller than the old horizon fork.
- **Orientation gotcha:** the bake camera's `up` is `-Z` so the RVT texel layout
  matches the main pass's `world_xz / world_size + 0.5` sampling. A `+Z` up
  stored every RVT channel (shadow, normal, material ID) rotated 180°.

### 4.1 Horizon map — REMOVED

The FFT horizon map is **removed**. Decision rationale:

- It exists to answer "is the sun occluded by distant terrain, for an
  **arbitrary** sun direction, cheaply." With a **fixed sun**, that per-direction
  data is redundant — a baked shadow texture that includes the terrain contains
  the exact same self-shadowing, for one direction, at a fraction of the storage
  (`W·H` vs the horizon map's `360·W·H·4`) and per-fragment cost (one fetch vs
  the `reconstruct_horizon` sample loop).
- **This is a permanent removal, not delete-now-re-add-later.** The future
  dynamic-sun path (§5) uses **on-the-fly ray-marched heightfield shadows**,
  which re-derive self-shadowing live — an *alternative* to the horizon map, not
  a consumer of it. The horizon map is part of neither phase.

**Removal touches:**
- `src/lib.rs`: `Clipmap.horizon` / `horizon_coeffs` fields, `GridMaterial`
  bindings 104/105/106, and the two material-construction sites passing them.
- `src/terrain.wgsl`: bindings 104–106, `reconstruct_horizon` (130-142), and the
  horizon block at 416-428 — `min(shadow, horizon_shadow)` simplifies to
  `shadow` (later `min(shadow, baked_static_shadow)`).
- `convert/clipmap.py`: the `horizon` subcommand.
- `examples/basic.rs` + `README.md`: horizon asset load and docs.

---

## 5. Lighting & shadows — dynamic tier (optional, high-power VR)

The **optional dynamic tier**, toggled by the quality preset (§7) for
higher-powered VR machines that want a moving sun / day-night. **Not built right
away.** The AAA VR-viable answer for a dynamic sun is a **hybrid**, merged into one
`sun_visibility` scalar:

| Range / caster                | Technique                                             | Status |
| ----------------------------- | ----------------------------------------------------- | ------ |
| Near/mid dynamic objects      | Cascaded Shadow Maps (Bevy, have it) + contact shadows (Bevy 0.19) | available |
| Terrain self-shadow, all dist | **Ray-marched heightfield** (maximum-mipmap of the heightmap, arbitrary sun) | build later |
| Far *object* shadows          | Distance Field (SDF) ray-march                        | build later, optional |

VR optimization: even with a moving sun, most shadow change is slow. Keep the
RVT sun-visibility channel but **refresh it lazily** (every N frames as the sun
creeps) for the slow parts (terrain self-shadow + static-object occlusion at the
current angle); only genuinely dynamic objects use live CSM every frame.

**Rejected for VR-first:**
- **Virtual Shadow Maps (VSM)** — Unreal's flagship dynamic-sun system; ~9ms
  shadow pass on console even after UE5.7 improvements, cache invalidation from
  head motion + stereo, and not present in Bevy. Known AAA reference, wrong tool
  here.
- **Hardware ray-traced / path-traced shadows** — highest quality, but
  perf-hostile for mainstream VR; Bevy's Solari is experimental.

---

## 6. Foundational cleanup (do before piling on features)

### 6.1 Shrink or eliminate the copied lighting shader
`terrain.wgsl:158-650` is a hand-copied fork of Bevy's `apply_pbr_lighting`,
which only exists to inject the horizon term at line 428. It is the single
biggest maintenance liability (re-taxed every Bevy upgrade — already felt in the
0.18→0.19 jump).

- Removing the horizon map (§4.1) removes the fork's original reason. It should
  shrink to a tiny baked-shadow sample inside the directional loop, or be
  eliminated by applying the baked shadowmask another way.
- Whatever remains: mark the divergence with a loud banner, keep it minimal, and
  maintain a `RESYNC.md` re-sync checklist. Pin the exact Bevy minor.

### 6.2 Cargo / version hygiene
`Cargo.toml` declares `bevy = "0.19.0"`, but the README compat table lists
0.18/0.17. Reconcile.

---

## 7. Quality scaling — VR vs flatscreen

Mechanism: a **`TerrainQualityKey`** driving pipeline specialization + shader
defs, generalizing the existing `WireframeKey` / `specialize()` pattern
(`src/lib.rs:463-539`), plus **crate feature flags** (`hextile`, `triplanar`,
`rvt`, `parallax`) so the heavy pieces are opt-in at the library level. That
feature-flag layout *is* the "optional increased visuals" deliverable.

Because RVT decouples per-pixel cost from material complexity, the **material
graph is identical across tiers** — the quality knob is mostly *RVT texel
density + bake frequency + AA + shadow resolution + near-ring effects*, not a
material rewrite.

| Knob                     | VR preset            | Flatscreen "ultra"      |
| ------------------------ | -------------------- | ----------------------- |
| RVT texel density        | moderate             | high                    |
| ring count / levels      | fewer                | more                    |
| hex-tiling               | nearest 1 ring       | several rings out       |
| triplanar                | slopes only / off    | slopes on               |
| detail-texture distance  | short                | long                    |
| parallax / POM           | off                  | on                      |
| shadow map resolution    | moderate             | high                    |
| anti-aliasing            | MSAA (forward)       | TAA/DLSS-class          |

---

## 8. VR correctness & longer-horizon items

- **Center the clipmap on the HMD head, not an eye.** `update_grids` follows a
  single `clipmap.target` entity (`src/lib.rs:420`). In stereo, both eyes share
  one head — point `target` at the head/tracking-space origin so both eyes share
  one clipmap and one RVT.
- **Verify multiview-safety** once stereo is added (via a community OpenXR crate;
  no first-party Bevy XR in 0.19). Per-material uniforms like `translation` are
  view-independent — good.
- **Forward + MSAA for the VR path; keep deferred for flatscreen.** Both
  pipelines already exist (`deferred_output` in `terrain.wgsl`). Treat the choice
  as part of the quality preset.
- **Stable shadows.** Snap CSM texels to world grid; prefer stable techniques —
  shadow shimmer is nausea-inducing in a headset.
- **Later — stream source data.** RVT caches *shading*; source heightmap/color
  are still single resident images. For planet-scale worlds, stream tiled
  heightmap + material IDs feeding the RVT bake. Keep the `TerrainLayerSet` asset
  and bake pass agnostic to whether inputs are resident or streamed.

---

## 9. Build order (agreed)

0. ✅ **Remove the horizon map** and eliminate the `apply_pbr_lighting` fork
   (§4.1, §6.1); reconcile Cargo/README versions. *(done)*
1a. ✅ **Splat blending** — `TerrainLayer` config, albedo texture array, control
   map, height blend, slope auto-rock, macro variation map (§3.1–3.2). *(done)*
1b. ✅ **Per-layer normal + ORM arrays** — tangent-space normals reoriented onto
   the geometry normal, per-layer roughness/metallic/AO (§3.1). *(done)*
2. ✅ **Distance→macro vista blend** — capped blend toward the macro color map at
   range (§3.2–3.3). Per-fragment hex tiling **deferred into the RVT bake**
   (step 4) — too costly for VR forward rendering. *(done)*
3. ✅ **Static RVT bake** (§2, §3.4) — the performance play: bake the splat into
   full-terrain textures (albedo+AO, octahedral normal+roughness+metallic), main
   pass collapses ~14 samples → ~2. Built via camera → `RenderTarget::Image` (no
   render-graph nodes) for maintainability. *(done; verified same FPS as the old
   single-texture terrain despite the full 4-layer material)*
4. ✅ **Near-range detail overlay** (§3.5) — the close-up VR fidelity. v1 (generic
   detail, distance-faded) and v2 (per-material rock/sand detail via a material ID
   baked into the RVT) both **done**. *(procedural stand-in textures; real
   photographic detail textures drop into the same array slots)*
5. ✅ **Baked hex tiling in the RVT bake** (§3.3) — Mikkelsen stochastic hex
   tiling in `bake.wgsl` de-tiles the base; zero runtime cost. *(done; see the
   §3.3 cell-size gotcha)*
6. ✅ **Baked terrain self-shadow** (§4) — heightfield ray-march against a fixed
   sun in the RVT bake, stored in the RVT albedo alpha; a compact lighting fork
   multiplies it into only the direct sun term. *(done)*
   - Also landed alongside: **sentinel-gated bake** (§2, replaced the 60-frame
     wait), **`load_terrain_array` + mip generation** and real CC0 textures
     (§3.6), sun-facing-slope acne fix (adaptive march).
7. **Baked static-object shadows** (§4) — render props from the sun into the same
   sun-visibility channel. *(active next step)*
8. **`TerrainQualityKey` specialization + feature flags** (§7) — gate the above
   into VR / flatscreen presets, incl. the **dynamic shadow tier** toggle.

**Known follow-ups:** RVT edge-stripe smear where the clipmap mesh extends past
heightmap coverage (clamp sun-visibility to 1 / fade outside coverage).

**Deferred / triggered:**
- **(High-power tier)** Dynamic shadows (§5) — live heightfield ray-march (dynamic
  sun) + CSM for objects + contact shadows; toggled by the quality preset.
- **(Stretch)** Triplanar on steep slopes — does not amortize into RVT (§3.3).
- **(Scaling)** Camera-centered toroidal RVT — only if the world outgrows a static
  RVT or goes streamed/procedural (§2.1).

The material + RVT + de-tiling + close-up-detail stack **and** baked terrain
self-shadow (steps 0–6) are done. Baked static-object shadows (step 7) are the
active next work.

---

## 10. Key code anchor points (current)

| Location                                    | Role                                                    |
| ------------------------------------------- | ------------------------------------------------------- |
| `src/bake.wgsl` `splat_terrain` / `hex_sample` | Layer splat + hex de-tiling; writes the RVT channels |
| `src/bake.wgsl` `sun_visibility`            | Adaptive heightfield ray-march (baked terrain shadow)   |
| `src/terrain.wgsl` `terrain_apply_lighting` | Compact PBR fork injecting baked sun-visibility          |
| `src/lib.rs` `load_terrain_array`           | Decode + stack + mip real texture sets (public API)     |
| `src/lib.rs` `init_rvt` / `drive_rvt_bake`  | Sentinel-gated RVT bake lifecycle                       |
| `src/lib.rs` `TerrainLayer` / `TerrainParams` | Layer config + GPU packing                            |
| `src/lib.rs` `WireframeKey` / `specialize()`  | Pattern to extend for `TerrainQualityKey`             |
| `src/lib.rs` `update_grids` / `clipmap.target` | Clipmap follow; center on HMD head for stereo        |

---

## 11. References

- [Runtime Virtual Texturing in Unreal Engine](https://dev.epicgames.com/documentation/unreal-engine/runtime-virtual-texturing-in-unreal-engine?lang=en-US)
- [GPU Geometry Clipmaps (NVIDIA GPU Gems 2)](https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-2-terrain-rendering-using-gpu-based-geometry)
- [Terrain shader generation deep-dive (80.lv)](https://80.lv/articles/using-next-gen-terrain-engines-for-games-production-009snw)
- [Hex-tiling demo (Mikkelsen)](https://github.com/mmikk/hextile-demo)
- [Fun with horizon maps](https://dasilvagf.github.io/posts/2020/08/fun-with-horizon-maps/)
- [Maximum-mipmap terrain shadows (arXiv)](https://arxiv.org/pdf/2005.06671)
- [UE Distance Field Shadows](https://dev.epicgames.com/documentation/en-us/unreal-engine/using-distance-field-shadows-in-unreal-engine)
- [UE Virtual Shadow Maps](https://dev.epicgames.com/documentation/en-us/unreal-engine/virtual-shadow-maps-in-unreal-engine)
- [Bevy 0.19 release notes](https://bevy.org/news/bevy-0-19/)
- [bevy_mesh_terrain (splat + texture array)](https://github.com/ethereumdegen/bevy_mesh_terrain)
- [bevy_triplanar_splatting](https://github.com/bonsairobo/bevy_triplanar_splatting)
