# Terrain Material & Shadowing — Design Doc

Status: **Draft / agreed direction** · Last updated: 2026-07-08

This document records the decisions for adding an Unreal-style multi-material
landscape system (splat-blended textures) and a terrain shadowing pipeline to
`bevy-clipmap`. It captures *what* we're building, *why*, and the *order* to
build it. It is the reference for future sessions and contributors.

> **2026-07-08 update.** A camera-centered **toroidal RVT ring** (double-buffered,
> amortized strip bake) and a decoupled **shadow + sky-AO** texture were built,
> evaluated, and **shelved** — see §2.1 and §4.2 for the verdicts. The adopted
> baseline stays the **static bake-once RVT**, now at **8192²** (~1 m/texel).
> Since then: material placement went **fully procedural** (slope/height bands,
> control map removed, §3.2), the **macro albedo map was removed** (§3.2), the
> detail overlay **blends the top-2 materials** (§3.5), and **no realtime
> shadows** became a governing rule (§4.2). Landscapes are **user-generated**,
> which drives the procedural/no-authored-maps direction throughout.

---

## 1. Goals

- **Unreal-style landscape multi-material**: blend multiple textures (grass,
  rock, dirt, snow, …) across the terrain — placed **procedurally** from slope
  and height (no authored control map; landscapes are **user-generated**, so
  nothing may require hand-painted per-terrain data).
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
- **Realtime shadows/AO of any kind** — CSM, SSAO/GTAO, live ray-march, VSM,
  ray-traced. Governing rule, not just a deferral: everything static bakes once;
  anything dynamic gets **faked** (blob/decal shadows). See §4.2.

---

## 2. Chosen architecture — Runtime Virtual Texture (RVT), "Tier 2 / AAA"

We use the **RVT approach**: rather than blending N material layers every frame
for every pixel, **bake the blended result into a texture** and have the main
terrain pass do a single fetch. This **decouples per-pixel shading cost from
material complexity** — the lever that lets one material serve both a tight VR
budget and a lavish flatscreen budget (see §7).

**The world is finite-ish, so the RVT is a static, full-terrain, bake-once
texture — NOT a camera-centered clipmap.** (Re-affirmed after actually building
the toroidal alternative — see §2.1.) The targets are **8192²** over the ~8192 m
example world → **~1 m/texel** (bumped from 4096²/2 m; sampling cost is
unchanged, VRAM ~4×, judged worth it for the mid-ground). What gets baked:

- Albedo (sRGB) + **sun-visibility** in alpha (baked shadow — see §4)
- Octahedral world normal + roughness + **packed material ids** in alpha (§3.5)

The bake runs via top-down orthographic cameras → `RenderTarget::Image`, then
stops. The main `terrain.wgsl` pass collapses from ~14 samples to ~2 fetches
(+ the near-only detail samples, §3.5) — the unused source-array bindings were
removed from the main material, freeing 4 sampled-texture slots against the
~16-per-stage ceiling on mobile-VR GPUs.

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

### 2.1 Camera-centered ring RVT — BUILT, EVALUATED, SHELVED (2026-07-08)

A camera-following high-density ring (1024 m footprint, 0.25 m/texel) was fully
prototyped on an exploration branch: ring-relative UV + ring→macro fade,
target-following re-bake, **double-buffering** (bake into a back buffer, swap on
completion — no tear), and an **amortized bake** (interleaved scanlines over N
frames into a non-cleared target — no hitch). It worked and looked good.

**Shelved for the VR baseline anyway.** The ring re-bakes whenever the head
moves — recurring per-frame GPU work that competes with the scene render inside
a Quest-class 8–11 ms budget, unverifiable without on-device profiling — and
double-buffering doubles ring VRAM. The static bake-once RVT has **zero**
per-frame bake cost, the safest possible VR profile; at 8192² its mid-ground is
good enough that the ring's extra density wasn't missed. Classic
sharpness-vs-schedulability trade — VR chose schedulability.

Revisit only if: the world outgrows a static RVT (>10–15 km at 1 m/texel),
terrain becomes streamed/effectively infinite, or a **flatscreen tier** wants the
extra near density (§7). The prototype (including the multi-ring "texture
clipmap riding the geometry levels" design) lives in this doc's git history and
the abandoned exploration commits.

---

## 3. Material system — procedural splat blending

### 3.1 Layer configuration

Layers are configured on the `Clipmap` component (`TerrainLayer`): per-layer
`tiling_scale`, `height_blend`, `normal_strength`, `roughness`, plus the
placement bands below. (The earlier `TerrainLayerSet` hot-reloadable-asset idea
and RGBA `control_maps` are superseded — see §3.2.)

### 3.2 Placement & blending rules — procedural, no control map (2026-07-08)

**The painted RGBA control map is REMOVED.** Landscapes are user-generated, so
placement must derive from the terrain itself. Each layer's weight is computed
in the bake as the overlap of two optional **bands** (`band()` in `bake.wgsl`):

- **`SlopeRule { min_deg, max_deg, blend_deg }`** — slope-angle band: grass on
  flat, dirt mid-slope, rock steep. Weight is 1 inside the band, ramping out
  over `blend_deg` at each edge.
- **`HeightRule { min, max, blend }`** — world-height band: snow above a
  snowline, cliffs staying bare rock at any height.
- A layer with neither rule is present **everywhere** — the "base layer"
  pattern: giving grass no slope band lets it compete on cliffs and poke through
  the rock via height-blend scoring, which is what makes boundaries look natural
  (an exclusive-bands-only setup read as sterile).
- **Height/depth blending** (not linear alpha) for crisp transitions, as before.
- **Gotcha — do NOT jitter the band inputs with value noise.** An fbm-of-value-
  noise jitter was added to make boundaries wander; thresholding lattice noise at
  a boundary makes the edge hug the noise's square cells → meters-scale square
  "brush-stroke" patchwork at **every** material edge. Removed; the base-layer
  competition provides the organic look on its own. If boundary wander is ever
  wanted, use a lattice-free (gradient, rotated-octave) noise.
- **Macro variation map — REMOVED.** With real CC0 texture sets + hex de-tiling,
  the layers carry the look; the macro multiply and the distance→macro fade were
  dead weight (and a texture slot). `Clipmap::{color, macro_*}` deleted.

**Guardrails (baked in):** arrays are `Handle<Image>`, so BC7/BC5 mipmapped KTX2
work unchanged; the tiling sampler uses repeat + anisotropic filtering; albedo is
sampled sRGB, height linear; weights are normalized in-shader; extending past 4
layers means widening the `TerrainParams` band vectors (top-4 per pixel).
**Normal maps are OpenGL convention (`nor_gl`)**, and the bake/overlay flip the
normal's X to match the reorientation tangent (which runs −X vs the +X tiling
UV) — without the flip, cracks light as ridges. Documented in the README.

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
- ~~**Distance→macro vista blend**~~ — built, then **removed with the macro map**
  (§3.2): hex de-tiling + the 8K RVT carry the vistas without it.
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
  detail albedo + normal + ORM arrays, selected by a layer ID baked into the
  RVT's metallic slot (terrain is never metallic). Real photographic detail
  textures drop straight into the same `Handle<Image>` array slots.
- **v3 — top-2 material blend + far-field skip** ✅ done (2026-07-08). A single
  dominant ID snapped at material boundaries (dirt wearing rock's relief; hard
  detail seams while the base blended smoothly underneath). Now the bake packs
  the **two dominant layer indices + a 4-bit blend weight into the material-id
  byte (2+2+4)** and the main pass lerps both materials' detail normal / albedo
  / ORM. Two hard-won gotchas:
  - **Read the packed byte NEAREST (`textureLoad`).** Bilinear-filtering a
    packed id byte sweeps through garbage id/weight combos → banding strips.
    Per-texel selection steps at ~1 m instead, which reads as natural mottling.
  - **The whole detail block is gated behind `detail_fade > 0`** with derivatives
    hoisted out of the branch (`textureSampleGrad`) for correct mips — far
    terrain now pays **zero** detail samples. Net: near +2 samples for the
    blend, far −3.

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

Everything here is **baked** — see the governing rule in §4.2.

Two distinct shadow problems, two (baked) tools:
- **Terrain self-shadow** (mountains → valleys): **heightfield ray-march** — march
  the heightmap toward the sun and test occlusion, once, in the bake. A shadow map
  is the *wrong* tool here (resolution / peter-panning, and it would need the
  displaced terrain to cast).
- **Objects → terrain** (buildings, rocks, props): render the static props from
  the sun **during the bake** into the same sun-visibility channel (step 7). Not
  CSM — realtime shadow maps are ruled out (§4.2).

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

### 4.2 Governing rule — NO realtime shadows/AO; and the sky-AO verdict

**Hard constraint (2026-07-08): avoid realtime shadows and occlusion at all
costs.** VR is ~90–120 Hz × two eyes; a per-frame shadow pass risks hitches, and
shadow shimmer is nausea-inducing in a headset. This rules out *every* per-frame
technique — CSM, SSAO/GTAO, live heightfield march, VSM, ray-traced. The system
is **one baked pass plus fakes**:

| Case                                | Approach (zero per-frame shadow work)              |
| ----------------------------------- | -------------------------------------------------- |
| Terrain self sun-shadow             | Baked sun-visibility (RVT alpha). ✅ done           |
| Static object → terrain sun-shadow  | Baked into the same channel (sun-view pass, step 7)|
| Static object's own shadow/AO       | Baked per-object (vertex AO / lightmap) at load    |
| **Dynamic** objects (if any)        | **Faked** — soft blob/decal. Never a shadow map.   |

**Sky-AO — tried and dropped (2026-07-08).** A horizon-march sky-occlusion bake
(`G` channel of a decoupled `RG8` shadow texture, applied to the ambient term
only) was built and evaluated. On open heightfield terrain it was **nearly
invisible** — hemispherical occlusion only bites in crevices/ravines, and open
ground has no nearby occluders — so it wasn't worth the extra channel and the
expensive multi-azimuth bake march. **Revisit when static objects exist**:
object-contact AO (the dark hug where a rock meets the ground, baked via
voxel/SDF or a hemispherical pass into that same channel) is very visible, and
that's when the channel earns its keep. The implementation is in the abandoned
exploration commits.

---

## 5. Lighting & shadows — dynamic tier (SUPERSEDED by §4.2)

> **Superseded.** The realtime techniques below (CSM, live ray-march, lazy
> refresh) are **not** pursued even as a high-power tier — §4.2's no-realtime
> rule stands for every target. Kept as a record of the evaluation. A moving sun
> under the current rule would mean **re-baking** the sun-visibility channel
> (lazily, budgeted), not live shadowing.

The original evaluation follows. The AAA VR-viable answer for a dynamic sun was a
**hybrid**, merged into one `sun_visibility` scalar:

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
| RVT resolution           | 4096² (measure!)     | 8192²                   |
| toroidal ring RVT (§2.1) | off (static bake)    | optional near ring      |
| triplanar                | slopes only / off    | slopes on               |
| detail-texture distance  | short                | long                    |
| parallax / POM           | off                  | on                      |
| baked shadow/AO channels | sun-vis only         | + object AO (§4.2)      |
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
7. **Baked static-object shadows** (§4, §4.2) — render props from the sun into
   the same sun-visibility channel during the bake. *(active next step; also the
   trigger to revisit the sky/contact-AO channel, §4.2)*
8. **`TerrainQualityKey` specialization + feature flags** (§7) — gate the above
   into VR / flatscreen presets.

**2026-07-08 exploration round (after step 6):**

- ✅ **Procedural material placement** (§3.2) — control map removed; slope/height
  bands (`SlopeRule` + `max_deg`, new `HeightRule`) evaluated in the bake; grass
  as an unbanded base layer for natural boundaries. Band-jitter noise added,
  then **removed** — its value-noise lattice caused square boundary patchwork
  (§3.2 gotcha).
- ✅ **Macro albedo map removed** (§3.2) — layers + hex de-tiling carry the look.
- ✅ **Detail overlay v3** (§3.5) — top-2 material blend via the packed id byte
  (read NEAREST), far-field detail skip.
- ✅ **RVT 4096² → 8192²** (~1 m/texel); normal-map convention fixed (X-flip,
  §3.2) with a README note.
- ⏸ **Toroidal ring RVT** — built through amortized double-buffered baking,
  **shelved** for the VR baseline (§2.1).
- ⏸ **Decoupled shadow + sky-AO texture** — built, AO judged invisible on open
  terrain, **shelved** until static objects exist (§4.2).

**Known follow-ups:** RVT edge-stripe smear where the clipmap mesh extends past
heightmap coverage (clamp sun-visibility to 1 / fade outside coverage); profile
the 8192² static build on-device (Quest) — the VRAM (~512 MB RVT) vs 4096² call
should be made from a headset, not a desktop; optionally have
`fetch_textures.py` pull `nor_dx` and drop the in-shader X-flip.

**Deferred / triggered:**
- **(Stretch)** Triplanar on steep slopes — does not amortize into RVT (§3.3).
- **(Scaling / flatscreen tier)** Camera-centered toroidal ring RVT — prototype
  exists; triggers in §2.1.
- **(With static objects)** Object sun-shadow into the bake (step 7) + the AO
  channel revival (§4.2). Dynamic objects: blob/decal fakes only.

Steps 0–6 plus the exploration round are done. Baked static-object shadows
(step 7) are the next feature work; on-device Quest profiling is the next
validation work.

---

## 10. Key code anchor points (current)

| Location                                    | Role                                                    |
| ------------------------------------------- | ------------------------------------------------------- |
| `src/bake.wgsl` `splat_terrain` / `hex_sample` | Layer splat + hex de-tiling; writes the RVT channels |
| `src/bake.wgsl` `band()`                    | Procedural slope/height placement bands (§3.2)          |
| `src/bake.wgsl` `sun_visibility`            | Adaptive heightfield ray-march (baked terrain shadow)   |
| `src/terrain.wgsl` `terrain_apply_lighting` | Compact PBR fork injecting baked sun-visibility          |
| `src/terrain.wgsl` packed-id unpack + detail block | Top-2 detail blend, NEAREST id read, far skip (§3.5) |
| `src/lib.rs` `load_terrain_array`           | Decode + stack + mip real texture sets (public API)     |
| `src/lib.rs` `init_rvt` / `drive_rvt_bake`  | Sentinel-gated RVT bake lifecycle                       |
| `src/lib.rs` `TerrainLayer` / `SlopeRule` / `HeightRule` | Layer config + placement bands (§3.2)      |
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
