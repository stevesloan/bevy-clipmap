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

We are targeting the **RVT approach**: rather than blending N material layers
every frame for every pixel, **bake the blended result into a camera-centered
texture clipmap** and have the main terrain pass do a single fetch.

Why this fits *this* project unusually well:

- RVT is itself a **texture clipmap with toroidal updates**. `bevy-clipmap`
  already owns that machinery for geometry (`update_grids`, `src/lib.rs`). The
  expensive part of RVT is mostly already built here.
- It **decouples per-pixel shading cost from material complexity** — which is
  the single lever that lets one material serve both a tight VR budget and a
  lavish flatscreen budget (see §7).

What gets baked into the RVT (camera-centered ring textures, mirroring the
geometry LOD levels):

- Albedo (sRGB)
- Packed normal + ORM (occlusion/roughness/metallic) (linear)
- **Sun-visibility** channel (baked shadow — see §5)

The bake runs only for the **dirty toroidal region** when the window scrolls,
amortized over many frames. The main `terrain.wgsl` fragment pass collapses to a
single array fetch.

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

### 3.3 Close-range VR fidelity (near LOD rings only)

Gated on `grid_lod` (the material already carries per-level LOD) and/or shader
defs, so cost is spent only where the camera can see it:

- **Stochastic / hex tiling** on near layers to eliminate visible repetition.
  Costs ~3× samples → near rings only.
- **Distance-based macro/micro detail**: near rings get a high-frequency detail
  texture + tighter tiling; far rings drop it.
- **Triplanar on steep slopes only** (full triplanar is 3× sampling — too costly
  for VR everywhere; gate on the slope value already computed).

### 3.4 Precomputed normals

Stop reconstructing normals per-fragment from 4 heightmap samples
(`terrain.wgsl:105-113`). That is wasteful **and a shimmer source that VR
exaggerates**. Bake the normal once into the RVT / a normal clipmap and sample
it.

---

## 4. Lighting & shadows — starting point (baked, fixed sun)

**Phase 1 ships with baked lighting and a FIXED directional sun.**

- Static object shadows onto terrain are baked into the **sun-visibility
  channel** of the RVT (a directional shadowmask).
- The bake **must include the terrain itself** (render terrain from the sun's
  view), not only props — otherwise terrain-on-terrain shadows (mountain over
  valley) are lost.
- Near-field dynamic objects can still receive live **cascaded shadow maps** —
  `terrain.wgsl` already calls `fetch_directional_shadow`, so the terrain is
  already a CSM receiver if/when needed.

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

## 5. Lighting & shadows — future (dynamic sun / time-of-day)

Planned upgrade, **not built right away**. The AAA VR-viable answer for a
dynamic sun is a **hybrid**, merged into one `sun_visibility` scalar:

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
1b. **Per-layer normal + ORM arrays** — tangent-space normals and per-layer
   roughness/metallic/AO (§3.1).
2. **`TerrainQualityKey` specialization + feature flags** (§7).
3. **RVT bake pass + precomputed normals** (§2, §3.4).
4. **Unified `sun_visibility` + baked static-object shadows** (fixed sun) (§4).
5. **(Later)** ray-marched heightfield shadows + hybrid shadow stack for a
   dynamic sun (§5).

Steps 0–2 are the foundation; do them before RVT so the bake pass and quality
system have somewhere to plug in.

---

## 10. Key code anchor points (current)

| Location                                    | Role                                                    |
| ------------------------------------------- | ------------------------------------------------------- |
| `src/terrain.wgsl` `splat_terrain`          | Layer splat / height-blend / slope; material injection  |
| `src/terrain.wgsl` fragment normal          | Per-fragment normal reconstruction (→ precompute in RVT) |
| `src/lib.rs` `TerrainLayer` / `TerrainParams` | Layer config + GPU packing (extend for normal/ORM)    |
| `src/lib.rs` `WireframeKey` / `specialize()`  | Pattern to extend for `TerrainQualityKey`             |
| `src/lib.rs` `update_grids`                 | Toroidal clipmap machinery (reuse for RVT)              |
| `src/lib.rs:420`             | `clipmap.target` — center on HMD head for stereo            |

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
