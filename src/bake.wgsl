// Bakes the terrain splat material into a texture indexed by world XZ. Runs the
// same splat blend as the main pass but writes raw channels unlit, so the main
// pass can sample the result instead of blending per-fragment. Rendered by a
// top-down orthographic camera over a flat quad covering the terrain.
#import bevy_pbr::forward_io::VertexOutput

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var heightmap_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var heightmap_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> texel_size: f32;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<uniform> minmax: vec2<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var albedo_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var albedo_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(8) var<uniform> params: TerrainParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(9) var normal_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(10) var normal_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(11) var orm_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(12) var orm_sampler: sampler;
// 0 = albedo target (rgb albedo, a sun-visibility); 1 = normal target (rg
// octahedral world normal, b roughness, a material id).
@group(#{MATERIAL_BIND_GROUP}) @binding(13) var<uniform> output_mode: u32;
// Normalized direction toward the fixed sun.
@group(#{MATERIAL_BIND_GROUP}) @binding(14) var<uniform> sun_direction: vec3<f32>;

struct TerrainParams {
    tiling_scale: vec4<f32>,
    height_blend: vec4<f32>,
    roughness: vec4<f32>,
    normal_strength: vec4<f32>,
    slope_min: vec4<f32>,
    slope_max: vec4<f32>,
    slope_blend: vec4<f32>,
    height_min: vec4<f32>,
    height_max: vec4<f32>,
    height_range_blend: vec4<f32>,
    layer_count: u32,
}

// Weight of a band: 1 inside [lo, hi], ramping to 0 over `blend` just outside
// each edge. `lo` at/below the input's min (or `hi` at/above its max) makes that
// side open-ended (weight stays 1 there).
fn band(x: f32, lo: f32, hi: f32, blend: f32) -> f32 {
    let up = smoothstep(lo - blend, lo, x);
    let down = 1.0 - smoothstep(hi, hi + blend, x);
    return up * down;
}

// Value noise + fbm, used to break up the clean slope/height bands. Baked once,
// so cost is irrelevant at runtime.
fn vhash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = vhash(i);
    let b = vhash(i + vec2<f32>(1.0, 0.0));
    let c = vhash(i + vec2<f32>(0.0, 1.0));
    let d = vhash(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var pp = p;
    for (var i = 0; i < 4; i++) {
        v += amp * vnoise(pp);
        pp *= 2.0;
        amp *= 0.5;
    }
    return v;
}

const MAX_LAYERS: u32 = 4u;

struct SplatResult {
    color: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    // Dominant layer index, normalized to 0..1 (baked into the RVT's metallic
    // slot, which terrain is never; used to pick the per-material detail).
    material_id: f32,
    occlusion: f32,
}

fn control_uv(world_xz: vec2<f32>) -> vec2<f32> {
    let texture_size = vec2<f32>(textureDimensions(heightmap_texture));
    return world_xz / (texture_size * texel_size) + 0.5;
}

fn geo_normal(world_xz: vec2<f32>) -> vec3<f32> {
    let texture_size = vec2<f32>(textureDimensions(heightmap_texture));
    let uv = world_xz / (texture_size * texel_size) + 0.5;
    let step = 1.0 / texture_size;
    let h_r = textureSample(heightmap_texture, heightmap_sampler, uv + vec2(step.x, 0.0)).r;
    let h_l = textureSample(heightmap_texture, heightmap_sampler, uv - vec2(step.x, 0.0)).r;
    let h_t = textureSample(heightmap_texture, heightmap_sampler, uv + vec2(0.0, step.y)).r;
    let h_b = textureSample(heightmap_texture, heightmap_sampler, uv - vec2(0.0, step.y)).r;
    let scale = (minmax.y - minmax.x) / (2.0 * texel_size);
    let dh_dx = (h_r - h_l) * scale;
    let dh_dy = (h_t - h_b) * scale;
    return normalize(vec3(-dh_dx, 1.0, -dh_dy));
}

fn height_bilinear(uv: vec2<f32>, lod: i32) -> f32 {
    let tex_size = vec2<f32>(textureDimensions(heightmap_texture, lod));
    let pos = uv * tex_size;
    let p0 = vec2<i32>(floor(pos));
    let f = pos - floor(pos);
    let h00 = textureLoad(heightmap_texture, p0, lod).r;
    let h10 = textureLoad(heightmap_texture, p0 + vec2(1, 0), lod).r;
    let h01 = textureLoad(heightmap_texture, p0 + vec2(0, 1), lod).r;
    let h11 = textureLoad(heightmap_texture, p0 + vec2(1, 1), lod).r;
    let hx0 = mix(h00, h10, f.x);
    let hx1 = mix(h01, h11, f.x);
    return mix(hx0, hx1, f.y);
}

// World-space terrain height at a position.
fn terrain_height(world_xz: vec2<f32>) -> f32 {
    let h = height_bilinear(control_uv(world_xz), 0);
    return h * (minmax.y - minmax.x) + minmax.x;
}

// Terrain self-shadow: soft-march the heightmap toward the sun. 0 = shadowed,
// 1 = lit. Baked once (static terrain, fixed sun), so the march is affordable.
fn sun_visibility(world_xz: vec2<f32>) -> f32 {
    const STEPS = 96;
    const MAX_DIST = 6000.0;
    const SOFTNESS = 10.0;      // lower = softer penumbra
    const NORMAL_BIAS = 12.0;   // lift the ray off the surface to avoid acne
    const STEP0 = 3.0;          // fine near-field step (resolves steep sun-facing slopes)
    const GROWTH = 1.12;        // geometric growth -> long reach without huge step count
    // Bias the start along the surface normal so sun-facing slopes don't
    // self-shadow.
    let origin = vec3<f32>(world_xz.x, terrain_height(world_xz), world_xz.y)
        + geo_normal(world_xz) * NORMAL_BIAS;
    var vis = 1.0;
    var step = STEP0;
    var t = STEP0;
    for (var i = 0; i < STEPS; i++) {
        if t > MAX_DIST {
            break;
        }
        let p = origin + sun_direction * t;
        if p.y > minmax.y {
            break; // above the highest terrain -> can't be occluded
        }
        // Soft shadow: penumbra widens with distance to the occluder.
        let clearance = p.y - terrain_height(p.xz);
        vis = min(vis, clamp(SOFTNESS * clearance / t, 0.0, 1.0));
        if vis <= 0.001 {
            break;
        }
        step *= GROWTH;
        t += step;
    }
    return vis;
}

// --- Stochastic hex tiling (Mikkelsen) ---
// Samples a tiled texture 3× on a randomized hex lattice and blends, hiding the
// repetition. Only run in the bake (one-time), never per-frame — it needs 3×
// samples and `textureSampleGrad` (per-cell offsets break implicit derivatives).

fn hex_hash(p: vec2<f32>) -> vec2<f32> {
    let r = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(r) * 43758.5453);
}

// Skewed triangle (hex) grid: barycentric weights + the 3 cell vertices for `uv`.
fn triangle_grid(
    uv: vec2<f32>,
    w: ptr<function, vec3<f32>>,
    v1: ptr<function, vec2<f32>>,
    v2: ptr<function, vec2<f32>>,
    v3: ptr<function, vec2<f32>>,
) {
    // Cells span ~1 texture repeat: each cell is a differently-offset crop.
    // Smaller cells speckle under minification; larger ones show repetition.
    let p = uv;
    let skewed = vec2<f32>(p.x - 0.57735027 * p.y, 1.15470054 * p.y);
    let base = floor(skewed);
    let f = fract(skewed);
    let fz = 1.0 - f.x - f.y;
    let s = step(0.0, -fz);
    let s2 = 2.0 * s - 1.0;
    *w = vec3<f32>(-fz * s2, s - f.y * s2, s - f.x * s2);
    *v1 = base + vec2<f32>(s, s);
    *v2 = base + vec2<f32>(s, 1.0 - s);
    *v3 = base + vec2<f32>(1.0 - s, s);
}

fn hex_sample(
    tex: texture_2d_array<f32>,
    samp: sampler,
    layer: u32,
    uv: vec2<f32>,
    ddx: vec2<f32>,
    ddy: vec2<f32>,
) -> vec4<f32> {
    var w: vec3<f32>;
    var v1: vec2<f32>;
    var v2: vec2<f32>;
    var v3: vec2<f32>;
    triangle_grid(uv, &w, &v1, &v2, &v3);
    let c1 = textureSampleGrad(tex, samp, uv + hex_hash(v1), layer, ddx, ddy);
    let c2 = textureSampleGrad(tex, samp, uv + hex_hash(v2), layer, ddx, ddy);
    let c3 = textureSampleGrad(tex, samp, uv + hex_hash(v3), layer, ddx, ddy);
    // Sharpen the barycentric weights to keep transitions crisp (avoid ghosting).
    let ws = pow(w, vec3<f32>(7.0)) + vec3<f32>(1e-6);
    let wn = ws / (ws.x + ws.y + ws.z);
    return c1 * wn.x + c2 * wn.y + c3 * wn.z;
}

// Mirror of `splat_terrain` in terrain.wgsl, plus hex tiling. Transitional
// duplication: only the bake runs the splat; the main pass samples the result.
fn splat_terrain(world_xz: vec2<f32>, normal: vec3<f32>) -> SplatResult {
    // Procedural placement: each layer's weight is the overlap of its slope band
    // (grass→dirt→rock) and its world-height band (e.g. snow above a snowline).
    // Baked noise jitters the band inputs so the boundaries wander naturally
    // instead of reading as clean iso-slope / iso-height lines.
    let slope = acos(clamp(normal.y, -1.0, 1.0));
    let height = terrain_height(world_xz);
    let slope_j = slope + (fbm(world_xz * 0.02) - 0.5) * 0.6;
    let height_j = height + (fbm(world_xz * 0.02 + vec2<f32>(53.0, 17.0)) - 0.5) * 250.0;

    var w = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var i = 0u; i < MAX_LAYERS; i++) {
        if i >= params.layer_count {
            continue;
        }
        w[i] = band(slope_j, params.slope_min[i], params.slope_max[i], params.slope_blend[i])
             * band(height_j, params.height_min[i], params.height_max[i], params.height_range_blend[i]);
    }

    let wsum = w[0] + w[1] + w[2] + w[3];
    if wsum < 1e-4 {
        w[0] = 1.0;
    } else {
        for (var i = 0u; i < MAX_LAYERS; i++) {
            w[i] = w[i] / wsum;
        }
    }

    var colors = array<vec3<f32>, 4>();
    var normals = array<vec3<f32>, 4>();
    var orms = array<vec3<f32>, 4>();
    var scores = array<f32, 4>();
    var maxs = -1e9;
    for (var i = 0u; i < MAX_LAYERS; i++) {
        let tile_uv = world_xz / params.tiling_scale[i];
        let ddx = dpdx(tile_uv);
        let ddy = dpdy(tile_uv);
        let a = hex_sample(albedo_array, albedo_sampler, i, tile_uv, ddx, ddy);
        // Flip X: the reorientation tangent runs -X vs the +X tiling UV, so the
        // normal's red axis is mirrored — concave features (cracks) light up as
        // convex (bumps/veins) without this.
        let n = (hex_sample(normal_array, normal_sampler, i, tile_uv, ddx, ddy).xyz * 2.0 - 1.0)
            * vec3<f32>(-1.0, 1.0, 1.0);
        colors[i] = a.rgb;
        normals[i] = vec3<f32>(n.xy * params.normal_strength[i], n.z);
        orms[i] = hex_sample(orm_array, orm_sampler, i, tile_uv, ddx, ddy).rgb;
        let mask = select(0.0, 1.0, w[i] > 1e-4);
        scores[i] = (w[i] + a.a * params.height_blend[i]) * mask - (1.0 - mask) * 1e9;
        maxs = max(maxs, scores[i]);
    }

    const TRANSITION = 0.2;
    var rgb = vec3<f32>(0.0);
    var tn = vec3<f32>(0.0);
    var orm = vec3<f32>(0.0);
    var rough = 0.0;
    var bsum = 0.0;
    // Track the top two contributing layers (b0 >= b1) so the detail overlay can
    // lerp their detail normals instead of snapping at material boundaries.
    var b0 = -1.0;
    var i0 = 0u;
    var b1 = -1.0;
    var i1 = 0u;
    for (var i = 0u; i < MAX_LAYERS; i++) {
        let b = max(0.0, scores[i] - (maxs - TRANSITION));
        rgb += colors[i] * b;
        tn += normals[i] * b;
        orm += orms[i] * b;
        rough += params.roughness[i] * b;
        bsum += b;
        if b > b0 {
            b1 = b0; i1 = i0;
            b0 = b; i0 = i;
        } else if b > b1 {
            b1 = b; i1 = i;
        }
    }
    let inv = 1.0 / max(bsum, 1e-4);
    rgb *= inv;
    orm *= inv;
    tn = normalize(tn);

    let ref_axis = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.z) > 0.99);
    let tangent = normalize(cross(ref_axis, normal));
    let bitangent = cross(normal, tangent);

    var out: SplatResult;
    out.color = rgb;
    out.normal = normalize(tangent * tn.x + bitangent * tn.y + normal * tn.z);
    out.occlusion = orm.r;
    out.roughness = (rough * inv) * orm.g;
    // Pack the two dominant layer indices + their blend into 8 bits (2+2+4). The
    // blend `t2` (0..1) maps to a detail-normal lerp of 0..0.5 in the main pass.
    let t2 = clamp(2.0 * b1 / max(b0 + b1, 1e-4), 0.0, 1.0);
    let q = floor(t2 * 15.0 + 0.5);
    out.material_id = (f32(i0) + f32(i1) * 4.0 + q * 16.0) / 255.0;
    return out;
}

// Octahedral encode of a unit vector into 0..1 (2 channels).
fn oct_wrap(v: vec2<f32>) -> vec2<f32> {
    return (1.0 - abs(v.yx)) * select(vec2(-1.0), vec2(1.0), v >= vec2(0.0));
}

fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let m = abs(n.x) + abs(n.y) + abs(n.z);
    var v = n.xy / m;
    v = select(oct_wrap(v), v, n.z >= 0.0);
    return v * 0.5 + 0.5;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let world_xz = in.world_position.xz;
    let splat = splat_terrain(world_xz, geo_normal(world_xz));
    if output_mode == 0u {
        return vec4<f32>(splat.color, sun_visibility(world_xz));
    }
    return vec4<f32>(oct_encode(splat.normal), splat.roughness, splat.material_id);
}
