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
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var control_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(7) var control_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(8) var<uniform> params: TerrainParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(9) var normal_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(10) var normal_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(11) var orm_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(12) var orm_sampler: sampler;
// 0 = albedo target (rgb albedo, a occlusion); 1 = normal target (rg octahedral
// world normal, b roughness, a metallic).
@group(#{MATERIAL_BIND_GROUP}) @binding(13) var<uniform> output_mode: u32;

struct TerrainParams {
    tiling_scale: vec4<f32>,
    height_blend: vec4<f32>,
    roughness: vec4<f32>,
    normal_strength: vec4<f32>,
    slope_min: vec4<f32>,
    slope_blend: vec4<f32>,
    layer_count: u32,
    macro_strength: f32,
    macro_near: f32,
    macro_far: f32,
}

const MAX_LAYERS: u32 = 4u;

struct SplatResult {
    color: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    metallic: f32,
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

// Mirror of `splat_terrain` in terrain.wgsl. Transitional duplication: once the
// RVT covers all rings, only the bake runs the splat and the main pass drops it.
fn splat_terrain(world_xz: vec2<f32>, uv: vec2<f32>, normal: vec3<f32>) -> SplatResult {
    let control = textureSample(control_texture, control_sampler, uv);
    var w = array<f32, 4>(control.x, control.y, control.z, control.w);

    let slope = acos(clamp(normal.y, -1.0, 1.0));

    for (var i = 0u; i < MAX_LAYERS; i++) {
        if (i >= params.layer_count) {
            w[i] = 0.0;
            continue;
        }
        let smin = params.slope_min[i];
        if (smin < 3.15) {
            let t = smoothstep(smin, smin + params.slope_blend[i], slope);
            w[i] = mix(w[i], 1.0, t);
        }
    }

    let wsum = w[0] + w[1] + w[2] + w[3];
    if (wsum < 1e-4) {
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
        let a = textureSample(albedo_array, albedo_sampler, tile_uv, i);
        let n = textureSample(normal_array, normal_sampler, tile_uv, i).xyz * 2.0 - 1.0;
        colors[i] = a.rgb;
        normals[i] = vec3<f32>(n.xy * params.normal_strength[i], n.z);
        orms[i] = textureSample(orm_array, orm_sampler, tile_uv, i).rgb;
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
    for (var i = 0u; i < MAX_LAYERS; i++) {
        let b = max(0.0, scores[i] - (maxs - TRANSITION));
        rgb += colors[i] * b;
        tn += normals[i] * b;
        orm += orms[i] * b;
        rough += params.roughness[i] * b;
        bsum += b;
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
    out.metallic = orm.b;
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
    let splat = splat_terrain(world_xz, control_uv(world_xz), geo_normal(world_xz));
    if output_mode == 0u {
        return vec4<f32>(splat.color, splat.occlusion);
    }
    return vec4<f32>(oct_encode(splat.normal), splat.roughness, splat.metallic);
}
