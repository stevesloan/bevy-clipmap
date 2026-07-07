#import bevy_pbr::mesh_functions
#import bevy_pbr::pbr_fragment::pbr_input_from_standard_material
#import bevy_pbr::view_transformations::position_world_to_clip

#ifdef MESHLET_MESH_MATERIAL_PASS
#import bevy_pbr::meshlet_visibility_buffer_resolve::VertexOutput
#else ifdef PREPASS_PIPELINE
#import bevy_pbr::prepass_io::{Vertex, VertexOutput, FragmentOutput}
#import bevy_pbr::pbr_deferred_functions::deferred_output;
#else   // PREPASS_PIPELINE
#import bevy_pbr::forward_io::{Vertex, VertexOutput, FragmentOutput}
#import bevy_pbr::pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing}
#endif  // PREPASS_PIPELINE

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var color_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var color_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var heightmap_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var heightmap_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(107) var<uniform> grid_lod: u32;
@group(#{MATERIAL_BIND_GROUP}) @binding(108) var<uniform> texel_size: f32;
@group(#{MATERIAL_BIND_GROUP}) @binding(109) var<uniform> minmax: vec2<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(110) var<uniform> translation: vec2<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(111) var<uniform> wireframe: u32;
@group(#{MATERIAL_BIND_GROUP}) @binding(112) var albedo_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(113) var albedo_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(114) var control_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(115) var control_sampler: sampler;

// Per-layer splat parameters. `vec4` lanes index the (up to 4) layers.
struct TerrainParams {
    tiling_scale: vec4<f32>,
    height_blend: vec4<f32>,
    roughness: vec4<f32>,
    slope_min: vec4<f32>,
    slope_blend: vec4<f32>,
    layer_count: u32,
    macro_strength: f32,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(116) var<uniform> params: TerrainParams;

const MAX_LAYERS: u32 = 4u;

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

@vertex
fn vertex(vertex: Vertex, @builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let model = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_position = model * vec4<f32>(vertex.position, 1.0);

    let texture_size = vec2<f32>(textureDimensions(heightmap_texture));
    let world_size = texel_size * texture_size;

    let height_uv = out.world_position.xz / world_size + 0.5;
    let height = height_bilinear(height_uv, 0);

    out.world_position.y = height * (minmax.y - minmax.x) + minmax.x;
    out.position = position_world_to_clip(out.world_position.xyz);

    return out;
}

struct SplatResult {
    color: vec3<f32>,
    roughness: f32,
}

// Blends the material layers via control-map weights, slope rules, and
// height-based (depth) blending. `world_xz` tiles the textures, `uv` (0..1
// across the terrain) samples the control map, `normal` drives slope rules.
// Every layer is sampled unconditionally for uniform control flow; zero-weight
// layers are masked out of the blend.
fn splat_terrain(world_xz: vec2<f32>, uv: vec2<f32>, normal: vec3<f32>) -> SplatResult {
    let control = textureSample(control_texture, control_sampler, uv);
    var w = array<f32, 4>(control.x, control.y, control.z, control.w);

    // Slope in radians: 0 = flat, HALF_PI = vertical.
    let slope = acos(clamp(normal.y, -1.0, 1.0));

    // Mask inactive layers and apply slope-based placement.
    for (var i = 0u; i < MAX_LAYERS; i++) {
        if (i >= params.layer_count) {
            w[i] = 0.0;
            continue;
        }
        let smin = params.slope_min[i];
        if (smin < 3.15) { // sentinel >= PI disables the rule
            let t = smoothstep(smin, smin + params.slope_blend[i], slope);
            w[i] = mix(w[i], 1.0, t);
        }
    }

    // Normalize; fall back to layer 0 where the control map is empty.
    let wsum = w[0] + w[1] + w[2] + w[3];
    if (wsum < 1e-4) {
        w[0] = 1.0;
    } else {
        for (var i = 0u; i < MAX_LAYERS; i++) {
            w[i] = w[i] / wsum;
        }
    }

    // Sample every layer and compute height-blend scores.
    var colors = array<vec3<f32>, 4>();
    var scores = array<f32, 4>();
    var maxs = -1e9;
    for (var i = 0u; i < MAX_LAYERS; i++) {
        let tile_uv = world_xz / params.tiling_scale[i];
        let s = textureSample(albedo_array, albedo_sampler, tile_uv, i);
        colors[i] = s.rgb;
        // weight + height relief; zero-weight layers are pushed far below the max.
        let mask = select(0.0, 1.0, w[i] > 1e-4);
        scores[i] = (w[i] + s.a * params.height_blend[i]) * mask - (1.0 - mask) * 1e9;
        maxs = max(maxs, scores[i]);
    }

    // Only layers within TRANSITION of the top contribute (depth blend).
    const TRANSITION = 0.2;
    var rgb = vec3<f32>(0.0);
    var rough = 0.0;
    var bsum = 0.0;
    for (var i = 0u; i < MAX_LAYERS; i++) {
        let b = max(0.0, scores[i] - (maxs - TRANSITION));
        rgb += colors[i] * b;
        rough += params.roughness[i] * b;
        bsum += b;
    }
    bsum = max(bsum, 1e-4);

    var out: SplatResult;
    out.color = rgb / bsum;
    out.roughness = rough / bsum;
    return out;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    if wireframe != 0 {
        var out: FragmentOutput;
        out.color = vec4(1.0);
        return out;
    }

    var in_modified = in;

    let texture_size = vec2<f32>(textureDimensions(heightmap_texture));
    let world_size = texture_size * texel_size;

    let uv = in.world_position.xz / world_size + 0.5;
    let step = 1.0 / texture_size;
    let h_r = textureSample(heightmap_texture, heightmap_sampler, uv + vec2(step.x, 0.0)).r;
    let h_l = textureSample(heightmap_texture, heightmap_sampler, uv - vec2(step.x, 0.0)).r;
    let h_t = textureSample(heightmap_texture, heightmap_sampler, uv + vec2(0.0, step.y)).r;
    let h_b = textureSample(heightmap_texture, heightmap_sampler, uv - vec2(0.0, step.y)).r;

    let scale = (minmax.y - minmax.x) / (2.0 * texel_size);
    let dh_dx = (h_r - h_l) * scale;
    let dh_dy = (h_t - h_b) * scale;
    in_modified.world_normal = normalize(vec3(-dh_dx, 1.0, -dh_dy));

    var pbr_input = pbr_input_from_standard_material(in_modified, is_front);

    // Splat-blend the terrain material layers.
    let splat = splat_terrain(in.world_position.xz, uv, in_modified.world_normal);
    var albedo = splat.color;
    // Macro variation multiply: large-scale color break-up over the tiled detail.
    let macro_col = textureSample(color_texture, color_sampler, uv).rgb;
    albedo *= mix(vec3<f32>(1.0), 2.0 * macro_col, params.macro_strength);

    pbr_input.material.base_color = vec4<f32>(albedo, 1.0);
    pbr_input.material.perceptual_roughness = splat.roughness;

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in_modified, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}
