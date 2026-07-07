#import bevy_pbr::mesh_functions
#import bevy_pbr::pbr_fragment::pbr_input_from_standard_material
#import bevy_pbr::view_transformations::position_world_to_clip
#import bevy_pbr::mesh_view_bindings::view

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
@group(#{MATERIAL_BIND_GROUP}) @binding(117) var normal_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(118) var normal_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(119) var orm_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(120) var orm_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(121) var rvt_albedo_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(122) var rvt_albedo_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(123) var rvt_normal_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(124) var rvt_normal_sampler: sampler;

// Per-layer splat parameters. `vec4` lanes index the (up to 4) layers.
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

// Octahedral decode of a 0..1 encoded unit vector (2 channels).
fn oct_decode(f: vec2<f32>) -> vec3<f32> {
    let e = f * 2.0 - 1.0;
    let n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    let t = max(-n.z, 0.0);
    let xy = n.xy + select(vec2(t), vec2(-t), n.xy >= vec2(0.0));
    return normalize(vec3<f32>(xy, n.z));
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

    // Sample the baked RVT material instead of blending the splat per-fragment:
    // albedo + occlusion, and octahedral world normal + roughness + metallic.
    let rvt_a = textureSample(rvt_albedo_texture, rvt_albedo_sampler, uv);
    let rvt_n = textureSample(rvt_normal_texture, rvt_normal_sampler, uv);

    in_modified.world_normal = oct_decode(rvt_n.rg);

    var pbr_input = pbr_input_from_standard_material(in_modified, is_front);

    var albedo = rvt_a.rgb;
    let macro_col = textureSample(color_texture, color_sampler, uv).rgb;
    // Near: subtle macro tint. Far: partial blend toward the macro color.
    albedo *= mix(vec3<f32>(1.0), 2.0 * macro_col, params.macro_strength);
    let cam_dist = distance(view.world_position, in.world_position.xyz);
    let macro_t = smoothstep(params.macro_near, params.macro_far, cam_dist) * 0.4;
    albedo = mix(albedo, macro_col, macro_t);

    pbr_input.material.base_color = vec4<f32>(albedo, 1.0);
    pbr_input.material.perceptual_roughness = rvt_n.b;
    pbr_input.material.metallic = rvt_n.a;
    pbr_input.diffuse_occlusion = vec3<f32>(rvt_a.a);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in_modified, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}
