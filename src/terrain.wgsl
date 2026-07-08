#import bevy_pbr::mesh_functions
#import bevy_pbr::pbr_fragment::pbr_input_from_standard_material
#import bevy_pbr::view_transformations::position_world_to_clip
#import bevy_pbr::mesh_view_bindings::{view, lights}
#import bevy_pbr::{
    pbr_types,
    mesh_view_types,
    lighting,
    lighting::LAYER_BASE,
    clustered_forward as clustering,
    shadows,
    ambient,
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
}

#ifdef ENVIRONMENT_MAP
#import bevy_pbr::environment_map
#endif

#ifdef MESHLET_MESH_MATERIAL_PASS
#import bevy_pbr::meshlet_visibility_buffer_resolve::VertexOutput
#else ifdef PREPASS_PIPELINE
#import bevy_pbr::prepass_io::{Vertex, VertexOutput, FragmentOutput}
#import bevy_pbr::pbr_deferred_functions::deferred_output;
#else   // PREPASS_PIPELINE
#import bevy_pbr::forward_io::{Vertex, VertexOutput, FragmentOutput}
#import bevy_pbr::pbr_functions::main_pass_post_lighting_processing
#endif  // PREPASS_PIPELINE

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
@group(#{MATERIAL_BIND_GROUP}) @binding(125) var detail_albedo_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(126) var detail_albedo_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(127) var detail_normal_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(128) var detail_normal_sampler: sampler;

// Near-range detail overlay parameters.
struct DetailParams {
    tiling: f32,
    normal_strength: f32,
    albedo_strength: f32,
    near: f32,
    far: f32,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(129) var<uniform> detail: DetailParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(130) var detail_orm_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(131) var detail_orm_sampler: sampler;

// Per-layer splat parameters. `vec4` lanes index the (up to 4) layers.
struct TerrainParams {
    tiling_scale: vec4<f32>,
    height_blend: vec4<f32>,
    roughness: vec4<f32>,
    normal_strength: vec4<f32>,
    slope_min: vec4<f32>,
    slope_blend: vec4<f32>,
    layer_count: u32,
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

// Terrain PBR lighting with baked sun-visibility injected on the directional sun.
// Compact fork of `apply_pbr_lighting`: base layer only, directional + ambient +
// environment map (no clearcoat / transmission / anisotropy / point lights — the
// terrain doesn't use them). `sun_vis` (0..1) attenuates only the direct sun, so
// shadowed valleys still receive sky/ambient light.
fn terrain_apply_lighting(in: pbr_types::PbrInput, sun_vis: f32) -> vec4<f32> {
    let base_color = in.material.base_color;
    let metallic = in.material.metallic;
    let perceptual_roughness = in.material.perceptual_roughness;
    let roughness = lighting::perceptualRoughnessToRoughness(perceptual_roughness);
    let reflectance = in.material.reflectance;
    let diffuse_color = base_color.rgb * (1.0 - metallic);
    let NdotV = max(dot(in.N, in.V), 0.0001);
    let R = reflect(-in.V, in.N);
    let F_ab = lighting::F_AB(perceptual_roughness, NdotV);
    let F0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + base_color.rgb * metallic;

    var li: lighting::LightingInput;
    li.layers[LAYER_BASE].NdotV = NdotV;
    li.layers[LAYER_BASE].N = in.N;
    li.layers[LAYER_BASE].R = R;
    li.layers[LAYER_BASE].perceptual_roughness = perceptual_roughness;
    li.layers[LAYER_BASE].roughness = roughness;
    li.P = in.world_position.xyz;
    li.V = in.V;
    li.diffuse_color = diffuse_color;
    li.metallic = metallic;
    li.F0_dielectric = 0.16 * reflectance * reflectance;
    li.F0_metallic = base_color.rgb;
    li.F_ab = F_ab;

    let view_z = dot(vec4<f32>(
        view.view_from_world[0].z,
        view.view_from_world[1].z,
        view.view_from_world[2].z,
        view.view_from_world[3].z,
    ), in.world_position);

    // Directional lights (the sun), attenuated by CSM shadow × baked sun-visibility.
    var direct = vec3<f32>(0.0);
    let n_dir = lights.n_directional_lights;
    for (var i = 0u; i < n_dir; i++) {
        var shadow = 1.0;
        if ((in.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u
            && (lights.directional_lights[i].flags & mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
            shadow = shadows::fetch_directional_shadow(i, in.world_position, in.world_normal, view_z, in.frag_coord.xy);
        }
        direct += lighting::directional_light(i, &li, true) * shadow * sun_vis;
    }

    // Indirect: environment map + ambient.
    var indirect = vec3<f32>(0.0);
    let cluster_index = clustering::view_fragment_cluster_index(in.frag_coord.xy, view_z, in.is_orthographic);
    var ranges = clustering::unpack_clusterable_object_index_ranges(cluster_index);
#ifdef ENVIRONMENT_MAP
    let env = environment_map::environment_map_light(&li, &ranges, false);
    indirect += env.diffuse * in.diffuse_occlusion + env.specular * in.specular_occlusion;
#endif
    indirect += ambient::ambient_light(in.world_position, in.N, in.V, NdotV, diffuse_color, F0, perceptual_roughness, in.diffuse_occlusion);

    let emissive = in.material.emissive.rgb * base_color.a;
    let color = view.exposure * (direct + indirect) + emissive;
    return vec4<f32>(color, base_color.a);
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

    let cam_dist = distance(view.world_position, in.world_position.xyz);
    let base_normal = oct_decode(rvt_n.rg);
    // Dominant material id baked into the RVT's (otherwise unused) metallic slot.
    let material_id = u32(clamp(rvt_n.a * 4.0, 0.0, 3.0));

    // Per-material near-range detail overlay: relief + grain + micro roughness,
    // faded with distance — close-up surface the RVT's density can't hold.
    let detail_fade = 1.0 - smoothstep(detail.near, detail.far, cam_dist);
    let dtile = in.world_position.xz / detail.tiling;
    let dn = (textureSample(detail_normal_array, detail_normal_sampler, dtile, material_id).xyz * 2.0 - 1.0)
        * vec3<f32>(-1.0, 1.0, 1.0);
    let dn_scaled = vec3<f32>(dn.xy * detail.normal_strength * detail_fade, dn.z);
    let ref_axis = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(base_normal.z) > 0.99);
    let dt = normalize(cross(ref_axis, base_normal));
    let db = cross(base_normal, dt);
    in_modified.world_normal = normalize(dt * dn_scaled.x + db * dn_scaled.y + base_normal * dn_scaled.z);

    var pbr_input = pbr_input_from_standard_material(in_modified, is_front);

    var albedo = rvt_a.rgb;
    // Per-material detail albedo grain (faded).
    let da = textureSample(detail_albedo_array, detail_albedo_sampler, dtile, material_id).rgb;
    albedo *= mix(vec3<f32>(1.0), 2.0 * da, detail.albedo_strength * detail_fade);

    // Per-material detail ORM: micro roughness + occlusion, faded. (Base AO is
    // gone — rvt_a.a now holds baked sun-visibility.)
    let dorm = textureSample(detail_orm_array, detail_orm_sampler, dtile, material_id);
    let rough = mix(rvt_n.b, dorm.g, detail_fade);
    let ao = mix(1.0, dorm.r, detail_fade);

    pbr_input.material.base_color = vec4<f32>(albedo, 1.0);
    pbr_input.material.perceptual_roughness = rough;
    pbr_input.material.metallic = 0.0;
    pbr_input.diffuse_occlusion = vec3<f32>(ao);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in_modified, pbr_input);
#else
    var out: FragmentOutput;
    out.color = terrain_apply_lighting(pbr_input, rvt_a.a);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}
