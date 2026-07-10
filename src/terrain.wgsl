#import bevy_pbr::mesh_functions
#import bevy_pbr::pbr_fragment::pbr_input_from_standard_material
#import bevy_pbr::view_transformations::position_world_to_clip
#import bevy_pbr::mesh_view_bindings::{view, lights, clustered_lights}
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
@group(#{MATERIAL_BIND_GROUP}) @binding(108) var<uniform> texel_size: f32;
@group(#{MATERIAL_BIND_GROUP}) @binding(109) var<uniform> minmax: vec2<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(111) var<uniform> wireframe: u32;
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

fn height_bilinear(uv: vec2<f32>, lod: i32) -> f32 {
    let tex_size = vec2<f32>(textureDimensions(heightmap_texture, lod));
    let pos = uv * tex_size;
    let p0 = vec2<i32>(floor(pos));
    let f = pos - floor(pos);

    // Clamp so uv == 1.0 (the world's far edge) doesn't read out of bounds.
    let hi = vec2<i32>(tex_size) - 1;
    let h00 = textureLoad(heightmap_texture, clamp(p0, vec2(0), hi), lod).r;
    let h10 = textureLoad(heightmap_texture, clamp(p0 + vec2(1, 0), vec2(0), hi), lod).r;
    let h01 = textureLoad(heightmap_texture, clamp(p0 + vec2(0, 1), vec2(0), hi), lod).r;
    let h11 = textureLoad(heightmap_texture, clamp(p0 + vec2(1, 1), vec2(0), hi), lod).r;

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
    let height = height_bilinear(clamp(height_uv, vec2(0.0), vec2(1.0)), 0);
    let world_y = height * (minmax.y - minmax.x) + minmax.x;

    // Out past the heightmap coverage the coarse LOD skirt would render as a wall
    // at the edge height. Drop those vertices to the height floor so the stripe
    // stays low and out of sight (the sample above is clamped, not read OOB).
    let in_coverage = all(height_uv >= vec2(0.0)) && all(height_uv <= vec2(1.0));
    out.world_position.y = select(minmax.x, world_y, in_coverage);
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
// Compact fork of `apply_pbr_lighting`: base layer only, directional + clustered
// point/spot lights + ambient + environment map (no clearcoat / transmission /
// anisotropy — the terrain doesn't use them). `sun_vis` (0..1) attenuates only the
// direct sun, so shadowed valleys still receive sky/ambient light; point and spot
// lights are unaffected by it (it's the sun's baked occlusion, not theirs).
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

    // Clusterable lights (point + spot) touching this fragment's cluster. The same
    // `ranges` also feeds the environment-map lookup below.
    let cluster_index = clustering::view_fragment_cluster_index(in.frag_coord.xy, view_z, in.is_orthographic);
    var ranges = clustering::unpack_clusterable_object_index_ranges(cluster_index);

    // Point lights, each attenuated by its own shadow map (if enabled). Not touched
    // by sun_vis — the baked sun occlusion doesn't occlude local lights.
    for (var i = ranges.first_point_light_index_offset; i < ranges.first_spot_light_index_offset; i++) {
        let light_id = clustering::get_clusterable_object_id(i);
        var shadow = 1.0;
        if ((in.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u
            && (clustered_lights.data[light_id].flags & mesh_view_types::POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
            shadow = shadows::fetch_point_shadow(light_id, in.world_position, in.world_normal, in.frag_coord.xy);
        }
        direct += lighting::point_light(light_id, &li, true, true) * shadow;
    }

    // Spot lights, likewise shadowed per-light.
    for (var i = ranges.first_spot_light_index_offset; i < ranges.first_reflection_probe_index_offset; i++) {
        let light_id = clustering::get_clusterable_object_id(i);
        var shadow = 1.0;
        if ((in.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u
            && (clustered_lights.data[light_id].flags & mesh_view_types::POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
            shadow = shadows::fetch_spot_shadow(
                light_id,
                in.world_position,
                in.world_normal,
                clustered_lights.data[light_id].shadow_map_near_z,
                in.frag_coord.xy,
            );
        }
        direct += lighting::spot_light(light_id, &li, true) * shadow;
    }

    // Indirect: environment map + ambient.
    var indirect = vec3<f32>(0.0);
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

    // Sample the baked RVT instead of blending the splat per-fragment:
    // albedo + baked sun-visibility (alpha); octahedral world normal + roughness
    // + packed material ids (alpha, read NEAREST below).
    let rvt_a = textureSample(rvt_albedo_texture, rvt_albedo_sampler, uv);
    let rvt_n = textureSample(rvt_normal_texture, rvt_normal_sampler, uv);

    let cam_dist = distance(view.world_position, in.world_position.xyz);
    let base_normal = oct_decode(rvt_n.rg);
    // Two dominant material ids + their blend, packed (2+2+4 bits) into the RVT's
    // metallic slot. `mblend` (0..0.5) lerps the two materials' detail normals.
    // Read it NEAREST (textureLoad) — the packed byte can't be linearly filtered,
    // or the bilinear sweep through id/weight combos shows as banding strips.
    let rvt_dims = vec2<f32>(textureDimensions(rvt_normal_texture));
    let mid_idx = clamp(vec2<i32>(uv * rvt_dims), vec2(0), vec2<i32>(rvt_dims) - 1);
    let mid = u32(textureLoad(rvt_normal_texture, mid_idx, 0).a * 255.0 + 0.5);
    let id0 = mid & 3u;
    let id1 = (mid >> 2u) & 3u;
    let mblend = f32((mid >> 4u) & 15u) / 15.0 * 0.5;

    // Near-range detail overlay, faded with distance. Skipped entirely past the
    // fade range so far terrain pays none of the detail samples. Derivatives are
    // taken outside the branch so the guarded samples keep correct mip selection.
    let detail_fade = 1.0 - smoothstep(detail.near, detail.far, cam_dist);
    let dtile = in.world_position.xz / detail.tiling;
    let ddx = dpdx(dtile);
    let ddy = dpdy(dtile);

    var world_normal = base_normal;
    var albedo = rvt_a.rgb;
    var rough = rvt_n.b;
    var ao = 1.0;
    if detail_fade > 0.001 {
        // Detail normal: lerp the two dominant materials' detail normals so the
        // relief blends across boundaries instead of snapping. (Flip X — see bake.)
        let dn0 = textureSampleGrad(detail_normal_array, detail_normal_sampler, dtile, id0, ddx, ddy).xyz * 2.0 - 1.0;
        let dn1 = textureSampleGrad(detail_normal_array, detail_normal_sampler, dtile, id1, ddx, ddy).xyz * 2.0 - 1.0;
        let dn = mix(dn0, dn1, mblend) * vec3<f32>(-1.0, 1.0, 1.0);
        let dn_scaled = vec3<f32>(dn.xy * detail.normal_strength * detail_fade, dn.z);
        let ref_axis = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(base_normal.z) > 0.99);
        let dt = normalize(cross(ref_axis, base_normal));
        let db = cross(base_normal, dt);
        world_normal = normalize(dt * dn_scaled.x + db * dn_scaled.y + base_normal * dn_scaled.z);

        // Detail albedo grain + ORM: blend the same two materials as the normal
        // so grain / roughness / AO don't snap at boundaries either.
        let da0 = textureSampleGrad(detail_albedo_array, detail_albedo_sampler, dtile, id0, ddx, ddy).rgb;
        let da1 = textureSampleGrad(detail_albedo_array, detail_albedo_sampler, dtile, id1, ddx, ddy).rgb;
        let da = mix(da0, da1, mblend);
        albedo *= mix(vec3<f32>(1.0), 2.0 * da, detail.albedo_strength * detail_fade);
        let dorm0 = textureSampleGrad(detail_orm_array, detail_orm_sampler, dtile, id0, ddx, ddy);
        let dorm1 = textureSampleGrad(detail_orm_array, detail_orm_sampler, dtile, id1, ddx, ddy);
        let dorm = mix(dorm0, dorm1, mblend);
        rough = mix(rvt_n.b, dorm.g, detail_fade);
        ao = mix(1.0, dorm.r, detail_fade);
    }
    in_modified.world_normal = world_normal;

    var pbr_input = pbr_input_from_standard_material(in_modified, is_front);
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
