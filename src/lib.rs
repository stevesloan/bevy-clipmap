use std::f32::consts::{FRAC_PI_2, PI};

use bevy::{
    asset::{AssetPath, embedded_asset, embedded_path},
    camera::{
        RenderTarget, ScalingMode,
        primitives::Aabb,
        visibility::{NoAutoAabb, RenderLayers},
    },
    core_pipeline::tonemapping::Tonemapping,
    ecs::system::SystemParam,
    light::NotShadowCaster,
    pbr::{ExtendedMaterial, Material, MaterialExtension},
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{AsBindGroup, ShaderType, TextureFormat, TextureUsages},
    },
    shader::{ShaderRef, load_shader_library},
};

mod height_fog;
mod mesh;
mod mesh_fog;
mod texture;
use mesh::{ClipmapPart, ClipmapParts, build_clipmap_parts};
pub use height_fog::{HeightFog, HeightFogParams, HeightFogPlugin};
pub use mesh_fog::HeightFogExtension;
pub use texture::{build_terrain_array, load_terrain_array};

/// Render layers isolating the RVT bake cameras/quads from the main view.
const RVT_ALBEDO_LAYER: usize = 1;
const RVT_NORMAL_LAYER: usize = 2;
/// Macro AO + bent normal + cavity gather target (bake mode 2).
const RVT_AO_LAYER: usize = 3;
/// Layer offset for the tiny sentinel targets that detect bake readiness. Must
/// exceed the number of bake targets so sentinel layers can't collide with a
/// target's base layer.
const RVT_SENTINEL_LAYER_OFFSET: usize = 3;

pub struct ClipmapPlugin;

impl Plugin for ClipmapPlugin {
    fn build(&self, app: &mut App) {
        // Shared fog math, imported by terrain.wgsl (inline VR fog) and the
        // height_fog.wgsl post-process (flatscreen fog).
        load_shader_library!(app, "fog_functions.wgsl");
        embedded_asset!(app, "terrain.wgsl");
        embedded_asset!(app, "bake.wgsl");
        embedded_asset!(app, "mesh_fog.wgsl");

        app.add_plugins(MaterialPlugin::<
            ExtendedMaterial<StandardMaterial, GridMaterial>,
        >::default())
            .add_plugins(MaterialPlugin::<BakeMaterial>::default())
            .add_plugins(MaterialPlugin::<
                ExtendedMaterial<StandardMaterial, HeightFogExtension>,
            >::default())
            .init_resource::<TerrainFog>()
            .init_resource::<TerrainQuality>()
            .init_resource::<InlineFog>()
            .add_systems(PreUpdate, (init_clipmaps, init_grids))
            .add_systems(
                Update,
                (
                    update_grids,
                    init_rvt,
                    drive_rvt_bake,
                    apply_terrain_quality,
                    fog_new_mesh_materials,
                ),
            );

        // Demo A/B keybinds for the AO/bent-normal experiment (B/N/V). Off by
        // default so the library ships no input systems; enable `dev-controls`.
        #[cfg(feature = "dev-controls")]
        app.add_systems(Update, (debug_cycle_view, toggle_ao, toggle_bent));
    }
}

/// Maximum number of terrain material layers, blended per pixel by their
/// procedural slope/height weights (§3.2). Bounded by the `Vec4` lanes in
/// [`TerrainParams`]; widen those to raise it.
pub const MAX_TERRAIN_LAYERS: usize = 4;

/// Slope-angle band a layer occupies (degrees from horizontal), e.g. grass on
/// flat ground, dirt on mid slopes, rock on cliffs. The layer's weight ramps in
/// over `blend_deg` above `min_deg` and out over `blend_deg` below `max_deg`.
/// Use `min_deg = 0` for "no lower bound" and `max_deg = 90` for "up to vertical".
#[derive(Clone, Debug)]
pub struct SlopeRule {
    pub min_deg: f32,
    pub max_deg: f32,
    /// Angular range (degrees) over which the layer blends in/out at each edge.
    pub blend_deg: f32,
}

/// World-height band a layer occupies (meters), e.g. snow above a snowline. The
/// weight ramps in over `blend` above `min` and out over `blend` below `max`.
/// Use a very negative `min` / very large `max` for an open-ended band.
#[derive(Clone, Debug)]
pub struct HeightRule {
    pub min: f32,
    pub max: f32,
    pub blend: f32,
}

/// A single tiling material layer in a [`Clipmap`]'s splat set. Placement is
/// procedural: a layer appears where its optional [`SlopeRule`] and [`HeightRule`]
/// bands overlap (a layer with neither is present everywhere). No control map.
#[derive(Clone, Debug)]
pub struct TerrainLayer {
    /// World-space size of one texture tile, in meters.
    pub tiling_scale: f32,
    /// Strength of this layer's height relief in height-based blending.
    /// 0 falls back to weight blending; higher lets the layer's alpha-channel
    /// height dominate transitions.
    pub height_blend: f32,
    /// Detail-normal perturbation strength. 0 disables the normal map, 1 is full.
    pub normal_strength: f32,
    /// Multiplier on the layer's ORM roughness channel.
    pub roughness: f32,
    /// Slope-angle band this layer occupies (none = any slope).
    pub slope: Option<SlopeRule>,
    /// World-height band this layer occupies (none = any height).
    pub height: Option<HeightRule>,
}

/// Per-layer parameters packed for the GPU. `Vec4` lanes index the layers.
#[derive(Clone, Copy, Debug, Default, ShaderType, Reflect)]
struct TerrainParams {
    tiling_scale: Vec4,
    height_blend: Vec4,
    roughness: Vec4,
    normal_strength: Vec4,
    /// Slope band per layer (radians): appears between `slope_min` and `slope_max`,
    /// ramping over `slope_blend` at each edge. No rule = wide-open band (all slopes).
    slope_min: Vec4,
    slope_max: Vec4,
    slope_blend: Vec4,
    /// World-height band per layer (meters): same shape as the slope band.
    height_min: Vec4,
    height_max: Vec4,
    height_range_blend: Vec4,
    layer_count: u32,
}

impl TerrainParams {
    fn from_clipmap(clipmap: &Clipmap) -> Self {
        let mut tiling_scale = [1.0f32; MAX_TERRAIN_LAYERS];
        let mut height_blend = [0.0f32; MAX_TERRAIN_LAYERS];
        let mut roughness = [1.0f32; MAX_TERRAIN_LAYERS];
        let mut normal_strength = [1.0f32; MAX_TERRAIN_LAYERS];
        // No-rule defaults: a band so wide the ramps never fire (weight factor 1).
        let mut slope_min = [-10.0f32; MAX_TERRAIN_LAYERS];
        let mut slope_max = [10.0f32; MAX_TERRAIN_LAYERS];
        let mut slope_blend = [0.01f32; MAX_TERRAIN_LAYERS];
        let mut height_min = [-1.0e9f32; MAX_TERRAIN_LAYERS];
        let mut height_max = [1.0e9f32; MAX_TERRAIN_LAYERS];
        let mut height_range_blend = [1.0f32; MAX_TERRAIN_LAYERS];
        for (i, layer) in clipmap.layers.iter().take(MAX_TERRAIN_LAYERS).enumerate() {
            tiling_scale[i] = layer.tiling_scale.max(1e-3);
            height_blend[i] = layer.height_blend;
            roughness[i] = layer.roughness;
            normal_strength[i] = layer.normal_strength;
            if let Some(slope) = &layer.slope {
                slope_min[i] = slope.min_deg.to_radians();
                slope_max[i] = slope.max_deg.to_radians();
                slope_blend[i] = slope.blend_deg.to_radians().max(1e-3);
            }
            if let Some(h) = &layer.height {
                height_min[i] = h.min;
                height_max[i] = h.max;
                height_range_blend[i] = h.blend.max(1e-3);
            }
        }
        Self {
            tiling_scale: Vec4::from_array(tiling_scale),
            height_blend: Vec4::from_array(height_blend),
            roughness: Vec4::from_array(roughness),
            normal_strength: Vec4::from_array(normal_strength),
            slope_min: Vec4::from_array(slope_min),
            slope_max: Vec4::from_array(slope_max),
            slope_blend: Vec4::from_array(slope_blend),
            height_min: Vec4::from_array(height_min),
            height_max: Vec4::from_array(height_max),
            height_range_blend: Vec4::from_array(height_range_blend),
            layer_count: clipmap.layers.len().min(MAX_TERRAIN_LAYERS) as u32,
        }
    }
}

/// The near-range detail overlay: per-material high-frequency textures blended
/// over the RVT near the camera and faded out with distance, for close-up
/// fidelity the RVT's texel density can't hold.
#[derive(Clone, Debug)]
pub struct DetailConfig {
    /// Per-material detail albedo array (`2d_array`, one slice per layer).
    pub albedo_array: Handle<Image>,
    /// Per-material detail normal array (`2d_array`), for close-up relief.
    pub normal_array: Handle<Image>,
    /// Per-material detail ORM array (`2d_array`), for close-up roughness/AO.
    pub orm_array: Handle<Image>,
    /// World size of one detail tile, in meters.
    pub tiling: f32,
    /// Detail-normal perturbation strength.
    pub normal_strength: f32,
    /// Detail-albedo grain strength.
    pub albedo_strength: f32,
    /// Camera distances (meters) over which the overlay fades out.
    pub near: f32,
    pub far: f32,
}

/// Near-range detail-overlay parameters (packed for the GPU).
#[derive(Clone, Copy, Debug, Default, ShaderType, Reflect)]
struct DetailParams {
    tiling: f32,
    normal_strength: f32,
    albedo_strength: f32,
    /// Camera distances (m) over which the overlay fades out.
    near: f32,
    far: f32,
}

impl DetailParams {
    fn from_config(d: &DetailConfig) -> Self {
        Self {
            tiling: d.tiling.max(1e-3),
            normal_strength: d.normal_strength,
            albedo_strength: d.albedo_strength,
            near: d.near,
            far: d.far,
        }
    }
}

/// The component defining a clipmap.
/// https://hhoppe.com/gpugcm.pdf
#[derive(Component)]
pub struct Clipmap {
    /// Half width of the grid
    /// Stored as half because the full width must be even.
    pub half_width: u32,

    /// Number of LOD levels to generate.
    /// Each next level covers 2x area of previous one.
    pub levels: u32,

    /// Base scale of the LOD square in world units.
    pub base_scale: f32,

    /// Physical size of one texel in meters.
    pub texel_size: f32,

    /// The entity to follow.
    pub target: Entity,

    /// Heightmap texture.
    pub heightmap: Handle<Image>,

    /// Albedo texture array (`2d_array`), one slice per layer. The alpha channel
    /// stores per-texel height, used for height-based blending.
    pub albedo_array: Handle<Image>,

    /// Tangent-space normal-map array (`2d_array`), one slice per layer.
    pub normal_array: Handle<Image>,

    /// ORM array (`2d_array`): R = occlusion, G = roughness, B = metallic.
    pub orm_array: Handle<Image>,

    /// Material layers, placed procedurally by slope/height (up to
    /// [`MAX_TERRAIN_LAYERS`]) — see [`TerrainLayer`].
    pub layers: Vec<TerrainLayer>,

    /// Near-range detail overlay (arrays + tiling/strength/fade).
    pub detail: DetailConfig,

    /// Height bounds.
    pub min: f32,
    pub max: f32,

    /// Enable wireframe.
    pub wireframe: bool,
}

#[derive(Component)]
struct ClipmapGrid {
    level: u32,
    trim: Entity,
}

impl ClipmapGrid {
    fn scale(&self, base_scale: f32) -> f32 {
        base_scale * 2u32.pow(self.level) as f32
    }
}

fn init_clipmaps(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, GridMaterial>>>,
    fog: Res<TerrainFog>,
    quality: Res<TerrainQuality>,
    clipmaps: Query<(Entity, &Clipmap), Added<Clipmap>>,
) {
    // Born with the current inline fog so terrain spawned after startup (when
    // `apply_terrain_quality` won't re-fire) still matches the active tier.
    let initial_fog = inline_fog_params(&fog, quality.fog);
    let size = quality.rvt_size;
    for (entity, clipmap) in clipmaps {
        let parts = build_clipmap_parts(&mut meshes, clipmap.half_width);

        let rvt_albedo = images.add(Image::new_target_texture(
            size,
            size,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        let rvt_normal =
            images.add(Image::new_target_texture(size, size, TextureFormat::Rgba8Unorm, None));
        // Macro AO (R) + bent normal world X/Z (GB) + cavity (A). Linear. When the
        // ambient gather is disabled it's a 4×4 stub — the binding stays valid but
        // the full-size target (and its bake + sample) are skipped.
        let ao_size = if quality.ambient_gather { size } else { 4 };
        let rvt_ao = images.add(Image::new_target_texture(
            ao_size,
            ao_size,
            TextureFormat::Rgba8Unorm,
            None,
        ));

        // Quality bits packed into `flags` alongside the per-material wireframe bit
        // (bit1 = ambient gather, bit2 = single-layer detail).
        let quality_bits =
            ((quality.ambient_gather as u32) << 1) | ((quality.detail_layers <= 1) as u32) << 2;
        // One material per clipmap, shared by every LOD grid (identical across
        // levels). `wireframe` is the only per-material variant.
        let mut make_material = |wireframe: u32| {
            materials.add(ExtendedMaterial {
                base: StandardMaterial::default(),
                extension: GridMaterial {
                    heightmap: clipmap.heightmap.clone(),
                    rvt_albedo: rvt_albedo.clone(),
                    rvt_normal: rvt_normal.clone(),
                    rvt_ao: rvt_ao.clone(),
                    // Macro AO + bent-normal ambient experiment, independently
                    // toggled (B / N). 0.0 = disabled; 1.0 = full effect.
                    ao_strength: 1.0,
                    bent_strength: 1.0,
                    debug_view: 0,
                    fog: initial_fog.clone(),
                    detail_albedo_array: clipmap.detail.albedo_array.clone(),
                    detail_normal_array: clipmap.detail.normal_array.clone(),
                    detail: DetailParams::from_config(&clipmap.detail),
                    detail_orm_array: clipmap.detail.orm_array.clone(),
                    texel_size: clipmap.texel_size,
                    minmax: Vec2::new(clipmap.min, clipmap.max),
                    flags: wireframe | quality_bits,
                },
            })
        };
        let clipmap_materials = ClipmapMaterials {
            solid: make_material(0),
            wireframe: make_material(1),
        };

        commands.entity(entity).insert((
            Transform::default(),
            Visibility::default(),
            clipmap_materials,
            ClipmapRvt {
                albedo: rvt_albedo,
                normal: rvt_normal,
                ao: rvt_ao,
                initialized: false,
                pending_bakes: 0,
                sun_direction: Vec3::ZERO,
            },
            parts,
        ));

        for level in 0..clipmap.levels {
            commands.entity(entity).with_child(ClipmapGrid {
                level,
                trim: Entity::PLACEHOLDER,
            });
        }
    }
}

fn init_grids(
    mut commands: Commands,
    clipmaps: Query<(&Clipmap, &ClipmapParts, &ClipmapMaterials)>,
    mut grids: Query<(Entity, &mut ClipmapGrid, &ChildOf), Added<ClipmapGrid>>,
) {
    for (entity, mut grid, child_of) in &mut grids {
        let (clipmap, parts, mats) = clipmaps.get(child_of.parent()).unwrap();

        let filler_width = 2 - clipmap.half_width as i32 % 2;
        let square_width = (clipmap.half_width as i32 - filler_width) / 2;

        commands.entity(entity).insert((
            Transform::from_scale(Vec3::splat(grid.scale(clipmap.base_scale))),
            Visibility::default(),
        ));

        // Height-corrected AABB for frustum culling (the mesh is flat; the vertex
        // shader displaces it). Constant per level, so set once — not per frame.
        let aabb_scale = 2u32.pow(1 + grid.level) as f32;
        let cy = (clipmap.max + clipmap.min) / aabb_scale;
        let hy = (clipmap.max - clipmap.min) / aabb_scale;
        let fix_aabb = |base: &Aabb| {
            let mut a = *base;
            a.center.y = cy;
            a.half_extents.y = hy;
            a
        };

        // Spawn one clipmap part as a child grid mesh (+ a wireframe overlay when
        // enabled), returning its entity. Shares the clipmap's materials.
        let spawn_part = |commands: &mut Commands, part: &ClipmapPart, transform: Transform| {
            let aabb = fix_aabb(&part.aabb);
            let mut e = commands.spawn((
                Mesh3d(part.handle.clone()),
                MeshMaterial3d(mats.solid.clone()),
                NotShadowCaster,
                transform,
                NoAutoAabb,
                aabb,
                ChildOf(entity),
            ));
            if clipmap.wireframe {
                e.with_child((
                    Mesh3d(part.handle.clone()),
                    MeshMaterial3d(mats.wireframe.clone()),
                    NoAutoAabb,
                    aabb,
                ));
            }
            e.id()
        };

        for xy in 0..4 * 4 {
            let x = xy % 4;
            let y = xy / 4;
            if grid.level != 0 && (x == 1 || x == 2) && (y == 1 || y == 2) {
                continue;
            }
            let offset_x = if x >= 2 { filler_width as f32 } else { 0.0 };
            let offset_y = if y >= 2 { filler_width as f32 } else { 0.0 };
            spawn_part(
                &mut commands,
                &parts.square,
                Transform::from_xyz(
                    (x - 2) as f32 * square_width as f32 + offset_x,
                    0.0,
                    (y - 2) as f32 * square_width as f32 + offset_y,
                ),
            );
        }

        let corner =
            Transform::from_xyz(-2.0 * square_width as f32, 0.0, -2.0 * square_width as f32);
        if grid.level == 0 {
            spawn_part(&mut commands, &parts.center, corner);
        } else {
            spawn_part(&mut commands, &parts.filler, corner);
            spawn_part(
                &mut commands,
                &parts.stitch,
                Transform::from_xyz(-square_width as f32, 0.0, -square_width as f32)
                    .with_scale(Vec3::splat(0.5)),
            );
        }

        grid.trim = spawn_part(&mut commands, &parts.trim, corner);
    }
}

/// Per-frame: snap each LOD grid (and its trim) to the target's toroidal grid.
/// The RVT samples by world position, so nothing per-material updates here.
fn update_grids(
    mut transforms: Query<&mut Transform>,
    clipmaps: Query<&Clipmap>,
    grids: Query<(Entity, &ClipmapGrid, &ChildOf), With<Transform>>,
) {
    for (entity, grid, child_of) in grids {
        // A grid can outlive the entities it references (parent clipmap, its
        // `target`, or `trim`) for a frame during despawn. Skip it rather than panic.
        let Ok(clipmap) = clipmaps.get(child_of.parent()) else {
            continue;
        };
        let filler_width = 2 - clipmap.half_width as i32 % 2;
        let snap_scale = grid.scale(clipmap.base_scale) * filler_width as f32;
        let Ok(target_pos) = transforms.get(clipmap.target).map(|t| t.translation) else {
            continue;
        };
        let snap_factor = (target_pos / snap_scale).floor().as_ivec3().xz();
        let snap_pos = snap_factor.as_vec2() * snap_scale;
        transforms.get_mut(entity).unwrap().translation = snap_pos.extend(0.0).xzy();

        let snap_mod2 = ((snap_factor % 2) + 2) % 2;
        let Ok(mut trim_transform) = transforms.get_mut(grid.trim) else {
            continue;
        };
        trim_transform.translation = {
            let offset_0 = filler_width as f32 - clipmap.half_width as f32;
            let offset_1 = clipmap.half_width as f32;
            Vec3 {
                x: if snap_mod2.x == 0 { offset_0 } else { offset_1 },
                y: 0.0,
                z: if snap_mod2.y == 0 { offset_0 } else { offset_1 },
            }
        };
        trim_transform.rotation = Quat::from_rotation_y(match snap_mod2 {
            IVec2 { x: 0, y: 0 } => 0.0,
            IVec2 { x: 0, y: 1 } => FRAC_PI_2,
            IVec2 { x: 1, y: 0 } => -FRAC_PI_2,
            IVec2 { x: 1, y: 1 } => PI,
            _ => unreachable!(),
        });
    }
}

#[repr(C)]
#[derive(Eq, PartialEq, Hash, Copy, Clone)]
struct WireframeKey {
    wireframe: bool,
}

impl From<&GridMaterial> for WireframeKey {
    fn from(material: &GridMaterial) -> Self {
        Self {
            wireframe: material.flags & 1 != 0,
        }
    }
}

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
#[bind_group_data(WireframeKey)]
struct GridMaterial {
    #[texture(102)]
    #[sampler(103)]
    heightmap: Handle<Image>,
    #[texture(121)]
    #[sampler(122)]
    rvt_albedo: Handle<Image>,
    #[texture(123)]
    #[sampler(124)]
    rvt_normal: Handle<Image>,
    #[texture(132)]
    #[sampler(133)]
    rvt_ao: Handle<Image>,
    /// Macro AO strength (toggled by `toggle_ao`, B key): 0 off, 1 full.
    #[uniform(110)]
    ao_strength: f32,
    /// Bent-normal strength (toggled by `toggle_bent`, N key): 0 off, 1 full.
    #[uniform(113)]
    bent_strength: f32,
    /// Debug channel isolation (cycled by `debug_cycle_view`): 0 lit, 1 macro AO,
    /// 2 bent normal, 3 cavity.
    #[uniform(112)]
    debug_view: u32,
    /// Inline height fog (`Low` tier): `density > 0` fogs in the terrain shader —
    /// free, terrain-only. `disabled()` skips it (`High` uses `HeightFogPlugin`).
    #[uniform(114)]
    fog: HeightFogParams,
    #[texture(125, dimension = "2d_array")]
    #[sampler(126)]
    detail_albedo_array: Handle<Image>,
    #[texture(127, dimension = "2d_array")]
    #[sampler(128)]
    detail_normal_array: Handle<Image>,
    #[uniform(129)]
    detail: DetailParams,
    #[texture(130, dimension = "2d_array")]
    #[sampler(131)]
    detail_orm_array: Handle<Image>,
    #[uniform(108)]
    texel_size: f32,
    #[uniform(109)]
    minmax: Vec2,
    /// Packed flags: bit0 wireframe, bit1 ambient gather, bit2 single-layer detail.
    /// Quality knobs ride here rather than adding uniforms (this material is at the
    /// bind-group binding limit — extra uniforms silently break its pipeline).
    #[uniform(111)]
    flags: u32,
}

impl MaterialExtension for GridMaterial {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("terrain.wgsl")).with_source("embedded"),
        )
    }

    fn deferred_vertex_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("terrain.wgsl")).with_source("embedded"),
        )
    }

    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("terrain.wgsl")).with_source("embedded"),
        )
    }

    fn deferred_fragment_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("terrain.wgsl")).with_source("embedded"),
        )
    }

    fn specialize(
        _: &bevy::pbr::MaterialExtensionPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _: &bevy::mesh::MeshVertexBufferLayoutRef,
        key: bevy::pbr::MaterialExtensionKey<Self>,
    ) -> std::result::Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        if key.bind_group_data.wireframe {
            descriptor.primitive.polygon_mode = bevy::render::render_resource::PolygonMode::Line;
            descriptor.depth_stencil.as_mut().unwrap().bias.slope_scale = 1.0;
        }
        Ok(())
    }
}

/// The clipmap's terrain material, solid + wireframe. Identical across all LOD
/// levels (nothing per-level survives in `GridMaterial`), so it's built once per
/// clipmap and shared by every grid, not rebuilt per level.
#[derive(Component)]
struct ClipmapMaterials {
    solid: Handle<ExtendedMaterial<StandardMaterial, GridMaterial>>,
    wireframe: Handle<ExtendedMaterial<StandardMaterial, GridMaterial>>,
}

/// Marker inserted on a [`Clipmap`] entity once its RVT bake has finished — the
/// terrain's material + self-shadowing are baked and the main pass will render
/// it fully. Callers can gate a loading screen on this so the (one-time) bake
/// pipeline compilation + bake render land before gameplay starts rather than
/// stalling the first live frame.
#[derive(Component)]
pub struct ClipmapReady;

/// Terrain sun-visibility at an arbitrary world point — the CPU counterpart of the
/// self-shadow the RVT bakes for the terrain *surface*.
///
/// The baked RVT channel is a 2D function of world XZ, valid only *on* the surface,
/// so it's wrong for a point at altitude. This marches the heightmap from the given
/// 3D point toward the fixed sun instead, correct at any height — the query flying
/// characters need (design doc §4.3).
///
/// O(points marched), not per-pixel: call it **once per entity**, never per
/// fragment; throttle or cache for more headroom.
///
/// ```no_run
/// # use bevy::prelude::*;
/// # use bevy_clipmap::SunVisibility;
/// fn shade_flyers(sun: SunVisibility, flyers: Query<&GlobalTransform>) {
///     for xf in &flyers {
///         if let Some(vis) = sun.sample(xf.translation()) {
///             // vis: 1 = full sun, 0 = fully shadowed by terrain.
///         }
///     }
/// }
/// ```
#[derive(SystemParam)]
pub struct SunVisibility<'w, 's> {
    clipmaps: Query<'w, 's, (&'static Clipmap, &'static ClipmapRvt)>,
    images: Res<'w, Assets<Image>>,
}

impl SunVisibility<'_, '_> {
    /// Sun visibility at `world_pos`: `1.0` = full sun, `0.0` = fully shadowed,
    /// soft penumbra between. Marches from `world_pos` itself, so pass a point above
    /// the surface (an entity's position) — an on-surface point reads a self-shadow.
    ///
    /// `None` if the point is outside every clipmap, the bake hasn't initialized, or
    /// the heightmap isn't CPU-resident (needs the default `MAIN_WORLD` asset usage).
    pub fn sample(&self, world_pos: Vec3) -> Option<f32> {
        for (clipmap, rvt) in &self.clipmaps {
            if !rvt.initialized {
                continue;
            }
            let Some(image) = self.images.get(&clipmap.heightmap) else {
                continue;
            };
            let Some(field) = Heightfield::new(image, clipmap.texel_size, clipmap.min, clipmap.max)
            else {
                continue;
            };
            if !field.contains(world_pos) {
                continue;
            }
            return Some(field.sun_visibility(world_pos, rvt.sun_direction));
        }
        None
    }
}

/// CPU view over a clipmap heightmap. Mirrors `bake.wgsl`'s `terrain_height` /
/// `sun_visibility` so the CPU march agrees with the GPU bake — keep the two in sync.
struct Heightfield<'a> {
    texels: &'a [u8],
    width: usize,
    height: usize,
    texel_size: f32,
    min: f32,
    max: f32,
}

impl<'a> Heightfield<'a> {
    fn new(image: &'a Image, texel_size: f32, min: f32, max: f32) -> Option<Self> {
        // Single-channel 16-bit heightmap (see `convert/clipmap.py`); other formats
        // aren't decoded — the query reports `None`.
        if image.texture_descriptor.format != TextureFormat::R16Unorm {
            return None;
        }
        Some(Self {
            texels: image.data.as_deref()?,
            width: image.width() as usize,
            height: image.height() as usize,
            texel_size,
            min,
            max,
        })
    }

    /// Half the world extent on each axis; the world is centered on the origin.
    fn half_extent(&self) -> Vec2 {
        Vec2::new(self.width as f32, self.height as f32) * self.texel_size * 0.5
    }

    fn contains(&self, p: Vec3) -> bool {
        let h = self.half_extent();
        p.x >= -h.x && p.x <= h.x && p.z >= -h.y && p.z <= h.y
    }

    /// One texel's normalized height (0..1), edge-clamped like the shader's
    /// `clamp(p0, 0, hi)`.
    fn texel(&self, x: i64, y: i64) -> f32 {
        let x = x.clamp(0, self.width as i64 - 1) as usize;
        let y = y.clamp(0, self.height as i64 - 1) as usize;
        let i = (y * self.width + x) * 2;
        u16::from_le_bytes([self.texels[i], self.texels[i + 1]]) as f32 / 65535.0
    }

    /// World-space terrain height at `xz` — bilinear, matching `terrain_height`.
    fn height(&self, xz: Vec2) -> f32 {
        let uv = xz / (Vec2::new(self.width as f32, self.height as f32) * self.texel_size) + 0.5;
        let pos = uv * Vec2::new(self.width as f32, self.height as f32);
        let base = pos.floor();
        let f = pos - base;
        let (x0, y0) = (base.x as i64, base.y as i64);
        let h00 = self.texel(x0, y0);
        let h10 = self.texel(x0 + 1, y0);
        let h01 = self.texel(x0, y0 + 1);
        let h11 = self.texel(x0 + 1, y0 + 1);
        let h = (h00 * (1.0 - f.x) + h10 * f.x) * (1.0 - f.y)
            + (h01 * (1.0 - f.x) + h11 * f.x) * f.y;
        h * (self.max - self.min) + self.min
    }

    /// Soft-march toward the sun from `origin` — the CPU twin of `bake.wgsl`
    /// `sun_visibility`, minus its surface normal bias (`origin` is a real 3D point).
    fn sun_visibility(&self, origin: Vec3, sun_direction: Vec3) -> f32 {
        const STEPS: u32 = 96;
        const MAX_DIST: f32 = 6000.0;
        const SOFTNESS: f32 = 10.0;
        const STEP0: f32 = 3.0;
        const GROWTH: f32 = 1.12;
        let mut vis = 1.0f32;
        let mut step = STEP0;
        let mut t = STEP0;
        for _ in 0..STEPS {
            if t > MAX_DIST {
                break;
            }
            let p = origin + sun_direction * t;
            if p.y > self.max {
                break; // above the highest terrain -> can't be occluded
            }
            let clearance = p.y - self.height(p.xz());
            vis = vis.min((SOFTNESS * clearance / t).clamp(0.0, 1.0));
            if vis <= 0.001 {
                break;
            }
            step *= GROWTH;
            t += step;
        }
        vis
    }
}

/// RVT (runtime virtual texture) state for a clipmap: the baked material texture
/// the main pass samples instead of blending the splat per-fragment.
#[derive(Component)]
struct ClipmapRvt {
    albedo: Handle<Image>,
    normal: Handle<Image>,
    ao: Handle<Image>,
    initialized: bool,
    /// Bake targets not yet finished. Set when the bake cameras spawn; each
    /// decrements as it completes, and [`ClipmapReady`] is inserted at zero.
    pending_bakes: u32,
    /// Sun direction the shadow was baked against (from the scene `DirectionalLight`,
    /// resolved in `init_rvt`), so [`SunVisibility`] marches toward the same sun.
    /// `ZERO` until `initialized`.
    sun_direction: Vec3,
}

/// A bake camera stays inactive until its sentinel readback proves the bake
/// pipeline is compiled and source textures are on the GPU (`ready`), then
/// renders a few frames (the full-coverage RVT is static) and deactivates.
/// Pipeline compilation and asset upload take a machine-dependent number of
/// frames; a camera that renders before they finish produces an empty target.
#[derive(Component)]
struct RvtBakeCamera {
    ready: bool,
    frames: u32,
    /// The clipmap entity this camera bakes for, so completion can be tallied.
    clipmap: Entity,
}

/// Active frames rendered once ready; > 1 only as safety margin.
const RVT_BAKE_FRAMES: u32 = 2;

fn drive_rvt_bake(
    mut commands: Commands,
    mut cameras: Query<(&mut Camera, &mut RvtBakeCamera)>,
    mut rvts: Query<&mut ClipmapRvt>,
) {
    for (mut camera, mut state) in &mut cameras {
        if !state.ready {
            continue;
        }
        if state.frames > 0 {
            camera.is_active = true;
            state.frames -= 1;
        } else if camera.is_active {
            camera.is_active = false;
            // This target is baked. When the clipmap's last one finishes, mark
            // it ready. Guarded by `is_active`, so this fires exactly once per
            // camera.
            if let Ok(mut rvt) = rvts.get_mut(state.clipmap) {
                rvt.pending_bakes = rvt.pending_bakes.saturating_sub(1);
                if rvt.pending_bakes == 0 {
                    commands.entity(state.clipmap).insert(ClipmapReady);
                }
            }
        }
    }
}

/// Press V to cycle the terrain debug view: lit → macro AO → bent normal →
/// cavity → lit. Renders the raw baked RVT-AO channel unlit so it reads as a
/// literal value. Experiment-only inspection aid for the AO/bent-normal bake.
#[cfg(feature = "dev-controls")]
fn debug_cycle_view(
    keys: Res<ButtonInput<KeyCode>>,
    mut materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, GridMaterial>>>,
    clipmaps: Query<&Clipmap>,
    mut commands: Commands,
) {
    if !keys.just_pressed(KeyCode::KeyV) {
        return;
    }
    let mut next = 0u32;
    let mut computed = false;
    for (_, material) in materials.iter_mut() {
        if !computed {
            next = (material.extension.debug_view + 1) % 4;
            computed = true;
        }
        material.extension.debug_view = next;
    }
    // The debug channels output raw 0..1 values; bypass the filmic tonemapper
    // while one is active so they read faithfully (AO ~0.9 shows near-white, not
    // gray-compressed). Restore the default tonemapper for the lit view.
    let tonemapping = if next == 0 {
        Tonemapping::default()
    } else {
        Tonemapping::None
    };
    for clipmap in &clipmaps {
        commands.entity(clipmap.target).insert(tonemapping);
    }
    let name = match next {
        1 => "macro AO",
        2 => "bent normal",
        3 => "cavity",
        _ => "off (lit terrain)",
    };
    info!("terrain debug view: {name}");
}

/// Press B to toggle the macro AO on/off in the lit render (flips `ao_strength`
/// 1 ↔ 0), so its contribution can be A/B'd on its own.
#[cfg(feature = "dev-controls")]
fn toggle_ao(
    keys: Res<ButtonInput<KeyCode>>,
    mut materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, GridMaterial>>>,
) {
    if !keys.just_pressed(KeyCode::KeyB) {
        return;
    }
    let mut next = 1.0f32;
    let mut computed = false;
    for (_, material) in materials.iter_mut() {
        if !computed {
            next = if material.extension.ao_strength > 0.5 {
                0.0
            } else {
                1.0
            };
            computed = true;
        }
        material.extension.ao_strength = next;
    }
    info!(
        "terrain macro AO: {}",
        if next > 0.5 { "on" } else { "off" }
    );
}

/// Press N to toggle the bent-normal ambient direction on/off (flips
/// `bent_strength` 1 ↔ 0), independently of the macro AO.
#[cfg(feature = "dev-controls")]
fn toggle_bent(
    keys: Res<ButtonInput<KeyCode>>,
    mut materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, GridMaterial>>>,
) {
    if !keys.just_pressed(KeyCode::KeyN) {
        return;
    }
    let mut next = 1.0f32;
    let mut computed = false;
    for (_, material) in materials.iter_mut() {
        if !computed {
            next = if material.extension.bent_strength > 0.5 {
                0.0
            } else {
                1.0
            };
            computed = true;
        }
        material.extension.bent_strength = next;
    }
    info!(
        "terrain bent-normal ambient: {}",
        if next > 0.5 { "on" } else { "off" }
    );
}

/// Authored height-fog parameters — how the fog *looks* (an art knob, separate
/// from the [`TerrainQuality`] performance profile). The crate realizes these
/// inline in the terrain (`FogTier::Low`) or via the fullscreen [`HeightFogPlugin`]
/// post-process (`FogTier::High`); both paths stay in sync. `HeightFog::default()
/// .density > 0`, so fog is on by default — set `density: 0.0` to disable.
#[derive(Resource, Clone, Default)]
pub struct TerrainFog(pub HeightFog);

/// How the fog is rendered (a field of [`TerrainQuality`]). Live-switchable —
/// both paths are always compiled.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FogTier {
    /// Fullscreen fog post-process (fogs the sky too, no seam) at the cost of one
    /// framebuffer pass. Forces MSAA **off** — that pass samples single-sample
    /// depth. Desktop budget.
    #[default]
    High,
    /// Inline terrain fog (virtually free, terrain-only, no extra pass) + MSAA 4×.
    /// Tiled GPUs punish fullscreen passes, and standalone VR wants stable MSAA — so
    /// this trades sky-fog for AA.
    Low,
}

/// Terrain performance profile — set **once at startup** from device detection (a
/// standalone headset → [`LOW`](TerrainQuality::LOW), desktop → [`HIGH`]
/// (TerrainQuality::HIGH)). Use a preset or hand-tune. Only [`fog`](Self::fog)
/// applies live; the bake-time fields
/// (`rvt_size`, `ambient_gather`, `detail_layers`) are read when a clipmap bakes —
/// changing them after has no effect (it would need a rebake).
#[derive(Resource, Clone, Copy, Debug)]
pub struct TerrainQuality {
    /// Fog method + MSAA (live-switchable). See [`FogTier`].
    pub fog: FogTier,
    /// RVT bake resolution (square). The dominant VRAM cost — three targets of
    /// `size²·4` bytes each (8192² ≈ 768 MB total; 4096² ≈ 192 MB).
    pub rvt_size: u32,
    /// Bake + sample the macro-AO / bent-normal / cavity channel. Off drops a
    /// whole RVT target (VRAM + a slow bake gather) and a per-fragment sample; the
    /// effect is subtle on open terrain, so it's the first thing to cut for VR.
    pub ambient_gather: bool,
    /// Near-detail overlay: blend the top `1` (cheapest, ~3 fewer samples) or `2`
    /// (smoothest boundaries) materials per fragment.
    pub detail_layers: u8,
}

impl TerrainQuality {
    /// Cheapest — inline fog + MSAA, 2048² RVT, no ambient gather, single-layer
    /// detail. Standalone VR / low-end.
    pub const LOW: Self = Self {
        fog: FogTier::Low,
        rvt_size: 2048,
        ambient_gather: false,
        detail_layers: 1,
    };
    /// Middle — fullscreen fog, 4096² RVT, ambient gather, single-layer detail.
    pub const MEDIUM: Self = Self {
        fog: FogTier::High,
        rvt_size: 4096,
        ambient_gather: true,
        detail_layers: 1,
    };
    /// Best — fullscreen fog, 8192² RVT, ambient gather, top-2 detail. Desktop.
    pub const HIGH: Self = Self {
        fog: FogTier::High,
        rvt_size: 8192,
        ambient_gather: true,
        detail_layers: 2,
    };
}

impl Default for TerrainQuality {
    fn default() -> Self {
        Self::HIGH
    }
}

/// The active tier's inline fog params — apply these to fog **your own** opaque
/// materials on fast per-material uniforms (no shared buffer / SSBO cost on tiled
/// VR GPUs). The crate's terrain + [`HeightFogExtension`] use it automatically.
///
/// - **Opaque** (buildings, characters): `#import bevy_clipmap::fog_functions`,
///   embed a `#[uniform(N)] HeightFogParams`, copy this in on `.is_changed()`.
///   It's density-0 on `High` (the fullscreen pass fogs opaques there), so it's
///   correct on both tiers.
/// - **Transparent** (particles, explosions): the fullscreen pass can't fog them,
///   so fog on *both* tiers from [`TerrainFog`] (`HeightFogParams::from(&fog.0)`).
#[derive(Resource, Clone, Default)]
pub struct InlineFog(pub HeightFogParams);

/// Inline-terrain fog params for the active tier: the authored fog with density
/// gated to 0 outside the `Low` tier (`High` uses the fullscreen pass instead).
fn inline_fog_params(fog: &TerrainFog, tier: FogTier) -> HeightFogParams {
    let density = if tier == FogTier::Low {
        fog.0.density
    } else {
        0.0
    };
    HeightFogParams::from(&fog.0).with_density(density)
}

/// Realizes [`TerrainFog`] across both fog paths for the active [`TerrainQuality::fog`]
/// tier whenever either changes: writes the inline params into every terrain
/// material, and (if the target camera has a [`HeightFog`], i.e. the fullscreen
/// path is installed) drives its density and the camera MSAA to match the tier.
fn apply_terrain_quality(
    quality: Res<TerrainQuality>,
    fog: Res<TerrainFog>,
    mut inline_fog: ResMut<InlineFog>,
    mut materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, GridMaterial>>>,
    mut mesh_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, HeightFogExtension>>>,
    clipmaps: Query<&Clipmap>,
    mut cameras: Query<(&mut Msaa, &mut HeightFog)>,
) {
    if !quality.is_changed() && !fog.is_changed() {
        return;
    }
    let low = quality.fog == FogTier::Low;
    let inline = inline_fog_params(&fog, quality.fog);
    // Publish for the game to fog its own materials (change-detected).
    inline_fog.0 = inline.clone();
    for (_, material) in materials.iter_mut() {
        material.extension.fog = inline.clone();
    }
    // Meshes (characters/props) using HeightFogExtension get the same inline fog,
    // so they don't render as unfogged cutouts on the Low tier.
    for (_, material) in mesh_materials.iter_mut() {
        material.extension.fog = inline.clone();
    }
    for clipmap in &clipmaps {
        if let Ok((mut msaa, mut camera_fog)) = cameras.get_mut(clipmap.target) {
            *camera_fog = fog.0.clone();
            camera_fog.density = if low { 0.0 } else { fog.0.density };
            *msaa = if low { Msaa::Sample4 } else { Msaa::Off };
        }
    }
}

/// Fog mesh materials the moment they're created, so a character/prop spawned at
/// runtime picks up the tier's fog immediately (else an unfogged cutout on `Low`
/// until the next tier change). Touches only new materials — free at steady state.
fn fog_new_mesh_materials(
    mut events: MessageReader<AssetEvent<ExtendedMaterial<StandardMaterial, HeightFogExtension>>>,
    quality: Res<TerrainQuality>,
    fog: Res<TerrainFog>,
    mut mesh_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, HeightFogExtension>>>,
) {
    let inline = inline_fog_params(&fog, quality.fog);
    for event in events.read() {
        if let AssetEvent::Added { id } = event {
            if let Some(mut material) = mesh_materials.get_mut(*id) {
                material.extension.fog = inline.clone();
            }
        }
    }
}

/// Material that bakes the terrain splat into the RVT. Runs the same source data
/// as `GridMaterial` but outputs raw channels unlit; see `bake.wgsl`.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
struct BakeMaterial {
    #[texture(0)]
    #[sampler(1)]
    heightmap: Handle<Image>,
    #[uniform(2)]
    texel_size: f32,
    #[uniform(3)]
    minmax: Vec2,
    #[texture(4, dimension = "2d_array")]
    #[sampler(5)]
    albedo_array: Handle<Image>,
    #[uniform(8)]
    params: TerrainParams,
    #[texture(9, dimension = "2d_array")]
    #[sampler(10)]
    normal_array: Handle<Image>,
    #[texture(11, dimension = "2d_array")]
    #[sampler(12)]
    orm_array: Handle<Image>,
    #[uniform(13)]
    output_mode: u32,
    #[uniform(14)]
    sun_direction: Vec3,
}

impl Material for BakeMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("bake.wgsl")).with_source("embedded"),
        )
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}

/// Once the heightmap is loaded, spawn the top-down bake camera and quad that
/// render the splat into the clipmap's RVT texture (full-terrain coverage).
fn init_rvt(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut bake_materials: ResMut<Assets<BakeMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut clipmaps: Query<(Entity, &Clipmap, &mut ClipmapRvt)>,
    suns: Query<&GlobalTransform, With<DirectionalLight>>,
    quality: Res<TerrainQuality>,
) {
    for (clipmap_entity, clipmap, mut rvt) in &mut clipmaps {
        if rvt.initialized {
            continue;
        }
        let Some(heightmap) = images.get(&clipmap.heightmap) else {
            continue;
        };
        let Some(sun_direction) = suns.iter().next().map(|t| t.back().as_vec3()) else {
            continue;
        };
        let world_size = clipmap.texel_size * heightmap.texture_descriptor.size.width as f32;
        rvt.initialized = true;
        rvt.sun_direction = sun_direction;
        // Albedo + normal/ORM always; the AO/bent-normal/cavity target only when
        // the ambient gather is enabled (see `TerrainQuality`). All must finish
        // before the clipmap is marked ready — kept in sync with the loop below.
        rvt.pending_bakes = if quality.ambient_gather { 3 } else { 2 };

        let mut make_bake = |mode: u32| {
            bake_materials.add(BakeMaterial {
                heightmap: clipmap.heightmap.clone(),
                texel_size: clipmap.texel_size,
                minmax: Vec2::new(clipmap.min, clipmap.max),
                albedo_array: clipmap.albedo_array.clone(),
                params: TerrainParams::from_clipmap(clipmap),
                normal_array: clipmap.normal_array.clone(),
                orm_array: clipmap.orm_array.clone(),
                output_mode: mode,
                sun_direction,
            })
        };
        let quad = meshes.add(Plane3d::default().mesh().size(world_size, world_size));

        // Shared by the real bake camera and its sentinel; identical projection
        // keeps the two renders interchangeable.
        let projection = || {
            Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::Fixed {
                    width: world_size,
                    height: world_size,
                },
                near: 0.0,
                far: 20000.0,
                ..OrthographicProjection::default_3d()
            })
        };
        // Up is -Z so the image's texel layout matches the main pass's
        // `world_xz / world_size + 0.5` sampling (u -> +X, v -> +Z).
        let camera_transform =
            Transform::from_xyz(0.0, 10000.0, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z);

        // Two bake targets: albedo (mode 0) and normal/ORM (mode 1). Bevy's
        // camera-to-image is single-target, so each is its own quad + camera on
        // its own render layer, rendered before the main view.
        //
        // Each bake camera starts inactive behind a sentinel: the same material
        // rendered to a tiny readback target. The first non-black readback
        // proves the bake pipeline is compiled and the source textures are on
        // the GPU — only then does the expensive full-res bake render (frame
        // counts are machine-dependent and get it wrong either way).
        let mut targets = vec![
            (
                0u32,
                rvt.albedo.clone(),
                TextureFormat::Rgba8UnormSrgb,
                RVT_ALBEDO_LAYER,
                -3isize,
            ),
            (
                1u32,
                rvt.normal.clone(),
                TextureFormat::Rgba8Unorm,
                RVT_NORMAL_LAYER,
                -2isize,
            ),
        ];
        if quality.ambient_gather {
            targets.push((
                2u32,
                rvt.ao.clone(),
                TextureFormat::Rgba8Unorm,
                RVT_AO_LAYER,
                -1isize,
            ));
        }
        for (mode, target, format, layer, order) in targets {
            let material = make_bake(mode);
            let sentinel_layer = layer + RVT_SENTINEL_LAYER_OFFSET;

            commands.spawn((
                Mesh3d(quad.clone()),
                MeshMaterial3d(material.clone()),
                Transform::default(),
                RenderLayers::layer(layer),
            ));
            let bake_camera = commands
                .spawn((
                    Camera3d::default(),
                    Camera {
                        order,
                        clear_color: Color::BLACK.into(),
                        // Inactive until the sentinel readback flips `ready`.
                        is_active: false,
                        ..default()
                    },
                    RenderTarget::Image(target.into()),
                    projection(),
                    Tonemapping::None,
                    Msaa::Off,
                    camera_transform,
                    RenderLayers::layer(layer),
                    RvtBakeCamera {
                        ready: false,
                        frames: RVT_BAKE_FRAMES,
                        clipmap: clipmap_entity,
                    },
                ))
                .id();

            // Sentinel: same mesh/material/format, 4x4 target read back each
            // frame. Must match the bake's pipeline key (target format, MSAA,
            // tonemapping) so its first successful draw implies the bake's
            // pipeline is ready too.
            let mut sentinel_image = Image::new_target_texture(4, 4, format, None);
            sentinel_image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
            let sentinel_target = images.add(sentinel_image);
            let sentinel_quad = commands
                .spawn((
                    Mesh3d(quad.clone()),
                    MeshMaterial3d(material),
                    Transform::default(),
                    RenderLayers::layer(sentinel_layer),
                ))
                .id();
            let sentinel_camera = commands
                .spawn((
                    Camera3d::default(),
                    Camera {
                        // Well below every bake camera's order so sentinel and
                        // bake orders never tie (all still render before main).
                        order: order - 10,
                        clear_color: Color::BLACK.into(),
                        ..default()
                    },
                    RenderTarget::Image(sentinel_target.clone().into()),
                    projection(),
                    Tonemapping::None,
                    Msaa::Off,
                    camera_transform,
                    RenderLayers::layer(sentinel_layer),
                ))
                .id();
            let readback = commands.spawn(Readback::texture(sentinel_target)).id();
            commands.entity(readback).observe(
                move |event: On<ReadbackComplete>,
                      mut cameras: Query<&mut RvtBakeCamera>,
                      mut commands: Commands| {
                    // Still clear color -> not rendered yet; keep polling.
                    let baked = event
                        .event()
                        .data
                        .chunks(4)
                        .any(|px| px[0] != 0 || px[1] != 0 || px[2] != 0);
                    if !baked {
                        return;
                    }
                    // The readback can fire again before these despawns flush.
                    // `ready` is set synchronously (query writes apply now, unlike
                    // deferred commands), so it latches the teardown to run once —
                    // otherwise the second fire re-despawns and warns.
                    let Ok(mut camera) = cameras.get_mut(bake_camera) else {
                        return;
                    };
                    if camera.ready {
                        return;
                    }
                    camera.ready = true;
                    commands.entity(sentinel_camera).despawn();
                    commands.entity(sentinel_quad).despawn();
                    commands.entity(readback).despawn();
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::{Extent3d, TextureDimension};

    /// A 16×16 R16Unorm heightmap; `h(x,y)` gives each texel's raw 16-bit height.
    fn heightmap(h: impl Fn(usize, usize) -> u16) -> Image {
        const N: usize = 16;
        let mut data = Vec::with_capacity(N * N * 2);
        for y in 0..N {
            for x in 0..N {
                data.extend_from_slice(&h(x, y).to_le_bytes());
            }
        }
        Image::new(
            Extent3d {
                width: N as u32,
                height: N as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data,
            TextureFormat::R16Unorm,
            RenderAssetUsages::MAIN_WORLD,
        )
    }

    #[test]
    fn height_decodes_and_centers_on_origin() {
        // Wall (max height) on the +X columns, flat (0) elsewhere. min..max = 0..100.
        let img = heightmap(|x, _| if x >= 12 { u16::MAX } else { 0 });
        let field = Heightfield::new(&img, 1.0, 0.0, 100.0).unwrap();
        // texel_size 1, width 16 -> world spans [-8, 8]; column 13 is world x = 5.
        assert!((field.height(Vec2::new(5.0, 0.0)) - 100.0).abs() < 1e-2);
        assert!(field.height(Vec2::new(-5.0, 0.0)).abs() < 1e-2);
        assert!(field.contains(Vec3::new(7.0, 0.0, 0.0)));
        assert!(!field.contains(Vec3::new(9.0, 0.0, 0.0)));
    }

    #[test]
    fn flat_terrain_is_fully_lit() {
        let img = heightmap(|_, _| 0);
        let field = Heightfield::new(&img, 1.0, 0.0, 100.0).unwrap();
        let sun = Vec3::new(1.0, 0.3, 0.0).normalize();
        // A point above flat ground sees the sun unobstructed.
        assert!((field.sun_visibility(Vec3::new(0.0, 5.0, 0.0), sun) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn tall_wall_casts_shadow_toward_the_sun() {
        // Wall along +X; sun is low in the +X sky, so points on the -X side of the
        // wall are occluded.
        let img = heightmap(|x, _| if x >= 12 { u16::MAX } else { 0 });
        let field = Heightfield::new(&img, 1.0, 0.0, 100.0).unwrap();
        let sun = Vec3::new(1.0, 0.3, 0.0).normalize();
        let shadowed = field.sun_visibility(Vec3::new(-6.0, 2.0, 0.0), sun);
        assert!(shadowed < 0.5, "expected shadow behind the wall, got {shadowed}");
        // Above the wall's height, nothing occludes the same column.
        let lit = field.sun_visibility(Vec3::new(-6.0, 150.0, 0.0), sun);
        assert!((lit - 1.0).abs() < 1e-3, "expected full sun above the wall, got {lit}");
    }
}
