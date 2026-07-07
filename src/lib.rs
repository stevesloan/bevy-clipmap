use std::{
    collections::HashMap,
    f32::consts::{FRAC_PI_2, PI},
};

use bevy::{
    asset::{AssetPath, RenderAssetUsages, embedded_asset, embedded_path},
    camera::{
        RenderTarget, ScalingMode,
        primitives::Aabb,
        visibility::{NoAutoAabb, RenderLayers},
    },
    core_pipeline::tonemapping::Tonemapping,
    light::NotShadowCaster,
    mesh::{Indices, PrimitiveTopology},
    pbr::{ExtendedMaterial, Material, MaterialExtension},
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType, TextureFormat},
    shader::ShaderRef,
};

/// Render layers isolating the RVT bake cameras/quads from the main view.
const RVT_ALBEDO_LAYER: usize = 1;
const RVT_NORMAL_LAYER: usize = 2;
/// Resolution of the RVT bake target textures.
const RVT_SIZE: u32 = 4096;

pub struct ClipmapPlugin;

struct ClipmapPart {
    handle: Handle<Mesh>,
    aabb: Aabb,
}

impl ClipmapPart {
    fn build(meshes: &mut ResMut<Assets<Mesh>>, builder: MeshBuilder) -> Self {
        let mut min = Vec3::from_slice(&builder.vertices[0]);
        let mut max = min;
        for v in builder.vertices.iter().map(|v| Vec3::from_slice(v)) {
            min = min.min(v);
            max = max.max(v);
        }
        Self {
            handle: meshes.add(builder.build()),
            aabb: Aabb::from_min_max(min, max),
        }
    }
}

#[derive(Component)]
struct ClipmapParts {
    square: ClipmapPart,
    filler: ClipmapPart,
    center: ClipmapPart,
    trim: ClipmapPart,
    stitch: ClipmapPart,
}

impl Plugin for ClipmapPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "terrain.wgsl");
        embedded_asset!(app, "bake.wgsl");

        app.add_plugins(MaterialPlugin::<
            ExtendedMaterial<StandardMaterial, GridMaterial>,
        >::default())
            .add_plugins(MaterialPlugin::<BakeMaterial>::default())
            .add_systems(PreUpdate, (init_clipmaps, init_grids))
            .add_systems(Update, (update_grids, init_rvt, stop_rvt_bake));
    }
}

struct MeshBuilder {
    unique_vertices: HashMap<(i32, i32), u32>,
    vertices: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    fn new() -> Self {
        Self {
            unique_vertices: HashMap::new(),
            vertices: vec![],
            indices: vec![],
        }
    }

    fn add_vertex(&mut self, x: i32, y: i32) -> u32 {
        if let Some(index) = self.unique_vertices.get(&(x, y)) {
            *index
        } else {
            let index = self.vertices.len() as u32;
            self.vertices.push([x as f32, 0.0, y as f32]);
            self.unique_vertices.insert((x, y), index);
            index
        }
    }

    fn add_triangle(&mut self, x1: i32, y1: i32, x2: i32, y2: i32, x3: i32, y3: i32) {
        let p1 = self.add_vertex(x1, y1);
        let p2 = self.add_vertex(x2, y2);
        let p3 = self.add_vertex(x3, y3);
        self.indices.extend([p1, p2, p3]);
    }

    fn add_square(&mut self, x: i32, y: i32) {
        let p1 = self.add_vertex(x, y);
        let p2 = self.add_vertex(x, y + 1);
        let p3 = self.add_vertex(x + 1, y + 1);
        let p4 = self.add_vertex(x + 1, y);
        self.indices.extend([p1, p2, p3]);
        self.indices.extend([p1, p3, p4]);
    }

    fn build(self) -> Mesh {
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.vertices)
            .with_inserted_indices(Indices::U32(self.indices))
    }
}

/// Maximum number of terrain material layers blended per pixel.
/// One RGBA control map supplies up to this many weights.
pub const MAX_TERRAIN_LAYERS: usize = 4;

/// Rule for automatically placing a layer on steep terrain (e.g. rock on cliffs).
#[derive(Clone, Debug)]
pub struct SlopeRule {
    /// Slope angle (degrees from horizontal) at which the layer starts to appear.
    pub min_deg: f32,
    /// Angular range (degrees) over which the layer blends in.
    pub blend_deg: f32,
}

/// A single tiling material layer in a [`Clipmap`]'s splat set.
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
    /// Optional automatic slope-based placement.
    pub slope: Option<SlopeRule>,
}

/// Per-layer parameters packed for the GPU. `Vec4` lanes index the layers.
#[derive(Clone, Copy, Debug, Default, ShaderType, Reflect)]
struct TerrainParams {
    tiling_scale: Vec4,
    height_blend: Vec4,
    roughness: Vec4,
    normal_strength: Vec4,
    /// Slope-rule onset in radians; a sentinel > PI disables the rule.
    slope_min: Vec4,
    /// Slope-rule blend range in radians.
    slope_blend: Vec4,
    layer_count: u32,
    macro_strength: f32,
    macro_near: f32,
    macro_far: f32,
}

impl TerrainParams {
    fn from_clipmap(clipmap: &Clipmap) -> Self {
        let mut tiling_scale = [1.0f32; MAX_TERRAIN_LAYERS];
        let mut height_blend = [0.0f32; MAX_TERRAIN_LAYERS];
        let mut roughness = [1.0f32; MAX_TERRAIN_LAYERS];
        let mut normal_strength = [1.0f32; MAX_TERRAIN_LAYERS];
        // Sentinel > PI means "no slope rule": the shader's smoothstep never fires.
        let mut slope_min = [10.0f32; MAX_TERRAIN_LAYERS];
        let mut slope_blend = [0.1f32; MAX_TERRAIN_LAYERS];
        for (i, layer) in clipmap.layers.iter().take(MAX_TERRAIN_LAYERS).enumerate() {
            tiling_scale[i] = layer.tiling_scale.max(1e-3);
            height_blend[i] = layer.height_blend;
            roughness[i] = layer.roughness;
            normal_strength[i] = layer.normal_strength;
            if let Some(slope) = &layer.slope {
                slope_min[i] = slope.min_deg.to_radians();
                slope_blend[i] = slope.blend_deg.to_radians().max(1e-3);
            }
        }
        Self {
            tiling_scale: Vec4::from_array(tiling_scale),
            height_blend: Vec4::from_array(height_blend),
            roughness: Vec4::from_array(roughness),
            normal_strength: Vec4::from_array(normal_strength),
            slope_min: Vec4::from_array(slope_min),
            slope_blend: Vec4::from_array(slope_blend),
            layer_count: clipmap.layers.len().min(MAX_TERRAIN_LAYERS) as u32,
            macro_strength: clipmap.macro_strength,
            macro_near: clipmap.macro_near,
            macro_far: clipmap.macro_far,
        }
    }
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
    fn from_clipmap(clipmap: &Clipmap) -> Self {
        Self {
            tiling: clipmap.detail_tiling.max(1e-3),
            normal_strength: clipmap.detail_normal_strength,
            albedo_strength: clipmap.detail_albedo_strength,
            near: clipmap.detail_near,
            far: clipmap.detail_far,
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

    /// Macro variation map, multiplied over the blended splat to add large-scale
    /// color variation and break up tiling (see `macro_strength`).
    pub color: Handle<Image>,

    /// Strength of the macro variation multiply. 0 ignores `color`.
    pub macro_strength: f32,

    /// Camera distances (meters) over which distant terrain blends toward the
    /// macro color map, adding large-scale color variation to vistas.
    pub macro_near: f32,
    pub macro_far: f32,

    /// Heightmap texture.
    pub heightmap: Handle<Image>,

    /// Albedo texture array (`2d_array`), one slice per layer. The alpha channel
    /// stores per-texel height, used for height-based blending.
    pub albedo_array: Handle<Image>,

    /// Tangent-space normal-map array (`2d_array`), one slice per layer.
    pub normal_array: Handle<Image>,

    /// ORM array (`2d_array`): R = occlusion, G = roughness, B = metallic.
    pub orm_array: Handle<Image>,

    /// RGBA control map; each channel is the weight of the matching layer.
    pub control: Handle<Image>,

    /// Material layers blended via the control map (up to [`MAX_TERRAIN_LAYERS`]).
    pub layers: Vec<TerrainLayer>,

    /// Per-material detail albedo array (`2d_array`, one slice per layer), overlaid
    /// near the camera for close-up grain/color.
    pub detail_albedo_array: Handle<Image>,

    /// Per-material detail normal array (`2d_array`), for close-up relief.
    pub detail_normal_array: Handle<Image>,

    /// Per-material detail ORM array (`2d_array`), for close-up roughness/AO.
    pub detail_orm_array: Handle<Image>,

    /// World size of one detail tile, in meters (~0.5–1).
    pub detail_tiling: f32,

    /// Detail-normal perturbation strength.
    pub detail_normal_strength: f32,

    /// Detail-albedo grain strength.
    pub detail_albedo_strength: f32,

    /// Camera distances (meters) over which the detail overlay fades out.
    pub detail_near: f32,
    pub detail_far: f32,

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
    clipmaps: Query<(Entity, &Clipmap), Added<Clipmap>>,
) {
    for (entity, clipmap) in clipmaps {
        let builder_width = clipmap.half_width as i32 * 2;
        let filler_width = 2 - clipmap.half_width as i32 % 2;
        let square_width = (clipmap.half_width as i32 - filler_width) / 2;

        let mut square = MeshBuilder::new();
        let mut filler = MeshBuilder::new();
        let mut center = MeshBuilder::new();
        let mut trim = MeshBuilder::new();
        let mut stitch = MeshBuilder::new();

        for xy in 0..builder_width.pow(2) {
            let x = xy % builder_width;
            let y = xy / builder_width;
            if x < square_width && y < square_width {
                square.add_square(x, y);
            }
            let range = square_width * 2..square_width * 2 + filler_width;
            if (range.contains(&x) || range.contains(&y))
                && x < builder_width - filler_width
                && y < builder_width - filler_width
            {
                center.add_square(x, y);
                let range = square_width..builder_width - square_width - filler_width;
                if !range.contains(&x) || !range.contains(&y) {
                    filler.add_square(x, y);
                }
            }
            if x >= builder_width - filler_width || y >= builder_width - filler_width {
                trim.add_square(x, y);
            }
        }

        for x in 0..builder_width / 2 {
            let x = x * 2;
            stitch.add_triangle(x, 0, x + 1, 0, x + 2, 0);
            stitch.add_triangle(x + 2, builder_width, x + 1, builder_width, x, builder_width);
            stitch.add_triangle(0, x + 2, 0, x + 1, 0, x);
            stitch.add_triangle(builder_width, x, builder_width, x + 1, builder_width, x + 2);
        }

        let rvt_albedo = images.add(Image::new_target_texture(
            RVT_SIZE,
            RVT_SIZE,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        let rvt_normal = images.add(Image::new_target_texture(
            RVT_SIZE,
            RVT_SIZE,
            TextureFormat::Rgba8Unorm,
            None,
        ));

        commands.entity(entity).insert((
            Transform::default(),
            Visibility::default(),
            ClipmapRvt {
                albedo: rvt_albedo,
                normal: rvt_normal,
                initialized: false,
            },
            ClipmapParts {
                square: ClipmapPart::build(&mut meshes, square),
                filler: ClipmapPart::build(&mut meshes, filler),
                center: ClipmapPart::build(&mut meshes, center),
                trim: ClipmapPart::build(&mut meshes, trim),
                stitch: ClipmapPart::build(&mut meshes, stitch),
            },
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
    mut materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, GridMaterial>>>,
    clipmaps: Query<(&Clipmap, &ClipmapParts, &ClipmapRvt)>,
    mut grids: Query<(Entity, &mut ClipmapGrid, &ChildOf), Added<ClipmapGrid>>,
) {
    for (entity, mut grid, clipmap) in &mut grids {
        let (clipmap, parts, rvt) = clipmaps.get(clipmap.parent()).unwrap();

        let filler_width = 2 - clipmap.half_width as i32 % 2;
        let square_width = (clipmap.half_width as i32 - filler_width) / 2;

        commands.entity(entity).insert((
            Transform::from_scale(Vec3::splat(grid.scale(clipmap.base_scale))),
            Visibility::default(),
        ));

        let terrain_material = materials.add(ExtendedMaterial {
            base: StandardMaterial::default(),
            extension: GridMaterial {
                color: clipmap.color.clone(),
                heightmap: clipmap.heightmap.clone(),
                albedo_array: clipmap.albedo_array.clone(),
                control: clipmap.control.clone(),
                params: TerrainParams::from_clipmap(clipmap),
                normal_array: clipmap.normal_array.clone(),
                orm_array: clipmap.orm_array.clone(),
                rvt_albedo: rvt.albedo.clone(),
                rvt_normal: rvt.normal.clone(),
                detail_albedo_array: clipmap.detail_albedo_array.clone(),
                detail_normal_array: clipmap.detail_normal_array.clone(),
                detail: DetailParams::from_clipmap(clipmap),
                detail_orm_array: clipmap.detail_orm_array.clone(),
                lod: grid.level,
                texel_size: clipmap.texel_size,
                minmax: Vec2 {
                    x: clipmap.min,
                    y: clipmap.max,
                },
                translation: Vec2::ZERO,
                wireframe: 0,
            },
        });

        let terrain_material_w = materials.add(ExtendedMaterial {
            base: StandardMaterial::default(),
            extension: GridMaterial {
                color: clipmap.color.clone(),
                heightmap: clipmap.heightmap.clone(),
                albedo_array: clipmap.albedo_array.clone(),
                control: clipmap.control.clone(),
                params: TerrainParams::from_clipmap(clipmap),
                normal_array: clipmap.normal_array.clone(),
                orm_array: clipmap.orm_array.clone(),
                rvt_albedo: rvt.albedo.clone(),
                rvt_normal: rvt.normal.clone(),
                detail_albedo_array: clipmap.detail_albedo_array.clone(),
                detail_normal_array: clipmap.detail_normal_array.clone(),
                detail: DetailParams::from_clipmap(clipmap),
                detail_orm_array: clipmap.detail_orm_array.clone(),
                lod: grid.level,
                texel_size: clipmap.texel_size,
                minmax: Vec2 {
                    x: clipmap.min,
                    y: clipmap.max,
                },
                translation: Vec2::ZERO,
                wireframe: 1,
            },
        });

        for xy in 0..4 * 4 {
            let x = xy % 4;
            let y = xy / 4;

            if grid.level != 0 && (x == 1 || x == 2) && (y == 1 || y == 2) {
                continue;
            }

            let offset_x = if x >= 2 { filler_width as f32 } else { 0.0 };
            let offset_y = if y >= 2 { filler_width as f32 } else { 0.0 };

            commands.entity(entity).with_children(|c| {
                let mut e = c.spawn((
                    Mesh3d(parts.square.handle.clone()),
                    MeshMaterial3d(terrain_material.clone()),
                    NotShadowCaster,
                    Transform::from_xyz(
                        (x - 2) as f32 * square_width as f32 + offset_x,
                        0.0,
                        (y - 2) as f32 * square_width as f32 + offset_y,
                    ),
                    NoAutoAabb,
                    parts.square.aabb.clone(),
                ));
                if clipmap.wireframe {
                    e.with_child((
                        Mesh3d(parts.square.handle.clone()),
                        MeshMaterial3d(terrain_material_w.clone()),
                        NoAutoAabb,
                        parts.square.aabb.clone(),
                    ));
                }
            });
        }

        if grid.level == 0 {
            commands.entity(entity).with_children(|c| {
                let mut e = c.spawn((
                    Mesh3d(parts.center.handle.clone()),
                    MeshMaterial3d(terrain_material.clone()),
                    NotShadowCaster,
                    Transform::from_xyz(
                        -2.0 * square_width as f32,
                        0.0,
                        -2.0 * square_width as f32,
                    ),
                    NoAutoAabb,
                    parts.center.aabb,
                ));
                if clipmap.wireframe {
                    e.with_child((
                        Mesh3d(parts.center.handle.clone()),
                        MeshMaterial3d(terrain_material_w.clone()),
                        NoAutoAabb,
                        parts.center.aabb,
                    ));
                }
            });
        } else {
            commands.entity(entity).with_children(|c| {
                let mut e = c.spawn((
                    Mesh3d(parts.filler.handle.clone()),
                    MeshMaterial3d(terrain_material.clone()),
                    NotShadowCaster,
                    Transform::from_xyz(
                        -2.0 * square_width as f32,
                        0.0,
                        -2.0 * square_width as f32,
                    ),
                    NoAutoAabb,
                    parts.filler.aabb,
                ));
                if clipmap.wireframe {
                    e.with_child((
                        Mesh3d(parts.filler.handle.clone()),
                        MeshMaterial3d(terrain_material_w.clone()),
                        NoAutoAabb,
                        parts.filler.aabb,
                    ));
                }
            });
            commands.entity(entity).with_children(|c| {
                let mut e = c.spawn((
                    Mesh3d(parts.stitch.handle.clone()),
                    MeshMaterial3d(terrain_material.clone()),
                    NotShadowCaster,
                    Transform::from_xyz(-square_width as f32, 0.0, -square_width as f32)
                        .with_scale(Vec3::splat(0.5)),
                    NoAutoAabb,
                    parts.stitch.aabb,
                ));
                if clipmap.wireframe {
                    e.with_child((
                        Mesh3d(parts.stitch.handle.clone()),
                        MeshMaterial3d(terrain_material_w.clone()),
                        NoAutoAabb,
                        parts.stitch.aabb,
                    ));
                }
            });
        }

        let mut trim = commands.spawn((
            Mesh3d(parts.trim.handle.clone()),
            MeshMaterial3d(terrain_material.clone()),
            NotShadowCaster,
            Transform::from_xyz(-2.0 * square_width as f32, 0.0, -2.0 * square_width as f32),
            NoAutoAabb,
            parts.trim.aabb,
        ));
        if clipmap.wireframe {
            trim.with_child((
                Mesh3d(parts.trim.handle.clone()),
                MeshMaterial3d(terrain_material_w.clone()),
                NoAutoAabb,
                parts.trim.aabb,
            ));
        }
        grid.trim = trim.id();
        commands.entity(entity).add_child(grid.trim);
    }
}

fn update_grids(
    mut transforms: Query<&mut Transform>,
    mut aabbs: Query<&mut Aabb>,
    mut terrain_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, GridMaterial>>>,
    terrain_material_handles: Query<
        &MeshMaterial3d<ExtendedMaterial<StandardMaterial, GridMaterial>>,
    >,
    clipmaps: Query<&Clipmap>,
    children: Query<&Children>,
    grids: Query<(Entity, &ClipmapGrid, &ChildOf), With<Transform>>,
) {
    for (entity, grid, clipmap) in grids {
        let clipmap = clipmaps.get(clipmap.parent()).unwrap();
        let filler_width = 2 - clipmap.half_width as i32 % 2;
        let snap_scale = grid.scale(clipmap.base_scale) * filler_width as f32;
        let target_pos = transforms.get(clipmap.target).unwrap().translation;
        let snap_factor = (target_pos / snap_scale).floor().as_ivec3().xz();
        let snap_pos = snap_factor.as_vec2() * snap_scale;
        transforms.get_mut(entity).unwrap().translation = snap_pos.extend(0.0).xzy();

        let snap_mod2 = ((snap_factor % 2) + 2) % 2;
        let mut trim_transform = transforms.get_mut(grid.trim).unwrap();
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

        let grid_pos = (snap_pos.extend(0.0).xzy() + trim_transform.translation * snap_scale).xz();
        let aabb_scale = 2u32.pow(1 + grid.level) as f32;
        for child in children.iter_descendants(entity) {
            let Ok(material) = terrain_material_handles.get(child) else {
                continue;
            };
            let Some(mut material) = terrain_materials.get_mut(material) else {
                continue;
            };
            let Ok(mut aabb) = aabbs.get_mut(child) else {
                continue;
            };
            material.extension.translation = grid_pos;
            aabb.center.y = (clipmap.max + clipmap.min) / aabb_scale;
            aabb.half_extents.y = (clipmap.max - clipmap.min) / aabb_scale;
        }
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
            wireframe: material.wireframe != 0,
        }
    }
}

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
#[bind_group_data(WireframeKey)]
struct GridMaterial {
    #[texture(100)]
    #[sampler(101)]
    color: Handle<Image>,
    #[texture(102)]
    #[sampler(103)]
    heightmap: Handle<Image>,
    #[texture(112, dimension = "2d_array")]
    #[sampler(113)]
    albedo_array: Handle<Image>,
    #[texture(114)]
    #[sampler(115)]
    control: Handle<Image>,
    #[uniform(116)]
    params: TerrainParams,
    #[texture(117, dimension = "2d_array")]
    #[sampler(118)]
    normal_array: Handle<Image>,
    #[texture(119, dimension = "2d_array")]
    #[sampler(120)]
    orm_array: Handle<Image>,
    #[texture(121)]
    #[sampler(122)]
    rvt_albedo: Handle<Image>,
    #[texture(123)]
    #[sampler(124)]
    rvt_normal: Handle<Image>,
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
    #[uniform(107)]
    lod: u32,
    #[uniform(108)]
    texel_size: f32,
    #[uniform(109)]
    minmax: Vec2,
    #[uniform(110)]
    translation: Vec2,
    #[uniform(111)]
    wireframe: u32,
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

/// RVT (runtime virtual texture) state for a clipmap: the baked material texture
/// the main pass samples instead of blending the splat per-fragment.
#[derive(Component)]
struct ClipmapRvt {
    albedo: Handle<Image>,
    normal: Handle<Image>,
    initialized: bool,
}

/// A bake camera renders a few frames (enough for source textures to upload) then
/// deactivates — the full-coverage RVT is static. Camera-centered rings will
/// re-bake on movement instead.
#[derive(Component)]
struct RvtBakeCamera {
    frames: u32,
}

fn stop_rvt_bake(mut cameras: Query<(&mut Camera, &mut RvtBakeCamera)>) {
    for (mut camera, mut bake) in &mut cameras {
        if bake.frames > 0 {
            bake.frames -= 1;
        } else if camera.is_active {
            camera.is_active = false;
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
    #[texture(6)]
    #[sampler(7)]
    control: Handle<Image>,
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
    images: Res<Assets<Image>>,
    mut clipmaps: Query<(&Clipmap, &mut ClipmapRvt)>,
) {
    for (clipmap, mut rvt) in &mut clipmaps {
        if rvt.initialized {
            continue;
        }
        let Some(heightmap) = images.get(&clipmap.heightmap) else {
            continue;
        };
        let world_size = clipmap.texel_size * heightmap.texture_descriptor.size.width as f32;
        rvt.initialized = true;

        let mut make_bake = |mode: u32| {
            bake_materials.add(BakeMaterial {
                heightmap: clipmap.heightmap.clone(),
                texel_size: clipmap.texel_size,
                minmax: Vec2::new(clipmap.min, clipmap.max),
                albedo_array: clipmap.albedo_array.clone(),
                control: clipmap.control.clone(),
                params: TerrainParams::from_clipmap(clipmap),
                normal_array: clipmap.normal_array.clone(),
                orm_array: clipmap.orm_array.clone(),
                output_mode: mode,
            })
        };
        let quad = meshes.add(Plane3d::default().mesh().size(world_size, world_size));

        // Two bake targets: albedo (mode 0) and normal/ORM (mode 1). Bevy's
        // camera-to-image is single-target, so each is its own quad + camera on
        // its own render layer, rendered before the main view.
        for (mode, target, layer, order) in [
            (0u32, rvt.albedo.clone(), RVT_ALBEDO_LAYER, -2isize),
            (1u32, rvt.normal.clone(), RVT_NORMAL_LAYER, -1isize),
        ] {
            commands.spawn((
                Mesh3d(quad.clone()),
                MeshMaterial3d(make_bake(mode)),
                Transform::default(),
                RenderLayers::layer(layer),
            ));
            commands.spawn((
                Camera3d::default(),
                Camera {
                    order,
                    clear_color: Color::BLACK.into(),
                    ..default()
                },
                RenderTarget::Image(target.into()),
                Projection::Orthographic(OrthographicProjection {
                    scaling_mode: ScalingMode::Fixed {
                        width: world_size,
                        height: world_size,
                    },
                    near: 0.0,
                    far: 20000.0,
                    ..OrthographicProjection::default_3d()
                }),
                Tonemapping::None,
                Msaa::Off,
                Transform::from_xyz(0.0, 10000.0, 0.0).looking_at(Vec3::ZERO, Vec3::Z),
                RenderLayers::layer(layer),
                RvtBakeCamera { frames: 60 },
            ));
        }
    }
}
