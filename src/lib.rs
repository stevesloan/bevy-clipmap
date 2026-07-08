use std::{
    collections::HashMap,
    f32::consts::{FRAC_PI_2, PI},
};

use std::path::Path;

use bevy::{
    asset::{AssetPath, RenderAssetUsages, embedded_asset, embedded_path},
    camera::{
        RenderTarget, ScalingMode,
        primitives::Aabb,
        visibility::{NoAutoAabb, RenderLayers},
    },
    core_pipeline::tonemapping::Tonemapping,
    image::{
        CompressedImageFormats, ImageAddressMode, ImageFilterMode, ImageSampler,
        ImageSamplerDescriptor, ImageType,
    },
    light::NotShadowCaster,
    mesh::{Indices, PrimitiveTopology},
    pbr::{ExtendedMaterial, Material, MaterialExtension},
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{
            AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat, TextureUsages,
        },
    },
    shader::ShaderRef,
};

/// Render layers isolating the RVT bake cameras/quads from the main view.
const RVT_ALBEDO_LAYER: usize = 1;
const RVT_NORMAL_LAYER: usize = 2;
/// Layer offset for the tiny sentinel targets that detect bake readiness.
const RVT_SENTINEL_LAYER_OFFSET: usize = 2;
/// Resolution of the RVT bake target textures.
const RVT_SIZE: u32 = 8192;

/// Repeat + anisotropic sampler for the tiling terrain layer arrays.
fn terrain_tiling_sampler() -> ImageSampler {
    ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 8,
        ..default()
    })
}

/// Decodes one image file per layer and stacks them into a tiling `2d_array`
/// for a [`Clipmap`]'s `albedo_array` / `normal_array` / `orm_array`, generating
/// a full mip chain (file formats like PNG carry none, and the RVT bake samples
/// these heavily minified — without mips the result aliases into noise).
///
/// Pass one path per terrain layer, in the same order as [`Clipmap::layers`].
/// All images must share the same dimensions (this decodes but does not
/// resample — export your set at a single resolution).
///
/// Set `srgb` to `true` for color/albedo maps and `false` for normal and ORM
/// maps, which hold linear data. ORM maps pack occlusion, roughness, metallic
/// into R, G, B (metallic is ~0 for terrain); build them from the separate
/// AO/roughness files that texture sites ship.
///
/// This reads files synchronously and is meant for one-time setup. It panics on
/// a missing/undecodable file or a dimension mismatch — asset-authoring errors
/// worth surfacing immediately at startup.
///
/// ```no_run
/// # use bevy::prelude::*;
/// # use bevy_clipmap::load_terrain_array;
/// # fn setup(mut images: ResMut<Assets<Image>>) {
/// let albedo = load_terrain_array(
///     &mut images,
///     &["terrain/grass_albedo.png", "terrain/rock_albedo.png"],
///     true,
/// );
/// # }
/// ```
pub fn load_terrain_array(
    images: &mut Assets<Image>,
    paths: &[impl AsRef<Path>],
    srgb: bool,
) -> Handle<Image> {
    assert!(
        !paths.is_empty(),
        "load_terrain_array needs at least one layer"
    );
    let format = if srgb {
        TextureFormat::Rgba8UnormSrgb
    } else {
        TextureFormat::Rgba8Unorm
    };

    let mut stacked = Vec::new();
    let mut dims: Option<(u32, u32)> = None;
    for path in paths {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("load_terrain_array: reading {}: {e}", path.display()));
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_else(|| panic!("load_terrain_array: {} has no file extension", path.display()));
        let image = Image::from_buffer(
            &bytes,
            ImageType::Extension(ext),
            CompressedImageFormats::NONE,
            srgb,
            ImageSampler::Default,
            RenderAssetUsages::RENDER_WORLD,
        )
        .unwrap_or_else(|e| panic!("load_terrain_array: decoding {}: {e:?}", path.display()));
        // PNG/JPEG decode straight to 8-bit RGBA in the requested color space;
        // convert anything else (e.g. 16-bit) so every layer matches `format`.
        let image = if image.texture_descriptor.format == format {
            image
        } else {
            image.convert(format).unwrap_or_else(|| {
                panic!(
                    "load_terrain_array: {} is {:?}, which can't convert to RGBA8 — re-export as 8-bit PNG",
                    path.display(),
                    image.texture_descriptor.format,
                )
            })
        };

        let size = (image.width(), image.height());
        if let Some(first) = dims {
            assert!(
                first == size,
                "load_terrain_array: {} is {size:?} but earlier layers are {first:?}; all layers must share dimensions",
                path.display(),
            );
        } else {
            dims = Some(size);
        }
        // Layer-major: each layer's full mip chain, then the next layer's.
        let mip0 = image
            .data
            .as_deref()
            .expect("decoded image is uncompressed and has pixel data");
        stacked.extend_from_slice(mip0);
        let mut level = mip0.to_vec();
        let (mut w, mut h) = size;
        while w > 1 || h > 1 {
            level = downsample_rgba8(&level, w, h, srgb);
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            stacked.extend_from_slice(&level);
        }
    }

    let (width, height) = dims.unwrap();
    let mut array = Image::default();
    array.data = Some(stacked);
    array.texture_descriptor.size = Extent3d {
        width,
        height,
        depth_or_array_layers: paths.len() as u32,
    };
    array.texture_descriptor.dimension = TextureDimension::D2;
    array.texture_descriptor.format = format;
    array.texture_descriptor.mip_level_count = 32 - width.max(height).leading_zeros();
    array.asset_usage = RenderAssetUsages::RENDER_WORLD;
    array.sampler = terrain_tiling_sampler();
    images.add(array)
}

/// Box-filters one RGBA8 mip level into the next. sRGB data is averaged in
/// roughly-linear space (averaging encoded bytes skews dark); gamma 2.0
/// (square/sqrt) stands in for the sRGB curve — indistinguishable for mip
/// averaging and much cheaper than the exact transfer function. Alpha is
/// always averaged linearly.
fn downsample_rgba8(src: &[u8], w: u32, h: u32, srgb: bool) -> Vec<u8> {
    let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
    let mut out = Vec::with_capacity((nw * nh * 4) as usize);
    for y in 0..nh {
        for x in 0..nw {
            // Clamp so odd dimensions reuse the last row/column.
            let (x0, y0) = (2 * x, 2 * y);
            let (x1, y1) = ((2 * x + 1).min(w - 1), (2 * y + 1).min(h - 1));
            for c in 0..4 {
                let at = |px: u32, py: u32| src[((py * w + px) * 4 + c) as usize] as u32;
                let (a, b, cc, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
                let avg = if srgb && c < 3 {
                    (((a * a + b * b + cc * cc + d * d) as f32 / 4.0).sqrt() + 0.5) as u32
                } else {
                    (a + b + cc + d) / 4
                };
                out.push(avg as u8);
            }
        }
    }
    out
}

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
            .add_systems(Update, (update_grids, init_rvt, drive_rvt_bake));
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

    /// Normalized direction *toward* the (fixed) sun, used to bake terrain
    /// self-shadowing into the RVT.
    pub sun_direction: Vec3,

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
                heightmap: clipmap.heightmap.clone(),
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
                heightmap: clipmap.heightmap.clone(),
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
    #[texture(102)]
    #[sampler(103)]
    heightmap: Handle<Image>,
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

/// A bake camera stays inactive until its sentinel readback proves the bake
/// pipeline is compiled and source textures are on the GPU (`ready`), then
/// renders a few frames (the full-coverage RVT is static) and deactivates.
/// Pipeline compilation and asset upload take a machine-dependent number of
/// frames; a camera that renders before they finish produces an empty target.
#[derive(Component)]
struct RvtBakeCamera {
    ready: bool,
    frames: u32,
}

/// Active frames rendered once ready; > 1 only as safety margin.
const RVT_BAKE_FRAMES: u32 = 2;

fn drive_rvt_bake(mut cameras: Query<(&mut Camera, &mut RvtBakeCamera)>) {
    for (mut camera, mut state) in &mut cameras {
        if !state.ready {
            continue;
        }
        if state.frames > 0 {
            camera.is_active = true;
            state.frames -= 1;
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
                params: TerrainParams::from_clipmap(clipmap),
                normal_array: clipmap.normal_array.clone(),
                orm_array: clipmap.orm_array.clone(),
                output_mode: mode,
                sun_direction: clipmap.sun_direction.normalize_or_zero(),
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
        for (mode, target, format, layer, order) in [
            (
                0u32,
                rvt.albedo.clone(),
                TextureFormat::Rgba8UnormSrgb,
                RVT_ALBEDO_LAYER,
                -2isize,
            ),
            (
                1u32,
                rvt.normal.clone(),
                TextureFormat::Rgba8Unorm,
                RVT_NORMAL_LAYER,
                -1isize,
            ),
        ] {
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
                        order: order - 2,
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
                    if let Ok(mut camera) = cameras.get_mut(bake_camera) {
                        camera.ready = true;
                    }
                    commands.entity(sentinel_camera).despawn();
                    commands.entity(sentinel_quad).despawn();
                    commands.entity(readback).despawn();
                },
            );
        }
    }
}
