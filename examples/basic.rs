use bevy::{
    asset::RenderAssetUsages,
    camera::Exposure,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    color::palettes::css::ALICE_BLUE,
    image::{
        ImageAddressMode, ImageFilterMode, ImageLoaderSettings, ImageSampler,
        ImageSamplerDescriptor,
    },
    light::{
        Atmosphere, AtmosphereEnvironmentMapLight, SunDisk, atmosphere::ScatteringMedium,
        light_consts::lux,
    },
    pbr::AtmosphereSettings,
    post_process::bloom::Bloom,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

use bevy_clipmap::{Clipmap, ClipmapPlugin, SlopeRule, TerrainLayer};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FreeCameraPlugin)
        .add_plugins(ClipmapPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
    commands.spawn(Atmosphere::earth(
        scattering_mediums.add(ScatteringMedium::earth(256, 256)),
    ));

    let target = commands
        .spawn((
            Camera3d::default(),
            Projection::from(PerspectiveProjection {
                fov: 90.0_f32.to_radians(),
                ..Default::default()
            }),
            Bloom::NATURAL,
            AtmosphereSettings {
                aerial_view_lut_max_distance: 16384.0,
                ..Default::default()
            },
            AtmosphereEnvironmentMapLight::default(),
            Exposure::SUNLIGHT,
            Transform::from_xyz(0.0, 150.0, 0.0).looking_at(Vec3::new(0.0, 150.0, -1000.0), Vec3::Y),
            FreeCamera {
                walk_speed: 500.0,
                run_speed: 1000.0,
                ..Default::default()
            },
        ))
        .id();

    // Fixed sun for baked terrain self-shadowing: 7pm North American summer —
    // west-northwest, ~17 degrees above the horizon (+X east, -Z north).
    let sun_direction = Vec3::new(-0.92, 0.3, -0.25).normalize();
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: false,
            illuminance: lux::RAW_SUNLIGHT,
            color: ALICE_BLUE.into(),
            ..Default::default()
        },
        SunDisk {
            angular_size: SunDisk::EARTH.angular_size * 3.0,
            intensity: 30.0,
        },
        Transform::from_translation(sun_direction * 1000.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Procedurally generated so the example runs with no downloads. To use real
    // textures (e.g. CC0 sets from polyhaven.com / ambientcg.com), drop one file
    // per layer into `assets/` and load them with `load_terrain_array` instead —
    // same layer order as `Clipmap::layers`, all files at the same resolution:
    //
    //     let albedo_array = load_terrain_array(&mut images, &[
    //         "terrain/grass_albedo.png", "terrain/dirt_albedo.png",
    //         "terrain/rock_albedo.png",  "terrain/snow_albedo.png",
    //     ], true);  // srgb = true for color, false for normal / ORM
    //
    // ORM packs occlusion, roughness, metallic into R, G, B (metallic ~0);
    // build it from the separate AO/roughness files those sites ship.
    let albedo_array = make_albedo_array(&mut images);
    let normal_array = make_normal_array(&mut images);
    let orm_array = make_orm_array(&mut images);
    let control = make_control_map(&mut images);
    let detail_albedo_array = make_detail_albedo_array(&mut images);
    let detail_normal_array = make_detail_normal_array(&mut images);
    let detail_orm_array = make_detail_orm_array(&mut images);

    commands.spawn(Clipmap {
        half_width: 128,
        levels: 7,
        base_scale: 1.0,
        texel_size: 8.0,
        target,
        color: asset_server.load("color_2048x2048.png"),
        macro_strength: 0.5,
        macro_near: 800.0,
        macro_far: 5000.0,
        heightmap: asset_server
            .load_builder()
            .with_settings(|settings: &mut ImageLoaderSettings| {
                settings.is_srgb = false;
            })
            .load("heightmap_1024x1024.ktx2"),
        albedo_array,
        normal_array,
        orm_array,
        control,
        layers: vec![
            // grass
            TerrainLayer {
                tiling_scale: 32.0,
                height_blend: 0.3,
                normal_strength: 0.8,
                roughness: 0.9,
                slope: None,
            },
            // dirt
            TerrainLayer {
                tiling_scale: 24.0,
                height_blend: 0.5,
                normal_strength: 1.0,
                roughness: 0.85,
                slope: None,
            },
            // rock — auto-placed on steep terrain
            TerrainLayer {
                tiling_scale: 20.0,
                height_blend: 0.8,
                normal_strength: 1.3,
                roughness: 0.7,
                slope: Some(SlopeRule {
                    min_deg: 32.0,
                    blend_deg: 18.0,
                }),
            },
            // snow
            TerrainLayer {
                tiling_scale: 48.0,
                height_blend: 0.4,
                normal_strength: 0.4,
                roughness: 0.5,
                slope: None,
            },
        ],
        detail_albedo_array,
        detail_normal_array,
        detail_orm_array,
        detail_tiling: 10.0,
        detail_normal_strength: 0.9,
        detail_albedo_strength: 0.3,
        detail_near: 60.0,
        detail_far: 400.0,
        sun_direction,
        min: -1312.5,
        max: 1312.5,
        wireframe: false,
    });
}

fn hash(x: u32, y: u32, seed: u32) -> f32 {
    let mut h = x
        .wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(seed.wrapping_mul(2246822519));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^= h >> 16;
    (h & 0xffff) as f32 / 65535.0
}

/// Smooth value noise in `0..1`, interpolated from a coarse grid.
fn value_noise(x: u32, y: u32, seed: u32) -> f32 {
    const CELL: u32 = 16;
    let gx = x / CELL;
    let gy = y / CELL;
    let fx = (x % CELL) as f32 / CELL as f32;
    let fy = (y % CELL) as f32 / CELL as f32;
    let a = hash(gx, gy, seed);
    let b = hash(gx + 1, gy, seed);
    let c = hash(gx, gy + 1, seed);
    let d = hash(gx + 1, gy + 1, seed);
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);
    let top = a + (b - a) * sx;
    let bot = c + (d - c) * sx;
    top + (bot - top) * sy
}

/// Tileable value noise in `0..1`. `freq` is the number of grid cells across the
/// texture (must divide `LAYER_TEX_SIZE`); wrapping the grid nodes modulo `freq`
/// makes opposite edges match, so the result repeats seamlessly.
fn noise_tileable(x: u32, y: u32, freq: u32, seed: u32) -> f32 {
    let cell = LAYER_TEX_SIZE / freq;
    let gx = x / cell;
    let gy = y / cell;
    let fx = (x % cell) as f32 / cell as f32;
    let fy = (y % cell) as f32 / cell as f32;
    let a = hash(gx % freq, gy % freq, seed);
    let b = hash((gx + 1) % freq, gy % freq, seed);
    let c = hash(gx % freq, (gy + 1) % freq, seed);
    let d = hash((gx + 1) % freq, (gy + 1) % freq, seed);
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);
    let top = a + (b - a) * sx;
    let bot = c + (d - c) * sx;
    top + (bot - top) * sy
}

const LAYER_TEX_SIZE: u32 = 256;
const LAYER_COUNT: u32 = 4;

/// Repeat + anisotropic sampler used for all tiling layer arrays.
fn tiling_sampler() -> ImageSampler {
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

/// Turn per-layer RGBA8 data (laid out slice by slice) into a tiling `2d_array`.
fn layer_array(images: &mut Assets<Image>, format: TextureFormat, data: Vec<u8>) -> Handle<Image> {
    let mut image = Image::new(
        Extent3d {
            width: LAYER_TEX_SIZE,
            height: LAYER_TEX_SIZE * LAYER_COUNT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image
        .reinterpret_stacked_2d_as_array(LAYER_COUNT)
        .expect("valid stacked layer array");
    image.sampler = tiling_sampler();
    images.add(image)
}

/// 4-slice albedo array (grass / dirt / rock / snow); RGB is sRGB color, alpha
/// is per-texel height for height blending.
fn make_albedo_array(images: &mut Assets<Image>) -> Handle<Image> {
    let bases = [
        [0.24, 0.34, 0.12], // grass
        [0.35, 0.26, 0.16], // dirt
        [0.42, 0.40, 0.38], // rock
        [0.90, 0.92, 0.96], // snow
    ];
    let mut data = Vec::with_capacity((LAYER_TEX_SIZE * LAYER_TEX_SIZE * LAYER_COUNT * 4) as usize);
    for layer in 0..LAYER_COUNT {
        for y in 0..LAYER_TEX_SIZE {
            for x in 0..LAYER_TEX_SIZE {
                let shade = 0.75 + 0.5 * noise_tileable(x, y, 16, layer);
                let base = bases[layer as usize];
                let height = noise_tileable(x, y, 32, layer + 9);
                data.push(((base[0] * shade).clamp(0.0, 1.0) * 255.0) as u8);
                data.push(((base[1] * shade).clamp(0.0, 1.0) * 255.0) as u8);
                data.push(((base[2] * shade).clamp(0.0, 1.0) * 255.0) as u8);
                data.push((height * 255.0) as u8);
            }
        }
    }
    layer_array(images, TextureFormat::Rgba8UnormSrgb, data)
}

/// 4-slice tangent-space normal array, derived from each layer's height field.
fn make_normal_array(images: &mut Assets<Image>) -> Handle<Image> {
    let mut data = Vec::with_capacity((LAYER_TEX_SIZE * LAYER_TEX_SIZE * LAYER_COUNT * 4) as usize);
    for layer in 0..LAYER_COUNT {
        for y in 0..LAYER_TEX_SIZE {
            for x in 0..LAYER_TEX_SIZE {
                let h = noise_tileable(x, y, 32, layer + 9);
                let hx = noise_tileable(x + 1, y, 32, layer + 9);
                let hy = noise_tileable(x, y + 1, 32, layer + 9);
                let dx = (hx - h) * 4.0;
                let dy = (hy - h) * 4.0;
                let inv = 1.0 / (dx * dx + dy * dy + 1.0).sqrt();
                data.push(((-dx * inv * 0.5 + 0.5) * 255.0) as u8);
                data.push(((-dy * inv * 0.5 + 0.5) * 255.0) as u8);
                data.push(((inv * 0.5 + 0.5) * 255.0) as u8);
                data.push(255);
            }
        }
    }
    layer_array(images, TextureFormat::Rgba8Unorm, data)
}

/// 4-slice ORM array: R = occlusion, G = roughness, B = metallic (0 for terrain).
fn make_orm_array(images: &mut Assets<Image>) -> Handle<Image> {
    let mut data = Vec::with_capacity((LAYER_TEX_SIZE * LAYER_TEX_SIZE * LAYER_COUNT * 4) as usize);
    for layer in 0..LAYER_COUNT {
        for y in 0..LAYER_TEX_SIZE {
            for x in 0..LAYER_TEX_SIZE {
                let occlusion = 0.85 + 0.15 * noise_tileable(x, y, 32, layer + 9);
                let roughness = 0.9 + 0.1 * noise_tileable(x, y, 16, layer + 21);
                data.push((occlusion.clamp(0.0, 1.0) * 255.0) as u8);
                data.push((roughness.clamp(0.0, 1.0) * 255.0) as u8);
                data.push(0);
                data.push(255);
            }
        }
    }
    layer_array(images, TextureFormat::Rgba8Unorm, data)
}

/// Per-material detail albedo array (grain around neutral, with per-material
/// contrast/tint). Slices: grass / dirt / rock / snow.
fn make_detail_albedo_array(images: &mut Assets<Image>) -> Handle<Image> {
    let tint = [
        [0.50, 0.52, 0.46],
        [0.52, 0.48, 0.43],
        [0.50, 0.50, 0.50],
        [0.50, 0.51, 0.53],
    ];
    let contrast = [0.12, 0.15, 0.28, 0.08];
    let mut data = Vec::with_capacity((LAYER_TEX_SIZE * LAYER_TEX_SIZE * LAYER_COUNT * 4) as usize);
    for layer in 0..LAYER_COUNT {
        let t = tint[layer as usize];
        let c = contrast[layer as usize];
        for y in 0..LAYER_TEX_SIZE {
            for x in 0..LAYER_TEX_SIZE {
                let n = (noise_tileable(x, y, 64, 300 + layer) - 0.5) * 2.0 * c;
                data.push(((t[0] + n).clamp(0.0, 1.0) * 255.0) as u8);
                data.push(((t[1] + n).clamp(0.0, 1.0) * 255.0) as u8);
                data.push(((t[2] + n).clamp(0.0, 1.0) * 255.0) as u8);
                data.push(255);
            }
        }
    }
    layer_array(images, TextureFormat::Rgba8UnormSrgb, data)
}

/// Per-material detail normal array (relief; rock strong, snow weak).
fn make_detail_normal_array(images: &mut Assets<Image>) -> Handle<Image> {
    let strength = [4.0, 5.0, 9.0, 2.0];
    let freq = [64u32, 64, 48, 80];
    let mut data = Vec::with_capacity((LAYER_TEX_SIZE * LAYER_TEX_SIZE * LAYER_COUNT * 4) as usize);
    for layer in 0..LAYER_COUNT {
        let s = strength[layer as usize];
        let f = freq[layer as usize];
        let seed = 400 + layer;
        for y in 0..LAYER_TEX_SIZE {
            for x in 0..LAYER_TEX_SIZE {
                let h = noise_tileable(x, y, f, seed);
                let hx = noise_tileable(x + 1, y, f, seed);
                let hy = noise_tileable(x, y + 1, f, seed);
                let dx = (hx - h) * s;
                let dy = (hy - h) * s;
                let inv = 1.0 / (dx * dx + dy * dy + 1.0).sqrt();
                data.push(((-dx * inv * 0.5 + 0.5) * 255.0) as u8);
                data.push(((-dy * inv * 0.5 + 0.5) * 255.0) as u8);
                data.push(((inv * 0.5 + 0.5) * 255.0) as u8);
                data.push(255);
            }
        }
    }
    layer_array(images, TextureFormat::Rgba8Unorm, data)
}

/// Per-material detail ORM array: R = occlusion, G = roughness, B = metallic (0).
fn make_detail_orm_array(images: &mut Assets<Image>) -> Handle<Image> {
    let rough = [0.9, 0.88, 0.78, 0.4];
    let mut data = Vec::with_capacity((LAYER_TEX_SIZE * LAYER_TEX_SIZE * LAYER_COUNT * 4) as usize);
    for layer in 0..LAYER_COUNT {
        let rb = rough[layer as usize];
        for y in 0..LAYER_TEX_SIZE {
            for x in 0..LAYER_TEX_SIZE {
                let ao = 0.8 + 0.2 * noise_tileable(x, y, 48, 600 + layer);
                let r = (rb + 0.15 * (noise_tileable(x, y, 64, 500 + layer) - 0.5)).clamp(0.0, 1.0);
                data.push((ao.clamp(0.0, 1.0) * 255.0) as u8);
                data.push((r * 255.0) as u8);
                data.push(0);
                data.push(255);
            }
        }
    }
    layer_array(images, TextureFormat::Rgba8Unorm, data)
}

/// Build an RGBA control map: `r`=grass, `g`=dirt, `a`=snow weights from noise.
/// The rock channel (`b`) stays 0 — rock is placed by the layer's slope rule.
fn make_control_map(images: &mut Assets<Image>) -> Handle<Image> {
    const SIZE: u32 = 512;
    let mut data = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let grass = 0.6 + 0.4 * value_noise(x, y, 1);
            let dirt = ((value_noise(x, y, 2) - 0.55) * 2.5).clamp(0.0, 1.0);
            let snow = ((value_noise(x, y, 3) - 0.7) * 3.0).clamp(0.0, 1.0);
            data.push((grass.clamp(0.0, 1.0) * 255.0) as u8);
            data.push((dirt * 255.0) as u8);
            data.push(0);
            data.push((snow * 255.0) as u8);
        }
    }
    images.add(Image::new(
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    ))
}
