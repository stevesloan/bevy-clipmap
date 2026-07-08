use bevy::{
    asset::RenderAssetUsages,
    camera::Exposure,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    color::palettes::css::ALICE_BLUE,
    image::ImageLoaderSettings,
    light::{
        Atmosphere, AtmosphereEnvironmentMapLight, SunDisk, atmosphere::ScatteringMedium,
        light_consts::lux,
    },
    pbr::AtmosphereSettings,
    post_process::bloom::Bloom,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

use bevy_clipmap::{Clipmap, ClipmapPlugin, SlopeRule, TerrainLayer, load_terrain_array};

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
            Transform::from_xyz(0.0, 150.0, 0.0)
                .looking_at(Vec3::new(0.0, 150.0, -1000.0), Vec3::Y),
            FreeCamera {
                walk_speed: 500.0,
                run_speed: 1000.0,
                ..Default::default()
            },
        ))
        .id();

    // Fixed sun for baked terrain self-shadowing: 8am North American summer —
    // east, slightly north, ~28 degrees above the horizon (+X east, -Z north).
    let sun_direction = Vec3::new(0.87, 0.47, -0.15).normalize();
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

    // CC0 texture sets from polyhaven.com, one file per layer in the same order
    // as `Clipmap::layers` — run `python3 assets/fetch_textures.py` once to
    // download them. srgb = true for color, false for linear normal / ORM.
    // ORM packs occlusion, roughness, metallic into R, G, B (metallic ~0).
    let layer = |suffix: &str| {
        ["grass", "dirt", "rock", "snow"].map(|name| format!("assets/terrain/{name}_{suffix}.png"))
    };
    let albedo_array = load_terrain_array(&mut images, &layer("albedo"), true);
    let normal_array = load_terrain_array(&mut images, &layer("normal"), false);
    let orm_array = load_terrain_array(&mut images, &layer("orm"), false);
    let control = make_control_map(&mut images);
    // Close-range detail reuses the same arrays at a finer tiling — the RVT
    // only holds ~2m texels, so all sub-2m structure comes from these.
    let detail_albedo_array = albedo_array.clone();
    let detail_normal_array = normal_array.clone();
    let detail_orm_array = orm_array.clone();

    commands.spawn(Clipmap {
        half_width: 128,
        levels: 7,
        base_scale: 1.0,
        texel_size: 8.0,
        target,
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
                tiling_scale: 100.0,
                height_blend: 0.3,
                normal_strength: 1.0,
                roughness: 0.9,
                slope: None,
            },
            // dirt — mid-slope band between flat grass and steep rock
            TerrainLayer {
                tiling_scale: 300.0,
                height_blend: 0.5,
                normal_strength: 1.3,
                roughness: 0.85,
                slope: Some(SlopeRule {
                    min_deg: 18.0,
                    blend_deg: 10.0,
                }),
            },
            // rock — auto-placed on steep terrain
            TerrainLayer {
                tiling_scale: 100.0,
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
                tiling_scale: 100.0,
                height_blend: 0.4,
                normal_strength: 0.4,
                roughness: 0.5,
                slope: None,
            },
        ],
        detail_albedo_array,
        detail_normal_array,
        detail_orm_array,
        detail_tiling: 70.0,
        detail_normal_strength: 0.8,
        detail_albedo_strength: 0.8,
        detail_near: 60.0,
        detail_far: 600.0,
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
