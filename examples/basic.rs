use bevy::{
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
};

use bevy_clipmap::{
    Clipmap, ClipmapPlugin, HeightRule, SlopeRule, TerrainLayer, load_terrain_array,
};

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
        // Fully procedural placement (no control map): grass/dirt/rock partition
        // by slope, snow by height. Snowline ~700 m (height range is ±1312.5).
        layers: vec![
            // grass — base layer, everywhere below the snowline. No slope band, so
            // it competes on cliffs and pokes through the rock (natural look).
            TerrainLayer {
                tiling_scale: 100.0,
                height_blend: 0.3,
                normal_strength: 1.0,
                roughness: 0.9,
                slope: None,
                height: Some(HeightRule {
                    min: -2000.0,
                    max: 700.0,
                    blend: 250.0,
                }),
            },
            // dirt — mid slopes, below the snowline
            TerrainLayer {
                tiling_scale: 300.0,
                height_blend: 0.5,
                normal_strength: 1.3,
                roughness: 0.85,
                slope: Some(SlopeRule {
                    min_deg: 25.0,
                    max_deg: 55.0,
                    blend_deg: 10.0,
                }),
                height: Some(HeightRule {
                    min: -2000.0,
                    max: 700.0,
                    blend: 250.0,
                }),
            },
            // rock — steep terrain at any height (cliffs stay bare above snow)
            TerrainLayer {
                tiling_scale: 100.0,
                height_blend: 0.8,
                normal_strength: 1.3,
                roughness: 0.7,
                slope: Some(SlopeRule {
                    min_deg: 45.0,
                    max_deg: 90.0,
                    blend_deg: 12.0,
                }),
                height: None,
            },
            // snow — above the snowline, on all but the steepest faces
            TerrainLayer {
                tiling_scale: 100.0,
                height_blend: 0.4,
                normal_strength: 0.4,
                roughness: 0.5,
                slope: Some(SlopeRule {
                    min_deg: 0.0,
                    max_deg: 35.0,
                    blend_deg: 12.0,
                }),
                height: Some(HeightRule {
                    min: 500.0,
                    max: 800.0,
                    blend: 250.0,
                }),
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
