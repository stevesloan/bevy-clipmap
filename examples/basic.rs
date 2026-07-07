use std::f32::consts::TAU;

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

use bevy_clipmap::{Clipmap, ClipmapPlugin};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FreeCameraPlugin)
        .add_plugins(ClipmapPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, update)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
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
            Transform::from_xyz(0.0, 150.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
            FreeCamera {
                walk_speed: 500.0,
                run_speed: 1000.0,
                ..Default::default()
            },
        ))
        .id();

    for _ in 0..2 {
        commands.spawn((
            DirectionalLight {
                shadow_maps_enabled: true,
                illuminance: lux::RAW_SUNLIGHT,
                color: ALICE_BLUE.into(),
                ..Default::default()
            },
            SunDisk {
                angular_size: SunDisk::EARTH.angular_size * 3.0,
                intensity: 30.0,
            },
            Transform::default(),
        ));
    }

    commands.spawn(Clipmap {
        half_width: 128,
        levels: 7,
        base_scale: 1.0,
        texel_size: 8.0,
        target,
        color: asset_server.load("color_2048x2048.png"),
        heightmap: asset_server
            .load_builder()
            .with_settings(|settings: &mut ImageLoaderSettings| {
                settings.is_srgb = false;
            })
            .load("heightmap_1024x1024.ktx2"),
        min: -1312.5,
        max: 1312.5,
        wireframe: false,
    });
}

fn update(mut lights: Query<&mut Transform, With<DirectionalLight>>, time: Res<Time>) {
    let cnt = lights.count();
    for (i, mut transform) in lights.iter_mut().enumerate() {
        let angle = 0.1 * time.elapsed_secs() + (TAU * i as f32 / cnt as f32);
        *transform =
            Transform::from_translation(Vec3::new(angle.cos(), angle.sin(), angle.sin() * 0.1))
                .looking_at(Vec3::ZERO, Vec3::Y);
    }
}
