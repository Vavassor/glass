use gfx_hal::window::Extent2D;
use gfx_hal::Instance;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

#[macro_use]
extern crate log;

mod bivector;
mod image_loading;
mod point;
mod renderer;
mod rotor;
mod trivector;
mod vector;

#[cfg(test)]
mod wedge_tests;

use point::Point3;
use renderer::Renderer;
use vector::Vector3;

fn triangle_area_sample() {
    let a = Point3::new(0.0, 1.0, 0.0);
    let b = Point3::new(-1.0, -2.0, 0.0);
    let c = Point3::new(2.5, -1.0, 0.0);
    let ab = b - a;
    let ac = c - a;
    let area = ab.wedge_vector3(ac).magnitude() / 2.0;
    info!("Area of triangle ABC {:?}", area);
}

fn trivector_sample() {
    let a = Vector3::new(0.0, 1.0, -1.0);
    let b = Vector3::new(-1.0, -1.0, 1.0);
    let c = Vector3::new(2.5, -1.0, 5.0);
    let d = Vector3::new(3.0, 3.0, 0.5);

    // bivector addition
    let ab = a.wedge_vector3(b);
    let ac = a.wedge_vector3(c);
    let bivector_sum = ab + ac;
    let formula = a.wedge_vector3(b + c);
    info!(
        "a ∧ b + a ∧ c = {:?} a ∧ (b + c) = {:?}",
        bivector_sum, formula
    );

    // dot product of bivectors
    let cd = c.wedge_vector3(d);
    info!(
        "(a ∧ b) ⋅ (c ∧ d) = {} (a ⋅ d)(b ⋅ c) - (a ⋅ c)(b ⋅ d) = {}",
        ab.dot(cd),
        a.dot(d) * b.dot(c) - a.dot(c) * b.dot(d)
    );
}

fn muck_around() {
    triangle_area_sample();
    trivector_sample();
}

fn main() {
    env_logger::init();

    muck_around();

    const APP_NAME: &'static str = "Glass";
    const MIN_WINDOW_SIZE: [u32; 2] = [64, 64];
    const WINDOW_SIZE: [u32; 2] = [1280, 720];

    let event_loop = EventLoop::new();

    let (logical_window_size, min_logical_window_size, physical_window_size) = {
        let dpi = match event_loop.primary_monitor() {
            Some(primary_monitor) => primary_monitor.scale_factor(),
            None => 1.0,
        };
        let logical: LogicalSize<u32> = WINDOW_SIZE.into();
        let min_logical: LogicalSize<u32> = MIN_WINDOW_SIZE.into();
        let physical: PhysicalSize<u32> = logical.to_physical(dpi);

        (logical, min_logical, physical)
    };

    let window = winit::window::WindowBuilder::new()
        .with_inner_size(logical_window_size)
        .with_min_inner_size(min_logical_window_size)
        .with_title(APP_NAME)
        .build(&event_loop)
        .unwrap();

    let (instance, surface, adapter) = {
        let instance =
            backend::Instance::create(APP_NAME, 1).expect("Failed to create an instance!");

        let surface = unsafe {
            instance
                .create_surface(&window)
                .expect("Failed to create a surface!")
        };

        let adapter = {
            let mut adapters = instance.enumerate_adapters();
            for adapter in &adapters {
                info!("{:?}", adapter.info);
            }
            adapters.remove(0)
        };

        (instance, surface, adapter)
    };

    let dimensions = Extent2D {
        height: WINDOW_SIZE[1],
        width: WINDOW_SIZE[0],
    };
    let mut renderer = Renderer::new(instance, surface, adapter, dimensions);

    renderer.render();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(dimensions) => {
                    renderer.dimensions = Extent2D {
                        width: dimensions.width,
                        height: dimensions.height,
                    };
                    renderer.recreate_swapchain();
                }
                _ => {}
            },
            Event::RedrawEventsCleared => {
                renderer.render();
            }
            _ => {}
        }
    });
}
