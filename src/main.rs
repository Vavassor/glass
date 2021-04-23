use gfx_hal::{window::Extent2D};
use gfx_hal::Instance;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

mod renderer;

use renderer::Renderer;

fn main() {
    const APP_NAME: &'static str = "Glass";
    const MIN_WINDOW_SIZE: [u32; 2] = [64, 64];
    const WINDOW_SIZE: [u32; 2] = [512, 512];

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
                println!("{:?}", adapter.info);
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
