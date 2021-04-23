use std::{mem::ManuallyDrop, ptr};
use gfx_hal::Instance;

pub struct Renderer<B: gfx_hal::Backend> {
    pub dimensions: gfx_hal::window::Extent2D,
    surface: ManuallyDrop<B::Surface>,
    // These members are dropped in the declaration order.
    adapter: gfx_hal::adapter::Adapter<B>,
    instance: B::Instance,
}

impl<B> Renderer<B>
where
    B: gfx_hal::Backend,
{
    pub fn new(
        instance: B::Instance,
        mut surface: B::Surface,
        adapter: gfx_hal::adapter::Adapter<B>,
        dimensions: gfx_hal::window::Extent2D,
    ) -> Renderer<B> {
        Renderer {
            adapter,
            dimensions,
            instance,
            surface: ManuallyDrop::new(surface),
        }
    }

    pub fn recreate_swapchain(&mut self) {}

    pub fn render(&mut self) {}
}

impl<B> Drop for Renderer<B>
where
    B: gfx_hal::Backend,
{
    fn drop(&mut self) {
        unsafe {
            let surface = ManuallyDrop::into_inner(ptr::read(&self.surface));
            self.instance.destroy_surface(surface);
        }
    }
}
