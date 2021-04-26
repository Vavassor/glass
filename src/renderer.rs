use gfx_hal::{
    adapter, buffer, command, format,
    format::{AsFormat, ChannelType, Rgba8Srgb, Swizzle},
    image,
    image::{SubresourceRange, ViewKind},
    memory, pass,
    pass::Subpass,
    pool,
    prelude::*,
    pso,
    pso::{PipelineStage, ShaderStageFlags, VertexInputRate},
    queue::QueueGroup,
    window, Features, Instance,
};
use std::{borrow::Borrow, io::Cursor, iter, mem, mem::ManuallyDrop, ptr};

use crate::image_loading::load_image_from_bytes;

#[derive(Debug, Clone, Copy)]
struct Vertex {
    position: [f32; 2],
    texcoord: [f32; 2],
}

const QUAD: [Vertex; 6] = [
    Vertex {
        position: [-0.5, 0.33],
        texcoord: [0.0, 1.0],
    },
    Vertex {
        position: [0.5, 0.33],
        texcoord: [1.0, 1.0],
    },
    Vertex {
        position: [0.5, -0.33],
        texcoord: [1.0, 0.0],
    },
    Vertex {
        position: [-0.5, 0.33],
        texcoord: [0.0, 1.0],
    },
    Vertex {
        position: [0.5, -0.33],
        texcoord: [1.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.33],
        texcoord: [0.0, 0.0],
    },
];

pub struct Renderer<B: gfx_hal::Backend> {
    buffer_memory: ManuallyDrop<B::Memory>,
    cmd_buffers: Vec<B::CommandBuffer>,
    cmd_pools: Vec<B::CommandPool>,
    descriptor_pool: ManuallyDrop<B::DescriptorPool>,
    descriptor_set: Option<B::DescriptorSet>,
    pub dimensions: window::Extent2D,
    format: gfx_hal::format::Format,
    frame: u64,
    framebuffer: ManuallyDrop<B::Framebuffer>,
    frames_in_flight: usize,
    image_memory: ManuallyDrop<B::Memory>,
    image_logo: ManuallyDrop<B::Image>,
    image_srv: ManuallyDrop<B::ImageView>,
    image_upload_buffer: ManuallyDrop<B::Buffer>,
    image_upload_memory: ManuallyDrop<B::Memory>,
    pipeline: ManuallyDrop<B::GraphicsPipeline>,
    pipeline_layout: ManuallyDrop<B::PipelineLayout>,
    render_pass: ManuallyDrop<B::RenderPass>,
    sampler: ManuallyDrop<B::Sampler>,
    set_layout: ManuallyDrop<B::DescriptorSetLayout>,
    submission_complete_fences: Vec<B::Fence>,
    submission_complete_semaphores: Vec<B::Semaphore>,
    surface: ManuallyDrop<B::Surface>,
    vertex_buffer: ManuallyDrop<B::Buffer>,
    viewport: pso::Viewport,
    // These members are dropped in the declaration order.
    device: B::Device,
    adapter: adapter::Adapter<B>,
    queue_group: QueueGroup<B>,
    instance: B::Instance,
}

struct ImageLayout {
    height: u32,
    row_pitch: u32,
    stride: usize,
    width: u32,
}

const PROGRAM_ENTRY_NAME: &str = "main";

impl<B> Renderer<B>
where
    B: gfx_hal::Backend,
{
    pub fn new(
        instance: B::Instance,
        mut surface: B::Surface,
        adapter: adapter::Adapter<B>,
        dimensions: window::Extent2D,
    ) -> Renderer<B> {
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.limits();

        let (device, mut queue_group, mut command_pool) = {
            let family = adapter
                .queue_families
                .iter()
                .find(|family| {
                    surface.supports_queue_family(family) && family.queue_type().supports_graphics()
                })
                .expect("No queue family supports presentation");

            let physical_device = &adapter.physical_device;
            let sparsely_bound = physical_device
                .features()
                .contains(Features::SPARSE_BINDING | Features::SPARSE_RESIDENCY_IMAGE_2D);
            let mut gpu = unsafe {
                physical_device
                    .open(
                        &[(family, &[1.0])],
                        if sparsely_bound {
                            Features::SPARSE_BINDING | Features::SPARSE_RESIDENCY_IMAGE_2D
                        } else {
                            Features::empty()
                        },
                    )
                    .unwrap()
            };
            let queue_group = gpu.queue_groups.pop().unwrap();
            let device = gpu.device;

            let command_pool = unsafe {
                device
                    .create_command_pool(queue_group.family, pool::CommandPoolCreateFlags::empty())
            }
            .expect("Can't create command pool");

            (device, queue_group, command_pool)
        };

        let set_layout = ManuallyDrop::new(
            unsafe {
                device.create_descriptor_set_layout(
                    vec![
                        pso::DescriptorSetLayoutBinding {
                            binding: 0,
                            ty: pso::DescriptorType::Image {
                                ty: pso::ImageDescriptorType::Sampled {
                                    with_sampler: false,
                                },
                            },
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                        pso::DescriptorSetLayoutBinding {
                            binding: 1,
                            ty: pso::DescriptorType::Sampler,
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                    ]
                    .into_iter(),
                    iter::empty(),
                )
            }
            .expect("Can't create descriptor set layout"),
        );

        let mut descriptor_pool = ManuallyDrop::new(
            unsafe {
                device.create_descriptor_pool(
                    1,
                    vec![
                        pso::DescriptorRangeDesc {
                            ty: pso::DescriptorType::Image {
                                ty: pso::ImageDescriptorType::Sampled {
                                    with_sampler: false,
                                },
                            },
                            count: 1,
                        },
                        pso::DescriptorRangeDesc {
                            ty: pso::DescriptorType::Sampler,
                            count: 1,
                        },
                    ]
                    .into_iter(),
                    pso::DescriptorPoolCreateFlags::empty(),
                )
            }
            .expect("Can't create descriptor pool"),
        );
        let mut descriptor_set = unsafe { descriptor_pool.allocate_one(&set_layout) }.unwrap();

        let non_coherent_alignment = limits.non_coherent_atom_size as u64;

        let (mut vertex_buffer, buffer_len, buffer_requirements) = {
            let buffer_stride = mem::size_of::<Vertex>() as u64;
            let buffer_len = QUAD.len() as u64 * buffer_stride;
            assert_ne!(buffer_len, 0);
            let padded_buffer_len = ((buffer_len + non_coherent_alignment - 1)
                / non_coherent_alignment)
                * non_coherent_alignment;

            let vertex_buffer = ManuallyDrop::new(
                unsafe { device.create_buffer(padded_buffer_len, buffer::Usage::VERTEX) }.unwrap(),
            );

            let buffer_requirements = unsafe { device.get_buffer_requirements(&vertex_buffer) };

            (vertex_buffer, buffer_len, buffer_requirements)
        };

        let upload_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                // type_mask is a bit field where each bit represents a memory type. If the bit is set
                // to 1 it means we can use that type for our buffer. So this code finds the first
                // memory type that has a `1` (or, is allowed), and is visible to the CPU.
                buffer_requirements.type_mask & (1 << id) != 0
                    && mem_type
                        .properties
                        .contains(memory::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        let buffer_memory = unsafe {
            let mut memory = device
                .allocate_memory(upload_type, buffer_requirements.size)
                .unwrap();
            device
                .bind_buffer_memory(&memory, 0, &mut vertex_buffer)
                .unwrap();
            let mapping = device
                .map_memory(&mut memory, memory::Segment::ALL)
                .unwrap();
            ptr::copy_nonoverlapping(QUAD.as_ptr() as *const u8, mapping, buffer_len as usize);
            device
                .flush_mapped_memory_ranges(iter::once((&memory, memory::Segment::ALL)))
                .unwrap();
            device.unmap_memory(&mut memory);
            ManuallyDrop::new(memory)
        };

        let (mut image_upload_buffer, image_mem_reqs, image_kind, img, image_layout) = {
            let img_data = include_bytes!("./assets/logo.png");
            let img = load_image_from_bytes(img_data);
            let (width, height) = img.dimensions();
            let kind = image::Kind::D2(width as image::Size, height as image::Size, 1, 1);
            let row_alignment_mask = limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
            let image_stride = 4usize;
            let row_pitch =
                (width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask;
            let upload_size = (height * row_pitch) as u64;
            let padded_upload_size = ((upload_size + non_coherent_alignment - 1)
                / non_coherent_alignment)
                * non_coherent_alignment;

            let upload_buffer = ManuallyDrop::new(
                unsafe { device.create_buffer(padded_upload_size, buffer::Usage::TRANSFER_SRC) }
                    .unwrap(),
            );
            let memory_requirements = unsafe { device.get_buffer_requirements(&upload_buffer) };

            let image_layout = ImageLayout {
                height,
                stride: image_stride,
                row_pitch,
                width,
            };

            (upload_buffer, memory_requirements, kind, img, image_layout)
        };

        let image_upload_memory = unsafe {
            let mut memory = device
                .allocate_memory(upload_type, image_mem_reqs.size)
                .unwrap();
            device
                .bind_buffer_memory(&memory, 0, &mut image_upload_buffer)
                .unwrap();
            let mapping = device
                .map_memory(&mut memory, memory::Segment::ALL)
                .unwrap();
            let height = image_layout.height;
            let image_stride = image_layout.stride;
            let row_pitch = image_layout.row_pitch;
            let width = image_layout.width;
            for y in 0..height as usize {
                let row = &(*img)[y * (width as usize) * image_stride
                    ..(y + 1) * (width as usize) * image_stride];
                ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    mapping.offset(y as isize * row_pitch as isize),
                    width as usize * image_stride,
                );
            }
            device
                .flush_mapped_memory_ranges(iter::once((&memory, memory::Segment::ALL)))
                .unwrap();
            device.unmap_memory(&mut memory);
            ManuallyDrop::new(memory)
        };

        let mut image_logo = ManuallyDrop::new(
            unsafe {
                device.create_image(
                    image_kind,
                    1,
                    Rgba8Srgb::SELF,
                    image::Tiling::Optimal,
                    image::Usage::TRANSFER_DST | image::Usage::SAMPLED,
                    image::ViewCapabilities::empty(),
                )
            }
            .unwrap(),
        );
        let image_req = unsafe { device.get_image_requirements(&image_logo) };

        let device_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, memory_type)| {
                image_req.type_mask & (1 << id) != 0
                    && memory_type
                        .properties
                        .contains(memory::Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();
        let image_memory = ManuallyDrop::new(
            unsafe { device.allocate_memory(device_type, image_req.size) }.unwrap(),
        );

        unsafe { device.bind_image_memory(&image_memory, 0, &mut image_logo) }.unwrap();
        let image_srv = ManuallyDrop::new(
            unsafe {
                device.create_image_view(
                    &image_logo,
                    ViewKind::D2,
                    Rgba8Srgb::SELF,
                    Swizzle::NO,
                    SubresourceRange {
                        aspects: gfx_hal::format::Aspects::COLOR,
                        ..Default::default()
                    },
                )
            }
            .unwrap(),
        );

        let sampler = ManuallyDrop::new(
            unsafe {
                device.create_sampler(&image::SamplerDesc::new(
                    image::Filter::Linear,
                    image::WrapMode::Clamp,
                ))
            }
            .expect("Can't create sampler"),
        );

        unsafe {
            device.write_descriptor_set(pso::DescriptorSetWrite {
                set: &mut descriptor_set,
                binding: 0,
                array_offset: 0,
                descriptors: vec![
                    pso::Descriptor::Image(&*image_srv, image::Layout::ShaderReadOnlyOptimal),
                    pso::Descriptor::Sampler(&*sampler),
                ]
                .into_iter(),
            });
        }

        // copy buffer to texture
        let mut copy_fence = device.create_fence(false).expect("Could not create fence");
        unsafe {
            let mut cmd_buffer = command_pool.allocate_one(command::Level::Primary);
            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            let image_barrier = memory::Barrier::Image {
                states: (image::Access::empty(), image::Layout::Undefined)
                    ..(
                        image::Access::TRANSFER_WRITE,
                        image::Layout::TransferDstOptimal,
                    ),
                target: &*image_logo,
                families: None,
                range: image::SubresourceRange {
                    aspects: format::Aspects::COLOR,
                    ..Default::default()
                },
            };

            cmd_buffer.pipeline_barrier(
                PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
                memory::Dependencies::empty(),
                iter::once(image_barrier),
            );

            let height = image_layout.height;
            let image_stride = image_layout.stride;
            let row_pitch = image_layout.row_pitch;
            let width = image_layout.width;

            cmd_buffer.copy_buffer_to_image(
                &image_upload_buffer,
                &image_logo,
                image::Layout::TransferDstOptimal,
                iter::once(command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: row_pitch / (image_stride as u32),
                    buffer_height: height as u32,
                    image_layers: image::SubresourceLayers {
                        aspects: format::Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: image::Offset { x: 0, y: 0, z: 0 },
                    image_extent: image::Extent {
                        width,
                        height,
                        depth: 1,
                    },
                }),
            );

            let image_barrier = memory::Barrier::Image {
                states: (
                    image::Access::TRANSFER_WRITE,
                    image::Layout::TransferDstOptimal,
                )
                    ..(
                        image::Access::SHADER_READ,
                        image::Layout::ShaderReadOnlyOptimal,
                    ),
                target: &*image_logo,
                families: None,
                range: image::SubresourceRange {
                    aspects: format::Aspects::COLOR,
                    ..Default::default()
                },
            };
            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                memory::Dependencies::empty(),
                iter::once(image_barrier),
            );

            cmd_buffer.finish();

            queue_group.queues[0].submit(
                iter::once(&cmd_buffer),
                iter::empty(),
                iter::empty(),
                Some(&mut copy_fence),
            );

            device
                .wait_for_fence(&copy_fence, !0)
                .expect("Can't wait for fence");
        }

        unsafe {
            device.destroy_fence(copy_fence);
        }

        let caps = surface.capabilities(&adapter.physical_device);
        let formats = surface.supported_formats(&adapter.physical_device);
        info!("formats: {:?}", formats);
        let format = formats.map_or(format::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let swap_config = window::SwapchainConfig::from_caps(&caps, format, dimensions);
        let fat = swap_config.framebuffer_attachment();
        info!("{:?}", swap_config);
        let extent = swap_config.extent;
        unsafe {
            surface
                .configure_swapchain(&device, swap_config)
                .expect("Can't configure swapchain");
        };

        let render_pass = {
            let attachment = pass::Attachment {
                format: Some(format),
                samples: 1,
                ops: pass::AttachmentOps::new(
                    pass::AttachmentLoadOp::Clear,
                    pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: pass::AttachmentOps::DONT_CARE,
                layouts: image::Layout::Undefined..image::Layout::Present,
            };

            let subpass = pass::SubpassDesc {
                colors: &[(0, image::Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            ManuallyDrop::new(
                unsafe {
                    device.create_render_pass(
                        iter::once(attachment),
                        iter::once(subpass),
                        iter::empty(),
                    )
                }
                .expect("Can't create render pass"),
            )
        };

        let framebuffer = ManuallyDrop::new(unsafe {
            device
                .create_framebuffer(
                    &render_pass,
                    iter::once(fat),
                    image::Extent {
                        width: dimensions.width,
                        height: dimensions.height,
                        depth: 1,
                    },
                )
                .unwrap()
        });

        // Define maximum number of frames we want to be able to be "in flight" (being computed
        // simultaneously) at once
        let frames_in_flight = 3;

        // The number of the rest of the resources is based on the frames in flight.
        let mut submission_complete_semaphores = Vec::with_capacity(frames_in_flight);
        let mut submission_complete_fences = Vec::with_capacity(frames_in_flight);
        // Note: We don't really need a different command pool per frame in such a simple demo like this,
        // but in a more 'real' application, it's generally seen as optimal to have one command pool per
        // thread per frame. There is a flag that lets a command pool reset individual command buffers
        // which are created from it, but by default the whole pool (and therefore all buffers in it)
        // must be reset at once. Furthermore, it is often the case that resetting a whole pool is actually
        // faster and more efficient for the hardware than resetting individual command buffers, so it's
        // usually best to just make a command pool for each set of buffers which need to be reset at the
        // same time (each frame). In our case, each pool will only have one command buffer created from it,
        // though.
        let mut cmd_pools = Vec::with_capacity(frames_in_flight);
        let mut cmd_buffers = Vec::with_capacity(frames_in_flight);

        cmd_pools.push(command_pool);
        for _ in 1..frames_in_flight {
            unsafe {
                cmd_pools.push(
                    device
                        .create_command_pool(
                            queue_group.family,
                            pool::CommandPoolCreateFlags::empty(),
                        )
                        .expect("Can't create command pool"),
                );
            }
        }

        for i in 0..frames_in_flight {
            submission_complete_semaphores.push(
                device
                    .create_semaphore()
                    .expect("Could not create semaphore"),
            );
            submission_complete_fences
                .push(device.create_fence(true).expect("Could not create fence"));
            cmd_buffers.push(unsafe { cmd_pools[i].allocate_one(command::Level::Primary) });
        }

        let pipeline_layout = ManuallyDrop::new(
            unsafe { device.create_pipeline_layout(iter::once(&*set_layout), iter::empty()) }
                .expect("Can't create pipeline layout"),
        );
        let pipeline = {
            let vs_module = {
                let spirv = gfx_auxil::read_spirv(Cursor::new(
                    &include_bytes!("./assets/quad.vert.spv")[..],
                ))
                .unwrap();
                unsafe { device.create_shader_module(&spirv) }.unwrap()
            };
            let fs_module = {
                let spirv = gfx_auxil::read_spirv(Cursor::new(
                    &include_bytes!("./assets/quad.frag.spv")[..],
                ))
                .unwrap();
                unsafe { device.create_shader_module(&spirv) }.unwrap()
            };

            let pipeline = {
                let (vs_entry, fs_entry) = (
                    pso::EntryPoint {
                        entry: PROGRAM_ENTRY_NAME,
                        module: &vs_module,
                        specialization: gfx_hal::spec_const_list![0.8f32],
                    },
                    pso::EntryPoint {
                        entry: PROGRAM_ENTRY_NAME,
                        module: &fs_module,
                        specialization: pso::Specialization::default(),
                    },
                );

                let subpass = Subpass {
                    index: 0,
                    main_pass: &*render_pass,
                };

                let vertex_buffers = vec![pso::VertexBufferDesc {
                    binding: 0,
                    stride: mem::size_of::<Vertex>() as u32,
                    rate: VertexInputRate::Vertex,
                }];

                let attributes = vec![
                    pso::AttributeDesc {
                        location: 0,
                        binding: 0,
                        element: pso::Element {
                            format: format::Format::Rg32Sfloat,
                            offset: 0,
                        },
                    },
                    pso::AttributeDesc {
                        location: 1,
                        binding: 0,
                        element: pso::Element {
                            format: format::Format::Rg32Sfloat,
                            offset: 8,
                        },
                    },
                ];

                let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
                    pso::PrimitiveAssemblerDesc::Vertex {
                        buffers: &vertex_buffers,
                        attributes: &attributes,
                        input_assembler: pso::InputAssemblerDesc {
                            primitive: pso::Primitive::TriangleList,
                            with_adjacency: false,
                            restart_index: None,
                        },
                        vertex: vs_entry,
                        geometry: None,
                        tessellation: None,
                    },
                    pso::Rasterizer::FILL,
                    Some(fs_entry),
                    &*pipeline_layout,
                    subpass,
                );

                pipeline_desc.blender.targets.push(pso::ColorBlendDesc {
                    mask: pso::ColorMask::ALL,
                    blend: Some(pso::BlendState::ALPHA),
                });

                unsafe { device.create_graphics_pipeline(&pipeline_desc, None) }
            };

            unsafe {
                device.destroy_shader_module(vs_module);
            }
            unsafe {
                device.destroy_shader_module(fs_module);
            }

            ManuallyDrop::new(pipeline.unwrap())
        };

        // Rendering setup
        let viewport = pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as _,
                h: extent.height as _,
            },
            depth: 0.0..1.0,
        };

        Renderer {
            adapter,
            buffer_memory,
            cmd_pools,
            cmd_buffers,
            descriptor_pool,
            descriptor_set: Some(descriptor_set),
            device,
            dimensions,
            format,
            frame: 0,
            framebuffer,
            frames_in_flight,
            image_logo,
            image_memory,
            image_srv,
            image_upload_buffer,
            image_upload_memory,
            instance,
            pipeline,
            pipeline_layout,
            queue_group,
            render_pass,
            sampler,
            set_layout,
            surface: ManuallyDrop::new(surface),
            submission_complete_fences,
            submission_complete_semaphores,
            vertex_buffer,
            viewport,
        }
    }

    pub fn recreate_swapchain(&mut self) {
        let caps = self.surface.capabilities(&self.adapter.physical_device);
        let swap_config = window::SwapchainConfig::from_caps(&caps, self.format, self.dimensions);
        info!("{:?}", swap_config);

        let extent = swap_config.extent.to_extent();
        self.viewport.rect.w = extent.width as _;
        self.viewport.rect.h = extent.height as _;

        unsafe {
            self.device
                .destroy_framebuffer(ManuallyDrop::into_inner(ptr::read(&self.framebuffer)));
            self.framebuffer = ManuallyDrop::new(
                self.device
                    .create_framebuffer(
                        &self.render_pass,
                        iter::once(swap_config.framebuffer_attachment()),
                        extent,
                    )
                    .unwrap(),
            )
        };

        unsafe {
            self.surface
                .configure_swapchain(&self.device, swap_config)
                .expect("Can't create swapchain");
        }
    }

    pub fn render(&mut self) {
        let surface_image = unsafe {
            match self.surface.acquire_image(!0) {
                Ok((image, _)) => image,
                Err(_) => {
                    self.recreate_swapchain();
                    return;
                }
            }
        };

        // Compute index into our resource ring buffers based on the frame number
        // and number of frames in flight. Pay close attention to where this index is needed
        // versus when the swapchain image index we got from acquire_image is needed.
        let frame_idx = self.frame as usize % self.frames_in_flight;

        // Wait for the fence of the previous submission of this frame and reset it; ensures we are
        // submitting only up to maximum number of frames_in_flight if we are submitting faster than
        // the gpu can keep up with. This would also guarantee that any resources which need to be
        // updated with a CPU->GPU data copy are not in use by the GPU, so we can perform those updates.
        // In this case there are none to be done, however.
        unsafe {
            let fence = &mut self.submission_complete_fences[frame_idx];
            self.device
                .wait_for_fence(fence, !0)
                .expect("Failed to wait for fence");
            self.device
                .reset_fence(fence)
                .expect("Failed to reset fence");
            self.cmd_pools[frame_idx].reset(false);
        }

        // Rendering
        let cmd_buffer = &mut self.cmd_buffers[frame_idx];
        unsafe {
            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            cmd_buffer.set_viewports(0, iter::once(self.viewport.clone()));
            cmd_buffer.set_scissors(0, iter::once(self.viewport.rect));
            cmd_buffer.bind_graphics_pipeline(&self.pipeline);
            cmd_buffer.bind_vertex_buffers(
                0,
                iter::once((&*self.vertex_buffer, buffer::SubRange::WHOLE)),
            );
            cmd_buffer.bind_graphics_descriptor_sets(
                &self.pipeline_layout,
                0,
                self.descriptor_set.as_ref().into_iter(),
                iter::empty(),
            );

            cmd_buffer.begin_render_pass(
                &self.render_pass,
                &self.framebuffer,
                self.viewport.rect,
                iter::once(command::RenderAttachmentInfo {
                    image_view: surface_image.borrow(),
                    clear_value: command::ClearValue {
                        color: command::ClearColor {
                            float32: [0.8, 0.8, 0.8, 1.0],
                        },
                    },
                }),
                command::SubpassContents::Inline,
            );
            cmd_buffer.draw(0..6, 0..1);
            cmd_buffer.end_render_pass();
            cmd_buffer.finish();

            self.queue_group.queues[0].submit(
                iter::once(&*cmd_buffer),
                iter::empty(),
                iter::once(&self.submission_complete_semaphores[frame_idx]),
                Some(&mut self.submission_complete_fences[frame_idx]),
            );

            // present frame
            let result = self.queue_group.queues[0].present(
                &mut self.surface,
                surface_image,
                Some(&mut self.submission_complete_semaphores[frame_idx]),
            );

            if result.is_err() {
                self.recreate_swapchain();
            }
        }

        // Increment our frame
        self.frame += 1;
    }
}

impl<B> Drop for Renderer<B>
where
    B: gfx_hal::Backend,
{
    fn drop(&mut self) {
        self.device.wait_idle().unwrap();
        unsafe {
            let _ = self.descriptor_set.take();
            self.device
                .destroy_descriptor_pool(ManuallyDrop::into_inner(ptr::read(
                    &self.descriptor_pool,
                )));
            self.device
                .destroy_descriptor_set_layout(ManuallyDrop::into_inner(ptr::read(
                    &self.set_layout,
                )));

            self.device
                .destroy_buffer(ManuallyDrop::into_inner(ptr::read(&self.vertex_buffer)));
            self.device
                .destroy_buffer(ManuallyDrop::into_inner(ptr::read(
                    &self.image_upload_buffer,
                )));
            self.device
                .destroy_image(ManuallyDrop::into_inner(ptr::read(&self.image_logo)));
            self.device
                .destroy_image_view(ManuallyDrop::into_inner(ptr::read(&self.image_srv)));
            self.device
                .destroy_sampler(ManuallyDrop::into_inner(ptr::read(&self.sampler)));
            for p in self.cmd_pools.drain(..) {
                self.device.destroy_command_pool(p);
            }
            for s in self.submission_complete_semaphores.drain(..) {
                self.device.destroy_semaphore(s);
            }
            for f in self.submission_complete_fences.drain(..) {
                self.device.destroy_fence(f);
            }
            self.device
                .destroy_render_pass(ManuallyDrop::into_inner(ptr::read(&self.render_pass)));
            self.device
                .destroy_framebuffer(ManuallyDrop::into_inner(ptr::read(&self.framebuffer)));
            self.surface.unconfigure_swapchain(&self.device);
            self.device
                .free_memory(ManuallyDrop::into_inner(ptr::read(&self.buffer_memory)));
            self.device
                .free_memory(ManuallyDrop::into_inner(ptr::read(&self.image_memory)));
            self.device.free_memory(ManuallyDrop::into_inner(ptr::read(
                &self.image_upload_memory,
            )));
            self.device
                .destroy_graphics_pipeline(ManuallyDrop::into_inner(ptr::read(&self.pipeline)));
            self.device
                .destroy_pipeline_layout(ManuallyDrop::into_inner(ptr::read(
                    &self.pipeline_layout,
                )));

            let surface = ManuallyDrop::into_inner(ptr::read(&self.surface));
            self.instance.destroy_surface(surface);
        }
    }
}
