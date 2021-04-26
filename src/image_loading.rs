use std::{io::Cursor, vec};

pub fn load_image_from_bytes(
    image_data: &[u8],
) -> image::ImageBuffer<image::Rgba<u8>, vec::Vec<u8>> {
    let img = image::load(Cursor::new(&image_data[..]), image::ImageFormat::Png)
        .unwrap()
        .to_rgba8();

    img
}
