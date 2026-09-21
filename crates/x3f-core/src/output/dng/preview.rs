//! IFD0 preview image for DNG output.
//!
//! Finder, Quick Look, Photos, Lightroom and most other viewers show the
//! DNG's IFD0 preview instead of rendering the raw plane whenever it is
//! large enough for the requested size. The legacy writer rendered that
//! preview itself from the linear raw data (300 px wide, no camera profile
//! or tone curve, per-channel clipping), which produced flat, magenta-
//! highlighted thumbnails. Every X3F carries the camera's own full-size
//! JPEG rendering, so the preview is now that JPEG, decoded, box-
//! downsampled to at most [`MAX_LONG_EDGE`] pixels on the long edge and
//! re-encoded as a plain baseline JFIF stream (the camera's EXIF is not
//! duplicated into the preview). The rendered preview remains the fallback
//! for files without a decodable embedded JPEG.
//!
//! The camera JPEG's pixels are stored in sensor orientation, exactly like
//! the raw plane, so IFD0's `Orientation` tag applies to both unchanged.

use rayon::prelude::*;

/// Longest edge of the embedded preview, in pixels. Adobe's DNG Converter
/// writes 1024 px previews by default; 1600 px keeps Finder and Photos
/// thumbnails sharp on Retina displays while adding well under a megabyte
/// to each file.
pub(crate) const MAX_LONG_EDGE: u32 = 1600;

/// Baseline JPEG quality of the re-encoded preview.
const QUALITY: u8 = 90;

/// A JPEG-compressed preview ready to be written as a single IFD0 strip.
pub(crate) struct JpegPreview {
    /// Complete JFIF stream (SOI … EOI).
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// The preview that ends up in IFD0: the camera JPEG when it decodes, the
/// legacy raw-rendered RGB strip otherwise.
pub(crate) enum PreviewImage {
    Jpeg(JpegPreview),
    /// Uncompressed 8-bit interleaved RGB, one strip.
    Rgb {
        bytes: Vec<u8>,
        width: u32,
        height: u32,
    },
}

impl PreviewImage {
    pub(crate) fn bytes(&self) -> &[u8] {
        match self {
            PreviewImage::Jpeg(p) => &p.bytes,
            PreviewImage::Rgb { bytes, .. } => bytes,
        }
    }

    pub(crate) fn dimensions(&self) -> (u32, u32) {
        match self {
            PreviewImage::Jpeg(p) => (p.width, p.height),
            PreviewImage::Rgb { width, height, .. } => (*width, *height),
        }
    }
}

/// Whether `model` gets the camera-JPEG preview. Limited to the TRUE II
/// bodies, whose raw-rendered preview looked flat and magenta; other cameras
/// keep the legacy rendered preview.
pub(crate) fn uses_camera_jpeg(model: Option<&str>) -> bool {
    matches!(
        model.map(str::trim),
        Some("SIGMA DP1X") | Some("SIGMA DP2X") | Some("SIGMA SD15")
    )
}

/// Build the IFD0 preview from the camera's embedded JPEG. Returns `None`
/// when the stream cannot be decoded, in which case the caller falls back
/// to the raw-rendered preview.
pub(crate) fn from_camera_jpeg(jpeg: &[u8], max_long_edge: u32) -> Option<JpegPreview> {
    let (rgb, width, height) = decode_rgb(jpeg)?;
    let factor = width
        .max(height)
        .div_ceil(max_long_edge.max(1) as usize)
        .max(1);
    let (pixels, out_w, out_h) = if factor == 1 {
        (rgb, width, height)
    } else {
        box_downsample(&rgb, width, height, factor)
    };
    if out_w == 0 || out_h == 0 || out_w > u16::MAX as usize || out_h > u16::MAX as usize {
        return None;
    }

    let mut bytes = Vec::with_capacity(pixels.len() / 8);
    let mut encoder = jpeg_encoder::Encoder::new(&mut bytes, QUALITY);
    // 4:2:0 chroma subsampling; IFD0's YCbCrSubSampling tag must match.
    encoder.set_sampling_factor(jpeg_encoder::SamplingFactor::F_2_2);
    encoder
        .encode(
            &pixels,
            out_w as u16,
            out_h as u16,
            jpeg_encoder::ColorType::Rgb,
        )
        .ok()?;

    Some(JpegPreview {
        bytes,
        width: out_w as u32,
        height: out_h as u32,
    })
}

/// Decode a JPEG stream to interleaved 8-bit RGB.
fn decode_rgb(jpeg: &[u8]) -> Option<(Vec<u8>, usize, usize)> {
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;

    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(jpeg, options);
    let pixels = decoder.decode().ok()?;
    let (width, height) = decoder.dimensions()?;
    if width == 0 || height == 0 || pixels.len() != width * height * 3 {
        return None;
    }
    Some((pixels, width, height))
}

/// Average `factor`×`factor` blocks of an interleaved RGB image. Rows and
/// columns that do not fill a whole block are dropped, so the output is
/// `(width / factor) × (height / factor)`.
fn box_downsample(
    rgb: &[u8],
    width: usize,
    height: usize,
    factor: usize,
) -> (Vec<u8>, usize, usize) {
    let out_w = width / factor;
    let out_h = height / factor;
    let mut out = vec![0u8; out_w * out_h * 3];
    let block = (factor * factor) as u32;

    out.par_chunks_mut(out_w * 3)
        .enumerate()
        .for_each(|(oy, out_row)| {
            let y0 = oy * factor;
            for ox in 0..out_w {
                let x0 = ox * factor;
                let mut acc = [0u32; 3];
                for y in y0..y0 + factor {
                    let row = &rgb[y * width * 3..(y + 1) * width * 3];
                    for x in x0..x0 + factor {
                        let px = &row[x * 3..x * 3 + 3];
                        acc[0] += px[0] as u32;
                        acc[1] += px[1] as u32;
                        acc[2] += px[2] as u32;
                    }
                }
                let o = ox * 3;
                out_row[o] = ((acc[0] + block / 2) / block) as u8;
                out_row[o + 1] = ((acc[1] + block / 2) / block) as u8;
                out_row[o + 2] = ((acc[2] + block / 2) / block) as u8;
            }
        });

    (out, out_w, out_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a synthetic gradient as a JPEG, the way a camera would.
    fn synthetic_jpeg(width: u16, height: u16) -> Vec<u8> {
        let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                rgb.push((x as u32 * 255 / width as u32) as u8);
                rgb.push((y as u32 * 255 / height as u32) as u8);
                rgb.push(128);
            }
        }
        let mut out = Vec::new();
        jpeg_encoder::Encoder::new(&mut out, 95)
            .encode(&rgb, width, height, jpeg_encoder::ColorType::Rgb)
            .unwrap();
        out
    }

    #[test]
    fn downsamples_to_the_long_edge_limit() {
        let preview = from_camera_jpeg(&synthetic_jpeg(640, 480), 200).expect("decodes");
        // ceil(640 / 200) = 4 → 160 × 120.
        assert_eq!((preview.width, preview.height), (160, 120));
        assert_eq!(&preview.bytes[..2], &[0xFF, 0xD8], "SOI marker");
        assert_eq!(
            &preview.bytes[preview.bytes.len() - 2..],
            &[0xFF, 0xD9],
            "EOI marker"
        );

        // The stream must decode back to the advertised dimensions.
        let (_, w, h) = decode_rgb(&preview.bytes).expect("re-decodes");
        assert_eq!((w, h), (160, 120));
    }

    #[test]
    fn small_jpegs_are_re_encoded_at_full_size() {
        let preview = from_camera_jpeg(&synthetic_jpeg(300, 200), 1600).expect("decodes");
        assert_eq!((preview.width, preview.height), (300, 200));
    }

    #[test]
    fn box_downsample_averages_blocks() {
        // 4×2 image, factor 2 → 2×1. Left block is all 10s, right block all 250s.
        let rgb: Vec<u8> = [10u8; 6]
            .iter()
            .chain([250u8; 6].iter())
            .chain([10u8; 6].iter())
            .chain([250u8; 6].iter())
            .copied()
            .collect();
        let (out, w, h) = box_downsample(&rgb, 4, 2, 2);
        assert_eq!((w, h), (2, 1));
        assert_eq!(out, vec![10, 10, 10, 250, 250, 250]);
    }

    #[test]
    fn only_true2_bodies_use_the_camera_jpeg() {
        for m in ["SIGMA DP1X", "SIGMA DP2X", "SIGMA SD15", " SIGMA DP2X "] {
            assert!(uses_camera_jpeg(Some(m)), "{m}");
        }
        for m in [
            "SIGMA DP2",
            "SIGMA DP1S",
            "SIGMA SD14",
            "SIGMA DP2 Merrill",
            "SIGMA dp2 Quattro",
            "SIGMA sd Quattro H",
            "",
        ] {
            assert!(!uses_camera_jpeg(Some(m)), "{m}");
        }
        assert!(!uses_camera_jpeg(None));
    }

    #[test]
    fn garbage_is_rejected() {
        assert!(from_camera_jpeg(b"not a jpeg", 1600).is_none());
        assert!(from_camera_jpeg(&[], 1600).is_none());
    }
}
