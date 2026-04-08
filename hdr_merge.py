import os
import glob
import math
import cv2
import numpy as np

# Optional EXIF reader (pure python). Install if missing: pip install exifread
try:
    import exifread
    HAS_EXIFREAD = True
except ImportError:
    HAS_EXIFREAD = False


def _ratio_to_float(r):
    # exifread returns Ratio objects sometimes; also strings like "1/125"
    try:
        return float(r)
    except Exception:
        s = str(r)
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)


def read_exif_exposure(path):
    """
    Returns (shutter_seconds, f_number, iso) or (None, None, None) if not available.
    """
    if not HAS_EXIFREAD:
        return None, None, None

    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    def get(tagname):
        return tags.get(tagname)

    shutter = get("EXIF ExposureTime") or get("EXIF ShutterSpeedValue")
    fnum = get("EXIF FNumber") or get("EXIF ApertureValue")
    iso = get("EXIF ISOSpeedRatings") or get("EXIF ISO")

    # ExposureTime is best (already seconds)
    shutter_s = None
    if shutter is not None:
        shutter_s = _ratio_to_float(shutter)

    f_number = None
    if fnum is not None:
        # If ApertureValue is in APEX, this is wrong; FNumber is preferred.
        # Most files provide EXIF FNumber, so we use it as-is.
        f_number = _ratio_to_float(fnum)

    iso_val = None
    if iso is not None:
        try:
            iso_val = int(str(iso).split()[0])
        except Exception:
            iso_val = int(_ratio_to_float(iso))

    return shutter_s, f_number, iso_val


def effective_exposure_time(shutter_s, f_number, iso,
                            ref_f=None, ref_iso=None):
    """
    OpenCV's MergeDebevec expects a list of exposure times proportional to total exposure.
    If aperture or ISO changes, fold them into an 'effective time':
        exposure ∝ shutter * (1 / f^2) * ISO
    We can scale relative to a reference (first image) to keep numbers reasonable.
    """
    if shutter_s is None:
        return None

    # If no aperture/ISO, fall back to shutter only
    if f_number is None and iso is None:
        return shutter_s

    if ref_f is None:
        ref_f = f_number if f_number is not None else 1.0
    if ref_iso is None:
        ref_iso = iso if iso is not None else 100

    f = f_number if f_number is not None else ref_f
    I = iso if iso is not None else ref_iso

    # Scale shutter by aperture and ISO relative to reference:
    # exposure ratio = (t / t_ref) * (ref_f^2 / f^2) * (I / ref_I)
    # We can bake reference into the "time" list:
    t_eff = shutter_s * ((ref_f * ref_f) / (f * f)) * (I / ref_iso)
    return float(t_eff)


def main(input_glob, out_exr="hdr.exr"):
    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise SystemExit(f"No files match: {input_glob}")

    images = []
    times = []

    # Use first image as reference for aperture/ISO scaling
    ref_shutter, ref_f, ref_iso = read_exif_exposure(paths[0])
    if ref_iso is None:
        ref_iso = 100

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Could not read image: {p}")

        images.append(img)

        shutter_s, fnum, iso = read_exif_exposure(p)
        t_eff = effective_exposure_time(shutter_s, fnum, iso, ref_f=ref_f, ref_iso=ref_iso)

        # If we can't read EXIF, fall back to 1.0 and warn later
        times.append(1.0 if t_eff is None else t_eff)

    times = np.array(times, dtype=np.float32)
    print("Len TImes",len(times))
    print("Len images", len(images))

    # Alignment (works best if camera viewpoint is basically fixed)
    align = cv2.createAlignMTB()
    aligned_images = []
    align.process(images, aligned_images)
    print("Len aligned", len(aligned_images))

    # Merge to HDR radiance map
    merge = cv2.createMergeDebevec()
    hdr = merge.process(images, times=times)

    # Write EXR (32-bit float)
    cv2.imwrite("hdr.hdr", hdr)
    ok = cv2.imwrite(out_exr, hdr)
    if not ok:
        raise SystemExit(
            "Failed to write EXR. Your OpenCV build might not have OpenEXR support.\n"
            "Try writing .hdr instead, or install opencv with OpenEXR enabled."
        )

    print(f"Wrote {out_exr}")
    print("Exposure times used (effective):", times.tolist())


if __name__ == "__main__":
    # Example:
    # python hdr_merge.py "/path/to/exports/*.tif" out.exr
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hdr_merge.py '<glob>' [out.exr]")
        raise SystemExit(1)
    glob_in = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "hdr.exr"
    main(glob_in, out)