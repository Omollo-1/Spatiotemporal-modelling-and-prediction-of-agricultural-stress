# %%
# Cell - Download helpers: pull an Earth Engine image onto OUR grid in tiles and save one GeoTIFF
import io, time, requests
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom

NODATA = -9999.0
AOI_GRID_GEOMETRY = transform_geom("EPSG:4326", GRID["crs"], CFG["aoi"]["geometry"])


def tile_windows(width, height, tile=512):
    """Pixel windows (col0, row0, w, h) that cover the whole grid."""
    return [(c, r, min(tile, width - c), min(tile, height - r))
            for r in range(0, height, tile) for c in range(0, width, tile)]


def fetch_tile(image, g, col0, row0, w, h, retries=4):
    """Download one tile of an ee.Image as a numpy array [bands, rows, cols]."""
    res = g["res"]
    x0, y1 = g["xmin"] + col0 * res, g["ymax"] - row0 * res
    region = ee.Geometry.Rectangle([x0, y1 - h * res, x0 + w * res, y1], proj=g["crs"], geodesic=False)
    params = {"region": region, "crs": g["crs"], "crs_transform": [res, 0, x0, 0, -res, y1], "format": "GEO_TIFF"}
    for attempt in range(retries):
        try:
            r = requests.get(image.getDownloadURL(params), timeout=900)
            r.raise_for_status()
            with rasterio.MemoryFile(r.content) as mem, mem.open() as src:
                return src.read().astype("float32")
        except Exception as err:
            print(f"   tile retry {attempt + 1}/{retries}: {str(err)[:120]}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Tile download failed - try a smaller tile size or re-run (finished files are skipped).")


def download_image(image, out_path, names, g, tile=512):
    """Download `image` (bands must match `names`) onto grid `g`; skip if the file already exists."""
    out_path = Path(out_path)
    if out_path.exists():
        print("   exists, skipped:", out_path.name)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full = np.full((len(names), g["height"], g["width"]), NODATA, dtype="float32")
    for (c0, r0, w, h) in tile_windows(g["width"], g["height"], tile):
        arr = fetch_tile(image, g, c0, r0, w, h)
        if arr.shape[0] != len(names):
            raise ValueError(f"Expected {len(names)} bands, got {arr.shape[0]}")
        hh, ww = min(h, arr.shape[1]), min(w, arr.shape[2])      # guard against 1-pixel rounding at tile edges
        full[:, r0:r0 + hh, c0:c0 + ww] = arr[:, :hh, :ww]
    transform = from_origin(g["xmin"], g["ymax"], g["res"], g["res"])
    outside_aoi = geometry_mask(
        [AOI_GRID_GEOMETRY], out_shape=(g["height"], g["width"]), transform=transform
    )
    full[:, outside_aoi] = NODATA
    with rasterio.open(out_path, "w", driver="GTiff", height=g["height"], width=g["width"], count=len(names),
                       dtype="float32", crs=g["crs"], transform=transform, nodata=NODATA, compress="deflate") as dst:
        dst.write(full)
        dst.descriptions = tuple(names)
    print("   saved:", out_path.name)
