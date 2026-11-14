# %% [markdown]
# # 03 - Sentinel-1 interferograms with ASF HyP3
# **REAL mode only.** Finds Sentinel-1 IW SLC scenes over the study area, designs an interferogram network, and (optionally) orders
# the interferograms from the free on-demand HyP3 service at the Alaska Satellite Facility. You do **not** need to download raw SLC data
# or install heavy SAR software: HyP3 does the processing in the cloud (GAMMA software).
#
# **You need:** a free NASA Earthdata login (https://urs.earthdata.nasa.gov) and to have accepted the ASF EULA once at
# https://hyp3-api.asf.alaska.edu. HyP3 gives each user a monthly allowance of processing **credits**; each interferogram costs a few to
# ~15 credits depending on settings. Check the current allowance and prices in the HyP3 docs before you submit (they change).
#
# **Safe by default:** `SUBMIT = False` only searches, builds the network and prints the cost. Nothing is ordered until you set it to `True`.
#
# **Network design.** Each scene is paired with its next `MAX_NEIGHBOURS` scenes within `MAX_DT_DAYS`. Short temporal baselines keep
# coherence over vegetation; a few redundant links let the time-series inversion (MintPy, notebook 04) detect and remove unwrapping errors.
# Ascending and descending tracks are processed separately and combined later.

# @@boot_env@@

# @@boot_cfg@@

# %%
# Cell 3 - Packages
ensure("asf-search", "hyp3-sdk", "shapely", "pandas", "numpy", "matplotlib")
import numpy as np, pandas as pd
import asf_search as asf
import hyp3_sdk as sdk
from shapely.geometry import shape
from datetime import datetime, timedelta
INSAR = DATA / "insar"
INSAR.mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
print("asf_search", asf.__version__, "| hyp3_sdk", sdk.__version__)

# %% [markdown]
# ## 1. Settings

# %%
# Cell 4 - Edit these
SUBMIT = False                       # False = dry run (search + cost estimate only). True = really order the jobs.
START, END = f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31"
DIRECTIONS = ["ASCENDING", "DESCENDING"]
MAX_NEIGHBOURS = 2                   # pair each scene with its next N scenes
MAX_DT_DAYS = 36                     # ...but only if closer than this (12-day repeat => 12, 24, 36 day pairs)
LOOKS = "20x4"                       # "20x4" ~ 80 m pixels, cheaper. "10x2" ~ 40 m, more credits.
CREDITS_PER_JOB = 10                 # approximate for 20x4 - VERIFY in the HyP3 docs / hyp3.costs() below
MAX_PAIRS_PER_DIRECTION = 400        # safety cap so you never order thousands of jobs by accident
OVERRIDE = {"ASCENDING": None, "DESCENDING": None}   # force a track/frame, e.g. {"ASCENDING": (174, 458)} = (relative orbit, frame)
AOI_WKT = shape(CFG["aoi"]["geometry"]).wkt

# %% [markdown]
# ## 2. Search for Sentinel-1 scenes

# %%
# Cell 5 - Search ASF (no login needed for searching)
def search_scenes(direction):
    res = asf.geo_search(intersectsWith=AOI_WKT, platform=asf.PLATFORM.SENTINEL1, processingLevel=asf.PRODUCT_TYPE.SLC,
                         beamMode=asf.BEAMMODE.IW, flightDirection=direction, start=START, end=END)
    df = pd.DataFrame([r.properties for r in res])
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["startTime"]).dt.tz_localize(None)
    return df[["sceneName", "time", "pathNumber", "frameNumber", "platform", "flightDirection"]].sort_values("time")


scenes = {d: search_scenes(d) for d in DIRECTIONS}
for d, df in scenes.items():
    print(d, "->", len(df), "scenes")

# %%
# Cell 6 - Which track/frame? Count scenes per (relative orbit, frame) and per year
for d, df in scenes.items():
    if df.empty:
        continue
    print("==", d)
    summary = df.groupby(["pathNumber", "frameNumber"]).agg(n=("sceneName", "size"), first=("time", "min"), last=("time", "max"))
    print(summary.sort_values("n", ascending=False).head(5))

# %% [markdown]
# **Pick one track/frame per direction.** By default the one with the most scenes is used; check that it actually covers your AOI
# (look at the footprint on https://search.asf.alaska.edu) and override above if needed.

# %%
# Cell 7 - Select the track and report the yearly acquisition count (Sentinel-1B failed in Dec 2021 -> 12-day repeat becomes 24-day)
chosen = {}
for d, df in scenes.items():
    if df.empty:
        continue
    key = OVERRIDE[d] or df.groupby(["pathNumber", "frameNumber"]).size().idxmax()
    sel = df[(df["pathNumber"] == key[0]) & (df["frameNumber"] == key[1])].drop_duplicates("sceneName").reset_index(drop=True)
    chosen[d] = sel
    print(d, "track", key, "|", len(sel), "scenes | per year:", sel["time"].dt.year.value_counts().sort_index().to_dict())

# %% [markdown]
# ## 3. Interferogram network

# %%
# Cell 8 - Build pairs: each scene to its next MAX_NEIGHBOURS scenes within MAX_DT_DAYS
def build_pairs(sel, n_neighbours, max_dt_days):
    pairs = []
    for i in range(len(sel)):
        for j in range(i + 1, min(i + 1 + n_neighbours, len(sel))):
            if (sel.loc[j, "time"] - sel.loc[i, "time"]).days <= max_dt_days:
                pairs.append((sel.loc[i, "sceneName"], sel.loc[j, "sceneName"], sel.loc[i, "time"], sel.loc[j, "time"]))
    return pairs


networks = {d: build_pairs(sel, MAX_NEIGHBOURS, MAX_DT_DAYS) for d, sel in chosen.items()}
for d, p in networks.items():
    if len(p) > MAX_PAIRS_PER_DIRECTION:
        print(f"{d}: {len(p)} pairs exceeds the cap - keeping every {int(np.ceil(len(p) / MAX_PAIRS_PER_DIRECTION))}th link. "
              "Prefer shortening the period or MAX_NEIGHBOURS instead of thinning.")
        networks[d] = p[::int(np.ceil(len(p) / MAX_PAIRS_PER_DIRECTION))]
total = sum(len(p) for p in networks.values())
print({d: len(p) for d, p in networks.items()}, "| total jobs:", total, "| approx credits:", total * CREDITS_PER_JOB)

# %%
# Cell 9 - Plot the network (time vs. perpendicular-free simple view: link lengths in days)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, len(networks), figsize=(6 * max(len(networks), 1), 3.5), squeeze=False)
for a, (d, p) in zip(ax[0], networks.items()):
    for (_, _, t1, t2) in p:
        a.plot([t1, t2], [0, 0], lw=0.4, alpha=0.4, color="tab:blue")
    a.hist([t for (_, _, t, _) in p], bins=40, color="tab:orange", alpha=0.7)
    a.set_title(f"{d}: {len(p)} interferograms (histogram of start dates)")
fig.savefig(OUT / "figures" / "03_insar_network.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Log in and check credits

# %%
# Cell 10 - Earthdata login (credentials are never stored in the notebook)
import getpass
import os

earthdata_username = os.environ.get("EARTHDATA_USERNAME")
earthdata_password = os.environ.get("EARTHDATA_PASSWORD")
if bool(earthdata_username) != bool(earthdata_password):
    raise ValueError("Set both EARTHDATA_USERNAME and EARTHDATA_PASSWORD, or leave both unset for secure prompts.")
if not earthdata_username:
    print("Enter Earthdata credentials; input is hidden and values are not saved.")
    earthdata_username = getpass.getpass("Earthdata username: ")
    earthdata_password = getpass.getpass("Earthdata password: ")
    if not earthdata_username or not earthdata_password:
        raise ValueError("Both Earthdata username and password are required.")

hyp3 = sdk.HyP3(username=earthdata_username, password=earthdata_password)
del earthdata_username, earthdata_password
print("Credits remaining:", hyp3.check_credits())

# %% [markdown]
# ## 5. Submit (only if `SUBMIT = True`)

# %%
# Cell 11 - Order the interferograms. Existing jobs with the same name are skipped, so re-running never double-orders.
def job_name(tag, g1, g2):
    return f"naivasha_{tag}_{g1[17:25]}_{g2[17:25]}"          # scene names contain the date at characters 17-24


existing = {j.name for j in hyp3.find_jobs(job_type="INSAR_GAMMA") if j.name}
for d, pairs in networks.items():
    tag = d[:3].lower()
    prepared = [hyp3.prepare_insar_job(g1, g2, name=job_name(tag, g1, g2), looks=LOOKS, include_look_vectors=True,
                                       include_dem=True, apply_water_mask=True)
                for (g1, g2, _, _) in pairs if job_name(tag, g1, g2) not in existing]
    print(f"{d}: {len(prepared)} new jobs ({len(pairs) - len(prepared)} already exist)")
    if SUBMIT and prepared:
        batch = hyp3.submit_prepared_jobs(prepared)
        print("   submitted", len(batch), "jobs")
    elif prepared:
        print("   dry run - set SUBMIT = True to order them")

# %% [markdown]
# ## 6. Wait and download
# HyP3 usually finishes a job in 20-60 minutes, and jobs run in parallel. You can close the notebook and come back: results are found by name.

# %%
# Cell 12 - Progress report (run as often as you like)
for d in networks:
    tag = d[:3].lower()
    jobs = [j for j in hyp3.find_jobs(job_type="INSAR_GAMMA") if j.name and j.name.startswith(f"naivasha_{tag}_")]
    status = pd.Series([j.status_code for j in jobs]).value_counts().to_dict()
    print(d, "->", status)

# %%
# Cell 13 - Download finished products to data/insar/hyp3_<asc|desc>/ (zips are kept; skip those we already have)
for d in networks:
    tag = d[:3].lower()
    out = INSAR / f"hyp3_{tag}"
    out.mkdir(exist_ok=True)
    ready = [j for j in hyp3.find_jobs(job_type="INSAR_GAMMA") if j.name and j.name.startswith(f"naivasha_{tag}_") and j.succeeded()]
    todo = [j for j in ready if not (out / j.files[0]["filename"]).exists()]
    print(f"{d}: {len(ready)} finished, {len(todo)} to download")
    if todo:
        sdk.Batch(todo).download_files(out)

# %% [markdown]
# **Next:** `04_insar_mintpy_timeseries`. If you prefer not to use HyP3, you can instead drop a vertical velocity raster in
# `data/insar/velocity_vertical_mm_yr.tif` (for example from COMET LiCSBAS) and go straight to notebook 05.
