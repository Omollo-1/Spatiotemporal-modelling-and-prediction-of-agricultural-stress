# %%
# Cell 2 - Load the project configuration written by notebook 00
CFG_FILE = PROJECT_DIR / "config.json"
if not CFG_FILE.exists():
    raise FileNotFoundError(f"{CFG_FILE} not found - run 00_setup_and_config first.")
CFG = json.loads(CFG_FILE.read_text())
if not isinstance(CFG.get("aoi"), dict) or not isinstance(CFG["aoi"].get("geometry"), dict):
    raise ValueError(
        f"{CFG_FILE} does not contain the study-area geometry; run 00_setup_and_config "
        "in this project folder first."
    )
GRID = CFG["grid"]
DATA = PROJECT_DIR / CFG["paths"]["data"]
OUT = PROJECT_DIR / CFG["paths"]["outputs"]
MODE = CFG["mode"]                                   # "real" or "demo"
YEARS = list(range(CFG["start_year"], CFG["end_year"] + 1))
MONTHS = [f"{y}-{m:02d}" for y in YEARS for m in range(1, 13)]
print(f"Mode: {MODE} | years {YEARS[0]}-{YEARS[-1]} | grid {GRID['width']} x {GRID['height']} px @ {GRID['res']} m")
if MODE == "demo":
    print(">>> DEMO MODE: all data are SYNTHETIC. Outputs are for testing the pipeline only, not scientific results.")
