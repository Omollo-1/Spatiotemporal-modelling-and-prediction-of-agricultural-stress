# %%
# Cell - Connect to Google Earth Engine (a browser sign-in link appears the first time)
ensure("earthengine-api", "requests")
import ee
GEE_PROJECT = CFG["gee_project"]
try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()                 # follow the link, paste the code
    ee.Initialize(project=GEE_PROJECT)
print("Earth Engine ready, project:", GEE_PROJECT)
