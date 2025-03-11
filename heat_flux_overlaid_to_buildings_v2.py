# Creates a raster of buildings with the maximum heat flux each building is exposed to

import rasterio
import numpy as np

RHF_Raster = r'C:\Users\rma388\OneDrive - University of Canterbury\PhD\Spark\5_rosie_projects\NZ_ROS_versions\RHF_mapper\Max_RHF_Output.tif'
Building_File = r'C:\Users\rma388\OneDrive - University of Canterbury\PhD\Spark\5_rosie_projects\2-25-rhf_mapper_data\buildingID_for_heatflux_overlay.tif'
Output = r'C:\Users\rma388\OneDrive - University of Canterbury\PhD\Spark\5_rosie_projects\NZ_ROS_versions\Output_building_RHFv2.tif'

# Load building footprint raster
with rasterio.open(Building_File) as buildings_src:
    buildings = buildings_src.read(1)  # Read first band
    building_nodata = buildings_src.nodata  # Get NoData value
    profile = buildings_src.profile  # Save profile for output

# Load heat flux raster
with rasterio.open(RHF_Raster) as heat_flux_src:
    heat_flux = heat_flux_src.read(1)  # Read first band

# Ensure both rasters have the same shape
if buildings.shape != heat_flux.shape:
    raise ValueError("Rasters must have the same shape and alignment!")

# Handle NoData values in the building raster
if building_nodata is not None:
    buildings_masked = np.where(buildings == building_nodata, 0, buildings)
else:
    buildings_masked = buildings

# Create output array initialized with -999
output_array = np.full_like(buildings, -999, dtype=np.float32)

# Identify unique building IDs
unique_buildings = np.unique(buildings_masked)
unique_buildings = unique_buildings[unique_buildings > 0]  # Remove background (0)

# Process each building separately
for building_id in unique_buildings:
    # Find max heat flux for the current building
    max_flux = np.max(heat_flux[buildings_masked == building_id])
    
    # Assign max heat flux value to all pixels of the same building
    output_array[buildings_masked == building_id] = max_flux

# Update profile for output
profile.update(dtype=rasterio.float32, nodata=-999)

try:
    with rasterio.open(Output, "w", **profile) as dst:
        dst.write(output_array, 1)
    print(f"✅ Successfully saved raster: {Output}")
except Exception as e:
    print(f" Error saving raster: {e}")

print(f" Processed raster saved to {Output} with max heat flux assigned to each building.")