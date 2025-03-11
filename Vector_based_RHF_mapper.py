# Outputs the isocrones at 3600s and 7200s but not the others
# Fixed the bug that Barry mentioned
# Adding in the actual building heights
# if Test_Scenario == True:
 
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point, Polygon, MultiPolygon
import rasterio
import math
import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import median_filter
from rasterio.features import geometry_mask
from shapely.ops import split

# Function to split the LINESTRING into smaller segments
def split_linestring(line, max_length):
    segments = []  # To store resulting LineString segments

    # Total length of the LineString
    total_length = line.length

    # Iterate through the LineString and split it into 10m sections
    current_length = 0
    while current_length < total_length:
        next_length = min(current_length + max_length, total_length)  # Ensure we don't exceed total length
        point = line.interpolate(next_length)  # Find the point at next_length
        segments.append(LineString([line.interpolate(current_length).coords[0], point.coords[0]]))  # Create a segment
        current_length = next_length  # Update the current length to the new segment's end
    

    return segments


def gaussian_kernel(size=3, sigma=1):
    """Generate a 3x3 Gaussian kernel"""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(
            - ((x - (size - 1) / 2)**2 + (y - (size - 1) / 2)**2) / (2 * sigma**2)
        ),
        (size, size)
    )
    return kernel / np.sum(kernel)  # Normalize the kernel

def ReadRaster(InputFile,Band):
    with rasterio.open(InputFile) as src:

        print(f"Raster file {InputFile} has {src.count} bands.")  # Print number of bands

        # ✅ Check if the band is within the valid range
        if Band < 1 or Band > src.count:
            raise IndexError(f"Band index {Band} is out of range. The file has {src.count} band(s).")

        #band = src.read(Band)
        transform = src.transform
        profile = src.profile  # Save the metadata for the new TIFF
        nodata_value = src.nodata
        # Read the raster data
        data = src.read(Band)  # Read the selected band
        pixel_size = src.res[0]
        metadata = src.meta.copy()
        bounds = src.bounds
        width = src.width
        height = src.height
        crs = src.crs

    return data, transform, pixel_size, profile, transform, nodata_value, metadata, crs, bounds, width, height

def get_pixel_value_from_coordinates(coords, bounds, pixel_size, transform, data):
    """
    Convert coordinates to pixel indices and retrieve raster values.
    """
    row = math.floor((bounds[3] - coords[1])/pixel_size)
    col = math.floor((coords[0] - bounds[0])/pixel_size)
    height = data[row, col] 

    return height

def divide_polyline_into_segments(polyline,pixel_size):
    
    """
    Divide a polyline (LineString) into a given number of segments.
    """
    total_length = polyline.length
    if total_length <= pixel_size:
        num_segments = 2
    else:
        num_segments = math.ceil(total_length/pixel_size)

    
    segment_length = total_length / num_segments
    sub_segments = []
    
    for i in range(num_segments):
        start_distance = i * segment_length
        end_distance = (i + 1) * segment_length
        segment = polyline.interpolate(start_distance), polyline.interpolate(end_distance)
        sub_segments.append(LineString([segment[0], segment[1]]))
    
    return sub_segments


def get_raster_values_under_segments(data, bounds, pixel_size, transform, segment):
   
    sub_segments = divide_polyline_into_segments(segment,pixel_size)
    
    raster_values = []
    segment_values = []
    
    for sample in sub_segments:
        for point in sample.coords:
           
            value = get_pixel_value_from_coordinates(point, bounds, pixel_size, transform, data)
            if value < 0:
                value = 0
                
            segment_values.append(value)
    raster_values.append(segment_values)
    
    return raster_values


def GetIsoChromeData(
    IsoChromeFile, flame_data, flame_bounds, flame_pixel_size, flame_transform,
    dem_data, dem_bounds, dem_pixel_size, dem_transform, 
    Test_Scenario, Test_Flame_Base_Height, Test_Flame_Height, Maximum_Wall_Length, OutputFolder):

    print("Reading ISO Chrome Data...")

    # Read the shapefile
    iso_polys = gpd.read_file(IsoChromeFile)
    crs = iso_polys.crs
    iso_lines = iso_polys.geometry.boundary
    iso_lines = gpd.GeoDataFrame(geometry=iso_lines, crs=crs)

    # Dictionaries for storing results
    burn_area_geometry = {}
    firewall_geometry, firewall_id, firewall_height = {}, {}, {}
    firewall_normal_angle, firewall_normal_vector, firewall_width = {}, {}, {}
    firewall_area, firewall_base_height = {}, {}
    firewall_xy_angle_radians, firewall_xy_angle_degrees = {}, {}
    firewall_bounding_extents, firewall_normal_angle_degrees = {}, {}
    firewall_normal_angle_radians, projected_wall_length = {}, {}
    projected_wall_a, projected_wall_b = {}, {}

    # Constant values
    rad_45 = math.radians(45)
    projected_wall_segment = 100 * math.sin(rad_45)

    # Timesteps to skip due to None geometries
    #skip_timesteps = {3600, 10800, 14400}

    # Iterate through each line in the shapefile
    for idx, line in enumerate(iso_lines.geometry):
        time_value = int(iso_polys['value'].iloc[idx])

        # Skip specified timesteps
        #if time_value in skip_timesteps:
        #    print(f"Skipping ISO Chrome Time = {time_value} due to None geometry.")
        #    continue  

        # Skip None geometries
        if line is None:
            print(f"Warning: Skipping None geometry at index {idx}.")
            continue

        polygon = iso_polys.iloc[idx].geometry
        if polygon is None:
            print(f"Warning: Skipping None polygon at index {idx}.")
            continue

        print(f"Processing ISO Chrome Time = {time_value}")

        # Initialize storage for this timestep
        burn_area_geometry[time_value] = polygon
        firewall_geometry[time_value], firewall_id[time_value] = [], []
        firewall_height[time_value], firewall_width[time_value] = [], []
        firewall_area[time_value], firewall_base_height[time_value] = [], []
        firewall_xy_angle_radians[time_value], firewall_xy_angle_degrees[time_value] = [], []
        firewall_bounding_extents[time_value] = []
        firewall_normal_angle_degrees[time_value], firewall_normal_angle_radians[time_value] = [], []
        projected_wall_length[time_value], projected_wall_a[time_value], projected_wall_b[time_value] = [], [], []

        # Handle MultiLineString cases
        line_sub_list = list(line.geoms) if isinstance(line, MultiLineString) else [line]

        # Iterate through each sub-line in the geometry
        for subline in line_sub_list:
            if subline is None:
                print(f"Warning: Skipping None subline at index {idx}.")
                continue

            coords = list(subline.coords)
            for i in range(len(coords) - 1):
                full_wall = LineString([coords[i], coords[i + 1]])
                segments = split_linestring(full_wall, Maximum_Wall_Length)

                for q, segment in enumerate(segments):
                    firewall_geometry[time_value].append(segment)
                    firewall_id[time_value].append(f"{idx}_{i}_{q}")
                    firewall_width[time_value].append(segment.length)

                    # Get flame height values from raster
                    flame_height_values = get_raster_values_under_segments(flame_data, flame_bounds, flame_pixel_size, flame_transform, segment)
                    flame_val = np.mean(flame_height_values) if flame_height_values else 0

                    if Test_Scenario:
                        firewall_height[time_value].append(Test_Flame_Height)
                        base_val = Test_Flame_Base_Height
                    else:
                        firewall_height[time_value].append(flame_val)
                        flame_base_height_values = get_raster_values_under_segments(dem_data, dem_bounds, dem_pixel_size, dem_transform, segment)
                        base_val = np.mean(flame_base_height_values) if flame_base_height_values else 0

                    firewall_base_height[time_value].append(base_val)
                    firewall_area[time_value].append(flame_val * segment.length)

                    # Calculate wall angles and bounding extents
                    dx, dy = segment.coords[1][0] - segment.coords[0][0], segment.coords[1][1] - segment.coords[0][1]
                    if segment.coords[1][0] < segment.coords[0][0]:  
                        dx, dy = -dx, -dy

                    llc, lrc, ulc, urc = (
                        (segment.coords[0][0], segment.coords[0][1], base_val),
                        (segment.coords[1][0], segment.coords[1][1], base_val),
                        (segment.coords[0][0], segment.coords[0][1], base_val + flame_val),
                        (segment.coords[1][0], segment.coords[1][1], base_val + flame_val),
                    )

                    wall_xyz_list = [llc, lrc, ulc, urc]
                    firewall_bounding_extents[time_value].append(wall_xyz_list)

                    # Compute angles
                    line_angle_radians = math.atan2(dy, dx)
                    line_angle_degrees = (math.degrees(line_angle_radians) + 360) % 360

                    firewall_xy_angle_radians[time_value].append(math.radians(line_angle_degrees))
                    firewall_xy_angle_degrees[time_value].append(line_angle_degrees)

                    # Compute projected wall data
                    projected_wall_length[time_value].append(projected_wall_segment)
                    projected_wall_a[time_value].append([llc[0], llc[1]])
                    projected_wall_b[time_value].append([lrc[0], lrc[1]])

                    # Compute normal angle and vector
                    potential_normal_angle_degrees = (line_angle_degrees + 90) % 360
                    potential_normal_angle_radians = math.radians(potential_normal_angle_degrees)

                    mid_x, mid_y = (segment.coords[0][0] + segment.coords[1][0]) / 2, (segment.coords[0][1] + segment.coords[1][1]) / 2
                    normal_x, normal_y = np.cos(potential_normal_angle_radians), np.sin(potential_normal_angle_radians)
                    test_point = Point(mid_x + normal_x * 0.1, mid_y + normal_y * 0.1)

                    if polygon.contains(test_point):
                        potential_normal_angle_degrees = (potential_normal_angle_degrees + 180) % 360

                    normal_angle_radians = math.radians(potential_normal_angle_degrees)
                    firewall_normal_angle_degrees[time_value].append(potential_normal_angle_degrees)
                    firewall_normal_angle_radians[time_value].append(normal_angle_radians)

        # Save results to shapefile
        geo_data_building_walls = gpd.GeoDataFrame({
            'wall_id': firewall_id[time_value],
            'length': firewall_width[time_value],
            'geometry': firewall_geometry[time_value],
            'wall_height': firewall_height[time_value],
            'ground_height': firewall_base_height[time_value],
            'wall_area': firewall_area[time_value],
            'wall_xy_angle_degrees': firewall_xy_angle_degrees[time_value],
            'wall_xy_angle_radians': firewall_xy_angle_radians[time_value],
            'normal_angle_degrees': firewall_normal_angle_degrees[time_value],
            'normal_angle_radians': firewall_normal_angle_radians[time_value],
            'proj_length': projected_wall_length[time_value]
        }, crs=crs)

        output_filename = f"{OutputFolder}Split_Firefront_{time_value}.shp" if not Test_Scenario else f"{OutputFolder}Test_Split_Walls.shp"
        geo_data_building_walls.to_file(output_filename)

    return firewall_id, firewall_width, firewall_base_height, firewall_height, firewall_xy_angle_radians, firewall_xy_angle_degrees, firewall_normal_angle_radians, firewall_normal_angle_degrees, firewall_bounding_extents, projected_wall_length, projected_wall_a, projected_wall_b, projected_wall_segment, burn_area_geometry




def bresenham_line_subpixel(x1, y1, x2, y2):
    # Ensure that x1, y1 is the starting point, and x2, y2 is the end point.
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Calculate step direction for both x and y.
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    # Error calculation
    err = dx - dy

    # Initialize starting coordinates as float for better precision
    x, y = float(x1), float(y1)

    points = []

    while True:
        # Round the coordinates to nearest integer for rasterization
        
        if [round(x),round(y)] not in points:
            points.append([round(x), round(y)])
    
        if [round(x-1), round(y)] not in points:
            points.append([round(x-1), round(y)])
        if [round(x+1), round(y)] not in points:
            points.append([round(x+1), round(y)])
        if [round(x), round(y-1)] not in points:
            points.append([round(x), round(y-1)])
        if [round(x+1), round(y)] not in points:
            points.append([round(x), round(y+1)])
        
        
        # Break when the end point is reached
        if (round(x) == x2) and (round(y) == y2):
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x += sx

        if e2 < dx:
            err += dx
            y += sy

    return points


def TextBookViewFactorOFFCenter(flame_width,flame_height,flame_base_height,
    target_width,target_height,target_base_height,target_horizontal_distance):

    X = (flame_width)/target_horizontal_distance
    Y = flame_height/target_horizontal_distance

    view_factor = (1/(2*math.pi))*((X/(math.sqrt(1+X**2))) * math.atan((Y/(math.sqrt(1+X**2)))) + ((Y/math.sqrt(1+Y**2))) * math.atan((X/(math.sqrt(1+Y**2)))))

    return view_factor, Y, X#, q

def UpdatedRHF_Mask(flame_height_list, flame_base_height, flame_width1, flame_width2, 
                    target_base_height, target_height, target_width, target_horizontal_distance,wall_type, flame_type):
    
    temp_q = 0
    view_factor = view_factor1 = view_factor2 = view_factor3 = view_factor4 = 0.0

    temp_view_factor1 = temp_view_factor2 = 0.0

    if flame_type != 'projected':
    
        flame_height1 = flame_height_list[0]
        flame_height2 = flame_height_list[1]

        if flame_width1 > 0:
            view_factor1, Y, X=  TextBookViewFactorOFFCenter(flame_width1,flame_height1,flame_base_height,
                    target_width,target_height,target_base_height,target_horizontal_distance)
            view_factor2, Y, X=  TextBookViewFactorOFFCenter(flame_width1,flame_height2,flame_base_height,
                    target_width,target_height,target_base_height,target_horizontal_distance)
            
        if flame_width2 > 0:
            view_factor3, Y, X=  TextBookViewFactorOFFCenter(flame_width2,flame_height1,flame_base_height,
                    target_width,target_height,target_base_height,target_horizontal_distance)
            view_factor4, Y, X=  TextBookViewFactorOFFCenter(flame_width2,flame_height2,flame_base_height,
                    target_width,target_height,target_base_height,target_horizontal_distance)
    else:
        if wall_type == "Normal":
            flame_height1 = flame_height_list[0]
            flame_height2 = flame_height_list[1]

            if flame_width1 > 0:
                view_factor1, Y, X=  TextBookViewFactorOFFCenter(flame_width1,flame_height1,flame_base_height,
                        target_width,target_height,target_base_height,target_horizontal_distance)
                view_factor2, Y, X=  TextBookViewFactorOFFCenter(flame_width1,flame_height2,flame_base_height,
                        target_width,target_height,target_base_height,target_horizontal_distance)
            
            if flame_width2 > 0:
                view_factor3, Y, X=  TextBookViewFactorOFFCenter(flame_width2,flame_height1,flame_base_height,
                        target_width,target_height,target_base_height,target_horizontal_distance)
                view_factor4, Y, X=  TextBookViewFactorOFFCenter(flame_width2,flame_height2,flame_base_height,
                        target_width,target_height,target_base_height,target_horizontal_distance)

        else:

            width1 = flame_width1
            width2 = flame_width2
            width3 = flame_width1 - flame_width2
            flame_height1 = flame_height_list[0]
            flame_height2 = flame_height_list[1]
            view_factor1, Y, X=  TextBookViewFactorOFFCenter(width1,flame_height1,flame_base_height,
                target_width,target_height,target_base_height,target_horizontal_distance)
            view_factor2, Y, X=  TextBookViewFactorOFFCenter(width2,flame_height1,flame_base_height,
                target_width,target_height,target_base_height,target_horizontal_distance)
            view_factor3, Y, X=  TextBookViewFactorOFFCenter(width1,flame_height2,flame_base_height,
                target_width,target_height,target_base_height,target_horizontal_distance)
            view_factor4, Y, X=  TextBookViewFactorOFFCenter(width2,flame_height2,flame_base_height,
                target_width,target_height,target_base_height,target_horizontal_distance)



    if wall_type == "Normal":

        if flame_type == "normal":

            view_factor = view_factor1 + view_factor2 + view_factor3 + view_factor4
        
        else:

            view_factor = (view_factor1 - view_factor2) + (view_factor3 - view_factor4)
    
    elif wall_type == "Projected":
        
        if flame_type == "normal":
        
            view_factor = (view_factor1 - view_factor3) + (view_factor2 - view_factor4)
        else:
            #view_factor = view_factor1 - (view_factor2 + view_factor3)
            vfb = view_factor2 - view_factor4
            vfc = view_factor3 - view_factor4
            view_factor = view_factor1 - (vfb + vfc + view_factor4)
     
    
    #Making sure view factor doe not go beloew zero or greater than one
    if round(view_factor,2) < 0 or round(view_factor,2) > 1:
        print("CHECK THIS!!!")
        print("Wall Type = ",wall_type)
        print("Flame Type = ",flame_type)


    Temp = 1200#1095 #Flame temp estimate in Kelvin
    Stf = 5.67E-11#11
    emissivity = 0.95#0.95

    q = (((Stf * emissivity * (Temp **4 ) * view_factor)))

    q = round(q,2)

    return q, view_factor

def Calculate_Heat_Flux_Values(current_wall_id, point_list, flame_base_height, flame_midpoint, flame_height, firewall_width_1, firewall_width_2,
                    target_width, dem_data, outline_data, build_data, start_row, start_col, temp_mask, temp_viewfactor, wall_type, projected_wall_segment, anchor_start_row, anchor_start_col, 
                    Test_Scenario, Test_Target_Base_Height, Test_Target_Height): 

    for point in point_list:
        
        if Test_Scenario == True:
            target_base_height = Test_Target_Base_Height
            target_height = Test_Target_Height
        else:
            target_base_height = dem_data[point[1],point[0]]
            if outline_data[point[1],point[0]] != outline_nodata_value:
                target_height = build_data[point[1],point[0]]
            else:
                target_height = 2

        flame_height_list = []
        temp_q = 0
        view_factor = 0.0

        scan_row = point[1]
        scan_col = point[0]

        begin_scan = 1
        flame_type = 'begin'
        
        if wall_type == 'Projected':
            distance_check = math.sqrt((scan_row - anchor_start_row)**2 + (scan_col - anchor_start_col)**2)
            if distance_check > 100:
            
                begin_scan = 0
            
        
        if begin_scan == 1:

            target_horizontal_distance = math.sqrt((start_col - scan_col)**2 + (start_row - scan_row)**2) #maybe update to coordinates later
            
            if target_horizontal_distance > 0 and target_horizontal_distance <= 100:
                
                if temp_mask[scan_row,scan_col] == 0:#this makes sure not double counting when propogating over rows and cols with bresenham_line
                
                    #Scenario 1: Target within flame height but less than the mid point
                    if (target_base_height + target_height) >= flame_base_height and (target_base_height + target_height) < flame_midpoint:
                        
                        flame_type = "normal"

                        top_flame_segment = (flame_base_height + flame_height) - (target_base_height + target_height)
                        bottom_flame_segment = flame_height - top_flame_segment#(flame_base_height + flame_height) - top_flame_segment
                        flame_height_list.append(top_flame_segment)
                        flame_height_list.append(bottom_flame_segment)
                        
                        temp_q, view_factor = UpdatedRHF_Mask(flame_height_list, flame_base_height, firewall_width_1, firewall_width_2, 
                            target_base_height, target_height, target_width, target_horizontal_distance,wall_type,flame_type)
                    
                        if view_factor > 1:
                            print("1: ", view_factor)

                        temp_mask[scan_row,scan_col] = temp_q
                        temp_viewfactor[scan_row,scan_col] = view_factor
                        
                        
                    
                    #Scenario 2: Target height within bounds of flame but greater or equal to the mid point 
                    elif (target_base_height + target_height) >= flame_midpoint and target_base_height  <= flame_midpoint:
                        
                        flame_type = "normal"
                        
                        flame_height_list.append(flame_height/2)
                        flame_height_list.append(flame_height/2)

                        temp_q, view_factor = UpdatedRHF_Mask(flame_height_list, flame_base_height, firewall_width_1, firewall_width_2, 
                            target_base_height, target_height, target_width, target_horizontal_distance,wall_type,flame_type)                             
                    
                        if view_factor > 1:
                            print("2: ", view_factor)
                    
                        temp_mask[scan_row,scan_col] = temp_q
                        temp_viewfactor[scan_row,scan_col] = view_factor

                    #Scenario 3: Target in range of flame but base higher than midpoint
                    elif target_base_height > flame_midpoint and target_base_height <= (flame_base_height + flame_height):
                        
                        flame_type = "normal"

                        top_flame_segment = (flame_base_height + flame_height) - target_base_height
                        bottom_flame_segment = flame_height - top_flame_segment

                        #projected_flame_height = (flame_height) * modify_coeff
                        flame_height_list.append(top_flame_segment)
                        flame_height_list.append(bottom_flame_segment)


                        temp_q, view_factor = UpdatedRHF_Mask(flame_height_list, flame_base_height, firewall_width_1, firewall_width_2, 
                            target_base_height, target_height, target_width, target_horizontal_distance,wall_type, flame_type)
                        
                        if view_factor > 1:
                            print("3: ",view_factor)
                        
                        temp_mask[scan_row,scan_col] = temp_q
                        temp_viewfactor[scan_row,scan_col] = view_factor
                    
                    #Scenario 4 target is below the flame base height
                    elif (target_base_height + target_height) < flame_base_height:
                        
                        flame_type = "projected"
                        
                        if wall_type == 'Normal':
                            top_flame_segment = (flame_base_height + flame_height) - (target_base_height + target_height)
                            bottom_flame_segment = flame_base_height - (target_base_height + target_height)#top_flame_segment - (flame_base_height - (target_base_height +target_height))
                            flame_height_list.append(top_flame_segment) 
                            flame_height_list.append(bottom_flame_segment) 
                        else:
                            wall_height1 = (flame_base_height + flame_height) - (target_base_height + target_height)
                            wall_height2 = wall_height1 - flame_height
                            flame_height_list.append(wall_height1) 
                            flame_height_list.append(wall_height2) 

                        vertical_distance = flame_base_height - (target_base_height + target_height)
                        diagonal_distance = math.sqrt(vertical_distance**2 + target_horizontal_distance**2)


                        #check to make sure still within 100m even in vertical plabe
                        #if diagonal_distance < 100: 

                        temp_q, view_factor = UpdatedRHF_Mask(flame_height_list, flame_base_height, firewall_width_1, firewall_width_2, 
                            target_base_height, target_height, target_width, target_horizontal_distance,wall_type, flame_type)


                        if view_factor > 1:
                            print("4: ", view_factor)
                            

                        temp_mask[scan_row,scan_col] = temp_q
                        temp_viewfactor[scan_row,scan_col] = view_factor

                    #Scenario 5 target base is above the flame
                    elif target_base_height >= (flame_base_height + flame_height):

                        flame_type = "projected"

                        if wall_type == 'Normal':
                            top_flame_segment = target_base_height - flame_base_height
                            bottom_flame_segment = top_flame_segment - flame_height
                            flame_height_list.append(top_flame_segment)
                            flame_height_list.append(bottom_flame_segment)
                        else:
                            wall_height1 = target_base_height - flame_base_height
                            wall_height2 = wall_height1 - flame_height
                            flame_height_list.append(wall_height1) 
                            flame_height_list.append(wall_height2) 

                        vertical_distance = target_base_height - (flame_base_height + flame_height)
                        diagonal_distance = math.sqrt(vertical_distance**2 + target_horizontal_distance**2)

                        if diagonal_distance < 100:
                            temp_q, view_factor = UpdatedRHF_Mask(flame_height_list, flame_base_height, firewall_width_1, firewall_width_2, 
                                target_base_height, target_height, target_width, target_horizontal_distance,wall_type, flame_type)

                        if temp_q < 0:
                            print("5: ", temp_q)

                        temp_mask[scan_row,scan_col] = temp_q
                        temp_viewfactor[scan_row,scan_col] = view_factor

                    else:
                        
                        print("Unknown Scenario -- Please Check")
                        print(flame_base_height,", ",flame_midpoint, ", ",flame_height, " :: ",target_base_height, ", ",target_height)

    return temp_mask, temp_viewfactor

def ScanWallSections(burn_area_geometry, firewall_id, firewall_width, firewall_base_height, firewall_height, firewall_xy_angle_degrees, firewall_xy_angle_radians, firewall_normal_angle_radians, firewall_normal_angle_degrees, firewall_bounding_extents,
        dem_data,dem_bounds,dem_pixel_size,outline_data, outline_nodata_value, build_data,
            SimulationStartTime,SimulationEndTime,TimeStepSeconds,dem_profile,dem_transform,
            projected_wall_length, projected_wall_a, projected_wall_b, projected_wall_segment, 
            Test_Scenario, Test_Target_Base_Height, Test_Target_Height, OutputFolder):

    target_width = 1
    # Calculate raster width and height based on pixel size and extent
    minx, miny, maxx, maxy = dem_bounds
    width = int((maxx - minx) / dem_pixel_size)
    height = int((maxy - miny) / dem_pixel_size)
   
    
   
    max_flame_height = 0.0
    
    max_rhf_mask = np.zeros((height, width), dtype=np.float64)

    #for time_key in firewall_id.keys():
    for time_key, polygon in burn_area_geometry.items():

        burn_mask = np.ones((height, width), dtype=np.float64)

        burn_area_mask = geometry_mask([polygon], transform=dem_transform, invert=True, out_shape=(height, width))
        
        burn_mask[burn_area_mask] = 0
        #new blank rhf_mask at time-step
        
        rhf_mask = np.zeros((height, width), dtype=np.float64)
        viewfactor_mask =  np.zeros((height, width), dtype=np.float64)

        print("time = ", time_key)
        
        for i in range(0, len(firewall_width[time_key])):
            
            current_wall_id = firewall_id[time_key][i]
            """REAL FIREWALL DATA"""
            current_firewall_width = firewall_width[time_key][i]
            firewall_angle_radians = firewall_xy_angle_radians[time_key][i]
            current_firewall_normal_angle_radians = firewall_normal_angle_radians[time_key][i]

            firewall_start_x = firewall_bounding_extents[time_key][i][0][0]
            firewall_end_x = firewall_bounding_extents[time_key][i][1][0]
            firewall_start_y = firewall_bounding_extents[time_key][i][0][1]
            firewall_end_y = firewall_bounding_extents[time_key][i][1][1]


            if current_firewall_width < 2:
                firewall_steps = 4
            else:
                firewall_steps = int(current_firewall_width * 2)

            #firewall_step_list = np.linspace(0,current_firewall_width,firewall_steps)
            if current_firewall_width >=2:
                firewall_step_list = np.arange(0,math.floor(current_firewall_width),1)
            else:
                firewall_step_list = np.linspace(0,current_firewall_width,firewall_steps)
            
            firewall_dwall_length = firewall_step_list[1] -  firewall_step_list[0]


            """PROJECTED FIREWALL DATA"""
            projected_firewall_segment = projected_wall_length[time_key][i]
            projected_firewall_width = projected_firewall_segment #+ current_firewall_width
            projected_firewall_start_x_a = projected_wall_a[time_key][i][0]
            #projected_firewall_end_x_a = projected_wall_a[time_key][i][1][0]
            projected_firewall_start_y_a = projected_wall_a[time_key][i][1]
            #projected_firewall_end_y_a = projected_wall_a[time_key][i][1][1]
            projected_firewall_start_x_b = projected_wall_b[time_key][i][0]
            #projected_firewall_end_x_b = projected_wall_b[time_key][i][1][0]
            projected_firewall_start_y_b = projected_wall_b[time_key][i][1]
            #projected_firewall_end_y_b = projected_wall_b[time_key][i][1][1]

          
            projected_firewall_steps = int(projected_firewall_width * 4)
            projected_firewall_step_list = np.linspace(0,projected_firewall_width,projected_firewall_steps)
            projected_firewall_dwall_length = projected_firewall_step_list[1] -  projected_firewall_step_list[0]
            

            flame_height = firewall_height[time_key][i]
            
            if flame_height > max_flame_height:
                max_flame_height = flame_height

            flame_base_height = firewall_base_height[time_key][i]
            flame_midpoint = flame_base_height + (flame_height/2)


            temp_mask = np.zeros((height, width), dtype=np.float64)
            temp_viewfactor = np.zeros((height, width), dtype=np.float64)
            
            angle_scan = []

            for firewall_dyx in firewall_step_list:

                firewall_width_1 = current_firewall_width - firewall_dyx
                firewall_width_2 = current_firewall_width - firewall_width_1

                
                
                start_wall_x_coord = firewall_start_x + (firewall_dyx) * math.cos(firewall_angle_radians) 
                start_wall_y_coord = firewall_start_y + (firewall_dyx) * math.sin(firewall_angle_radians) 
                target_x_coord = start_wall_x_coord + 100 * math.cos(current_firewall_normal_angle_radians) 
                target_y_coord = start_wall_y_coord + 100 * math.sin(current_firewall_normal_angle_radians)           

                                                   
                start_row = math.floor((dem_bounds[3] - start_wall_y_coord)/dem_pixel_size)
                start_col = math.floor((start_wall_x_coord - dem_bounds[0])/dem_pixel_size)
                target_row = math.floor((dem_bounds[3] - target_y_coord)/dem_pixel_size)
                target_col = math.floor((target_x_coord - dem_bounds[0])/dem_pixel_size)
               
                #fetch all rows and cells in along normal line
                point_list = bresenham_line_subpixel(start_col, start_row, target_col, target_row)

                wall_type = "Normal"

                #temp_mask = np.zeros((height, width), dtype=np.float64)

                temp_mask, temp_viewfactor = Calculate_Heat_Flux_Values(current_wall_id, point_list, flame_base_height, flame_midpoint, flame_height, firewall_width_1, firewall_width_2,
                    target_width, dem_data,outline_data,build_data,start_row, start_col, temp_mask, temp_viewfactor, wall_type, projected_wall_segment,0,0, 
                    Test_Scenario,Test_Target_Base_Height, Test_Target_Height)

            z = 0


            for projected_firewall_dyx in projected_firewall_step_list:

                firewall_width_1 = current_firewall_width + projected_firewall_dyx
                firewall_width_2 = projected_firewall_dyx

                #check

                start_wall_x_coord = projected_firewall_start_x_a - (projected_firewall_dyx) * math.cos(firewall_angle_radians) 
                start_wall_y_coord = projected_firewall_start_y_a - (projected_firewall_dyx) * math.sin(firewall_angle_radians) 
                
                
                target_x_coord = start_wall_x_coord + 100 * math.cos(current_firewall_normal_angle_radians) 
                target_y_coord = start_wall_y_coord + 100 * math.sin(current_firewall_normal_angle_radians)

                
                    
                start_row = math.floor((dem_bounds[3] - start_wall_y_coord)/dem_pixel_size)
                start_col = math.floor((start_wall_x_coord - dem_bounds[0])/dem_pixel_size)

                anchor_start_row_a = start_row
                anchor_start_col_a = start_col
                

                target_row = math.floor((dem_bounds[3] - target_y_coord)/dem_pixel_size)
                target_col = math.floor((target_x_coord - dem_bounds[0])/dem_pixel_size)
            
                #fetch all rows and cells in along normal line
                point_list = bresenham_line_subpixel(start_col, start_row, target_col, target_row)

                wall_type = "Projected"


                temp_mask, temp_viewfactor = Calculate_Heat_Flux_Values(current_wall_id, point_list, flame_base_height, flame_midpoint, flame_height, firewall_width_1, firewall_width_2,
                    target_width, dem_data,outline_data,build_data,start_row, start_col, temp_mask, temp_viewfactor, wall_type, projected_wall_segment, anchor_start_row_a, anchor_start_col_a, 
                    Test_Scenario, Test_Target_Base_Height, Test_Target_Height)
                

                start_wall_x_coord = projected_firewall_start_x_b + (projected_firewall_dyx) * math.cos(firewall_angle_radians) 
                start_wall_y_coord = projected_firewall_start_y_b + (projected_firewall_dyx) * math.sin(firewall_angle_radians) 
                target_x_coord = start_wall_x_coord + 100 * math.cos(current_firewall_normal_angle_radians) 
                target_y_coord = start_wall_y_coord + 100 * math.sin(current_firewall_normal_angle_radians)

            
                start_row = math.floor((dem_bounds[3] - start_wall_y_coord)/dem_pixel_size)
                start_col = math.floor((start_wall_x_coord - dem_bounds[0])/dem_pixel_size)
                target_row = math.floor((dem_bounds[3] - target_y_coord)/dem_pixel_size)
                target_col = math.floor((target_x_coord - dem_bounds[0])/dem_pixel_size)

                anchor_start_row_b = start_row
                anchor_start_col_b = start_col
            
                #fetch all rows and cells in along normal line
                point_list = bresenham_line_subpixel(start_col, start_row, target_col, target_row)

                wall_type = "Projected"

                #temp_mask = np.zeros((height, width), dtype=np.float64)

                temp_mask, temp_viewfactor = Calculate_Heat_Flux_Values(current_wall_id, point_list, flame_base_height, flame_midpoint, flame_height, firewall_width_1, firewall_width_2,
                    target_width, dem_data,outline_data,build_data,start_row, start_col, temp_mask, temp_viewfactor, wall_type, projected_wall_segment, anchor_start_row_b, anchor_start_col_b, 
                    Test_Scenario, Test_Target_Base_Height, Test_Target_Height)
                    
                
            
            
            temp_mask *= burn_mask 
            #if current_wall_id == '0_28':
            rhf_mask += temp_mask    
                #get max view factor
            viewfactor_mask = np.where(temp_viewfactor > viewfactor_mask, temp_viewfactor, viewfactor_mask)
            
            #kernel = np.ones((3, 3)) / 9  # 3x3 matrix with all values = 1/9
            #kernel = gaussian_kernel(size=3, sigma=1)

            # Perform convolution (filtering) with the Gaussian kernel
            #filtered_raster = convolve(rhf_mask, kernel)
            # Apply the convolution (low-pass filter averaging)
            #filtered_raster = convolve(rhf_mask, kernel, mode='reflect')
            #filtered_raster = median_filter(rhf_mask, size=3)   

        max_rhf_mask = np.where(rhf_mask > max_rhf_mask, rhf_mask, max_rhf_mask) 
        

        
        if Test_Scenario == True:
            filename = OutputFolder + 'RHF_Test_Outputs_' + str(time_key) + '.tif'
        else:
            filename = OutputFolder + 'RHF_Model_Outputs_' + str(time_key) + '.tif'
        # Write the resampled data to a new file
        with rasterio.open(filename, 'w', **dem_profile) as dst:
            dst.write(rhf_mask, 1) #filtered_raster
            dst.nodata = 0.0
            dst.transform = dem_transform

        
        if Test_Scenario == True:
            filename = OutputFolder + 'Viewfactor_Test_Outputs' + str(time_key) + '.tif'
        else:
            filename = OutputFolder + 'Viewfactor_Model_Outputs' + str(time_key) + '.tif'

        # Write the resampled data to a new file
        with rasterio.open(filename, 'w', **dem_profile) as dst:
            dst.write(viewfactor_mask, 1)
            dst.nodata = 0.0
            dst.transform = dem_transform
            
                        
            #print(np.max(temp_mask))
                #print(point_list)
                #Test_Full_Run
    filename = OutputFolder + 'Max_RHF_Output.tif'
    # Write the resampled data to a new file
    with rasterio.open(filename, 'w', **dem_profile) as dst:
        dst.write(max_rhf_mask, 1)
        dst.nodata = 0.0
        dst.transform = dem_transform
            
        #break
    print("Max flame was ",max_flame_height)
    return

if __name__ == "__main__":

    #IsoChromeFile = r'C:\Users\bev25\OneDrive - University of Canterbury\Minority_Report\Spark\SpainData\TestTriangle.shp'
    #IsoChromeFile = r'C:\Users\bev25\OneDrive - University of Canterbury\Minority_Report\Spark\SpainData\TestIso.shp'
    MaxFlameHeightMap = r'C:\Users\rma388\OneDrive - University of Canterbury\PhD\Spark\5_rosie_projects\2-25-rhf_mapper_data\NZ_spark_outputs\output.tif'
    #ArrivalTimeMap = r'C:\Users\rma388\OneDrive - University of Canterbury\PhD\Spark\5_rosie_projects\2-25-rhf_mapper_data\NZ_spark_outputs\arrival_v2.tif'
    #IsoChromeFile = r'C:\Users\rma388\OneDrive - University of Canterbury\PhD\Spark\4_projects_and_analysis\1_port_hills_trails_angles\Output\reprojected_iso.shp'
    IsoChromeFile = r'C:\Users\rma388\OneDrive - University of Canterbury\PhD\Spark\5_rosie_projects\2-25-rhf_mapper_data\Data_from_arcGIS'
    DEMFile = r'C:\Users\rma388\OneDrive - University of Canterbury\PhD\Spark\5_rosie_projects\NZ_ROS_versions\Input\DEM_1m_res.tif'
    DEM_BuildingFile = r'C:\Users\rma388\OneDrive - University of Canterbury\PhD\Spark\5_rosie_projects\2-25-rhf_mapper_data\Building_height_data\Building_Raster_height.tif'
    Building_Outline = r'C:\Users\rma388\OneDrive - University of Canterbury\PhD\Spark\5_rosie_projects\2-25-rhf_mapper_data\Data_from_arcGIS\Building_Outline_v2.tif'
    OutputFolder = r'C:\\Users\\rma388\\OneDrive - University of Canterbury\\PhD\\Spark/5_rosie_projects/NZ_ROS_versions/RHF_mapper//'

    OutputFolder = r'C:\\Users\\rma388\\OneDrive - University of Canterbury\\PhD\\Spark/5_rosie_projects/NZ_ROS_versions/RHF_mapper//'


    TimeStepSeconds = 600 #not used yet
    SimulationStartTime = 600 #not use yet
    SimulationEndTime = 36000 #not used yet

    Maximum_Wall_Length = 8.0 #Specify the maximum length in m a wall segment can be. This then effectively chops up the isochrome lines where needed
   
    """This is used for testing the model for sythetic scenarios. If set to true the model will assume terrain is flat and use parameters below for target and flame properties"""
    Test_Scenario = False #Set to True if you want to test model with parameters below
    Test_Target_Base_Height = 0.0
    Test_Target_Height = 1.2
    Test_Flame_Base_Height = 0.0
    Test_Flame_Height = 17.0

    

    flame_data, flame_transform, flame_pixel_size, flame_profile, flame_transform, flame_nodata_value, flame_flame, arrival_crs, flame_bounds, flame_width, flame_height  = ReadRaster(MaxFlameHeightMap, 4)
    dem_data, dem_transform, dem_pixel_size, dem_profile, dem_transform, dem_nodata_value, dem_dem, arrival_crs, dem_bounds, dem_width, dem_height  = ReadRaster(DEMFile, 1)
    build_data, build_transform, build_pixel_size, build_profile, build_transform, build_nodata_value, build_build, arrival_crs, build_bounds, build_width, build_height  = ReadRaster(DEM_BuildingFile, 1)
    outline_data, outline_transform, outline_pixel_size, outline_profile, outline_transform, outline_nodata_value, outline_outline, arrival_crs, outline_bounds, outline_width, outline_height= ReadRaster(Building_Outline, 1)

    
    if Test_Scenario == True:
        #Sets SEM to be flat ground all zero
        minx, miny, maxx, maxy = dem_bounds
        width = int((maxx - minx) / dem_pixel_size)
        height = int((maxy - miny) / dem_pixel_size)
    
        dem_data = np.zeros((height, width), dtype=np.float64)

    firewall_id, firewall_width, firewall_base_height, firewall_height, firewall_xy_angle_radians, firewall_xy_angle_degrees, firewall_normal_angle_radians, firewall_normal_angle_degrees, firewall_bounding_extents, projected_wall_length, projected_wall_a, projected_wall_b, projected_wall_segment, burn_area_geometry = GetIsoChromeData(IsoChromeFile,flame_data, flame_bounds, flame_pixel_size, flame_transform, dem_data, dem_bounds, dem_pixel_size, dem_transform, Test_Scenario, Test_Flame_Base_Height, Test_Flame_Height, Maximum_Wall_Length, OutputFolder)

    ScanWallSections(burn_area_geometry, firewall_id, firewall_width, firewall_base_height, firewall_height, firewall_xy_angle_degrees, firewall_xy_angle_radians, firewall_normal_angle_radians, firewall_normal_angle_degrees, firewall_bounding_extents,
        dem_data,dem_bounds,dem_pixel_size,
        outline_data, outline_nodata_value, build_data,
        SimulationStartTime,SimulationEndTime,TimeStepSeconds,dem_profile,dem_transform,
        projected_wall_length, projected_wall_a, projected_wall_b, projected_wall_segment, Test_Scenario, Test_Target_Base_Height, Test_Target_Height, OutputFolder)
