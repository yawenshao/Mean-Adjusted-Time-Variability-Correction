"""
Script to plot Figure S9. Location of five global grid cells
The shapefile of the global map ('./Data/World_Continents/World_Continents.shp') is downloadable at https://hub.arcgis.com/datasets/CESJ::world-continents/explore

Author: Yawen Shao, created on March 17, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
import numpy as np

lats = [31,40,49.5,27,-80.5]
lons = [120.5,-120,22,10,30]

aus_map = gpd.read_file('./Data/World_Continents/World_Continents.shp')

# Create Points information
df = pd.DataFrame(data={'Lat':lats, 'Lon':lons})
crs = {'init': 'epsg:4326'}
geometry = [Point(xy) for xy in zip(df["Lon"], df['Lat'])]
geo_df = gpd.GeoDataFrame(df, # Specific data
                          crs = crs, # Specify our coordinate reference system
                          geometry = geometry) # Specify the geometry list we created

# Plot the points onto the aus map
fig, ax = plt.subplots(nrows = 1,ncols = 1,figsize=(8, 16))
aus_map.plot(ax=ax, alpha=0.25, edgecolor='black', linewidth=0.4)
geo_df.plot(ax=ax, markersize = 50, color="red", marker="o")

# Set longitude and latitude limit and labels
plt.ylim([-90,90])
plt.xlim([-180,180])

# Text cells on the map
for i, c in enumerate(['1','2','3','4','5']):
    plt.text(lons[i]-3, lats[i]+6, c, fontsize=11.5, fontweight='bold')

fname = "./Figures/Global_5case_map.jpeg"
fig.savefig(fname, bbox_inches='tight',dpi=300)