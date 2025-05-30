"""
Script to plot Figure 2. Location of four grid cells
The shapefile of the global map ('./Data/Aus_map/AUS_STATE_SHAPE.shp') is downloadable from the webpage of the Australian Bureau of Statistics https://www.abs.gov.au/book/export/25822/print

Author: Yawen Shao, created on November 21, 2024
"""
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
import numpy as np

# Coordinates of the selected grid cells
lats = [-23.5, -28, -32.5, -37]
lons = [133, 148, 116.5, 146.5]
prct_noP = [70, 50, 30, 11] # percentage of the days with no rain

# Australian map
aus_map = gpd.read_file('./Data/Aus_map/AUS_STATE_SHAPE.shp')

# Create Points information
df = pd.DataFrame(data={'Lat':lats, 'Lon':lons})
crs = {'init': 'epsg:4326'}
geometry = [Point(xy) for xy in zip(df["Lon"], df['Lat'])]
geo_df = gpd.GeoDataFrame(df, # Specific data
                          crs = crs, # Specify our coordinate reference system
                          geometry = geometry) # Specify the geometry list we created

# Plot the points onto the aus map
fig, ax = plt.subplots(nrows = 1,ncols = 1,figsize=(5, 5))
aus_map.plot(ax=ax, alpha=0.25, edgecolor='black', linewidth=0.4)
geo_df.plot(ax=ax, markersize = 60, color="red", marker="o")

# Set longitude and latitude limit and labels
plt.ylim([-45,-8.5])
plt.yticks(np.linspace(-40,-10, num=4, dtype=int), ('40$^\circ$S', '30$^\circ$S', '20$^\circ$S', '10$^\circ$S'))

plt.xlim([110,155])
plt.xticks(np.linspace(110,150, num=5, dtype=int), ('110$^\circ$E', '120$^\circ$E', '130$^\circ$E', '140$^\circ$E', '150$^\circ$E'))

# Text cells on the map
for i, c in enumerate(['A','B','C','D']):
    plt.text(lons[i]-0.6, lats[i]+1, c+' '+str(prct_noP[i])+'%', fontsize=10.5, fontweight='bold')

fname = "./Figures/Aus_4case_map.jpeg"
fig.savefig(fname, bbox_inches='tight',dpi=300)