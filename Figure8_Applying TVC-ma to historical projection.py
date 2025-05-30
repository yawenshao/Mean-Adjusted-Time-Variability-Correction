"""
Script to plot Figure 8. Australian map of historical statistics for application of TVC-ma trained on AGCD data
The shapefile of the global map ('./Figures/Aus_map/national.shp') is downloadable from the webpage of the Australian Bureau of Statistics https://www.abs.gov.au/book/export/25822/print

Author: Yawen Shao, created on November 21, 2024
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

shpFilePath = './Figures/Aus_map/national.shp'
shape_feature = ShapelyFeature(Reader(shpFilePath).geometries(),ccrs.Mercator(),edgecolor='black',facecolor='none')

font = {
        'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 12,
        }
matplotlib.rc('font', **font)

expr =  'historical'
models = pd.read_csv('./Data/CMIP6_model_name.csv', header=None)
models = models.to_numpy().flatten()
TVC_stat = xr.open_dataset('./Data/TVC_ma_stats_allmodels_'+expr+'_Aus.nc')
TVC_stat = TVC_stat.stats.isel(stat=[0,1,4,5,6,7]).mean(dim='model')

raw_stat = []
for m in models:
    raw = xr.open_dataset('./Data/'+m+'_'+expr+'_Aus_obs2.nc')
    raw_stat.append(raw.stats.isel(stat=[0,1,4,5,6,7]))

raw_stat = xr.concat(raw_stat, dim='model').mean(dim='model')
obs_stat = xr.open_dataset('./Data/AGCD_stats_Aus.nc')
obs_stat = obs_stat.stats.isel(stat=[0,1,4,5,6,7])

## For historical/ssp126/ssp585 period 6*3 plot
color = ['Wistia', 'Reds','Purples','Greys','GnBu','PuBu']

title_all = ['Mean','Variance','R10mm','CDD','Rx1day','Rx5day']
bar_label = ['Mean (mm)','Variance (mm$^2$)','R10mm (day/yr)','CDD (day/yr)','Rx1day (mm/yr)','Rx5day (mm/year)']
titles = ['AGCD','Raw-MMM', 'TVC-ma']
bar_y = [0.85, 0.69, 0.525, 0.36, 0.195, 0.03]
letters = ['a)','b)','c)','d)','e)','f)','g)','h)','i)','j)','k)','l)','m)','n)','o)','p)','q)','r)']
ylabel = [-33,-32,-32,-32,-33,-33]

fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(14,21), subplot_kw={'projection': ccrs.Mercator()})  

levels = [[0,0.5,1,1.5,2,3,4,5], [0,1,2,5,10,20,30,50,100], [0,5,10,15,20,30,40,50], [0,20,40,60,80,100,150,200], [0,5,10,15,25,35,45,60], [0,20,40,60,80,100,150,200]]
num = [8, 9, 9, 8, 9, 8]

lat = raw_stat.lat
lon = raw_stat.lon

for i in range(6):
    cmap_get = plt.get_cmap(color[i], num[i])
    cmap = matplotlib.colors.ListedColormap([cmap_get(k) for k in range(num[i]-1)])
    cmap.set_over(cmap_get(num[i]-1))
    norm = matplotlib.colors.BoundaryNorm(levels[i], cmap.N)
    
    im = ax[i,1].pcolormesh(
        lon,
        lat,
        raw_stat.isel(stat=i),
        norm=norm,
        cmap=cmap
    )

    ax[i,2].pcolormesh(
        lon,
        lat,
        TVC_stat.isel(stat=i),
        norm=norm,
        cmap=cmap
    )

    ax[i,0].pcolormesh(
        lon,
        lat,
        obs_stat.isel(stat=i),
        norm=norm,
        cmap=cmap
    )
    
    for r in range(3):
        ax[i,r].add_feature(shape_feature, linewidth=0.5)
        ax[i,r].set_extent([112,154,-44,-9],ccrs.Mercator())
        ax[i,r].text(x=112, y=-8.5, s=letters[r+3*i], fontsize=19, fontweight='bold')
        if i == 0:
            ax[i,r].set_title(titles[r], fontsize=19, fontweight='bold')
        if r == 0:
            ax[i,r].text(x=107, y=ylabel[i], s=title_all[i], fontsize=19, rotation=90, fontweight='bold')

    # Set colorbar
    cbaxes = fig.add_axes([0.92, bar_y[i], 0.02, 0.1])
    cbar = fig.colorbar(im, orientation='vertical', cax=cbaxes,
                        extend='max',
                        shrink=0.6,
                        ticks=levels[i]
                        )
    cbar.set_label(bar_label[i], fontdict=font)

fig.subplots_adjust(left=0.04,top=0.98,bottom=0.01,right=0.9, hspace=0.1, wspace=0.02)

fig.savefig('./Figures/Aus_map_4metrics_'+titles[1]+'_'+expr+'_Aus_r2_indices_paper_historical.jpeg',dpi=300)