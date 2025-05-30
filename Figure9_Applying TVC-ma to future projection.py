"""
Script to plot Figure 9. Australian map of future projection statistics for application of TVC-ma trained on AGCD data
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

# For ssp126 and ssp585 together - 6*4 plot
models = pd.read_csv('./Data/CMIP6_model_name.csv', header=None)
models = models.to_numpy().flatten()
color = ['Wistia', 'Reds','Purples','Greys','GnBu','PuBu']

title_all = ['Mean','Variance','R10mm','CDD','Rx1day','Rx5day']
bar_label = ['Mean (mm)','Variance (mm$^2$)','R10mm (day/yr)','CDD (day/yr)','Rx1day (mm/yr)','Rx5day (mm/year)']
titles = ['Raw-MMM', 'TVC-ma']
bar_y = [0.86, 0.69, 0.53, 0.36, 0.2, 0.03]
letters = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)','h)', 'i)','j)','k)','l)','m)','n)','o)','p)','q)','r)','s)','t)','u)','v)','w)','x)']
ylabel = [-29,-32,-31,-30,-32,-32]

fig, ax = plt.subplots(nrows=6, ncols=4, figsize=(19,23), subplot_kw={'projection': ccrs.Mercator()})  

levels = [[0,0.5,1,1.5,2,3,4,5], [0,1,2,5,10,20,30,50,100], [0,5,10,15,20,30,40,50], [0,20,40,60,80,100,150,200], [0,5,10,15,25,35,45,60], [0,20,40,60,80,100,150,200]] #[0,0.1,0.2,0.3,0.4,0.6,0.8,0.9],[0,0.1,0.2,0.3,0.4,0.5,0.6]
num = [8, 9, 9, 8, 9, 8]

for i in range(6):
    cmap_get = plt.get_cmap(color[i], num[i])
    cmap = matplotlib.colors.ListedColormap([cmap_get(k) for k in range(num[i]-1)])
    cmap.set_over(cmap_get(num[i]-1))
    norm = matplotlib.colors.BoundaryNorm(levels[i], cmap.N)
    
    for e, expr in enumerate(['ssp126','ssp585']):
        TVC_stat = xr.open_dataset('/g/data/w42/ys9723/rainfall/Result/TVC_ma_stats_allmodels_'+expr+'_Aus.nc')
        TVC_stat = TVC_stat.stats.isel(stat=[0,1,4,5,6,7]).mean(dim='model')
        
        raw_stat = []
        for m in models:
            raw = xr.open_dataset('/g/data/w42/ys9723/rainfall/Result/'+m+'_'+expr+'_Aus_obs2.nc')
            raw_stat.append(raw.isel(stat=[0,1,4,5,6,7]))
        raw_stat = xr.concat(raw_stat, dim='model').stats.mean(dim='model')
        
        lat = raw_stat.lat
        lon = raw_stat.lon
        
        TVC = TVC_stat.isel(stat=i)
        raw = raw_stat.isel(stat=i)

        im = ax[i,2*e].pcolormesh(
            lon,
            lat,
            raw,
            norm=norm,
            cmap=cmap
        )

        ax[i,2*e+1].pcolormesh(
            lon,
            lat,
            TVC,
            norm=norm,
            cmap=cmap
        )

        for r in range(2):
            ax[i,2*e+r].add_feature(shape_feature, linewidth=0.5)
            ax[i,2*e+r].set_extent([112,154,-44,-9],ccrs.Mercator())
            ax[i,2*e+r].text(x=113, y=-8.5, s=letters[i+6*r+12*e], fontsize=20, fontweight='bold')
            if i == 0:
                ax[i,2*e+r].set_title(expr+'  '+titles[r], fontsize=20, fontweight='bold')
            if e == 0 and r == 0:
                ax[i,2*e+r].text(x=107, y=ylabel[r], s=title_all[i], fontsize=20, rotation=90, fontweight='bold') 

    # Set colorbar
    cbaxes = fig.add_axes([0.94, bar_y[i], 0.015, 0.1])
    cbar = fig.colorbar(im, orientation='vertical', cax=cbaxes,
                        extend='max',
                        shrink=0.6,
                        ticks=levels[i]
                        )
    cbar.set_label(bar_label[i], fontdict=font)

fig.subplots_adjust(left=0.03,top=0.99,bottom=0.005,right=0.92, hspace=0.02, wspace=0.02)
fig.savefig('./Figures/Aus_map_4metrics_TVC_ma_ssp126_ssp585_Aus_indices_paper.jpeg',dpi=300)