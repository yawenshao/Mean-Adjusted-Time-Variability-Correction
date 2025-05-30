"""
Script to plot Figure 7. Plot probability density function of R10mm and CDD for CNRM-CM6-1, GFDL-ESM4, and MIROC6 obs, raw and TVC-ma results, and calculate total variance distance

Author: Yawen Shao, created on February 27, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib
from scipy.stats import rv_histogram
from scipy.stats import rv_continuous
from scipy.ndimage import gaussian_filter1d

font = {
        'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 14,
        }
matplotlib.rc('font', **font)

models = pd.read_csv('./Data/CMIP6_model_name.csv', header=None)
models = models.to_numpy().flatten()

names = ['R10mm (day/yr)','CDD (day/yr)']
bins = 20
sigma = 1  # Increasing sigma increases the smoothness
model_name = ['CNRM-CM6-1','GFDL-ESM4','MIROC6']
letters = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
ys = [0.0685, 0.065, 0.048, 0.0112, 0.0113, 0.0218]

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,9.5))
colors = [['lightcoral','orangered','red'],['lightgrey','grey','black'],['skyblue','royalblue','blue']]

bb = []
for j, expr in enumerate(['historical','ssp126','ssp585']):
    for i, truth in enumerate(model_name):
        # Import data
        obs_org = xr.open_dataset('./Data/'+truth+'_stat_'+expr+'_Aus_obs_pdf.nc')
        TVC_org = xr.open_dataset('./Data/MaT_stat_'+truth+'_allmodels_'+expr+'_Aus_TVC_ma_pdf.nc')
        raw_models = models[models!=truth]
        
        raw_all = []
        for m in raw_models:
            raw = xr.open_dataset('./Data/'+m+'_stat_'+expr+'_Aus_obs_pdf.nc')
            raw_all.append(raw)
        raw_all_org = xr.concat(raw_all, dim='model').assign_coords(model=raw_models)
        
        for name, varb in enumerate(names):
            obs = obs_org.stats.isel(stat=1-name).values
            obs = obs[~np.isnan(obs)]
            TVC = TVC_org.stats.isel(stat=1-name).values
            TVC = TVC[~np.isnan(TVC)]
            raw_all = raw_all_org.stats.isel(stat=1-name).values
            raw_all = raw_all[~np.isnan(raw_all)]

            obs_data = np.histogram(obs, bins=bins)
            obs_dist = rv_histogram(obs_data, density=False)
            raw_data = np.histogram(raw_all, bins=bins)
            raw_dist = rv_histogram(raw_data, density=False)
            TVC_data = np.histogram(TVC, bins=bins)
            TVC_dist = rv_histogram(TVC_data, density=False)

            if name == 0: #R10mm
                X = np.linspace(0, np.max(obs)/2, bins)
            elif name == 1: #CDD
                X = np.linspace(0, np.max(obs)-100, bins)

            # Evaluate the PDFs at the points in X
            pdf_obs = obs_dist.pdf(X)
            pdf_TVC = TVC_dist.pdf(X)
            pdf_raw = raw_dist.pdf(X)
            
            # Calculate total variance distance
            TVD_TVC = 0.5 * np.sum(abs(pdf_TVC - pdf_obs))
            TVD_raw = 0.5 * np.sum(abs(pdf_raw - pdf_obs))
            
            print(varb+' '+truth+' Raw '+expr+': '+str(TVD_raw))
            print(varb+' '+truth+' TVC '+expr+': '+str(TVD_TVC))
            print('--------------------')

            # Apply a Gaussian filter to smooth the PDF lines; adjust sigma as needed
            pdf_obs_smooth = gaussian_filter1d(pdf_obs, sigma=sigma)
            pdf_TVC_smooth = gaussian_filter1d(pdf_TVC, sigma=sigma)
            pdf_raw_smooth = gaussian_filter1d(pdf_raw, sigma=sigma)

            # Plot the smoothed PDFs
            b1 = ax[name, i].plot(X, pdf_obs_smooth, color = colors[0][j], label='Obs-'+expr)
            b2 = ax[name, i].plot(X, pdf_raw_smooth, color = colors[1][j], label='Raw-'+expr)
            b3 = ax[name, i].plot(X, pdf_TVC_smooth, color = colors[2][j], label='TVC-'+expr)

            if i == 0 and name == 0:
                bb.append([b1,b2,b3])

            if name == 0:
                ax[name, i].set_title(truth, fontsize=17, weight='bold',pad=6)

            if j == 0:
                ax[name, i].set_xlabel(varb, fontsize=16)

            if i == 0 and j == 0:
                ax[name, i].set_ylabel('Probability', fontsize=16)
            
            ax[name, i].text(x=-1, y=ys[i+3*name], s=letters[i+3*name], fontsize=17, fontweight='bold')

bb = [item for sublist in bb for pair in sublist for item in pair]
legend = plt.legend(handles=bb, bbox_to_anchor=(-0.05, -0.15),
          ncol=3, fontsize=14, handletextpad = 0.4,labelspacing=0.35,handlelength=1.3)
legend._legend_box.sep = 8
legend.get_frame().set_linewidth(0.5)
legend.get_frame().set_edgecolor('k')

fig.subplots_adjust(left=0.06,top=0.96,bottom=0.17,right=0.99, hspace=0.25)
fig.savefig('./Figures/PDF_plot_bin'+str(bins)+'_selected_models_paper.jpeg',dpi=300)