"""
Script to plot Figure S6-7 and Table S1. Plot probability density function of R10mm and CDD for all models, and calculate total variance distance

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

names = ['CDD (day/yr)','R10mm (day/yr)']
name = 0 #0-CDD, 1-R10
bins = 20
sigma = 1  # Increasing sigma increases the smoothness

fig, ax = plt.subplots(nrows=7, ncols=4, figsize=(19,24))
colors = [['lightcoral','orangered','red'],['lightgrey','grey','black'],['skyblue','royalblue','blue']] #obs, raw, TVC

bb = []
TVD_all = np.full((len(models)*2, 3), np.nan)
for j, expr in enumerate(['historical','ssp126','ssp585']):
    for i, truth in enumerate(models):
        obs = xr.open_dataset('./Data/'+truth+'_stat_'+expr+'_Aus_obs_pdf.nc')
        obs = obs.stats.isel(stat=name).values
        obs = obs[~np.isnan(obs)]
        TVC = xr.open_dataset('./Data/MaT_stat_'+truth+'_allmodels_'+expr+'_Aus_TVC_ma_pdf.nc')
        TVC = TVC.stats.isel(stat=name).values
        TVC = TVC[~np.isnan(TVC)]
        
        raw_models = models[models!=truth]

        raw_all = []
        for m in raw_models:
            raw = xr.open_dataset('./Data/'+m+'_stat_'+expr+'_Aus_obs_pdf.nc')
            raw_all.append(raw)
        raw_all = xr.concat(raw_all, dim='model').assign_coords(model=raw_models)
        raw_all = raw_all.stats.isel(stat=name).values
        raw_all = raw_all[~np.isnan(raw_all)]
        
        obs_data = np.histogram(obs, bins=bins)
        obs_dist = rv_histogram(obs_data, density=False)
        raw_data = np.histogram(raw_all, bins=bins)
        raw_dist = rv_histogram(raw_data, density=False)
        TVC_data = np.histogram(TVC, bins=bins)
        TVC_dist = rv_histogram(TVC_data, density=False)
        
        if name == 0:
            X = np.linspace(0, np.max(obs)-100, bins)
        elif name == 1:
            X = np.linspace(0, np.max(obs)/2, bins)
            
        # Evaluate the PDFs at the points in X
        pdf_obs = obs_dist.pdf(X)
        pdf_TVC = TVC_dist.pdf(X)
        pdf_raw = raw_dist.pdf(X)
        
        # Calculate total variance distance
        TVD_TVC = 0.5 * np.sum(abs(pdf_TVC - pdf_obs))
        TVD_raw = 0.5 * np.sum(abs(pdf_raw - pdf_obs))
        
        TVD_all[2*i,j] = TVD_raw
        TVD_all[2*i+1,j] = TVD_TVC

        # Apply a Gaussian filter to smooth the PDF lines; adjust sigma as needed
        pdf_obs_smooth = gaussian_filter1d(pdf_obs, sigma=sigma)
        pdf_TVC_smooth = gaussian_filter1d(pdf_TVC, sigma=sigma)
        pdf_raw_smooth = gaussian_filter1d(pdf_raw, sigma=sigma)

        # Plot the smoothed PDFs
        b1 = ax[i//4, i%4].plot(X, pdf_obs_smooth, color = colors[0][j], label='Obs-'+expr)
        b2 = ax[i//4, i%4].plot(X, pdf_raw_smooth, color = colors[1][j], label='Raw-'+expr)
        b3 = ax[i//4, i%4].plot(X, pdf_TVC_smooth, color = colors[2][j], label='TVC-'+expr)
        
        if i == 0:
            bb.append([b1,b2,b3])
        
        ax[i//4, i%4].set_title(truth, fontsize=17, weight='bold',pad=6)
        
        if i//4 == 6 and j == 0:
            ax[i//4, i%4].set_xlabel(names[name], fontsize=16)
        
        if i%4 == 0 and j == 0:
            ax[i//4, i%4].set_ylabel('Probability', fontsize=16)

bb = [item for sublist in bb for pair in sublist for item in pair]
legend = plt.legend(handles=bb, bbox_to_anchor=(0.8, -0.22),
          ncol=9, fontsize=14, handletextpad = 0.4,labelspacing=0.35,handlelength=1.3)
legend._legend_box.sep = 8
legend.get_frame().set_linewidth(0.5)
legend.get_frame().set_edgecolor('k')

fig.subplots_adjust(left=0.05,top=0.98,bottom=0.05,right=0.99, hspace=0.25)
fig.savefig('./Figures/PDF_plot_'+str(name)+'_bin'+str(bins)+'_paper.jpeg',dpi=300)