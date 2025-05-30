v"""
Script to calculate and plot the mean of the 365-day scale average time series

Author: Yawen Shao, created on November 21, 2024
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib

font = {
        'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 12,
        }
matplotlib.rc('font', **font)

def sep_time_scales_rolling(data, scales):
    '''
    The wavelet based transform approach to filter the original time series
    '''
    # The length of data and level of smoothers
    n = len(data)
    l = len(scales)

    y_Pkn = np.full((n, l), np.nan)
    y_Pkn_ano = np.full((n, l), np.nan)

    data_df = pd.DataFrame(data)
    
    # Filter the time series one by one
    # The Nan values are handled
    for k, smer in enumerate(scales): # timescale of evolution
        if k == 0:
            y_Pkn[:,k] = data_df.rolling(smer, center=False, min_periods=1).mean().to_numpy().flatten()
            y_Pkn_ano[:,k] = data - y_Pkn[:,k]
        else:
            y_ano_df = pd.DataFrame(y_Pkn_ano[:,k-1])
            y_Pkn[:,k] = y_ano_df.rolling(smer, center=False, min_periods=1).mean().to_numpy().flatten()
            y_Pkn_ano[:,k] = y_Pkn_ano[:,k-1] - y_Pkn[:,k]

        y_Pkn[np.where(np.isnan(y_Pkn_ano[:,k])),k] = np.nan

    # Obtain data with filtered time series
    init_ind = sum(scales)-l
    L_data = n - init_ind

    Y = np.full((L_data,l+1), np.nan)
    Y[:,-1] = y_Pkn_ano[init_ind:,-1]
    Y[:,:-1] = y_Pkn[init_ind:,:]

    return Y

def get_TVC_delta_mean(data, scales):
    '''
    Calculate the mean of all time scales
    '''
    if sum(~np.isnan(data)) == 0:
        return np.full((len(scales)+1), np.nan)
    data_m = sep_time_scales_rolling(data, scales)
    
    # Remove nan values
    filter_data = np.argwhere(np.isnan(data_m[:,0]))
    data_m = np.delete(data_m, filter_data, axis=0)

    # Calculate mean vector
    data_mean = np.mean(data_m, axis=0)

    return data_mean

def get_TVC_delta_mean_func(data, scales):
    array = xr.apply_ufunc(get_TVC_delta_mean,
                           data, scales,
                           input_core_dims=[['time'],['ind']],
                           output_core_dims=[['mu']],
                           output_sizes = {"mu":len(scales)+1},
                           output_dtypes = [float],
                           vectorize=True,
                           dask='parallelized'
                          )

    return array

if __name__ == "__main__" 
    scales = [365,183,92,46,23,12,6,3,2]
    init = sum(scales)-len(scales)

    models = pd.read_csv('./Data/CMIP6_model_name.csv', header=None)
    models = models.to_numpy().flatten()

    # Calculate and save the mean ratio for all obs, raw and TVC results
    ## Take ACCESS-ESM1-5 ssp126 scenario as an example
    for m in ['ACCESS-ESM1-5']:
        data_hist = xr.open_dataset('./Data/pr_day_'+m+'_historical_1950_2014_1.5_masked.nc', chunks={"lat":3, "lon":6})
        data_hist = data_hist*60*60*24
        mu_hist = get_TVC_delta_mean_func(data_hist, scales)

        for expr in ['ssp126','ssp585']:
            data_fut = xr.open_dataset('./Data/pr_day_'+m+'_'+expr+'_2015_2099_1.5_masked.nc', chunks={"lat":3, "lon":6})
            data_fut = data_fut*60*60*24
            mu_fut = get_TVC_delta_mean_func(data_fut, scales)

            ratio = mu_fut/mu_hist
            ratio = ratio.assign_coords(s=[365,183,92,46,23,12,6,3,2,1])
            
            ratio.to_netcdf('./Data/'+m+'_climate_change_mean_ratio_'+expr+'_Aus_obs_TVCma.nc')

    # Plot boxplot for mean ratio at the 365-day time scale
    color1 = ['yellow','gold']
    mcolor = 'orange'
    letters = ['a)','b)','c)','d)','e)','f)','g)','h)','i)','j)']
    yys = [1.9, 0.4, 2.7, 2.35, 2.72, 2.77, 2.6, 2.6, 2.52, 2.45]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,6))

    ax_tick = []
    for r in range(1):
        bb = []
        
        for k, m in enumerate(models):
            for s, expr in enumerate(['ssp126','ssp585']):
                data = xr.open_dataset('./Data/'+m+'_climate_change_mean_ratio_'+expr+'_Aus_obs_TVCma.nc')

                data_s = data.pr.isel(mu=r).values.flatten()
                data_s = data_s[~np.isnan(data_s)]

                b = ax.boxplot(data_s,
                       vert=True,
                       patch_artist=True, 
                       positions=[0.12+k+0.25*s],
                       widths=(0.13), 
                       showmeans = True,
                       showfliers = False,
                       whiskerprops=dict(linewidth=2, color=color1[s]),
                       capprops=dict(linewidth=0, color=color1[s]),
                       boxprops=dict(linewidth=2, edgecolor=color1[s], facecolor=color1[s]),
                       medianprops=dict(linewidth=2, color=mcolor),
                       meanprops=dict(marker="D", markersize=0)
                      )

                if k == 0:
                    bb.append(b)

                if s == 1 and r == 0:
                    ax_tick.append(m)
                    ax_tick.append(" ")

                ax.set_xticklabels([])

                ax.set_ylabel(r'$\mu_{'+str(scale_y[r])+'}$', fontsize=14)
                ax.set_title(str(data.isel(s=r).s.values)+'-day', fontsize=17, weight='bold',pad=6)

                if r == 0:
                    ax.set_ylim([0.5,1.7])
                else:
                    ax.set_ylim([-10,10])
                    
        ax.axhline(y=1, color='grey', linestyle='dashed', linewidth=1)
        ax.legend([bb[0]["boxes"][0], bb[1]["boxes"][0]], ['ssp126', 'ssp585'], loc='upper right', fontsize=15)
        ax.set_xticklabels(ax_tick, rotation=80, fontsize=15)
                
    fig.subplots_adjust(left=0.05,top=0.95,bottom=0.31,right=0.99)
    fig.savefig('./Figures/Boxplot_mean_MaT_allmodels_all_expr_Aus_TVC_ma_365day.jpeg',dpi=300)