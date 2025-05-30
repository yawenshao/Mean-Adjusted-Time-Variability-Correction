"""
Script to calculate climate change ratio matrix and plot diagonal elements of the climate change ratio matrix. 

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

def get_CC_ratio(cov_hist, cov_fut):
    '''
    Calculate the diagonal elements of the climate change ratio matrix
    '''
    if sum(~np.isnan(np.diag(cov_hist))) == 0:
        return np.full((len(scales)+1, ), np.nan)

    ratio = sqrtm(np.linalg.inv(cov_hist)) @ sqrtm(cov_fut) @ sqrtm(np.linalg.inv(cov_hist))

    return np.diag(ratio)

def get_ratio_scales(raw, fut):
    bc_array = xr.apply_ufunc(get_CC_ratio,
                              raw, fut,
                              input_core_dims=[['s1','s2'], ['s1','s2']],
                              output_core_dims=[['s']],
                              output_dtypes = [float],
                              output_sizes = {"s":len(raw.s1)},
                              vectorize=True,
                              dask='parallelized'
                              )
    return bc_array

def get_TVC_delta_cov(data, scales):
    '''
    Obtain the covariance matrix of the data series
    '''
    if sum(~np.isnan(data)) == 0:
        return np.full((len(scales)+1, len(scales)+1), np.nan)
    data_m = sep_time_scales_rolling(data, scales)
    
    # Remove nan values
    filter_data = np.argwhere(np.isnan(data_m[:,0]))
    data_m = np.delete(data_m, filter_data, axis=0)

    # Calculate mean vector
    data_mean = np.mean(data_m, axis=0)

    # Calculate covariance matrix
    cov_data = (data_m.T - data_mean[:,None]) @ (data_m - data_mean)/(data_m.shape[0]-1)

    return cov_data

def get_TVC_delta_cov_func(data, scales):
    array = xr.apply_ufunc(get_TVC_delta_cov,
                           data, scales,
                           input_core_dims=[['time'],['ind']],
                           output_core_dims=[['s1','s2']],
                           output_sizes = {"s1":len(scales)+1, "s2":len(scales)+1},
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

    # Calculate the metrics for all obs, raw and TVC results
    # Take ACCESS-ESM1-5 SSP126 scenario as an example
    m, expr = 'ACCESS-ESM1-5', 'ssp126'
    data_hist = xr.open_dataset('./Data/pr_day_'+m+'_historical_1950_2014_1.5_masked.nc', chunks={"lat":3, "lon":6})
    cov_hist = get_TVC_delta_cov_func(data_hist, scales)

    data_fut = xr.open_dataset('./Data/'+expr+'/pr_day_'+m+'_'+expr+'_2015_2099_1.5_masked.nc', chunks={"lat":3, "lon":6})
    cov_fut = get_TVC_delta_cov_func(data_fut, scales)

    ratio = get_ratio_scales(cov_hist, cov_fut)
    ratio = ratio.assign_coords(s=[365,183,92,46,23,12,6,3,2,1])
            
    ratio.to_netcdf('./Data/'+m+'_climate_change_ratio_'+expr+'_Aus_obs_TVCma.nc')
            
    # Plot boxplot of diaganol elements of lambda x
    scales = [365,183,92,46,23,12,6,3,2]
    scale_y = [365,183,92,46,23,12,6,3,2,1]
    init = sum(scales)-len(scales)
    models = pd.read_csv('./Data/CMIP6_model_name.csv', header=None)
    models = models.to_numpy().flatten()

    color1 = ['lightcoral','indianred']
    mcolor = 'red'
    letters = ['a)','b)','c)','d)','e)','f)','g)','h)','i)','j)']
    yys = [3.9, 2.4, 2.7, 2.35, 2.72, 2.77, 2.6, 2.6, 2.52, 2.45]

    fig, ax = plt.subplots(nrows=10, ncols=1, figsize=(13,28))

    ax_tick = []
    for r in range(len(scales)+1):
        bb = []
        
        for k, m in enumerate(models):
            for s, expr in enumerate(['ssp126','ssp585']):
                data = xr.open_dataset('./Data/'+m+'_climate_change_ratio_'+expr+'_Aus_obs_TVCma.nc')

                data_s = data.pr.isel(s=r).values.flatten()
                data_s = data_s[~np.isnan(data_s)]

                b = ax[r].boxplot(data_s,
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

                ax[r].set_xticklabels([])

                ax[r].set_ylabel(r'$\sigma^{2}_{'+str(scale_y[r])+'}$', fontsize=14)
                ax[r].set_title(str(data.isel(s=r).s.values)+'-day', fontsize=17, weight='bold',pad=6)
                    
        ax[r].axhline(y=1, color='grey', linestyle='dashed', linewidth=1)
        if r == 0:
            ax[r].legend([bb[0]["boxes"][0], bb[1]["boxes"][0]], ['ssp126', 'ssp585'], loc='upper right', fontsize=15)
        ax[r].text(x=-0.3, y=yys[r], s=letters[r], fontsize=15, fontweight='bold')
        
        if r == len(scales):
            ax[r].set_xticklabels(ax_tick, rotation=80, fontsize=15)
                
    fig.subplots_adjust(left=0.06,top=0.99,bottom=0.08,right=0.99, hspace=0.16)
    fig.savefig('./Figures/Boxplot_lambdaX_MaT_allmodels_all_expr_Aus_TVC_ma.jpeg',dpi=300)