"""
Script to run TVC-ma for selected global grid cells under the model-as-truth setup, and calculate percentage improvement in relevant statistics

Author: Yawen Shao, created on March 17, 2024
"""
import pandas as pd
import numpy as np
import xarray as xr
import time
from TVC_class import TVC

def TVC_func(obs, train, val, scale):
    '''
    Apply time variability correction to the validated data
    obs - observation series
    train - raw training series
    val - raw validation series
    scale - the smoothers list
    '''
    init = sum(scale)-len(scale)

    if sum(np.isnan(obs))>len(obs)/2:
        Z = np.full((len(val)-init, len(scale)+1), np.nan)
    else:
        TVC_model = TVC(obs, train, val, scale)
        Z = TVC_model.TVC_postprocess()

    return np.sum(Z, axis=1)

def perform_TVC(obs, train, val, scale):
    '''
    Function to perform TVC model for DataArray
    '''
    bc_array = xr.apply_ufunc(TVC_func,
                              obs, train, val, scale,
                              input_core_dims=[['time'],['time'],['time2'],['ind']],
                              output_core_dims=[['date']],
                              output_sizes = {"date":len(val.time2[init:])},
                              output_dtypes = [float],
                              vectorize=True,
                              dask='parallelized'
                              )
    return bc_array

def find_optimal_minus(obs, TVC):
    '''
    Find alpha to adjust TVC series so that the mean of the adjusted TVC time series is aligned with the observed mean
    obs - observation series
    TVC - TVC series
    '''
    obs_mu = np.mean(obs)
    mu_diff = obs_mu - np.mean(np.where(TVC>0, TVC, 0))
    a,b,c = mu_diff, mu_diff/2.0, 0 # bound: the initial guess of parameter

    TVC_a = TVC+a
    TVC_a = np.where(TVC_a>0, TVC_a, 0)
    
    while abs(np.mean(TVC_a)-obs_mu)>0.0001:
        if np.mean(TVC_a)-obs_mu<0:
            a=b
            b=(a+c)/2.0
        else:
            c = a
            a = a+(c-b)*2
            b = (a+c)/2.0
            
        TVC_a = TVC+a
        TVC_a = np.where(TVC_a>0, TVC_a, 0)

    return a

def find_optimal_minus_func(obs, TVC):
    array = xr.apply_ufunc(find_optimal_minus,
                      obs, TVC,
                      input_core_dims=[['time'],['time']],
                      output_dtypes = [float],
                      vectorize=True,
                      dask='parallelized'
                     )
    return array

def CDD_cal(data, start_y, end_y):
    '''
    Maximum length of dry spell: maximum number of consecutive days with RR < 1mm (year averaged)
    '''
    
    dates = pd.date_range(str(start_y)+'-01-01', str(end_y)+'-12-31', freq='D')
    data_y = data[-len(dates):]
    data_f = np.where(data_y < 1, 1, 0)

    data_df = pd.DataFrame({'dates':dates, 'data':data_f}, columns=['dates','data'])
    data_df = data_df['data'].groupby(data_df['dates'].dt.year)
    cdd = []
    for year, group in data_df:
        cc = group * (group.groupby((group != group.shift()).cumsum()).cumcount() + 1)
        cdd.append(cc.max())
    cdd = np.sum(cdd)/(end_y-start_y+1)
    
    return cdd

def R10mm_cal(data, start_y, end_y):
    '''
    R10mm: Annual count of days when precipitation ≥ 10 mm (averaged across years)
    '''
    dates = pd.date_range(str(start_y)+'-01-01', str(end_y)+'-12-31', freq='D')
    data_y = data[-len(dates):]
    
    data_df = pd.DataFrame({'dates':dates, 'data':data_y}, columns=['dates','data'])
    data_df = data_df[data_df['data'] >= 10]
    data_df = data_df.groupby(data_df['dates'].dt.year)
    
    R10mm = np.sum([group['data'].count() for year, group in data_df])/(end_y-start_y+1)
    
    return R10mm

def calculate_rx1day(data, start_y, end_y):
    '''
    Calculate Annual Maximum Consecutive 1-day Precipitation (Rx1day)
    '''
    dates = pd.date_range(str(start_y)+'-01-01', str(end_y)+'-12-31', freq='D')
    data_y = data[-len(dates):]

    data_df = pd.DataFrame({'dates':dates, 'data':data_y}, columns=['dates','data'])

    rx1day = data_df['data'].groupby(data_df['dates'].dt.year).max()

    return np.mean(rx1day)

def calculate_rx5day(data, start_y, end_y):
    '''
    Calculate Annual Maximum Consecutive 5-day Precipitation (averaged across years)
    '''
    dates = pd.date_range(str(start_y)+'-01-01', str(end_y)+'-12-31', freq='D')
    data_y = data[-len(dates):]

    data_df = pd.DataFrame({'dates':dates, 'data':data_y}, columns=['dates','data'])

    data_df['5day_sum'] = data_df['data'].rolling(window=5, min_periods=5).sum()
    rx5day = data_df.groupby(data_df['dates'].dt.year)['5day_sum'].max()

    return np.mean(rx5day)

def nday_discrete_average_df(data,n):
    '''
    Average of n-day data with length of len(data)//2
    '''

    data_df = pd.DataFrame(data)

    series = data_df.rolling(n, center=False, min_periods=1).mean().to_numpy().flatten()

    return series[n-1:][::n]

def metric_cal(data, start_y, end_y):
    '''
    Calculate the statistics for the given time series
    data - a given time series
    start_y - start year of the evaluation period
    end_y - end year of the evaluation period
    '''
    if sum(np.isnan(data))>len(data)/2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    ### mean
    mu = np.mean(data)

    ### variance
    var = np.var(data, ddof=1)
    
    ### lag-1 correlation
    lag1 = np.corrcoef(data[1:], data[:-1])[0,1]
    
    ### lag-5 correlation
    data_avg = nday_discrete_average_df(data,5)
    lag5 = np.corrcoef(data_avg[1:], data_avg[:-1])[0,1]
    
    ### R10mm: Annual count of days when precipitation ≥ 10 mm
    R10 = R10mm_cal(data, start_y, end_y)
    
    ### CDD: Maximum length of dry spell: maximum number of consecutive days with RR < 1mm
    max_dry_spell = CDD_cal(data, start_y, end_y)

    ### Rx1day: Maximum consecutive 1-day precipitation
    rx1day = calculate_rx1day(data, start_y, end_y)

    ### Rx5day: Maximum consecutive 5-day precipitation
    rx5day = calculate_rx5day(data, start_y, end_y)
    
    return mu, var, lag1, lag5, R10, max_dry_spell, rx1day, rx5day
   
def metric_all_chunk(data, start_y, end_y):
    bc_array = xr.apply_ufunc(metric_cal,
                              data, start_y, end_y,
                              input_core_dims=[['time'],[],[]],
                              output_core_dims=[[],[],[],[],[],[],[],[]],
                              output_dtypes = [float,float,float,float,float,float,float,float],
                              vectorize=True,
                              dask='parallelized'
                              )    
    return bc_array
 
def prct_improve_MAE(raw, TVC, obs):
    '''
    Calculate the percentage improvement in mean absolute error
    '''
    if sum(np.isnan(obs.flatten()))> len(obs.flatten())-1:
        return np.nan
    # MAE
    raw_MAE = np.nanmean(abs(raw-obs))
    TVC_MAE = np.nanmean(abs(TVC-obs))

    if raw_MAE == 0.0:
        return np.nan
    elif np.isnan(raw_MAE):
        return np.nan
    else:
        return (raw_MAE-TVC_MAE)/raw_MAE*100

def prct_improve(raw, TVC, obs):
    array = xr.apply_ufunc(prct_improve_MAE,
                      raw, TVC, obs,
                      input_core_dims=[['lat','lon'],['lat','lon'],['lat','lon']],
                      vectorize=True,
                      dask='parallelized'
                     )
    return array

def prct_improve_allmodels(raw, TVC, obs):
    array = xr.apply_ufunc(prct_improve_MAE,
                      raw, TVC, obs,
                      input_core_dims=[['model'],['model'],[]],
                      vectorize=True,
                      dask='parallelized'
                     )
    return array
    
if __name__ == "__main__" 
    scale = [365,183,92,46,23,12,6,3,2]
    init = sum(scale)-len(scale)

    # Coordinates information for four cells
    lats = [-23.5, -28, -32.5, -37]
    lons = [133, 148, 116.5, 146.5]
    
    names = ['mu', 'var', 'lag1', 'lag5', 'R10', 'max_dry_spell', 'Rx1day', 'Rx5day']
    comp = dict(zlib=True, complevel=9)

    TVC_ma_fut_stat_ssp_all = []
    obs_fut_stat_ssp_all = []

    # Import data of global cases
    data_hist_all = xr.open_dataset('./Data/pr_day_allmodels_historical_1.5_global.nc')
    data_hist_all = data_hist_all.pr

    for ssp in ['historical','ssp126','ssp585']:
        TVC_ma_fut_stat_all = []
        obs_fut_stat_all = []
        
        if ssp == 'historical':
            data_all = data_hist_all.copy()
        else:
            data_all = xr.open_dataset('./Data/pr_day_allmodels_'+ssp+'_1.5_global.nc')
            data_all = data_all.pr
        
        for m, truth in enumerate(models):
            raw_models = models[models != truth]
            obs = data_hist_all.sel(model=truth)
            raw = data_hist_all.sel(model=raw_models).chunk({"model": 3})
            
            obs_fut = data_all.sel(model=truth)
            raw_fut = data_all.sel(model=raw_models).chunk({"model": 3})

            Z_hist = perform_TVC(obs, raw, raw.rename(time='time2'), scale)
            Z_hist = xr.where(Z_hist>0, Z_hist, 0).rename(date='time')
                
            Z_fut = perform_TVC(obs, raw, raw_fut.rename(time='time2'), scale)
            Z_fut = xr.where(Z_fut>0, Z_fut, 0).rename(date='time')

            root = find_optimal_minus_func(obs.isel(time=slice(init,len(obs.time))), Z_hist)
                
            Z_fut_ma = root+Z_fut
            Z_fut_ma = xr.where(Z_fut_ma > 0, Z_fut_ma, 0)

            if ssp == 'historical':
                TVC_ma_fut_stat = metric_all_chunk(Z_fut_ma, 1952, 2014)
                obs_fut_stat = metric_all_chunk(obs_fut.isel(time=slice(init,len(obs_fut.time))), 1952, 2014)
            else:
                TVC_ma_fut_stat = metric_all_chunk(Z_fut_ma, 2017, 2099)
                obs_fut_stat = metric_all_chunk(obs_fut.isel(time=slice(init,len(obs_fut.time))), 2017, 2099)
                
            TVC_ma_fut_stat_all.append(xr.concat(TVC_ma_fut_stat, dim='stat').assign_coords(stat=names))
            obs_fut_stat_all.append(xr.concat(obs_fut_stat, dim='stat').assign_coords(stat=names))

        TVC_ma_fut_stat_ssp_all.append(xr.concat(TVC_ma_fut_stat_all, dim='truth').assign_coords(truth=models))
        obs_fut_stat_ssp_all.append(xr.concat(obs_fut_stat_all, dim='truth').assign_coords(truth=models))
        
    TVC_ma_fut_stat_ssp_all = xr.concat(TVC_ma_fut_stat_ssp_all, dim='ssp').assign_coords(ssp=['historical','ssp126','ssp585',])
    obs_fut_stat_ssp_all = xr.concat(obs_fut_stat_ssp_all, dim='ssp').assign_coords(ssp=['historical','ssp126','ssp585'])

    TVC_ma_fut_stat_ssp_all = TVC_ma_fut_stat_ssp_all.to_dataset(name='stats')
    encoding = {var: comp for var in TVC_ma_fut_stat_ssp_all.data_vars}
    TVC_ma_fut_stat_ssp_all.to_netcdf('./Data/TVC_ma_global_cases.nc', encoding=encoding)

    obs_fut_stat_ssp_all = obs_fut_stat_ssp_all.to_dataset(name='stats')
    encoding = {var: comp for var in obs_fut_stat_ssp_all.data_vars}
    obs_fut_stat_ssp_all.to_netcdf('./Data/Obs_global_cases.nc', encoding=encoding)
    
    
    ### Calculate and print averaged percentage improvement
    TVC_ma_fut_stat_ssp_all = xr.open_dataset('./Data/TVC_ma_global_cases.nc')
    obs_fut_stat_ssp_all = xr.open_dataset('./Data/Obs_global_cases.nc')

    prct_all_scenario = []
    for expr in ['historical','ssp126','ssp585']:
        prct_all = []
        for truth in models:
            raw_models = models[models != truth]
            obs_data = obs_fut_stat_ssp_all.stats.sel(truth=truth, ssp=expr)
            
            raw_data = []
            for model in raw_models:
                raw_data.append(obs_fut_stat_ssp_all.stats.sel(truth=model, ssp=expr))
            raw_data = xr.concat(raw_data, dim='model').assign_coords(model=raw_models)
            
            TVC_ma_data = TVC_ma_fut_stat_ssp_all.stats.sel(truth=truth, model=raw_models, ssp=expr)
            
            prct = prct_improve_allmodels(raw_data, TVC_ma_data, obs_data)
            prct_all.append(prct)
            
        prct_all = xr.concat(prct_all, dim='model').assign_coords(model=models)
        prct_all_scenario.append(prct_all)
        
        print(expr)
        print(prct_all.mean(dim=['model']).values)
    