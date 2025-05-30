"""
Script to run TVC-ma for selected grid cells, and calculate relevant statistics

Author: Yawen Shao, created on November 21, 2024
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
                              input_core_dims=[['time'],['time'],['time'],['ind']],
                              output_core_dims=[['date']],
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
    
if __name__ == "__main__" 
    scale = [365,183,92,46,23,12,6,3,2]
    init = sum(scale)-len(scale)

    # Coordinates information for four cells
    lats = [-23.5, -28, -32.5, -37]
    lons = [133, 148, 116.5, 146.5]

    # Import obs and raw data
    obs = xr.open_dataset('/g/data/w42/ys9723/rainfall/CMIP_aus/Result/agcd_v1_precip_four_cells.nc')
    raw = xr.open_dataset('/g/data/w42/ys9723/rainfall/CMIP_aus/Result/pr_day_ACCESS-ESM1-5_historical_four_cells.nc')

    stat_all = np.full((4, 8, 3), np.nan) # (No.of cells, No.of statistics, No.of variables)

    for g in range(4):
        print(g)
        obs_g = obs.precip.sel(lat=lats[g],lon=lons[g])
        raw_g = raw.pr.sel(lat=lats[g],lon=lons[g])
        
        s1 = time.time()
        # Perform TVC post-processing
        Z_hist = perform_TVC(obs_g, raw_g, raw_g, scale)
       
        # Introduce mean adjustment
        root = find_optimal_minus(obs_g.isel(time=slice(init,len(obs_g.time))).values, Z_hist.values)
        Z_hist_ma = root + Z_hist.values
        Z_hist_ma = np.where(Z_hist_ma > 0, Z_hist_ma, 0)
        s2 = time.time()
        print('time: '+str(s2-s1))
        
        obs_stat = metric_cal(obs_g.isel(time=slice(init,len(obs_g.time))).values, 1952, 2014)
        raw_stat = metric_cal(raw_g.isel(time=slice(init,len(raw_g.time))).values, 1952, 2014)
        TVC_ma_stat = metric_cal(Z_hist_ma, 1952, 2014)

        # Combine all results
        for i in range(len(obs_stat)):
            stat_all[g,i,0] = obs_stat[i]
            stat_all[g,i,1] = raw_stat[i]
            stat_all[g,i,2] = TVC_ma_stat[i]
        
            print(stat_all[g,i,:])