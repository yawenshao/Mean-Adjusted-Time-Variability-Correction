"""
Script to plot Figure 3-6, S2-S5. Percentage improvement in MAE of mean, variance, lag correlations, R10mm, consecutive dry days, Rx1day and Rx5day.

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

def prct_improve_MAE(raw, TVC, obs):
    '''
    Calculate percentage improvement in mean absolute error
    raw - raw model series
    TVC - TVC result series
    obs - observation series
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
    # Import data
    models = pd.read_csv('./Data/CMIP6_model_name.csv', header=None)
    models = models.to_numpy().flatten()

    # Calculate percentage improvement in MAE
    prct_all_scenario = []
    for expr in ['historical','ssp126','ssp585']:
        prct_all = []
        for truth in models:
            obs_data = xr.open_dataset('./Data/'+truth+'_'+expr+'_Aus_obs2.nc')
            raw_models = models[models != truth]
            raw_data = []

            for model in raw_models:
                raw_data.append(xr.open_dataset('./Data/'+model+'_'+expr+'_Aus_obs2.nc'))
            raw_data = xr.concat(raw_data, dim='model').assign_coords(model=raw_models)
            
            TVC_ma_data = xr.open_dataset('./Data/MaT_stats_'+truth+'_allmodels_'+expr+'_Aus_TVC_ma2.nc')
            
            prct = prct_improve_allmodels(raw_data, TVC_ma_data, obs_data)
            prct_all.append(prct.stats)
            
        prct_all = xr.concat(prct_all, dim='model').assign_coords(model=models)
        prct_all_scenario.append(prct_all)
        
        # Print averaged percentage improvement across all grid cells and all truths
        print(expr)
        print(prct_all.mean(dim=['model','lat','lon']).values)
        
    # Plot box plot of percentage improvement
    variable = ['mu', 'var', 'lag1', 'lag5', 'R10', 'CDD', 'rx1day', 'rx5day']
    var_name = ['Mean', 'Variance', 'Lag-1 Correlation', 'Lag-5 Correlation','R10mm', 'CDD', 'Rx1day', 'Rx5day']
    ylabel = ['mm','mm2',' ',' ','No. of days','No. of days']
    color1 = [['wheat','yellow','gold'],['salmon','lightcoral','indianred'],['lightgreen','limegreen','forestgreen'],
              ['palegreen','mediumseagreen','seagreen'],['plum','violet','mediumorchid'],['silver','gray','lightslategray'],['lightblue','deepskyblue','dodgerblue'],['cornflowerblue','royalblue','blue']]
    mcolor = ['orange','red','darkgreen','darkgreen','darkviolet','black','blue','darkblue']
    letters = ['a)','b)']

    for f in range(4):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(13,9.4))
        
        for r in range(2):
            i = f*2+r

            if r == 1:
                ax_tick = []
            bb = []
            
            for k in range(len(prct_all.model)):
                for s in range(3):
                    data = prct_all_scenario[s].isel(stat=i, model=k).values.flatten()             
                    data = data[~np.isnan(data)]
                    
                    b = ax[r].boxplot(data,
                           vert=True,
                           patch_artist=True,
                           positions=[0.12+k+0.25*s],
                           widths=(0.13), 
                           showmeans = True,
                           showfliers = False,
                           whiskerprops=dict(linewidth=2, color=color1[i][s]),
                           capprops=dict(linewidth=0, color=color1[i][s]),
                           boxprops=dict(linewidth=2, edgecolor=color1[i][s], facecolor=color1[i][s]), #, edgecolor=color
                           medianprops=dict(linewidth=2, color=mcolor[i]),
                           meanprops=dict(marker="D", markersize=0)
                          )
                    
                    if k == 0:
                        bb.append(b)
            
                if r == 1:
                    ax_tick.append(" ") 
                    ax_tick.append(models[k])
                    ax_tick.append(" ")
            
            ax[r].axhline(y=0, color='grey', linestyle='dashed', linewidth=1)
            ax[r].legend([bb[0]["boxes"][0], bb[1]["boxes"][0], bb[2]["boxes"][0]], ['Historical', 'ssp126', 'ssp585'], loc='lower right', fontsize=11)
            ax[r].text(x=-0.3, y=105, s=letters[r], fontsize=15, fontweight='bold')
            
            if r == 1:
                ax[r].set_xticklabels(ax_tick, rotation=80, fontsize=13)
            else:
                ax[r].set_xticklabels([])
            
            ax[r].set_ylabel('Percentage \n improvement (%)', fontsize=14)
            ax[r].set_title(var_name[i], fontsize=15, weight='bold',pad=6)
            
            if var_name[i] == 'Mean':
                ax[r].set_ylim([-80,100])
            elif var_name[i] == 'Variance':
                ax[r].set_ylim([-100,100])
            elif var_name[i] == 'Lag-1 Correlation' or var_name[i] == 'Lag-5 Correlation':
                ax[r].set_ylim([-70,100])
            elif var_name[i] == 'R10mm':
                ax[r].set_ylim([-100,100])
            elif var_name[i] == 'CDD':
                ax[r].set_ylim([-80,100])
                ax[r].legend([bb[0]["boxes"][0], bb[1]["boxes"][0], bb[2]["boxes"][0]], ['Historical', 'ssp126', 'ssp585'], loc='lower left', fontsize=11)
            elif var_name[i] == 'Rx1day':
                ax[r].set_ylim([-90,100])
                ax[r].legend([bb[0]["boxes"][0], bb[1]["boxes"][0], bb[2]["boxes"][0]], ['Historical', 'ssp126', 'ssp585'], loc='lower left', fontsize=11)
            elif var_name[i] == 'Rx5day':
                ax[r].set_ylim([-150,100])   
                ax[r].legend([bb[0]["boxes"][0], bb[1]["boxes"][0], bb[2]["boxes"][0]], ['Historical', 'ssp126', 'ssp585'], loc='lower left', fontsize=11)
        
        fig.subplots_adjust(left=0.08,top=0.97,bottom=0.19,right=0.98, hspace=0.14)
        fig.savefig('./Figures/Boxplot_metric'+str(2*f)+'_'+str(2*f+2)+'_MaT_allmodels_all_expr_Aus_r2_2plots_TVC_ma.jpeg',dpi=300)