# Mean-Adjusted-Time-Variability-Correction

This repository provides the Python code and associated data used in the manuscript:
Yawen Shao, Craig H. Bishop (2025). Improving CMIP6 Projections of Daily Precipitation Using a Mean-Adjusted Time Variability Correction Technique. Accepted in Climate Dynamics.

## Overview
The Mean-adjusted Time Variability Correction (TVC-ma) model is an extension of the original TVC method for post-processing daily precipitation projections. It introduces a mean adjustment procedure to eliminate negative precipitation values while ensuring alignment of the corrected mean and variability with observations in the historical training period.
This release includes Python scripts and data files for reproducing the key results and figures in the manuscript.

## Requirements
The following Python libraries are required:
numpy, pandas, xarray, scipy, calendar, operator, datetime, math, matplotlib, shapely, geopandas, time, cartopy

## Prerequisites
1.	Original TVC code
Please download TVC_class.py from Zenodo (https://zenodo.org/records/10212122) before running TVC-ma scripts.
2.	Shapefiles
•	Australia map: download from the Australian Bureau of Statistics https://www.abs.gov.au/book/export/25822/print.\n
•	Global map: download from https://hub.arcgis.com/datasets/CESJ::world-continents/explore

## Script Overview
‘Figure2_Locations of four selected grid cells.py’: plots locations of four grid cells on the Australian map.
‘Table2_Data processing and statistics calculation of four grid cells.py’: implement TVC-ma for selected grid cells and computes relevant statistics.
‘Figure3-6_S2-S5_percentage improvement of all statistics.py’: calculates and plots boxplots of percentage improvements in mean absolute error for mean, variance, lag correlations, R10mm, consecutive dry days, Rx1day and Rx5day (TVC-ma: Figures 3-6; TVC: Figures S2-S5).
‘Figure7_PDF of R10mm and CDD for selected models.py’: plots probability density functions of R10mm and CDD for selected models and computes total variance distance.
‘Figure8_Applying TVC-ma to historical projection.py’: plots Australian maps of historical statistics using TVC-ma trained on AGCD data. 
‘Figure9_Applying TVC-ma to future projection.py’: plots Australian map of future projection statistics using TVC-ma trained on AGCD data. 
‘FigureS1_Diagonal elements of the climate change ratio matrix.py’: computes and visualizes diagonal elements of the climate change ratio matrix.
‘FigureS6-7_PDF of R10mm and CDD for all models.py’: plots probability density functions of R10mm and CDD and calculates total variance distance values for all models.
‘TableS3_Data processing and statistics calculation of five global grid cells’: applies TVC-ma to five global grid cells and calculates percentage improvement.
‘FigureS8_Ratio of the mean of the 365-day time scale’: computes and plots the mean of 365-day averaged time series.
‘FigureS9_Locations of five selected global grid cells’: visualises the locations of five global grid cells.
