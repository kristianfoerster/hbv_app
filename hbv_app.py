#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 16:44:20 2025

@author: kristianfoerster
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from hbv import simulation as HBV
from hbv import bounds
import pandas as pd
from model_performance import model_performance

# read sample data
data = pd.read_csv('data/data.csv', index_col=0, parse_dates=[0])
obs=data['obs'].values
forcing=data[['Temp', 'Prec', 'Evap']]
infofile = 'data/source.txt'

info = ''
with open(infofile) as f:
    info+=f.read()
f.close()

days = np.arange(len(obs))

st.title("HBV Calibration App")

# Sidebar parameters
st.sidebar.header("Calibration Parameters")

param_names = [
    "BETA", "CET", "FC", "K0", "K1", "K2", "LP", "MAXBAS",
    "PERC", "UZL", "PCORR", "TT", "CFMAX", "SFCF", "CFR", "CWH"
]

params = []
bnds = bounds()
for name, (low, high) in zip(param_names, bnds):
    default = (low + high) / 2
    # detect whether the slider should be int or float
    if float(low).is_integer() and float(high).is_integer() and float(default).is_integer():
        val = st.sidebar.slider(name, int(low), int(high), int(default), step=1)
    else:
        val = st.sidebar.slider(name, float(low), float(high), float(default), step=(high - low) / 100)
    params.append(val)


# Run HBV model
sim = HBV(forcing, params)

#indexes
i1=0
i2=-1

if len(days)>=730:
    # ignore first year for warm-up
    i1=365

fig, axes = plt.subplots(
    3, 1, figsize=(14, 8), sharex=True,
    gridspec_kw={'height_ratios': [1, 1,2]}
)

# ---------------------------
# 1. Temperature subplot
# ---------------------------
ax_temp = axes[0]
ax_temp.plot(forcing.index[i1:i2], forcing['Temp'][i1:i2], label='Temperature (°C)', color='red')
ax_temp.set_ylabel('Temp (°C)')
ax_temp.grid(True, alpha=0.3)
ax_temp.legend(loc='upper left')

# ---------------------------
# 2. Precip + Evap subplot
# ---------------------------
ax_prec = axes[1]
ax_evap = ax_prec.twinx()

# precipitation (bars)
ax_prec.bar(forcing.index[i1:i2], forcing['Prec'][i1:i2], width=1.0, alpha=0.5,
            label='Precipitation')

# reverse precipitation axis
ax_prec.invert_yaxis()
ax_prec.set_ylabel('Precip (mm)')
ax_prec.grid(True, alpha=0.3)

# evaporation (line)
ax_evap.plot(forcing.index[i1:i2], forcing['Evap'][i1:i2], color='orange',
             label='Evapotranspiration')
ax_evap.set_ylabel('Evap (mm)')

# legends
lines1, labels1 = ax_prec.get_legend_handles_labels()
lines2, labels2 = ax_evap.get_legend_handles_labels()
ax_evap.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# ---------------------------
# 3. Runoff subplot
# ---------------------------
ax_runoff = axes[2]
ax_runoff.plot(forcing.index[i1:i2], sim[i1:i2], label='Simulated Runoff', color='red')
ax_runoff.plot(forcing.index[i1:i2], obs[i1:i2], label='Observed Runoff', color='blue')
ax_runoff.set_ylabel('Runoff (mm)')
ax_runoff.set_xlabel('Date')
ax_runoff.grid(True, alpha=0.3)
ax_runoff.legend()

plt.tight_layout()
st.pyplot(fig)

NSE, KGE, PBIAS, RMSE, RSR, r = model_performance(obs[i1:i2], sim[i1:i2])

st.code(
        'Model Performance\n'
        '=================\n'
        'n     = %8i\n' % len(obs[i1:i2])+
        'NSE   = %8.2f\n' % NSE+
        'KGE   = %8.2f\n' % KGE+
        'PBIAS = %8.2f\n' % PBIAS+
        'RMSE  = %8.2f\n' % RMSE+
        'RSR   = %8.2f\n' % RSR+
        'R     = %8.2f' % r
        )

st.code(info)
