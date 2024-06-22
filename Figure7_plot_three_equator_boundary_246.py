#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.interpolate import make_interp_spline
fig = plt.figure(figsize=(12, 12))
ssusi_boundary_data = scipy.io.loadmat(
    r".\ssusi_mean_data_2005_2016_boundary_results_kp246_round8_md6_thred_split_by_year_add_nightsidedata_02_45_sin_cos.mat")
XGBoost_boundary_data = scipy.io.loadmat(
    r".\XGBoost_model_boundary_2005_2016_results_kp246_round8_md6_thred_split_by_year_add_nightsidedata_02_45_sin_cos.mat")
zp_boundary_data = scipy.io.loadmat(
    r".\zhang_paxton_model_2005_2016_boundary_results_kp246_round8_md6_thred_split_by_year_add_nightsidedata_02_45.mat")

#================1-2++++++++++++++++=============
i_num=[1,2,3]
Kp_str=['1-2','3-4','5-6']
a_str=['(a)','(b)','(c)']
for i in i_num:
    print(i)
    mlt=ssusi_boundary_data['ssusi_eb_mlt'][0][i-1][0];
    fp = np.where(( mlt>= 15) |(mlt <10)&(mlt!=9.75))
    datadict = {'MLT': ssusi_boundary_data['ssusi_eb_mlt'][0][i-1][0][fp],'MLAT':ssusi_boundary_data['ssusi_eb_mlat'][0][i-1][0][fp]}
    ssusi_df = pd.DataFrame.from_dict(datadict)
    ssusi_mlat=ssusi_boundary_data['ssusi_eb_mlat'][0][i-1][0][fp];
    ssusi_mlt=ssusi_boundary_data['ssusi_eb_mlt'][0][i-1][0][fp]
    ssusi_boundary_data['ssusi_eb_mlt'][0][i - 1][0][fp]

    mlt = XGBoost_boundary_data['XGB_eb_mlt'][0][i - 1][0];
    fp = np.where((mlt >= 15) | (mlt < 10) & (mlt != 9.75))
    datadict = {'MLT': XGBoost_boundary_data['XGB_eb_mlt'][0][i - 1][0][fp],
                'MLAT': np.abs(XGBoost_boundary_data['XGB_eb_mlat'][0][i - 1][0][fp])}
    # xgb_mlat = np.abs(XGBoost_boundary_data['XGB_eb_mlat'][0][i - 1][0][fp]);
    xgb_df = pd.DataFrame.from_dict(datadict)

    mlt = zp_boundary_data['zp_eb_mlt'][0][i - 1][0];
    fp = np.where((mlt >= 15) | (mlt < 10) & (mlt != 9.75))
    datadict = {'MLT': zp_boundary_data['zp_eb_mlt'][0][i - 1][0][fp],
                'MLAT': np.abs(zp_boundary_data['zp_eb_mlat'][0][i - 1][0][fp])}
    # zp_mlat = np.abs(zp_boundary_data['zp_eb_mlat'][0][i - 1][0][fp]);
    zp_df = pd.DataFrame.from_dict(datadict)
    ssusi_mlat=[];zp_mlat=[];
    for j in xgb_df.MLT:
        fp_s=np.where(ssusi_df.MLT==j)
        fp_zp = np.where(zp_df.MLT == j)
        ssusi_mlat.append(ssusi_df.MLAT[fp_s[0][0]])
        zp_mlat.append(zp_df.MLAT[fp_zp[0][0]])

    fig.add_subplot(3,3,i)
    sn.regplot(x="MLT",y="MLAT",data=ssusi_df, order=4, ci=0, color='r',  x_estimator=np.mean, scatter_kws={'s': 30},label="SSUSI")
    sn.regplot(x="MLT", y="MLAT", data=xgb_df, order=4, ci=0, color='b',  x_estimator=np.mean,  scatter_kws={'s': 30},label="XGBoost model")
    sn.regplot(x="MLT", y="MLAT", data=zp_df, order=4, ci=0,  color='g',  x_estimator=np.mean,  scatter_kws={'s': 30},label="ZP08 model")

    if i==1:
        plt.legend(bbox_to_anchor=(-0, 1.05), loc=3, borderaxespad=0)
    plt.xlabel('MLT',fontsize=12,fontweight='bold')
    plt.ylabel('MLAT',fontsize=12,fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(53,77)
    plt.xlim(0,24)
    plt.grid()
    plt.minorticks_on()
    plt.text(16,75,'Kp='+Kp_str[i-1], fontsize=14,fontweight='bold')
    plt.text(2, 75, a_str[i-1], fontsize=14, fontweight='bold')
    plt.text(4, 56, 'MSE1=' + str(np.round((mean_squared_error(ssusi_mlat,xgb_df.MLAT)),2)), fontsize=12)
    plt.text(4, 54, 'MSE2=' + str(np.round((mean_squared_error(ssusi_mlat, zp_mlat)), 2)), fontsize=12
             )

plt.show( )
plt.tight_layout()
plt.savefig(
    '.\\' + 'three_method_results_2005_2016_equatorward_round8_md6_1000_add_nightsidedata_kp_246_only_equator_45_sin_cos'+'.png',bbox_inches='tight', format='png',dpi=600)
plt.savefig(
    '.\\' + 'three_method_results_2005_2016_equatorward_round8_md6_1000_add_nightsidedata_kp_246_only_equator_45_sin_cos'+'.pdf',bbox_inches='tight')




