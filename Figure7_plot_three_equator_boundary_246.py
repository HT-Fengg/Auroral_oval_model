#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/5 20:18
# @Author  : FHT
# @File    : plot_three_method_boundaries_246.py

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.metrics import mean_squared_error
# sn.set_style("whitegrid")
from scipy.interpolate import make_interp_spline
#
fig = plt.figure(figsize=(12, 12))
ssusi_boundary_data = scipy.io.loadmat(
    r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\LBHS_model\Figures\ssusi_mean_data_2005_2016_boundary_results_kp246_round8_md6_thred_split_by_year_add_nightsidedata_02_45_sin_cos.mat")
XGBoost_boundary_data = scipy.io.loadmat(
    r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\LBHS_model\Figures\XGBoost_model_boundary_2005_2016_results_kp246_round8_md6_thred_split_by_year_add_nightsidedata_02_45_sin_cos.mat")
zp_boundary_data = scipy.io.loadmat(
    r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\mlat_52\zhang_paxton_model_2005_2016_boundary_results_kp246_round8_md6_thred_split_by_year_add_nightsidedata_02_45.mat")


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
    # sn.pointplot(x="MLT",y="MLAT",data=df,ci=68)
    sn.regplot(x="MLT", y="MLAT", data=xgb_df, order=4, ci=0, color='b',  x_estimator=np.mean,  scatter_kws={'s': 30},label="XGBoost model")
    sn.regplot(x="MLT", y="MLAT", data=zp_df, order=4, ci=0,  color='g',  x_estimator=np.mean,  scatter_kws={'s': 30},label="ZP08 model")


    # plt.plot(ssusi_boundary_data['ssusi_eb_mlt'][0][i-1][0],ssusi_boundary_data['ssusi_eb_mlat'][0][i-1][0],'r--',linewidth=2,marker='o',markersize=6,color="r",label='DMSP/SSUSI')
    # plt.plot(XGBoost_boundary_data['XGB_eb_mlt'][0][i - 1][0],
    #          np.abs(XGBoost_boundary_data['XGB_eb_mlat'][0][i - 1][0]), "b--",marker='o',markersize=6, linewidth=2, label='XGBoost model')
    # plt.plot(zp_boundary_data['zp_eb_mlt'][0][i-1][0],np.abs(zp_boundary_data['zp_eb_mlat'][0][i-1][0]),"g--",marker='o',markersize=6,linewidth=2,label='Zhang-Paxton model')

    # plt.scatter(ssusi_boundary_data['ssusi_eb_mlt'][0][i-1][0],ssusi_boundary_data['ssusi_eb_mlat'][0][i-1][0],marker='o',s=30,color="r")
    # x = ssusi_boundary_data['ssusi_eb_mlt'][0][i-1][0];
    # y = ssusi_boundary_data['ssusi_eb_mlat'][0][i-1][0]
    # parameter = np.polyfit(x, y, 4)
    # y2 = parameter[0] * x ** 4 + parameter[1] * x ** 3 + parameter[2]*x**2+ parameter[3]*x+parameter[4]
    # plt.plot(x, y2, color='r', lw=3, label='SSUSI')
    #
    # plt.scatter(XGBoost_boundary_data['XGB_eb_mlt'][0][i - 1][0],
    #          np.abs(XGBoost_boundary_data['XGB_eb_mlat'][0][i - 1][0]) ,marker='o',s=30,color='b')
    # x = XGBoost_boundary_data['XGB_eb_mlt'][0][i - 1][0];
    # y = np.abs(XGBoost_boundary_data['XGB_eb_mlat'][0][i - 1][0])
    # parameter = np.polyfit(x, y, 4)
    # y2 = parameter[0] * x ** 4 + parameter[1] * x ** 3 + parameter[2] * x ** 2 + parameter[3] * x + parameter[4]
    # plt.plot(x, y2, color='b', lw=3, label='XGBoost model')
    #
    # plt.scatter(zp_boundary_data['zp_eb_mlt'][0][i-1][0],np.abs(zp_boundary_data['zp_eb_mlat'][0][i-1][0]),marker='o',s=30,color='g')
    # x = zp_boundary_data['zp_eb_mlt'][0][i-1][0];
    # y = np.abs(zp_boundary_data['zp_eb_mlat'][0][i-1][0])
    # parameter = np.polyfit(x, y, 4)
    # y2 = parameter[0] * x ** 4 + parameter[1] * x ** 3 + parameter[2] * x ** 2 + parameter[3] * x + parameter[4]
    # plt.plot(x, y2, color='g', lw=3, label='Zhang-Paxton model')

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
    # plt.text(4, 56, 'PCCs1=' + str(np.round(np.corrcoef(ssusi_mlat,xgb_df.MLAT)[0,1],2)), fontsize=12)
    # plt.text(4, 54, 'PCCs2=' + str(np.round(np.corrcoef(ssusi_mlat, zp_mlat)[0, 1], 2)), fontsize=12
    #          )
    plt.text(4, 56, 'MSE1=' + str(np.round((mean_squared_error(ssusi_mlat,xgb_df.MLAT)),2)), fontsize=12)
    plt.text(4, 54, 'MSE2=' + str(np.round((mean_squared_error(ssusi_mlat, zp_mlat)), 2)), fontsize=12
             )


    # if i ==1:
    #     plt.text(4, 48, 'PCCs1:The PCCs between SSUSI observation and prediction from XGBoost model' , fontsize=10)
    #     plt.text(4, 46, 'PCCs2:The PCCs between SSUSI observation and result from ZP08 model', fontsize=10)
    # plt.legend()

# ## poleward========
# ssusi_boundary_data = scipy.io.loadmat(
#     r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\mlat_52\ssusi_mean_data_2005_2016_boundary_results_kp246_round8_md6_thred_split_by_year_add_nightsidedata_02.mat")
# XGBoost_boundary_data = scipy.io.loadmat(
#     r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\mlat_52\XGBoost_model_boundary_2005_2016_results_kp246_round8_md6_thred_split_by_year_add_nightsidedata_02.mat")
# zp_boundary_data = scipy.io.loadmat(
#     r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\mlat_52\zhang_paxton_model_2005_2016_boundary_results_kp246_round8_md6_thred_split_by_year_add_nightsidedata_02.mat")
#
# #================++++++++++++++++++++++++++++++++++=============
# i_num=[4,5,6]
# Kp_str=['1-2','3-4','6-7']
# for i in i_num:
#     fig.add_subplot(2,3,i)
#     print(i)
#     plt.plot(ssusi_boundary_data['ssusi_eb_mlt'][0][i-4][0],np.abs(ssusi_boundary_data['ssusi_eb_mlat'][0][i-4][0]-XGBoost_boundary_data['XGB_eb_mlat'][0][i - 4][0]),"b-",linewidth=3,label='SSUSI')
#     plt.plot(ssusi_boundary_data['ssusi_eb_mlt'][0][i - 4][0], np.abs(
#         ssusi_boundary_data['ssusi_eb_mlat'][0][i - 4][0] - zp_boundary_data['zp_eb_mlat'][0][i-4][0]), "g-",
#              linewidth=3, label='SSUSI')
#
#     # plt.plot(XGBoost_boundary_data['XGB_pb_mlt'][0][i - 4][0],
#     #          np.abs(XGBoost_boundary_data['XGB_pb_mlat'][0][i - 4][0]), "b-", linewidth=3, label='XGBoost')
#     # plt.plot(zp_boundary_data['zp_pb_mlt'][0][i-4][0],np.abs(zp_boundary_data['zp_pb_mlat'][0][i-4][0]),"g-",linewidth=3,label='Z-P model')
#
#
#     if i==1:
#         plt.legend(bbox_to_anchor=(-0.15, 1.05), loc=3, borderaxespad=0)
#     plt.xlabel('MLT', fontsize=12)
#     plt.ylabel('MLAT', fontsize=12)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.ylim(60,80)
#     plt.xlim(0,24)
#     plt.grid()
#     plt.minorticks_on()
#     plt.text(17,78,'Kp='+Kp_str[i-4], fontsize=15,fontweight='bold')
#     # plt.legend()


plt.show( )
plt.tight_layout()
plt.savefig(
    'E:\\Feng_paperwork\\20_aurora_forcast\\meaching_learing_results_split_by_year\\mlat_larger_than_45\\LBHS_model\\Figures\\' + 'three_method_results_2005_2016_equatorward_round8_md6_1000_add_nightsidedata_kp_246_only_equator_45_sin_cos'+'.png',bbox_inches='tight', format='png',dpi=600)
plt.savefig(
    'E:\\Feng_paperwork\\20_aurora_forcast\\meaching_learing_results_split_by_year\\mlat_larger_than_45\\LBHS_model\\Figures\\' + 'three_method_results_2005_2016_equatorward_round8_md6_1000_add_nightsidedata_kp_246_only_equator_45_sin_cos'+'.pdf',bbox_inches='tight')




