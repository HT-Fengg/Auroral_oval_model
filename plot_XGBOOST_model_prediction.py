#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/6 18:36
# @Author  : FHT
# @File    : plot_model_predictions_for_KP_789.py

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import joblib,calendar
import pandas as pd

def polar_map(hf):
    ax = hf
    # ax = plt.gca(projection='polar')
    ax.set_thetagrids(np.arange(0.0, 360.0, 30))
    ax.set_thetamin(0.0)  # 设置极坐标图开始角度为0°
    ax.set_thetamax(360.0)  # 设置极坐标结束角度为360°
    ax.set_xticklabels(['06', '08', '10', '12', '14', '16', '18', '20', '22', '00', '02', '04'], fontsize=7,
                       family='times new roman', weight='bold')
    ax.set_rgrids(np.arange(0.0, 50.0, 10))
    ax.set_rlabel_position(270.0)  # 标签显示在0°
    ax.set_rlim(0.0, 40.0)  # 标签范围为[0, 5000)
    ax.set_yticklabels([' ', '80°', '70°', '60°', '50°'], fontsize=7, family='times new roman', weight='bold',
                       color='r')
    ax.grid(True, linestyle="-", color="k", linewidth=1, alpha=0.8)
    return hf
Kp=[3,4]
n=[1,3]
MLAT = [];
MLT = [];
for mlat in np.arange(50, 90, 0.05):
    for mlt in np.arange(0, 24, 0.05):
        MLAT.append(mlat);
        MLT.append(mlt);

fig = plt.figure(figsize=(8, 10))
for kp,N in zip(Kp,n):
    kp_n = np.ones((1, np.size(np.array(MLAT)))) * (kp) * 10
    data = {'MLAT': np.array(MLAT) / 90, 'cos_MLT': np.cos(np.array(MLT)*(np.pi/12)), 'sin_MLT': np.sin(np.array(MLT)*np.pi/12),'KP': kp_n[0] / 100}
    x_test = pd.DataFrame.from_dict(data)
    #==================get XGBOOST data  =====
    path=r".XGBoost_lbhs_prediction.model"
    model = joblib.load(path)
    y_pred = model.predict(x_test)
    MLAT_n =np.array(MLAT)
    MLT_n =np.array(MLT)
    pre_lbhs = y_pred[:, 3] * 6
    FPP=np.where(pre_lbhs>1);
    pre_lbhs[pre_lbhs < 0] = 0;
    rho = (90 - abs(np.array(MLAT_n)));
    theta = ((np.array(MLT_n) * 15 + 270) * np.pi / 180);

    hf4=fig.add_subplot(3 ,2 ,N,projection='polar')
    hf4=polar_map(hf4)
    # plot ssusi in NH
    ax=plt.scatter(theta,rho, s=2, c=pre_lbhs, cmap='jet', alpha=1,vmax=4,vmin=-1)#bwr
    date='XGBoost model'+'\n Kp='+str(kp)
    plt.title(date,fontsize=12,family='Arial',weight='bold',color='k',loc='left')

    cb1 = plt.colorbar(ax, fraction=0.035, pad=0.12)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([ -1,0, 1, 2, 3,4])
    cb1.update_ticks()
    cb1.ax.set_title('LBHS' + '\n' + 'log10 [R]', loc='center', pad=10, fontsize=8)


    hf4 = fig.add_subplot(3, 2, N+1, projection='polar')
    hf4 = polar_map(hf4)
    # plot ssusi in NH
    ax = plt.scatter(theta[FPP], rho[FPP], s=2, c=pre_lbhs[FPP], cmap='jet', alpha=1, vmax=4, vmin=-1)  # bwr
    date = 'Auroral Intensity > 10 R'
    plt.title(date, fontsize=12, family='Arial', weight='bold', color='k', loc='left')

    cb1 = plt.colorbar(ax, fraction=0.035, pad=0.12)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([-1, 0, 1, 2, 3, 4])
    cb1.update_ticks()
    cb1.ax.set_title('LBHS' + '\n' + 'log10 [R]', loc='center', pad=10, fontsize=8)

plt.subplots_adjust(hspace=0.8, wspace=1)
plt.tight_layout(pad=2, h_pad=0.5)
plt.show()
model_path='.\\Figures\\'
SAVEPATH=model_path+'model_results_kp_89_xgboost';
plt.savefig(SAVEPATH+'.pdf',bbox_inches='tight') #pdf
plt.close(fig)






