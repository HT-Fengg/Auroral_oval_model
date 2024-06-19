#!/usr/bin/env python
# encoding: utf-8
# @Author  : FHT
#@Time    : 2023/8/17 08:31

import os;
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
import matplotlib.pyplot as plt
from numpy import exp, linspace, cos, sin, ndarray, sum, argmin, abs, pi
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
import dmsp_mat_data
import pandas as pd
import scipy.io as scio
from datetime import datetime
from datetime import timedelta
import matplotlib.ticker as ticker


# a rewrite of Zhang and Paxton (2008) auroral model in python
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
def find_close_data(mlt,mlat,need_mlt):
    line_mlat=np.linspace(50,88,24)
        # [50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,88,86,84,82,80,78,76,74,72,70,68,66,64,62,60,58,56,54,52,50];
    line_mlt=np.ones(( np.size(line_mlat),1))*need_mlt;

    index=[];
    for line_mlt1,line_mlat1 in zip(line_mlt,line_mlat):
        data=np.sqrt((mlt-line_mlt1)**2+(mlat-line_mlat1)**2)
        index.append(np.where(data==np.min(data))[0][0])

    return index,line_mlat

# Year=['2014','2014','2014'];Doy=['007','054','050'];hour=['20','16','08'];minute=['54','14','13'];satellite=['f16','f18','f16'];orb=['52749','22436','53349'];Month=['01','02','02'];Day=['07','23','19'];
# Year=['2012','2012','2012'];Doy=['006','076','069'];hour=['20','19','13'];minute=['37','13','56'];satellite=['f16','f16','f16'];orb=['42408','43396','43294'];Month=['01','03','03'];Day=['06','16','09'];
Year=['2012','2012','2012'];Doy=['006','076','069'];hour=['20','19','12'];minute=['37','13','14'];satellite=['f16','f16','f16'];orb=['42408','43396','43293'];Month=['01','03','03'];Day=['06','16','09'];
hemis = 'NH'
# ==================== plot ssusi data ======================
# fig = plt.figure(figsize=(10, 6))
fig = plt.figure(figsize=(16, 13))
i=[0,4,8]
# i=[1,2,3]
h=[0.7,0.4,0.1]
N_mlt=[18,18,18]
N_mlt2=[6,6,6]
for satellite,Year,Month,Day,Doy,n,orb,hour,minute,h,need_mlt,need_mlt2 in zip(satellite,Year,Month,Day,Doy,i,orb,hour,minute,h,N_mlt,N_mlt2):
    amlt,amlat,amlong,aorb,aradiance,aut=dmsp_mat_data.read_ssusi_mat(satellite,Year,Month,Day,Doy)
    # ssj_mlt, ssj_mlat, ssj_mlong, ssj_utsod=dmsp_mat_data.read_ssj_mat(satellite,Year,Month,Day,Doy)
    # deal with ssusi data
    if hemis=='NH':
        fp=np.argwhere((amlat > 0) & (abs(amlat) >= 50)&(aorb==int(orb)))
        mlt=amlt[fp[:,0]];mlong = amlong[fp[:,0]];mlat = amlat[fp[:,0]];ut = aut[fp[:,0]];radiance = aradiance[fp[:,0],3];mlt = amlt[fp[:,0]]
        # get ssusi theta and rho
        NH_ssusi_theta = (mlt * 15 + 270) * np.pi / 180;NH_ssusi_rho = 90 - abs(mlat);NH_radiance = radiance

    if hemis=='SH':
        fp=np.argwhere((amlat < 0) & (abs(amlat) >= 50)&(aorb==int(orb)))
        mlt = amlt[fp[:,0]]; mlong = amlong[fp[:,0]];mlat = amlat[fp[:,0]];ut = aut[fp[:,0]];radiance = aradiance[fp[:,0],3];mlt = amlt[fp[:,0]]
        # get ssusi theta and rho
        SH_ssusi_theta = (mlt * 15 + 270) * np.pi / 180;SH_ssusi_rho = 90 - abs(mlat);SH_radiance = radiance
    ##=== get  ssj  orbit ===
    # fpp=np.where((ssj_utsod>((np.min(ut)-0.2)*3600))&(ssj_utsod<((np.min(ut)+0.5)*3600))&(np.abs(ssj_mlat)>50))
    # ssj_mlat_need=ssj_mlat[fpp[0]];ssj_mlt_need=ssj_mlt[fpp[0]]
    # ssj_theta = (ssj_mlt_need * 15 + 270) * np.pi / 180;
    # ssj_rho = 90 - abs(ssj_mlat_need);
    # ssusi_idx=[];ssusi_mlt=[];

    ## ==== get Kp ====
    def datetime2matlabdn(dt):
       mdn = dt + timedelta(days = 366)
       frac_seconds = (dt-datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
       frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
       return mdn.toordinal() + frac_seconds + frac_microseconds

    data=scio.loadmat(r'E:\Feng_paperwork\20_aurora_forcast\kp_dst_hp\kp_all_2010_2016.mat')
    kp_time=data['kp_time'];
    kp_index=data['kp_index'];
    t=datetime(int(Year),int(Month),int(Day),int(hour),int(minute),0)
    find_time=datetime2matlabdn(t)
    # find_time=t.timestamp()/86400 +719529.333333333;
    idx = np.argmin(np.abs(kp_time - find_time))
    kp=kp_index[idx][0]
    kp = np.ones((1, np.size(mlat))) * (kp)/100

    # plot panel 1 NH SSUSI ang cluster

    # # plot ssusi in NH
    NH_radiance[NH_radiance < 0] = 1;
    NH_radiance=np.log10(NH_radiance)
    index,line_mlat=find_close_data(mlt, mlat,need_mlt)
    index2, line_mlat2 = find_close_data(mlt, mlat, need_mlt2)
    hf1=fig.add_subplot(3, 4, n+1,projection='polar')
    hf1=polar_map(hf1)
    ax=plt.scatter(NH_ssusi_theta,NH_ssusi_rho, s=2, c=NH_radiance, cmap='jet',vmax=3,vmin=0)
    plt.scatter(NH_ssusi_theta[np.array(index)],NH_ssusi_rho[np.array(index)],s=4,c='yellow')
    plt.scatter(NH_ssusi_theta[np.array(index2)], NH_ssusi_rho[np.array(index2)], s=4, c='yellow')
    date=Year+'-'+Month+'-'+Day+' '+'DMSP/SSUSI-'+satellite+'\n'+'Kp='+str(round(kp[0][0]*10,1))
    plt.title(date,fontsize=12,family='Arial',weight='bold',color='k',loc='left')

    # position = fig.add_axes([0.08, h, 0.01, 0.1])
    # cb1=plt.colorbar(ax,cax=position,fraction=0.03, pad=0.05)
    cb1 = plt.colorbar(ax, fraction=0.035, pad=0.12)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([0, 1, 2, 3])
    cb1.update_ticks()
    cb1.ax.set_title('LBHS'+'\n'+'log [R]', loc='center',pad=10, fontsize=8)
    # plt.show(block=True)

    ssusi_radiance = NH_radiance[index];
    MLAT=[];MLT=[];cos_MLT=[];sin_MLT=[];
    for cmlat,cmlt in zip(mlat,mlt):
        MLAT.append(cmlat/90)
        MLT.append(cmlt/24)
        cos_MLT.append(np.cos(cmlt*(np.pi/12)))
        sin_MLT.append(np.sin(cmlt * (np.pi / 12)))
    #==================get RandomForest data  =====
    path = r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\LBHS_model\RF_lbhs_prediction.model"

    # path = r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\LBHL_model\RF_LBHL_prediction.model"

    model = joblib.load(path)
    data = {'MLAT':MLAT,'cos_MLT':cos_MLT, 'sin_MLT': sin_MLT, 'KP': kp[0][0]}
    x_test = pd.DataFrame.from_dict(data)
    y_pred = model.predict(x_test)
    # MLAT = y_pred[:, 0] * 90
    # MLT = y_pred[:, 1] * 24
    MLAT_n =np.array(MLAT) * 90
    MLT_n =np.array(MLT) * 24
    pre_lbhs = y_pred[:, 3] * 6
    pre_lbhs[pre_lbhs < 0] = 0;
    rho = (90 - abs(np.array(MLAT_n)));
    theta = ((np.array(MLT_n) * 15 + 270) * np.pi / 180);

    hf2=fig.add_subplot(3, 4, n+3,projection='polar')
    hf2=polar_map(hf2)
    # plot ssusi in NH
    ax=plt.scatter(theta,rho, s=2, c=pre_lbhs, cmap='jet', alpha=1,vmax=3,vmin=0)#bwr
    date='RF'
    plt.title(date,fontsize=13,family='Arial',weight='bold',color='k',loc='left')

    cb1 = plt.colorbar(ax, fraction=0.035, pad=0.12)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([0, 1, 2, 3])
    cb1.update_ticks()
    cb1.ax.set_title('LBHS' + '\n' + 'log [R]', loc='center', pad=10, fontsize=8)

    RF_radiance = pre_lbhs[index];

    #==================get KNeighborsRegressor data  =====
    path = r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\LBHS_model\KNN_lbhs_prediction.model"

    # path = r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\LBHL_model\KNN_LBHL_prediction.model"

    model = joblib.load(path)
    x_test = pd.DataFrame.from_dict(data)
    y_pred = model.predict(x_test)
    # MLAT = y_pred[:, 0] * 90
    # MLT = y_pred[:, 1] * 24
    MLAT_n =np.array(MLAT) * 90
    MLT_n=np.array(MLT) * 24
    pre_lbhs = y_pred[:, 3] * 6
    pre_lbhs[pre_lbhs < 0] = 0;
    rho = (90 - abs(np.array(MLAT_n)));
    theta = ((np.array(MLT_n) * 15 + 270) * np.pi / 180);

    hf3=fig.add_subplot(3, 4, n+2,projection='polar')
    hf3=polar_map(hf3)
    # plot ssusi in NH
    ax=plt.scatter(theta,rho, s=2, c=pre_lbhs, cmap='jet', alpha=1,vmax=3,vmin=0)#bwr
    date='KNN '
    plt.title(date,fontsize=13,family='Arial',weight='bold',color='k',loc='left')

    cb1 = plt.colorbar(ax, fraction=0.035, pad=0.12)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([ 0, 1, 2, 3])
    cb1.update_ticks()
    cb1.ax.set_title('LBHS' + '\n' + 'log [R]', loc='center', pad=10, fontsize=8)
    KN_radiance = pre_lbhs[index];

    #==================get XGBOOST data  =====
    path = r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\LBHS_model\XGBoost_lbhs_prediction.model"

    # path = r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\LBHL_model\XGBoost_LBHL_prediction.model"

    model = joblib.load(path)
    y_pred = model.predict(x_test)
    # MLAT = y_pred[:, 0] * 90
    # MLT = y_pred[:, 1] * 24
    MLAT_n =np.array(MLAT) * 90
    MLT_n =np.array(MLT) * 24
    pre_lbhs = y_pred[:, 3] * 6
    pre_lbhs[pre_lbhs < 0] = 0;
    rho = (90 - abs(np.array(MLAT_n)));
    theta = ((np.array(MLT_n) * 15 + 270) * np.pi / 180);

    hf4=fig.add_subplot(3, 4, n+4,projection='polar')
    hf4=polar_map(hf4)
    # plot ssusi in NH
    ax=plt.scatter(theta,rho, s=2, c=pre_lbhs, cmap='jet', alpha=1,vmax=3,vmin=0)#bwr
    date='XGBoost '
    plt.title(date,fontsize=13,family='Arial',weight='bold',color='k',loc='left')

    cb1 = plt.colorbar(ax, fraction=0.035, pad=0.12)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([0, 1, 2, 3])
    cb1.update_ticks()
    cb1.ax.set_title('LBHS' + '\n' + 'log [R]', loc='center', pad=10, fontsize=8)
    XGBoost_radiance = pre_lbhs[index];

    # hf5 = fig.add_subplot(2, 3, n)
    # # plot ssusi in NH==
    # plt.grid(b=True, linestyle="--", alpha=0.5,axis="both")
    # color = ['k', 'green', 'blue', 'r', 'm'];
    # plt.scatter(line_mlat,ssusi_radiance,marker='.',s=30, color=color[0])
    # x=line_mlat;y=ssusi_radiance
    # # parameter = np.polyfit(x, y, 3)
    # # y2 = parameter[0] * x ** 3 + parameter[1] * x ** 2 + parameter[2] * x + parameter[3]
    # parameter = np.polyfit(x, y, 4)
    # y2 = parameter[0] * x ** 4 + parameter[1] * x ** 3 + parameter[2] * x**2 + parameter[3]*x+parameter[4]
    # # parameter = np.polyfit(x, y, 2)
    # # y2 = parameter[0] * x ** 2 + parameter[1] * x ** 1 + parameter[2]
    # plt.plot(x, y2,color=color[0],lw=2,label='SSUSI')
    #
    # plt.scatter(line_mlat, RF_radiance, marker='.',s=30,color=color[1])
    # x = line_mlat;
    # y = RF_radiance
    # # parameter = np.polyfit(x, y, 3)
    # # y2 = parameter[0] * x ** 3 + parameter[1] * x ** 2 + parameter[2] * x + parameter[3]
    # parameter = np.polyfit(x, y, 4)
    # y2 = parameter[0] * x ** 4 + parameter[1] * x ** 3 + parameter[2] * x ** 2 + parameter[3] * x + parameter[4]
    # # parameter = np.polyfit(x, y, 2)
    # # y2 = parameter[0] * x ** 2 + parameter[1] * x ** 1 + parameter[2]
    # plt.plot(x, y2, color=color[1], lw=2, label='RF')
    #
    # plt.scatter(line_mlat, KN_radiance,  marker='.',s=30,color=color[2])
    # x = line_mlat;
    # y = KN_radiance
    # # parameter = np.polyfit(x, y, 3)
    # # y2 = parameter[0] * x ** 3 + parameter[1] * x ** 2 + parameter[2] * x + parameter[3]
    # parameter = np.polyfit(x, y, 4)
    # y2 = parameter[0] * x ** 4 + parameter[1] * x ** 3 + parameter[2] * x ** 2 + parameter[3] * x + parameter[4]
    # # parameter = np.polyfit(x, y, 2)
    # # y2 = parameter[0] * x ** 2 + parameter[1] * x ** 1 + parameter[2]
    # plt.plot(x, y2, color=color[2], lw=2, label='KNeighbors')
    #
    # plt.scatter(line_mlat, XGBoost_radiance, marker='.',s=30,color=color[3])
    # x = line_mlat;
    # y = XGBoost_radiance
    # # parameter = np.polyfit(x, y, 3)
    # # y2 = parameter[0] * x ** 3 + parameter[1] * x ** 2 + parameter[2] * x + parameter[3]
    # parameter = np.polyfit(x, y, 4)
    # y2 = parameter[0] * x ** 4 + parameter[1] * x ** 3 + parameter[2] * x ** 2 + parameter[3] * x + parameter[4]
    # # parameter = np.polyfit(x, y, 2)
    # # y2 = parameter[0] * x ** 2 + parameter[1] * x ** 1 + parameter[2]
    # plt.plot(x, y2, color=color[3], lw=2, label='XGBoost')
    #
    # # plt.scatter(line_mlat[0:20], RF_radiance[0:20], color=color[1], marker='o',s=20,label='RF')
    # # plt.scatter(line_mlat[0:20], KN_radiance[0:20], color=color[2], marker='*',s=20, label='KNeighbor')
    # # plt.scatter(line_mlat[0:20], XGBoost_radiance[0:20], color=color[3],  marker='d',s=20,label='XGBoost')
    # plt.legend(prop = {'size':6},loc=1)
    # plt.ylim([-1,4])
    # plt.xlabel('MLAT');
    # plt.ylabel('Radiance'+'\n'+'log [R]')

plt.subplots_adjust(hspace=0.3, wspace=2)
plt.tight_layout(pad=2, h_pad=0.5)
plt.show()
# SAVEPATH='E:\\Feng_paperwork\\20_aurora_forcast\\paper_figures\\'+'example_valiation_in_diff_mlt_x4';
# plt.savefig(SAVEPATH+'.pdf', bbox_inches='tight') #pdf
SAVEPATH='E:\\Feng_paperwork\\20_aurora_forcast\\meaching_learing_results_split_by_year\\mlat_larger_than_45\\LBHS_model\\Figures\\'+'example_valiation_all_jet_45_sin_cos_02';
plt.savefig(SAVEPATH+'.png', dpi=600,bbox_inches='tight') #pdf
plt.close(fig)

# path = r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\LBHL_model\XGBoost_LBHL_prediction.model"

