#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io as scio
from numpy import exp, linspace, cos, sin, ndarray, sum, argmin, abs, pi
import aacgmv2
import datetime
from netCDF4 import Dataset
import joblib
import matplotlib.pyplot as plt
import scipy.io
from matplotlib import colors


def polar_map(hf):
    ax=hf
    # ax = plt.gca(projection='polar')
    ax.set_thetagrids(np.arange(0.0, 360.0, 30))
    ax.set_thetamin(0.0)  
    ax.set_thetamax(360.0) 
    ax.set_xticklabels(['06', '08', '10', '12', '14', '16', '18', '20', '22', '00', '02', '04'], fontsize=10,family='times new roman', weight='bold')
    ax.set_rgrids(np.arange(0.0, 50.0, 10))
    ax.set_rlabel_position(270.0) 
    ax.set_rlim(0.0, 40.0)  
    ax.set_yticklabels([' ', '80°', '70°', '60°', '50°'], fontsize=10, family='times new roman', weight='bold',color='r')
    ax.grid(True, linestyle="-", color="k", linewidth=2, alpha=0.8)
    return hf
def get_ssusi_data(ssusi_mat_path):
    MLAT=[];MLT=[];MLONG=[];KP=[];LBHS_RADIANCE=[];
    for i in range(len(ssusi_mat_path)):
        data=scio.loadmat(ssusi_mat_path[i]);
        MLAT.append(data['nh_mlat_01']);
        MLT.append(data['nh_mlt_01']);
        MLONG.append(data['nh_mlong_01']);
        KP.append(data['KP']);
        LBHS_RADIANCE.append(data['nh_radiance_lbhs_01']);
    return MLAT,MLONG,MLT,KP,LBHS_RADIANCE
def get_HP(kp):
    A = [16.8244, 0.323365, -4.86128]
    B = [1.82336, 0.613192, 26.1798]
    if kp <= 6.9:
        HP = A[0] * exp(A[1] * kp) + A[2]
    else:
        HP = B[0] * exp(B[1] * kp) + B[2]
    return HP
def get_fitting_value(t, r, AA, BB):
    ncoeff, nkp = AA.shape
    ang = linspace(start=0, stop=nkp-1, num=nkp) * t
    cosang = cos(ang)
    sinang = sin(ang)
    coeff = ndarray(shape=ncoeff)
    for i in range(ncoeff):
        coeff[i] = sum(AA[i, :]*cosang + BB[i, :]*sinang)
    F1 = exp((r - coeff[1]) / coeff[2])
    F2 = 1 + exp((r - coeff[1]) / coeff[3])
    value = coeff[0] * F1 / F2**2
    return value
def get_guvi_kp_model(kp, mlat, mlt):
# input: kp, mlat, mlt (of same length)
# output: energy flux and mean energy
# guvi_auroral_model.nc contains paramters of the auroral model
    data = Dataset(filename='guvi_auroral_model.nc')
    kp_list = data['kp'][:].filled()
    AA_energy = data['AA_energy'][:].filled()
    BB_energy = data['BB_energy'][:].filled()
    AA_flux = data['AA_flux'][:].filled()
    BB_flux = data['BB_flux'][:].filled()
    data.close()

# simplified kp index calculation
    ida = argmin(abs(kp_list - kp))
    if kp_list[ida] > kp:
        ida -= 1
    idb = ida + 1

    x0 = (kp_list[idb] - kp) / (kp_list[idb] - kp_list[ida])
    x1 = (kp - kp_list[ida]) / (kp_list[idb] - kp_list[ida])

    HPa = get_HP(kp_list[ida])
    HPb = get_HP(kp_list[idb])
    HP = get_HP(kp)

    xa = (HPb - HP) / (HPb - HPa)
    xb = (HP - HPa) / (HPb - HPa)

    n = len(mlat)
    flux = ndarray(shape=n)
    energy = ndarray(shape=n)

    for i in range(n):
        r = 90 - mlat[i]
        t = mlt[i] * pi/12
        Va = get_fitting_value(t, r, AA_flux[:, ida, :], BB_flux[:, ida, :])
        Vb = get_fitting_value(t, r, AA_flux[:, idb, :], BB_flux[:, idb, :])
        flux[i] = Va*xa + Vb*xb
        Va = get_fitting_value(t, r, AA_energy[:, ida, :], BB_energy[:, ida, :])
        Vb = get_fitting_value(t, r, AA_energy[:, idb, :], BB_energy[:, idb, :])
        energy[i] = Va*x0 + Vb*x1

    return flux, energy
def zhang_paxton_kp_model_results(kp,dtime):
    MLAT = [];MLT = [];
    FLUX = [];ENERGY = [];
    for mlat in np.arange(50, 90, 0.25):
        for mlt in np.arange(0, 24, 0.1):
            flux, energy = get_guvi_kp_model(kp=np.array([kp]), mlat=np.array([mlat]), mlt=np.array([mlt]));
            MLAT.append(mlat);
            MLT.append(mlt);
            FLUX.append(flux);
            ENERGY.append(energy);
    rho = (90 - abs(np.array(MLAT)));
    theta = ((np.array(MLT) * 15 + 270) * np.pi / 180);
    MLONG = aacgmv2.convert_mlt(MLT, dtime, m2a=True)
    LOG_FLUX = np.array(np.log10(FLUX));
    ENERGY = np.array(ENERGY)
    energy_flux = np.array(LOG_FLUX) * np.array(ENERGY);
    return rho,theta,MLONG,LOG_FLUX,ENERGY,energy_flux,MLAT,MLT

    # fp = np.where(np.array(LOG_FLUX) > -1)
    # fp2 = np.where(np.array(energy_flux) > 0.05)

#========= SH DATA=====
# ========== high Kp
data = pd.read_csv(
    "E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_southern_hemisphere_IMF_geomagnetic_conditions_high_kp_test_data_for_NH_121417.csv")
data = data.dropna(how='any')
# # 提取特征和目标变量
sh_data_features_test = data[['KP', 'OMNI_BX', 'OMNI_BY', 'OMNI_BZ', 'FLOW_SPEED', 'PRESSURE', 'DENSITY']]
sh_data_targets_test = data[['sh_mlat', 'sh_mlt', 'sh_radiance_lbhs']];
# 数据处理 feature
sh_data_features_test['KP'].fillna(sh_data_features_test['KP'].mean(), inplace=True)
sh_data_features_test['OMNI_BX'] = sh_data_features_test['OMNI_BX'].round()
sh_data_features_test['OMNI_BY'] = sh_data_features_test['OMNI_BY'].round()
sh_data_features_test['OMNI_BZ'] = sh_data_features_test['OMNI_BZ'].round()
sh_data_features_test['PRESSURE'] = sh_data_features_test['PRESSURE'].round()
sh_data_features_test['FLOW_SPEED'] = sh_data_features_test['FLOW_SPEED'].round()
sh_data_features_test['DENSITY'] = sh_data_features_test['DENSITY'].round()
fp = np.where((sh_data_targets_test['sh_mlat'] > -50) & (sh_data_targets_test['sh_radiance_lbhs']) > 0)

# 处理数据 targets
sh_data_targets_test['sh_mlat'] = np.abs(sh_data_targets_test['sh_mlat'].round())
sh_data_targets_test['sh_mlt'] = sh_data_targets_test['sh_mlt'].round(1)
sh_data_targets_test['sh_radiance_lbhs'][sh_data_targets_test['sh_radiance_lbhs'] < 0.1] = 1
sh_data_targets_test['sh_radiance_lbhs'].values[fp] = 1
sh_data_targets_test['sh_radiance_lbhs'] = round(np.log10(sh_data_targets_test['sh_radiance_lbhs']), 8)

sh_data_targets_test.rename(
    columns={'sh_mlat': 'nh_mlat', 'sh_mlt': 'nh_mlt', 'sh_radiance_lbhs': 'nh_radiance_lbhs'}, inplace=True)

# ====== data_2014==============
data = pd.read_csv(
    "E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_southern_hemisphere_IMF_geomagnetic_conditions_2014_for_NH.csv")
data = data.dropna(how='any')
# # 提取特征和目标变量
sh_data_features_2014 = data[['KP', 'OMNI_BX', 'OMNI_BY', 'OMNI_BZ', 'FLOW_SPEED', 'PRESSURE', 'DENSITY']]
sh_data_targets_2014 = data[['sh_mlat', 'sh_mlt', 'sh_radiance_lbhs']];
# 数据处理 feature
sh_data_features_2014['KP'].fillna(sh_data_features_2014['KP'].mean(), inplace=True)
sh_data_features_2014['OMNI_BX'] = sh_data_features_2014['OMNI_BX'].round()
sh_data_features_2014['OMNI_BY'] = sh_data_features_2014['OMNI_BY'].round()
sh_data_features_2014['OMNI_BZ'] = sh_data_features_2014['OMNI_BZ'].round()
sh_data_features_2014['PRESSURE'] = sh_data_features_2014['PRESSURE'].round()
sh_data_features_2014['FLOW_SPEED'] = sh_data_features_2014['FLOW_SPEED'].round()
sh_data_features_2014['DENSITY'] = sh_data_features_2014['DENSITY'].round()
fp = np.where((sh_data_targets_2014['sh_mlat'] > -50) & (sh_data_targets_2014['sh_radiance_lbhs']) > 0)

# 处理数据 targets
sh_data_targets_2014['sh_mlat'] = np.abs(sh_data_targets_2014['sh_mlat'].round())
sh_data_targets_2014['sh_mlt'] = sh_data_targets_2014['sh_mlt'].round(1)
sh_data_targets_2014['sh_radiance_lbhs'][sh_data_targets_2014['sh_radiance_lbhs'] < 0.1] = 1
sh_data_targets_2014['sh_radiance_lbhs'].values[fp] = 1
sh_data_targets_2014['sh_radiance_lbhs'] = round(np.log10(sh_data_targets_2014['sh_radiance_lbhs']), 8)
sh_data_targets_2014.rename(
    columns={'sh_mlat': 'nh_mlat', 'sh_mlt': 'nh_mlt', 'sh_radiance_lbhs': 'nh_radiance_lbhs'}, inplace=True)

# #====== data_2014 ==============
data= pd.read_csv("E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_northern_hemisphere_IMF_geomagnetic_conditions_2014.csv")
data =data.dropna(how='any')
# # 提取特征和目标变量
# data_features_2014= data[['KP','OMNI_BX','OMNI_BY','OMNI_BZ','FLOW_SPEED','PRESSURE','DENSITY']]
data_features_2014= data[['KP']]#,'OMNI_BX','OMNI_BY','OMNI_BZ'
data_targets_2014=data[['nh_mlat','nh_mlt','nh_radiance_lbhs']];
# data_targets_2016= round(lbhs_2016,1)
# 数据处理 feature
data_features_2014['KP'].fillna(data_features_2014['KP'].mean(),inplace=True)
# data_features_2014['OMNI_BX']=data_features_2014['OMNI_BX'].round()
# data_features_2014['OMNI_BY']=data_features_2014['OMNI_BY'].round()
# data_features_2014['OMNI_BZ']=data_features_2014['OMNI_BZ'].round()
# data_features_2014['PRESSURE']=data_features_2014['PRESSURE'].round()
# data_features_2014['FLOW_SPEED']=data_features_2014['FLOW_SPEED'].round()
# data_features_2014['DENSITY']=data_features_2014['DENSITY'].round()
fp=np.where((data_targets_2014['nh_mlat']<50)&(data_targets_2014['nh_radiance_lbhs'])>0)

#处理数据 targets
data_targets_2014['nh_mlat']=data_targets_2014['nh_mlat'].round()
data_targets_2014['nh_mlt']=data_targets_2014['nh_mlt'].round(1)
data_targets_2014['nh_radiance_lbhs'][data_targets_2014['nh_radiance_lbhs']<0.1]=1
data_targets_2014['nh_radiance_lbhs'].values[fp]=1
data_targets_2014['nh_radiance_lbhs']=round(np.log10(data_targets_2014['nh_radiance_lbhs']),5)

#=== contact 2014-2016======
# data_targets=pd.concat([data_targets_2010,data_targets_2011,data_targets_2012,data_targets_2013,data_targets_2014,data_targets_2015,data_targets_2016])
# data_features=pd.concat([data_features_2010,data_features_2011,data_features_2012,data_features_2013,data_features_2014,data_features_2015,data_features_2016])
# data_targets=pd.concat([data_targets_2010,data_targets_2011,data_targets_2012,data_targets_2013,data_targets_2014,data_targets_2015,data_targets_2016])
# data_features=pd.concat([data_features_2010,data_features_2011,data_features_2012,data_features_2013,data_features_2014,data_features_2015,data_features_2016])
data_targets=pd.concat([data_targets_2014,sh_data_targets_2014,sh_data_targets_test])
data_features=pd.concat([data_features_2014,sh_data_features_2014,sh_data_features_test])
data_test= pd.concat([data_targets['nh_mlat']/90, np.cos(data_targets['nh_mlt']*np.pi/12),np.sin(data_targets['nh_mlt']*np.pi/12),data_features['KP']/100], axis=1)

 # np.cos(test_data_targets['nh_mlt']*np.pi/12), np.sin(test_data_targets['nh_mlt']*np.pi/12)
#=== value given===================
KP_SSUSI=data_features['KP'].values;
MLAT_SSUSI=data_targets['nh_mlat'].values;
MLT_SSUSI=data_targets['nh_mlt'].values;
LBHS_RADIANCE_SSUSI=data_targets['nh_radiance_lbhs'].values;

# # =============== load XGBOOST model ===================
# path = r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\mlat_50\MultiOutputRegressor_model_KP_2005_2016_md_6_1000_round8_add_nightsidedata_mlat45.model"
path = r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\LBHS_model\XGBoost_lbhs_prediction.model"

# path=r"E:\Feng_paperwork\20_aurora_forcast\meaching_learing_results_split_by_year\mlat_larger_than_45\LBHL_model\XGBoost_LBHL_prediction.model"

model = joblib.load(path)
y_pred = model.predict(data_test)
pre_MLAT = np.array(data_targets['nh_mlat'])
pre_MLT = np.array(data_targets['nh_mlt'])
pre_lbhs = y_pred[:, 3]*6

# # ======= plot different kp results =======
fig = plt.figure(figsize=(12, 12))
# # ================= ssusi-results =========
num=[1,4,7];
kp=[10,30,50];
# ===== boundary index ========
ssusi_pb_mlat=[];ssusi_pb_mlt=[];ssusi_eb_mlat=[];ssusi_eb_mlt=[];ssusi_kp=[];
for i,i_kp in zip(num,kp):
    hf1 = fig.add_subplot(3,3,i, projection='polar')
    polar_map(hf1)
    fp=np.where((KP_SSUSI>i_kp)&(KP_SSUSI<=i_kp+10))
    # fp = np.where((KP_SSUSI == i_kp))
    mlt_ssusi=MLT_SSUSI[fp];
    mlat_ssusi = MLAT_SSUSI[fp];
    randiance_lbhs=LBHS_RADIANCE_SSUSI[fp];
    randiance_lbhs=randiance_lbhs
    rho = (90 - abs(mlat_ssusi));
    theta = ((mlt_ssusi * 15 + 270) * np.pi / 180);

    mlat_point=[];mlt_point=[];ssusi_mean_lbhs=[];
    for mlat in np.arange(50, 90, 2):
        for mlt in np.arange(0, 24.5, 0.5):
            fpp=np.where((mlat_ssusi>=mlat)&(mlat_ssusi<mlat+2)&(mlt_ssusi>=mlt)&(mlt_ssusi<mlt+0.5))
            mlat_point.append(mlat+1);mlt_point.append(mlt+0.25);
            ssusi_mean_lbhs.append(np.nanmean(randiance_lbhs[fpp]));

    #== boundary index =====
    equator_boundary_mlat=[];equator_boundary_mlt=[];
    poleward_boundary_mlat=[];poleward_boundary_mlt=[];
    for mlt in np.arange(0.25, 24.25, 0.5):
        fb1=np.where(np.array(mlt_point)==mlt)
        if ((i_kp / 10) < 5)&(mlt<14.5)&(mlt>12):
            fb2=np.where((np.array(ssusi_mean_lbhs)[fb1]>=1)&(np.array(mlat_point)[fb1]>69))
        else:
            fb2 = np.where((np.array(ssusi_mean_lbhs)[fb1] >=1) & (np.array(mlat_point)[fb1] > 50))

        if (np.size(fb1)>0) & (np.size(fb2)>0):
            equator_boundary_mlat.append(np.min(np.array(mlat_point)[fb1][fb2]));
            equator_boundary_mlt.append(mlt);

            poleward_boundary_mlat.append(np.max(np.array(mlat_point)[fb1][fb2]));
            poleward_boundary_mlt.append(mlt);

    fb4 = np.where(np.array(poleward_boundary_mlat) <80)
    poleward_boundary_rho=(90 - abs(np.array(poleward_boundary_mlat)[fb4]));
    poleward_boundary_theta =((np.array(poleward_boundary_mlt)[fb4] * 15 + 270) * np.pi / 180);
    if (i_kp / 10) < 2:
        fb3 = np.where(np.array(equator_boundary_mlat) > 55)
        equator_boundary_rho = (90 - abs(np.array(equator_boundary_mlat)[fb3]));
        equator_boundary_theta = ((np.array(equator_boundary_mlt)[fb3] * 15 + 270) * np.pi / 180);
    else:
        equator_boundary_rho = (90 - abs(np.array(equator_boundary_mlat)));
        equator_boundary_theta = ((np.array(equator_boundary_mlt) * 15 + 270) * np.pi / 180);

    # plot
    ssusi_rho_point = (90 - abs(np.array(mlat_point)));
    ssusi_theta_point= ((np.array(mlt_point) * 15 + 270) * np.pi / 180);

    im = plt.scatter(ssusi_theta_point, ssusi_rho_point, s=60, c=np.array(ssusi_mean_lbhs), cmap='jet')
    cbar = fig.colorbar(im, ax=hf1, fraction=0.035, pad=0.12)
    plt.clim(-1, 4)
    cbar.ax.set_title('LBHS'+'\n'+'log [R]')
    cbar.set_ticks([-1,0,1, 2, 3, 4])
    cbar.set_ticklabels(['-1','0','1', '2', '3', '4'])
    # plt.plot(poleward_boundary_theta,poleward_boundary_rho,'r--')
    # plt.plot(equator_boundary_theta, equator_boundary_rho, 'r--')

    txt = 'SSUSI DATA\n Kp=' + str(int(i_kp/10))+'-'+ str(int(i_kp+10)/10);
    plt.title(txt, color='r', fontsize=10)

#====xgboost==============
    XGB_pb_mlat=[];XGB_pb_mlt=[];XGB_eb_mlat=[];XGB_eb_mlt=[];XGB_kp=[];
    hf1=fig.add_subplot(3, 3, i+1, projection='polar')
    polar_map(hf1)

    # ====== get specify_kp_data=======
    fp = np.where((KP_SSUSI >i_kp) & (KP_SSUSI <=i_kp + 10))
    # fp = np.where((KP_SSUSI == i_kp))
    pre_mlt_ssusi = pre_MLT[fp];
    pre_mlat_ssusi =pre_MLAT[fp];
    pre_randiance_lbhs = pre_lbhs[fp];
    # randiance_lbhs = randiance_lbhs
    # rho = (90 - abs(mlat_ssusi));
    # theta = ((mlt_ssusi * 15 + 270) * np.pi / 180);

    #======= get mean lbhs=======
    mlat_point=[];mlt_point=[];xgb_mean_lbhs=[];
    for mlat in np.arange(50, 90, 2):
        for mlt in np.arange(0, 24.5, 0.5):
            fpp=np.where((pre_mlat_ssusi>=mlat)&(pre_mlat_ssusi<mlat+2)&(pre_mlt_ssusi>=mlt)&(pre_mlt_ssusi<mlt+0.5))
            mlat_point.append(mlat+1);mlt_point.append(mlt+0.25);
            xgb_mean_lbhs.append(np.nanmean(pre_randiance_lbhs[fpp]));

    # 这一步是懒得修改以下程序，所以把平均后的值 赋值一下。
    MLT=np.array(mlt_point);MLAT=np.array(mlat_point);xgb_pre_mean_lbhs=np.array(xgb_mean_lbhs)
    xgb_rho = (90 - abs(np.array(MLAT)));
    xgb_theta = ((np.array(MLT) * 15 + 270) * np.pi / 180);
    #===== boundary index =====
    equator_boundary_mlat=[];equator_boundary_mlt=[];
    poleward_boundary_mlat=[];poleward_boundary_mlt=[];
    for mlt in np.arange(0.25, 24.25, 0.5):
        fb1=np.where(np.array(mlt_point)==mlt)
        # fb1=np.where((mlt<=np.array(MLT))&(np.array(MLT)<mlt+0.5)&(np.array(MLAT)<77)&(np.array(MLAT)>50))
        if np.size(fb1) > 0:
            fb2=np.where(np.array(xgb_pre_mean_lbhs)[fb1]>=1)
            if np.size(fb2) > 0:
                equator_boundary_mlat.append(np.min(MLAT[fb1][fb2]));
                equator_boundary_mlt.append(mlt);
                poleward_boundary_mlat.append(np.max(MLAT[fb1][fb2]));
                poleward_boundary_mlt.append(mlt);


    poleward_boundary_rho=(90 - abs(np.array(poleward_boundary_mlat)));
    poleward_boundary_theta =((np.array(poleward_boundary_mlt) * 15 + 270) * np.pi / 180);

    # fb3=np.where(np.array(equator_boundary_mlat) > 60)
    equator_boundary_rho = (90 - abs(np.array(equator_boundary_mlat)));
    equator_boundary_theta = ((np.array(equator_boundary_mlt) * 15 + 270) * np.pi / 180);

    # fp=np.where(y_pred[:,2]>0)
    im = plt.scatter(xgb_theta, xgb_rho, s=60, c=xgb_pre_mean_lbhs, cmap='jet')
    # cbar = fig.colorbar(im, ax=hf1, orientation='horizontal')
    cbar = fig.colorbar(im, ax=hf1, fraction=0.035, pad=0.12)
    cbar.ax.set_title('LBHS'+'\n'+'log [R]')
    plt.clim(-1, 4)
    cbar.set_ticks([-1,0,1, 2, 3, 4])
    cbar.set_ticklabels(['-1','0','1', '2', '3', '4'])

    # plt.plot(poleward_boundary_theta, poleward_boundary_rho, 'r--')
    # plt.plot(equator_boundary_theta, equator_boundary_rho, 'r--')

    txt = 'XGBoost model\n Kp=' + str((i_kp)/10)+'-'+str((i_kp+10)/10);
    plt.title(txt, color='r', fontsize=10)
    # plt.show()
#========== plot error=======
    hf1=fig.add_subplot(3, 3, i+2, projection='polar')
    polar_map(hf1)
    error_lbhs=ssusi_mean_lbhs-xgb_pre_mean_lbhs;
    color_list = ['lightseagreen','lawngreen','darkorange','gold']
    my_cmap = colors.LinearSegmentedColormap.from_list('jet', color_list,N=4)
    im = plt.scatter(xgb_theta, xgb_rho, s=60, c=error_lbhs, cmap='jet')
    cbar = fig.colorbar(im, ax=hf1, fraction=0.035, pad=0.12)
    cbar.ax.set_title('LBHS Error' + '\n' + 'log [R]')
    plt.clim(-2,2)
    cbar.set_ticks([-2,-1,0,1,2])
    cbar.set_ticklabels(['-2','-1', '0', '1','2'])
    txt = 'Error (SSUSI-XGBoost)\n Kp=' + str((i_kp) / 10) + '-' + str((i_kp + 10) / 10);
    plt.title(txt, color='r', fontsize=10)



# plt.show(block=True)
plt.tight_layout()
plt.savefig('E:\\Feng_paperwork\\20_aurora_forcast\\meaching_learing_results_split_by_year\\mlat_larger_than_45\\LBHS_model\\Figures\\' + 'three_method_results_2005_2016_kp246_with_boundary_round8_md6_thred_split_by_year_add_nightsidedata_error_45_sin_cos'+'.png',
bbox_inches='tight')
plt.savefig('E:\\Feng_paperwork\\20_aurora_forcast\\meaching_learing_results_split_by_year\\mlat_larger_than_45\\LBHS_model\\Figures\\' + 'three_method_results_2005_2016_kp246_with_boundary_round8_md6_thred_split_by_year_add_nightsidedata_error_45_sin_cos'+'.pdf',
bbox_inches='tight')

# plt.savefig('E:\\Feng_paperwork\\20_aurora_forcast\\meaching_learing_results_split_by_year\\mlat_larger_than_45\\LBHL_model\\' + 'three_method_results_2005_2016_kp246_with_boundary_round8_md6_thred_split_by_year_add_nightsidedata_error_45'+'.png',
# bbox_inches='tight')
# plt.savefig('E:\\Feng_paperwork\\20_aurora_forcast\\meaching_learing_results_split_by_year\\mlat_larger_than_45\\LBHL_model\\' + 'three_method_results_2005_2016_kp246_with_boundary_round8_md6_thred_split_by_year_add_nightsidedata_error_45'+'.pdf',
# bbox_inches='tight')
# plt.close()



