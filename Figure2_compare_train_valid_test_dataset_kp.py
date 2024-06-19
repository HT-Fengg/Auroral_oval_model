#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/7 22:04
# @Author  : FHT
# @File    : compare_train_valid_test_dataset_kp.py

import pandas as pd  # 数据分析
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import joblib
import numpy as np
import pandas as pd
import aacgmv2,time,datetime
import matplotlib.pyplot as plt
from matplotlib import colors

# southern data for northern 2011
def polar_map(hf):
    ax=hf
    # ax = plt.gca(projection='polar')
    ax.set_thetagrids(np.arange(0.0, 360.0, 30.0))
    ax.set_thetamin(0.0)  # 设置极坐标图开始角度为0°
    ax.set_thetamax(360.0)  # 设置极坐标结束角度为360°
    ax.set_xticklabels(['06', '08', '10', '12', '14', '16', '18', '20', '22', '00', '02', '04'], fontsize=10,family='times new roman', weight='bold')
    ax.set_rgrids(np.arange(0.0, 50.0, 10))
    ax.set_rlabel_position(270.0)  # 标签显示在0°
    ax.set_rlim(0.0, 40.0)  # 标签范围为[0, 5000)
    ax.set_yticklabels([' ', '80°', '70°', '60°', '50°'], fontsize=10, family='times new roman', weight='bold',color='r')
    ax.grid(True, linestyle="-", color="k", linewidth=2, alpha=0.5)
    return hf
def data_process():
    #=== high kp_for test ===
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
#============
    data = pd.read_csv(
        "E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_southern_hemisphere_IMF_geomagnetic_conditions_high_kp_for_NH.csv")
    data = data.dropna(how='any')
    # # 提取特征和目标变量
    sh_data_features_train = data[['KP', 'OMNI_BX', 'OMNI_BY', 'OMNI_BZ', 'FLOW_SPEED', 'PRESSURE', 'DENSITY']]
    sh_data_targets_train = data[['sh_mlat', 'sh_mlt', 'sh_radiance_lbhs']];
    # 数据处理 feature
    sh_data_features_train['KP'].fillna(sh_data_features_train['KP'].mean(), inplace=True)
    sh_data_features_train['OMNI_BX'] = sh_data_features_train['OMNI_BX'].round()
    sh_data_features_train['OMNI_BY'] = sh_data_features_train['OMNI_BY'].round()
    sh_data_features_train['OMNI_BZ'] = sh_data_features_train['OMNI_BZ'].round()
    sh_data_features_train['PRESSURE'] = sh_data_features_train['PRESSURE'].round()
    sh_data_features_train['FLOW_SPEED'] = sh_data_features_train['FLOW_SPEED'].round()
    sh_data_features_train['DENSITY'] = sh_data_features_train['DENSITY'].round()
    fp = np.where((sh_data_targets_train['sh_mlat'] > -50) & (sh_data_targets_train['sh_radiance_lbhs']) > 0)

    # 处理数据 targets
    sh_data_targets_train['sh_mlat'] = np.abs(sh_data_targets_train['sh_mlat'].round())
    sh_data_targets_train['sh_mlt'] = sh_data_targets_train['sh_mlt'].round(1)
    sh_data_targets_train['sh_radiance_lbhs'][sh_data_targets_train['sh_radiance_lbhs'] < 0.1] = 1
    sh_data_targets_train['sh_radiance_lbhs'].values[fp] = 1
    sh_data_targets_train['sh_radiance_lbhs'] = round(np.log10(sh_data_targets_train['sh_radiance_lbhs']), 8)

    sh_data_targets_train.rename(
        columns={'sh_mlat': 'nh_mlat', 'sh_mlt': 'nh_mlt', 'sh_radiance_lbhs': 'nh_radiance_lbhs'}, inplace=True)

    # southern data for northern 2011
    #====== data_2011==============
    data= pd.read_csv("E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_southern_hemisphere_IMF_geomagnetic_conditions_2011_for_NH.csv")
    data =data.dropna(how='any')
    # # 提取特征和目标变量
    sh_data_features_2011= data[['KP','OMNI_BX','OMNI_BY','OMNI_BZ','FLOW_SPEED','PRESSURE','DENSITY']]
    sh_data_targets_2011=data[['sh_mlat','sh_mlt','sh_radiance_lbhs']];
    # 数据处理 feature
    sh_data_features_2011['KP'].fillna(sh_data_features_2011['KP'].mean(),inplace=True)
    sh_data_features_2011['OMNI_BX']=sh_data_features_2011['OMNI_BX'].round()
    sh_data_features_2011['OMNI_BY']=sh_data_features_2011['OMNI_BY'].round()
    sh_data_features_2011['OMNI_BZ']=sh_data_features_2011['OMNI_BZ'].round()
    sh_data_features_2011['PRESSURE']=sh_data_features_2011['PRESSURE'].round()
    sh_data_features_2011['FLOW_SPEED']=sh_data_features_2011['FLOW_SPEED'].round()
    sh_data_features_2011['DENSITY']=sh_data_features_2011['DENSITY'].round()
    fp=np.where((sh_data_targets_2011['sh_mlat']>-55)&(sh_data_targets_2011['sh_radiance_lbhs'])>0)

    #处理数据 targets
    sh_data_targets_2011['sh_mlat']=np.abs(sh_data_targets_2011['sh_mlat'].round())
    sh_data_targets_2011['sh_mlt']=sh_data_targets_2011['sh_mlt'].round(1)
    sh_data_targets_2011['sh_radiance_lbhs'][sh_data_targets_2011['sh_radiance_lbhs']<0.1]=1
    sh_data_targets_2011['sh_radiance_lbhs'].values[fp]=1
    sh_data_targets_2011['sh_radiance_lbhs']=round(np.log10(sh_data_targets_2011['sh_radiance_lbhs']),8)

    sh_data_targets_2011.rename(columns={'sh_mlat':'nh_mlat','sh_mlt':'nh_mlt','sh_radiance_lbhs':'nh_radiance_lbhs'}, inplace=True)

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
    fp = np.where((sh_data_targets_2014['sh_mlat'] > -55) & (sh_data_targets_2014['sh_radiance_lbhs']) > 0)

    # 处理数据 targets
    sh_data_targets_2014['sh_mlat'] = np.abs(sh_data_targets_2014['sh_mlat'].round())
    sh_data_targets_2014['sh_mlt'] = sh_data_targets_2014['sh_mlt'].round(1)
    sh_data_targets_2014['sh_radiance_lbhs'][sh_data_targets_2014['sh_radiance_lbhs'] < 0.1] = 1
    sh_data_targets_2014['sh_radiance_lbhs'].values[fp] = 1
    sh_data_targets_2014['sh_radiance_lbhs'] = round(np.log10(sh_data_targets_2014['sh_radiance_lbhs']), 8)
    sh_data_targets_2014.rename(
        columns={'sh_mlat': 'nh_mlat', 'sh_mlt': 'nh_mlt', 'sh_radiance_lbhs': 'nh_radiance_lbhs'}, inplace=True)

    # ======= data_2005 =============
    data = pd.read_csv(
        "E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_northern_hemisphere_IMF_geomagnetic_conditions_2005.csv")
    data = data.dropna(how='any')
    # # 提取特征和目标变量
    data_features_2005 = data[['KP', 'OMNI_BX', 'OMNI_BY', 'OMNI_BZ', 'FLOW_SPEED', 'PRESSURE', 'DENSITY']]
    data_targets_2005= data[['nh_mlat', 'nh_mlt', 'nh_radiance_lbhs']];
    # 数据处理 feature
    data_features_2005['KP'].fillna(data_features_2005['KP'].mean(), inplace=True)
    data_features_2005['OMNI_BX'] = data_features_2005['OMNI_BX'].round()
    data_features_2005['OMNI_BY'] = data_features_2005['OMNI_BY'].round()
    data_features_2005['OMNI_BZ'] = data_features_2005['OMNI_BZ'].round()
    data_features_2005['PRESSURE'] = data_features_2005['PRESSURE'].round()
    data_features_2005['FLOW_SPEED'] = data_features_2005['FLOW_SPEED'].round()
    data_features_2005['DENSITY'] = data_features_2005['DENSITY'].round()
    fp = np.where((data_targets_2005['nh_mlat'] < 55) & (data_targets_2005['nh_radiance_lbhs']) > 0)
    # 处理数据 targets
    data_targets_2005['nh_mlat'] = data_targets_2005['nh_mlat'].round()
    data_targets_2005['nh_mlt'] = data_targets_2005['nh_mlt'].round(1)
    data_targets_2005['nh_radiance_lbhs'][data_targets_2005['nh_radiance_lbhs'] < 0.1] = 1
    data_targets_2005['nh_radiance_lbhs'].values[fp] = 1
    data_targets_2005['nh_radiance_lbhs'] = round(np.log10(data_targets_2005['nh_radiance_lbhs']), 8)

    # ======= data_2006 =============
    data = pd.read_csv(
        "E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_northern_hemisphere_IMF_geomagnetic_conditions_2006.csv")
    data = data.dropna(how='any')
    # # 提取特征和目标变量
    data_features_2006 = data[['KP', 'OMNI_BX', 'OMNI_BY', 'OMNI_BZ', 'FLOW_SPEED', 'PRESSURE', 'DENSITY']]
    data_targets_2006 = data[['nh_mlat', 'nh_mlt', 'nh_radiance_lbhs']];
    # 数据处理 feature
    data_features_2006['KP'].fillna(data_features_2006['KP'].mean(), inplace=True)
    data_features_2006['OMNI_BX'] = data_features_2006['OMNI_BX'].round()
    data_features_2006['OMNI_BY'] = data_features_2006['OMNI_BY'].round()
    data_features_2006['OMNI_BZ'] = data_features_2006['OMNI_BZ'].round()
    data_features_2006['PRESSURE'] = data_features_2006['PRESSURE'].round()
    data_features_2006['FLOW_SPEED'] = data_features_2006['FLOW_SPEED'].round()
    data_features_2006['DENSITY'] = data_features_2006['DENSITY'].round()
    fp = np.where((data_targets_2006['nh_mlat'] < 55) & (data_targets_2006['nh_radiance_lbhs']) > 0)
    # 处理数据 targets
    data_targets_2006['nh_mlat'] = data_targets_2006['nh_mlat'].round()
    data_targets_2006['nh_mlt'] = data_targets_2006['nh_mlt'].round(1)
    data_targets_2006['nh_radiance_lbhs'][data_targets_2006['nh_radiance_lbhs'] < 0.1] = 1
    data_targets_2006['nh_radiance_lbhs'].values[fp] = 1
    data_targets_2006['nh_radiance_lbhs'] = round(np.log10(data_targets_2006['nh_radiance_lbhs']), 8)

    # ======= data_2007 =============
    data = pd.read_csv(
        "E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_northern_hemisphere_IMF_geomagnetic_conditions_2007.csv")
    data = data.dropna(how='any')
    # # 提取特征和目标变量
    data_features_2007 = data[['KP', 'OMNI_BX', 'OMNI_BY', 'OMNI_BZ', 'FLOW_SPEED', 'PRESSURE', 'DENSITY']]
    data_targets_2007 = data[['nh_mlat', 'nh_mlt', 'nh_radiance_lbhs']];
    # 数据处理 feature
    data_features_2007['KP'].fillna(data_features_2007['KP'].mean(), inplace=True)
    data_features_2007['OMNI_BX'] = data_features_2007['OMNI_BX'].round()
    data_features_2007['OMNI_BY'] = data_features_2007['OMNI_BY'].round()
    data_features_2007['OMNI_BZ'] = data_features_2007['OMNI_BZ'].round()
    data_features_2007['PRESSURE'] = data_features_2007['PRESSURE'].round()
    data_features_2007['FLOW_SPEED'] = data_features_2007['FLOW_SPEED'].round()
    data_features_2007['DENSITY'] = data_features_2007['DENSITY'].round()
    fp = np.where((data_targets_2007['nh_mlat'] < 55) & (data_targets_2007['nh_radiance_lbhs']) > 0)
    # 处理数据 targets
    data_targets_2007['nh_mlat'] = data_targets_2007['nh_mlat'].round()
    data_targets_2007['nh_mlt'] = data_targets_2007['nh_mlt'].round(1)
    data_targets_2007['nh_radiance_lbhs'][data_targets_2007['nh_radiance_lbhs'] < 0.1] = 1
    data_targets_2007['nh_radiance_lbhs'].values[fp] = 1
    data_targets_2007['nh_radiance_lbhs'] = round(np.log10(data_targets_2007['nh_radiance_lbhs']), 8)

    # ======= data_2008 =============
    data = pd.read_csv(
        "E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_northern_hemisphere_IMF_geomagnetic_conditions_2008.csv")
    data = data.dropna(how='any')
    # # 提取特征和目标变量
    data_features_2008 = data[['KP', 'OMNI_BX', 'OMNI_BY', 'OMNI_BZ', 'FLOW_SPEED', 'PRESSURE', 'DENSITY']]
    data_targets_2008 = data[['nh_mlat', 'nh_mlt', 'nh_radiance_lbhs']];
    # 数据处理 feature
    data_features_2008['KP'].fillna(data_features_2008['KP'].mean(), inplace=True)
    data_features_2008['OMNI_BX'] = data_features_2008['OMNI_BX'].round()
    data_features_2008['OMNI_BY'] = data_features_2008['OMNI_BY'].round()
    data_features_2008['OMNI_BZ'] = data_features_2008['OMNI_BZ'].round()
    data_features_2008['PRESSURE'] = data_features_2008['PRESSURE'].round()
    data_features_2008['FLOW_SPEED'] = data_features_2008['FLOW_SPEED'].round()
    data_features_2008['DENSITY'] = data_features_2008['DENSITY'].round()
    fp = np.where((data_targets_2008['nh_mlat'] < 55) & (data_targets_2008['nh_radiance_lbhs']) > 0)

    # 处理数据 targets
    data_targets_2008['nh_mlat'] = data_targets_2008['nh_mlat'].round()
    data_targets_2008['nh_mlt'] = data_targets_2008['nh_mlt'].round(1)
    data_targets_2008['nh_radiance_lbhs'][data_targets_2008['nh_radiance_lbhs'] < 0.1] = 1
    data_targets_2008['nh_radiance_lbhs'].values[fp] = 1
    data_targets_2008['nh_radiance_lbhs'] = round(np.log10(data_targets_2008['nh_radiance_lbhs']), 8)

    #======= data_2009 =============
    data = pd.read_csv(
        "E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_northern_hemisphere_IMF_geomagnetic_conditions_2009.csv")
    data = data.dropna(how='any')
    # # 提取特征和目标变量
    data_features_2009 = data[['KP', 'OMNI_BX', 'OMNI_BY', 'OMNI_BZ', 'FLOW_SPEED', 'PRESSURE', 'DENSITY']]
    data_targets_2009 = data[['nh_mlat', 'nh_mlt', 'nh_radiance_lbhs']];
    # 数据处理 feature
    data_features_2009['KP'].fillna(data_features_2009['KP'].mean(), inplace=True)
    data_features_2009['OMNI_BX'] = data_features_2009['OMNI_BX'].round()
    data_features_2009['OMNI_BY'] = data_features_2009['OMNI_BY'].round()
    data_features_2009['OMNI_BZ'] = data_features_2009['OMNI_BZ'].round()
    data_features_2009['PRESSURE'] = data_features_2009['PRESSURE'].round()
    data_features_2009['FLOW_SPEED'] = data_features_2009['FLOW_SPEED'].round()
    data_features_2009['DENSITY'] = data_features_2009['DENSITY'].round()
    fp = np.where((data_targets_2009['nh_mlat'] < 55) & (data_targets_2009['nh_radiance_lbhs']) > 0)

    # 处理数据 targets
    data_targets_2009['nh_mlat'] = data_targets_2009['nh_mlat'].round()
    data_targets_2009['nh_mlt'] = data_targets_2009['nh_mlt'].round(1)
    data_targets_2009['nh_radiance_lbhs'][data_targets_2009['nh_radiance_lbhs'] < 0.1] = 1
    data_targets_2009['nh_radiance_lbhs'].values[fp] = 1
    data_targets_2009['nh_radiance_lbhs'] = round(np.log10(data_targets_2009['nh_radiance_lbhs']), 8)

    #====== data_2010==============
    data= pd.read_csv("E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_northern_hemisphere_IMF_geomagnetic_conditions_2010.csv")
    data =data.dropna(how='any')
    # # 提取特征和目标变量
    data_features_2010= data[['KP','OMNI_BX','OMNI_BY','OMNI_BZ','FLOW_SPEED','PRESSURE','DENSITY']]
    data_targets_2010=data[['nh_mlat','nh_mlt','nh_radiance_lbhs']];
    # 数据处理 feature
    data_features_2010['KP'].fillna(data_features_2010['KP'].mean(),inplace=True)
    data_features_2010['OMNI_BX']=data_features_2010['OMNI_BX'].round()
    data_features_2010['OMNI_BY']=data_features_2010['OMNI_BY'].round()
    data_features_2010['OMNI_BZ']=data_features_2010['OMNI_BZ'].round()
    data_features_2010['PRESSURE']=data_features_2010['PRESSURE'].round()
    data_features_2010['FLOW_SPEED']=data_features_2010['FLOW_SPEED'].round()
    data_features_2010['DENSITY']=data_features_2010['DENSITY'].round()
    fp=np.where((data_targets_2010['nh_mlat']<55)&(data_targets_2010['nh_radiance_lbhs'])>0)

    #处理数据 targets
    data_targets_2010['nh_mlat']=data_targets_2010['nh_mlat'].round()
    data_targets_2010['nh_mlt']=data_targets_2010['nh_mlt'].round(1)
    data_targets_2010['nh_radiance_lbhs'][data_targets_2010['nh_radiance_lbhs']<0.1]=1
    data_targets_2010['nh_radiance_lbhs'].values[fp]=1
    data_targets_2010['nh_radiance_lbhs']=round(np.log10(data_targets_2010['nh_radiance_lbhs']),8)

    #====== data_2011==============
    data= pd.read_csv("E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_northern_hemisphere_IMF_geomagnetic_conditions_2011.csv")
    data =data.dropna(how='any')
    # # 提取特征和目标变量
    data_features_2011= data[['KP','OMNI_BX','OMNI_BY','OMNI_BZ','FLOW_SPEED','PRESSURE','DENSITY']]
    data_targets_2011=data[['nh_mlat','nh_mlt','nh_radiance_lbhs']];
    # 数据处理 feature
    data_features_2011['KP'].fillna(data_features_2011['KP'].mean(),inplace=True)
    data_features_2011['OMNI_BX']=data_features_2011['OMNI_BX'].round()
    data_features_2011['OMNI_BY']=data_features_2011['OMNI_BY'].round()
    data_features_2011['OMNI_BZ']=data_features_2011['OMNI_BZ'].round()
    data_features_2011['PRESSURE']=data_features_2011['PRESSURE'].round()
    data_features_2011['FLOW_SPEED']=data_features_2011['FLOW_SPEED'].round()
    data_features_2011['DENSITY']=data_features_2011['DENSITY'].round()
    fp=np.where((data_targets_2011['nh_mlat']<55)&(data_targets_2011['nh_radiance_lbhs'])>0)

    #处理数据 targets
    data_targets_2011['nh_mlat']=data_targets_2011['nh_mlat'].round()
    data_targets_2011['nh_mlt']=data_targets_2011['nh_mlt'].round(1)
    data_targets_2011['nh_radiance_lbhs'][data_targets_2011['nh_radiance_lbhs']<0.1]=1
    data_targets_2011['nh_radiance_lbhs'].values[fp]=1
    data_targets_2011['nh_radiance_lbhs']=round(np.log10(data_targets_2011['nh_radiance_lbhs']),8)

    #====== data_2012==============
    data= pd.read_csv("E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_northern_hemisphere_IMF_geomagnetic_conditions_2012.csv")
    data =data.dropna(how='any')
    # # 提取特征和目标变量
    data_features_2012= data[['KP','OMNI_BX','OMNI_BY','OMNI_BZ','FLOW_SPEED','PRESSURE','DENSITY']]
    data_targets_2012=data[['nh_mlat','nh_mlt','nh_radiance_lbhs']];
    # 数据处理 feature
    data_features_2012['KP'].fillna(data_features_2012['KP'].mean(),inplace=True)
    data_features_2012['OMNI_BX']=data_features_2012['OMNI_BX'].round()
    data_features_2012['OMNI_BY']=data_features_2012['OMNI_BY'].round()
    data_features_2012['OMNI_BZ']=data_features_2012['OMNI_BZ'].round()
    data_features_2012['PRESSURE']=data_features_2012['PRESSURE'].round()
    data_features_2012['FLOW_SPEED']=data_features_2012['FLOW_SPEED'].round()
    data_features_2012['DENSITY']=data_features_2012['DENSITY'].round()
    fp=np.where((data_targets_2012['nh_mlat']<55)&(data_targets_2012['nh_radiance_lbhs'])>0)

    #处理数据 targets
    data_targets_2012['nh_mlat']=data_targets_2012['nh_mlat'].round()
    data_targets_2012['nh_mlt']=data_targets_2012['nh_mlt'].round(1)
    data_targets_2012['nh_radiance_lbhs'][data_targets_2012['nh_radiance_lbhs']<0.1]=1
    data_targets_2012['nh_radiance_lbhs'].values[fp]=1
    data_targets_2012['nh_radiance_lbhs']=round(np.log10(data_targets_2012['nh_radiance_lbhs']),8)


    #====== data_2013==============
    data= pd.read_csv("E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_northern_hemisphere_IMF_geomagnetic_conditions_2013.csv")
    data =data.dropna(how='any')
    # # 提取特征和目标变量
    data_features_2013= data[['KP','OMNI_BX','OMNI_BY','OMNI_BZ','FLOW_SPEED','PRESSURE','DENSITY']]
    data_targets_2013=data[['nh_mlat','nh_mlt','nh_radiance_lbhs']];
    # 数据处理 feature
    data_features_2013['KP'].fillna(data_features_2013['KP'].mean(),inplace=True)
    data_features_2013['OMNI_BX']=data_features_2013['OMNI_BX'].round()
    data_features_2013['OMNI_BY']=data_features_2013['OMNI_BY'].round()
    data_features_2013['OMNI_BZ']=data_features_2013['OMNI_BZ'].round()
    data_features_2013['PRESSURE']=data_features_2013['PRESSURE'].round()
    data_features_2013['FLOW_SPEED']=data_features_2013['FLOW_SPEED'].round()
    data_features_2013['DENSITY']=data_features_2013['DENSITY'].round()
    fp=np.where((data_targets_2013['nh_mlat']<55)&(data_targets_2013['nh_radiance_lbhs'])>0)

    #处理数据 targets
    data_targets_2013['nh_mlat']=data_targets_2013['nh_mlat'].round()
    data_targets_2013['nh_mlt']=data_targets_2013['nh_mlt'].round(1)
    data_targets_2013['nh_radiance_lbhs'][data_targets_2013['nh_radiance_lbhs']<0.1]=1
    data_targets_2013['nh_radiance_lbhs'].values[fp]=1
    data_targets_2013['nh_radiance_lbhs']=round(np.log10(data_targets_2013['nh_radiance_lbhs']),8)

    #====== data_2014 ==============
    data= pd.read_csv("E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_northern_hemisphere_IMF_geomagnetic_conditions_2014.csv")
    data =data.dropna(how='any')
    # # 提取特征和目标变量
    data_features_2014= data[['KP','OMNI_BX','OMNI_BY','OMNI_BZ','FLOW_SPEED','PRESSURE','DENSITY']]
    data_targets_2014=data[['nh_mlat','nh_mlt','nh_radiance_lbhs']];
    # 数据处理 feature
    data_features_2014['KP'].fillna(data_features_2014['KP'].mean(),inplace=True)
    data_features_2014['OMNI_BX']=data_features_2014['OMNI_BX'].round()
    data_features_2014['OMNI_BY']=data_features_2014['OMNI_BY'].round()
    data_features_2014['OMNI_BZ']=data_features_2014['OMNI_BZ'].round()
    data_features_2014['PRESSURE']=data_features_2014['PRESSURE'].round()
    data_features_2014['FLOW_SPEED']=data_features_2014['FLOW_SPEED'].round()
    data_features_2014['DENSITY']=data_features_2014['DENSITY'].round()
    fp=np.where((data_targets_2014['nh_mlat']<55)&(data_targets_2014['nh_radiance_lbhs'])>0)

    #处理数据 targets
    data_targets_2014['nh_mlat']=data_targets_2014['nh_mlat'].round()
    data_targets_2014['nh_mlt']=data_targets_2014['nh_mlt'].round(1)
    data_targets_2014['nh_radiance_lbhs'][data_targets_2014['nh_radiance_lbhs']<0.1]=1
    data_targets_2014['nh_radiance_lbhs'].values[fp]=1
    data_targets_2014['nh_radiance_lbhs']=round(np.log10(data_targets_2014['nh_radiance_lbhs']),8)


    #====== data_2015 ==============
    data= pd.read_csv("E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_northern_hemisphere_IMF_geomagnetic_conditions_2015_01.csv")
    data =data.dropna(how='any')
    # # 提取特征和目标变量
    data_features_2015= data[['KP','OMNI_BX','OMNI_BY','OMNI_BZ','FLOW_SPEED','PRESSURE','DENSITY']]
    data_targets_2015=data[['nh_mlat','nh_mlt','nh_radiance_lbhs']];
    # data_targets_2016= round(lbhs_2016,1)
    # 数据处理 feature
    data_features_2015['KP'].fillna(data_features_2015['KP'].mean(),inplace=True)
    data_features_2015['OMNI_BX']=data_features_2015['OMNI_BX'].round()
    data_features_2015['OMNI_BY']=data_features_2015['OMNI_BY'].round()
    data_features_2015['OMNI_BZ']=data_features_2015['OMNI_BZ'].round()
    data_features_2015['PRESSURE']=data_features_2015['PRESSURE'].round()
    data_features_2015['FLOW_SPEED']=data_features_2015['FLOW_SPEED'].round()
    data_features_2015['DENSITY']=data_features_2015['DENSITY'].round()
    fp=np.where((data_targets_2015['nh_mlat']<55)&(data_targets_2015['nh_radiance_lbhs'])>0)

    #处理数据 targe
    data_targets_2015['nh_mlat']=data_targets_2015['nh_mlat'].round()
    data_targets_2015['nh_mlt']=data_targets_2015['nh_mlt'].round(1)
    data_targets_2015['nh_radiance_lbhs'][data_targets_2015['nh_radiance_lbhs']<0.1]=1
    data_targets_2015['nh_radiance_lbhs'].values[fp]=1
    data_targets_2015['nh_radiance_lbhs']=round(np.log10(data_targets_2015['nh_radiance_lbhs']),8)

    #====== data_2016 ==============
    data= pd.read_csv("E:\\Feng_paperwork\\20_aurora_forcast\\aurora_forcast_ssusi_data_northern_hemisphere_IMF_geomagnetic_conditions_2016_new01.csv")
    data =data.dropna(how='any')
    # # 提取特征和目标变量
    data_features_2016= data[['KP','OMNI_BX','OMNI_BY','OMNI_BZ','FLOW_SPEED','PRESSURE','DENSITY']]
    data_targets_2016=data[['nh_mlat','nh_mlt','nh_radiance_lbhs']];
    # 数据处理 feature
    data_features_2016['KP'].fillna(data_features_2016['KP'].mean(),inplace=True)
    data_features_2016['OMNI_BX']=data_features_2016['OMNI_BX'].round()
    data_features_2016['OMNI_BY']=data_features_2016['OMNI_BY'].round()
    data_features_2016['OMNI_BZ']=data_features_2016['OMNI_BZ'].round()
    data_features_2016['PRESSURE']=data_features_2016['PRESSURE'].round()
    data_features_2016['FLOW_SPEED']=data_features_2016['FLOW_SPEED'].round()
    data_features_2016['DENSITY']=data_features_2016['DENSITY'].round()
    fp=np.where((data_targets_2016['nh_mlat']<55)&(data_targets_2016['nh_radiance_lbhs'])>0)
    #处理数据 targets
    data_targets_2016['nh_mlat']=data_targets_2016['nh_mlat'].round()
    data_targets_2016['nh_mlt']=data_targets_2016['nh_mlt'].round(1)
    data_targets_2016['nh_radiance_lbhs'][data_targets_2016['nh_radiance_lbhs']<0.1]=1
    data_targets_2016['nh_radiance_lbhs'].values[fp]=1
    data_targets_2016['nh_radiance_lbhs']=round(np.log10(data_targets_2016['nh_radiance_lbhs']),8)
    #=== contact 2014-2016======
    # data_targets=pd.concat([data_targets_2015])
    # data_features=pd.concat([data_features_2015])
    train_data_targets = pd.concat(
        [sh_data_targets_train,sh_data_targets_2011, data_targets_2005, data_targets_2006, data_targets_2007, data_targets_2008,
         data_targets_2009, data_targets_2010, data_targets_2011, data_targets_2013, data_targets_2015,
         data_targets_2016])
    train_data_features = pd.concat(
        [sh_data_features_train,sh_data_features_2011, data_features_2005, data_features_2006, data_features_2007, data_features_2008,
         data_features_2009, data_features_2010, data_features_2011, data_features_2013, data_features_2015,
         data_features_2016])

    valid_data_targets = pd.concat([data_targets_2012])
    valid_data_features = pd.concat([data_features_2012])

    test_data_targets = pd.concat([data_targets_2014, sh_data_targets_2014,sh_data_targets_test])
    test_data_features = pd.concat([data_features_2014, sh_data_targets_2014,sh_data_features_test])

    print('训练集平均数/中位数:'+str(round(np.nanmean(train_data_features['KP']),2))+','+str(round(np.nanmedian(train_data_features['KP']),2)))
    print('验证集平均数/中位数:'+str(round(np.nanmean(valid_data_features['KP']),2))+'-'+str(round(np.nanmedian(valid_data_features['KP']),2)))
    print('测试集平均数/中位数:'+str(round(np.nanmean(test_data_features['KP']),2))+'-'+str(round(np.nanmedian(test_data_features['KP']),2)))

    #=====combine features===========
    # combine_features = pd.concat([data_targets['nh_mlat'], data_targets['nh_mlt'],data_features['KP'],data_features['OMNI_BX'],data_features['OMNI_BY'],data_features['OMNI_BZ'],data_features['FLOW_SPEED'],data_features['PRESSURE'],data_features['DENSITY']], axis=1)
    # combine_features = pd.concat([data_targets_2016['nh_mlat'], data_targets_2016['nh_mlt'],data['KP']], axis=1)
    # combine_targets = pd.concat([data_targets['nh_mlat'], data_targets['nh_mlt'],data_targets['nh_radiance_lbhs']], axis=1) # 注意是两个相互独立的输出标签

    train_combine_features = pd.concat([train_data_targets['nh_mlat'], train_data_targets['nh_mlt'],train_data_features['KP']], axis=1)
    train_combine_targets = pd.concat([train_data_targets['nh_mlat'], train_data_targets['nh_mlt'],train_data_targets['nh_radiance_lbhs']], axis=1)

    valid_combine_features= pd.concat([valid_data_targets['nh_mlat'], valid_data_targets['nh_mlt'],valid_data_features['KP']], axis=1)
    valid_combine_targets = pd.concat([valid_data_targets['nh_mlat'], valid_data_targets['nh_mlt'],valid_data_targets['nh_radiance_lbhs']], axis=1)

    test_combine_features= pd.concat([test_data_targets['nh_mlat'], test_data_targets['nh_mlt'],test_data_features['KP']], axis=1)
    test_combine_targets = pd.concat([test_data_targets['nh_mlat'], test_data_targets['nh_mlt'],test_data_targets['nh_radiance_lbhs']], axis=1)

    return train_combine_features,train_combine_targets,valid_combine_features,valid_combine_targets,test_combine_features,test_combine_targets
#======= plot train_kp data======
train_combine_features, train_combine_targets, valid_combine_features, valid_combine_targets, test_combine_features, test_combine_targets = data_process();
fig = plt.figure(figsize=(12, 12))
hf1 = fig.add_subplot(331)
x=train_combine_features['KP']
ax=plt.hist(x,alpha=0.5)  #直方图关键操作
plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看
meankp=round(np.nanmean(x),1);
median=round(np.nanmedian(x));
plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
plt.xlabel('Kp*10')
plt.ylabel("Number of occurrences")
plt.title('Train set mean-median='+str(meankp)+'-'+str(median))

hf1 = fig.add_subplot(332)
x=valid_combine_features['KP']
ax=plt.hist(x,alpha=0.5)  #直方图关键操作
plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看
meankp=round(np.nanmean(x),1);
median=round(np.nanmedian(x));
plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
plt.xlabel('Kp*10')
plt.ylabel("Number of occurrences")
plt.title('Valid Set mean-median='+str(meankp)+'-'+str(median))


hf1 = fig.add_subplot(333)
x=test_combine_features['KP']
ax=plt.hist(x,alpha=0.5)  #直方图关键操作
plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看
meankp=round(np.nanmean(x),1);
median=round(np.nanmedian(x));
plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
plt.xlabel('Kp*10')
plt.ylabel(("Number of occurrences"))
plt.title('Test set mean-median='+str(meankp)+'-'+str(median))

hf1 = fig.add_subplot(3, 3, 4,projection='polar')
hf1 = polar_map(hf1)
mlt_test=np.array(train_combine_features['nh_mlt'])
mlat_test = np.array(train_combine_features['nh_mlat'])
# rho = (90 - abs(np.array(mlat_test)));
# theta = ((np.array(mlt_test) * 15 + 270) * np.pi / 180);
# im = plt.scatter(theta, rho,marker='o', s=5,alpha=0.5)
# plt.hist2d(mlt_test,mlat_test,bins=(12,20),cmap = "GnBu",  norm =  colors.PowerNorm(0.9))
rho_test = (90 - abs(np.array(mlat_test)));
theta_test = ((np.array(mlt_test) * 15 + 270) * np.pi / 180);
theta=((np.linspace(0, 24, 24)* 15 + 270) * np.pi / 180);
r=np.linspace(0,40,20)
r,theta=np.meshgrid(r,theta)
Z = np.zeros((24,20))      #随机生成填入网格的数据，根据网格随机生成一个30,100的矩阵
for row in range(0,24):
    for colum in range(0,20):
        theta_span=theta[row,colum]
        r_span=r[row,colum]
        count=np.sum((theta_test>=theta_span)&(theta_test<theta_span+0.27318)&(rho_test>=r_span)&(rho_test<r_span+2.1052631))/len(theta_test)
        Z[row,colum]=count
im=hf1.pcolormesh(theta, r, np.log10(Z),  cmap='Blues', edgecolor='black', linewidth=0.2)
c=fig.colorbar(im,ax=hf1)      #显示colorbar
c.set_label('Occurrence Rate [log]')
txt = 'Training Set';
plt.title(txt, color='black', fontsize=10)
# plt.xlabel('MLT', fontsize=10)
# plt.ylabel('MLAT', fontsize=10)


# cbar.set_ticks([10, 100, 1000,10000])
# cbar.set_ticklabels(['10^1', '10^2', '10^3', '10^4'], fontsize=10)

hf1 = fig.add_subplot(3, 3, 5,projection='polar')
hf1 = polar_map(hf1)
mlt_test = np.array(valid_combine_features['nh_mlt'])
mlat_test = np.array(valid_combine_features['nh_mlat'])
# rho = (90 - abs(np.array(mlat_test)));
# theta = ((np.array(mlt_test) * 15 + 270) * np.pi / 180);
# im = plt.scatter(theta, rho, marker='o', s=5,alpha=0.5)
# plt.hist2d(mlt_test,mlat_test,bins=(12,20),cmap = "GnBu",  norm =  colors.PowerNorm(0.9))
rho_test = (90 - abs(np.array(mlat_test)));
theta_test = ((np.array(mlt_test) * 15 + 270) * np.pi / 180);
theta=((np.linspace(0, 24, 24)* 15 + 270) * np.pi / 180);
r=np.linspace(0,40,20)
r,theta=np.meshgrid(r,theta)
Z = np.zeros((24,20))      #随机生成填入网格的数据，根据网格随机生成一个30,100的矩阵
for row in range(0,24):
    for colum in range(0,20):
        theta_span=theta[row,colum]
        r_span=r[row,colum]
        count=np.sum((theta_test>=theta_span)&(theta_test<theta_span+0.27318)&(rho_test>=r_span)&(rho_test<r_span+2.1052631))/len(theta_test)
        Z[row,colum]=count
im=hf1.pcolormesh(theta, r, np.log10(Z), cmap='Blues', edgecolor='black', linewidth=0.2)
c=fig.colorbar(im,ax=hf1)        #显示colorbar
c.set_label('Occurrence Rate [log]')
txt = 'Validation Set';
plt.title(txt, color='black', fontsize=10)
# plt.xlabel('MLT', fontsize=10)
# plt.ylabel('MLAT', fontsize=10)

# cbar.set_ticks([10, 100, 1000,10000])
# cbar.set_ticklabels(['10^1', '10^2', '10^3', '10^4'], fontsize=10)

hf1 = fig.add_subplot(3, 3, 6,projection='polar')
hf1 = polar_map(hf1)
mlt_test = np.array(test_combine_features['nh_mlt'])
mlat_test = np.array(test_combine_features['nh_mlat'])
# rho = (90 - abs(np.array(mlat_test)));
# theta = ((np.array(mlt_test) * 15 + 270) * np.pi / 180);
# im = plt.scatter(theta, rho, marker='o', s=5,alpha=0.5)
# plt.hist2d(mlt_test,mlat_test,bins=(12,20),cmap = "GnBu",  norm = colors.PowerNorm(0.9))
rho_test = (90 - abs(np.array(mlat_test)));
theta_test = ((np.array(mlt_test) * 15 + 270) * np.pi / 180);
theta=((np.linspace(0, 24, 24)* 15 + 270) * np.pi / 180);
r=np.linspace(0,40,20)
r,theta=np.meshgrid(r,theta)
Z = np.zeros((24,20))      #随机生成填入网格的数据，根据网格随机生成一个30,100的矩阵
for row in range(0,24):
    for colum in range(0,20):
        theta_span=theta[row,colum]
        r_span=r[row,colum]
        count=np.sum((theta_test>=theta_span)&(theta_test<theta_span+0.27318)&(rho_test>=r_span)&(rho_test<r_span+2.1052631))/len(theta_test)
        Z[row,colum]=count
im=hf1.pcolormesh(theta, r, np.log10(Z), cmap='Blues', edgecolor='black', linewidth=0.2)
c=fig.colorbar(im,ax=hf1)       #显示colorbar
c.set_label('Occurrence Rate [log]')
txt = 'Test Set';
plt.title(txt, color='black', fontsize=10)
# plt.xlabel('MLT', fontsize=10)
# plt.ylabel('MLAT', fontsize=10)

# cbar.set_ticks([10, 100, 1000,10000])
# cbar.set_ticklabels(['10^1', '10^2', '10^3', '10^4'], fontsize=10)
# cbar.set_ticks([1, 2, 3,4,5,6])
# cbar.set_ticklabels(['10^1', '10^2', '10^3'], fontsize=10)

plt.show( )
plt.tight_layout( )
plt.savefig(
    'E:\\Feng_paperwork\\20_aurora_forcast\\paper_figures\\' + 'compare_train_valid_test_set_kp_distribution_add_nigheside_data_event_location_03' + '.pdf',
    bbox_inches='tight', format='pdf',dpi=600)

plt.savefig(
    'E:\\Feng_paperwork\\20_aurora_forcast\\paper_figures\\' + 'compare_train_valid_test_set_kp_distribution_add_nigheside_data_event_location_03' + '.png',
    bbox_inches='tight', format='png',dpi=600)

# hf1 = fig.add_subplot(334)
# x=data_features_2013['KP']
# plt.hist(x)  #直方图关键操作
# plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看
# meankp=round(np.nanmean(x),1);
# median=round(np.nanmedian(x));
# plt.xlabel('Kp')
# plt.ylabel('Number of Events')
# plt.title('2013 mean-median'+str(meankp)+'-'+str(median))
#
# hf1 = fig.add_subplot(335)
# x=data_features_2014['KP']
# plt.hist(x)  #直方图关键操作
# plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看
# meankp=round(np.nanmean(x),1);
# median=round(np.nanmedian(x));
# plt.xlabel('Kp')
# plt.ylabel('Number of Events')
# plt.title('2014 mean-median'+str(meankp)+'-'+str(median))
#
# hf1 = fig.add_subplot(336)
# x=data_features_2015['KP']
# plt.hist(x)  #直方图关键操作
# plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看
# meankp=round(np.nanmean(x),1);
# median=round(np.nanmedian(x));
# plt.xlabel('Kp')
# plt.ylabel('Number of Events')
# plt.title('2015 mean-median'+str(meankp)+'-'+str(median))
#
# hf1 = fig.add_subplot(337)
# x=data_features_2016['KP']
# plt.hist(x)  #直方图关键操作
# plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看
# meankp=round(np.nanmean(x),1);
# median=round(np.nanmedian(x));
# plt.xlabel('Kp')
# plt.ylabel('Number of Events')
# plt.title('2016 mean-median'+str(meankp)+'-'+str(median))