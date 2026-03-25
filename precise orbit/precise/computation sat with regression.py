# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:15:25 2021

@author: fkarlitepe
"""
import pandas as pd
import numpy as np

sp3=pd.read_csv("SHA0MGXRAP_20200330000_01D_15M_ORB.SP3", header=None)
sp3=np.array(sp3.iloc[32:-1])

for i in range(len(sp3)):
    if sp3[i][0].startswith('PG'):
        sp3[i][0]=sp3[i][0].replace('    ',' ')
        sp3[i][0]=sp3[i][0].replace('   ',' ')
        sp3[i][0]=sp3[i][0].replace('  ',' ')
        sp3[i][0]=sp3[i][0].replace('  ',' ')
        sp3[i][0]=sp3[i][0].replace('  ',' ')
sp3_=np.array([])
for i in range(len(sp3)):
    if sp3[i][0].startswith('PG'):
        sp3_=np.append(sp3_,sp3[i][0][2:50].split(' '))
        
kk=np.where(sp3_!='')
sp3_=sp3_[kk].reshape(int(len(sp3_)/5),5).astype(float)
time=np.arange(0,85500+900,900).reshape(96,1)
# time_=np.array([])
# for i in time:
#     time_=np.append(time_,np.full(32,i))
# sp3=np.c_[time_,sp3_]

#Random Forest Regressor ################################################ 
satRF=np.array([])
from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

for j in range(1,85500):# j time sutunun sonu
    for i in range(1,33):# i uydu sayısını döndürecek
        kk1=np.where(sp3_[:,0]==i)
        sat=np.c_[time,sp3_[kk1]]
    
        lin_reg=LinearRegression()
        lin_reg.fit(sat[:,0:1], sat[:,0:1]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_epok=lin_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz        
        
       
        # sat1_x_train,sat1_x_test,sat1_y_train,sat1_y_test=train_test_split(sat1[:,0:1],sat1[:,2:5],test_size=0.33,random_state=0)
        rf_reg=RandomForestRegressor(n_estimators=150, random_state=0)        
        rf_reg.fit(sat[:,0:1], sat[:,2:5]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_xyz=rf_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz

        rf_reg=RandomForestRegressor(n_estimators=150, random_state=0) 
        rf_reg.fit(sat[:,0:1], sat[:,5:6]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_t=rf_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz
        
        sat1=np.c_[sat1_epok,i,sat1_xyz,sat1_t]  
   
        satRF=np.append(satRF,sat1)
    satRF=satRF.reshape(int(len(satRF)/6),6)
    satRF[:,2:5]=satRF[:,2:5]*1000



####SVM Regressor########################################################
satSVM=np.array(())        
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


for j in range(1,85500):
    for i in range(1,33):
        kk1=np.where(sp3_[:,0]==i)
        sat=np.c_[time,sp3_[kk1]]

        sc1=StandardScaler()
        x_olcekli=sc1.fit_transform(sat[:,2:3])
        sc2=StandardScaler()
        y_olcekli=sc2.fit_transform(sat[:,3:4])
        sc3=StandardScaler()
        z_olcekli=sc3.fit_transform(sat[:,4:5])

        
        sat_scaler=np.c_[x_olcekli,y_olcekli,z_olcekli]
        
        lin_reg=LinearRegression()
        lin_reg.fit(sat[:,0:1], sat[:,0:1]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_epok=lin_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz  
        
        svr_reg=SVR(kernel="rbf")
        # svr_reg=SVR(kernel="linear")
        # svr_reg=SVR(kernel="poly")
        # svr_reg=SVR(kernel="sigmoid")
        # svr_reg=SVR(kernel="precomputed")
        svr_reg.fit(sat[:,0:1], sat_scaler[:,0:1]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_x=svr_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz
        
        svr_reg=SVR(kernel="rbf")
        # svr_reg=SVR(kernel="linear")
        # svr_reg=SVR(kernel="poly")
        # svr_reg=SVR(kernel="sigmoid")
        # svr_reg=SVR(kernel="precomputed")
        svr_reg.fit(sat[:,0:1], sat_scaler[:,1:2]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_y=svr_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz
        
        svr_reg=SVR(kernel="rbf")
        # svr_reg=SVR(kernel="linear")
        # svr_reg=SVR(kernel="poly")
        # svr_reg=SVR(kernel="sigmoid")
        # svr_reg=SVR(kernel="precomputed")
        svr_reg.fit(sat[:,0:1], sat_scaler[:,2:3]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_z=svr_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz
        
        lin_reg=LinearRegression()
        lin_reg.fit(sat[:,0:1], sat[:,5:6]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_t=lin_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz
        
        sat1=np.c_[sat1_epok,i,sc1.inverse_transform(sat1_x.reshape(1,1)),sc2.inverse_transform(sat1_y.reshape(1,1)),sc3.inverse_transform(sat1_z.reshape(1,1)),sat1_t]
        
        
        satSVM=np.append(satSVM,sat1)
    satSVM=satSVM.reshape(int(len(satSVM)/6),6)
    satSVM[:,2:5]=satSVM[:,2:5]*1000
####KNeighbors Regression####################################################
satKNN=np.array([])       
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

for j in range(1,85500):
    for i in range(1,33):
        kk1=np.where(sp3_[:,0]==i)
        sat=np.c_[time,sp3_[kk1]]

        lin_reg=LinearRegression()
        lin_reg.fit(sat[:,0:1], sat[:,0:1]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_epok=lin_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz        
        
       
        # sat1_x_train,sat1_x_test,sat1_y_train,sat1_y_test=train_test_split(sat1[:,0:1],sat1[:,2:5],test_size=0.33,random_state=0)
        kn_reg=KNeighborsRegressor(n_neighbors=12)        
        kn_reg.fit(sat[:,0:1], sat[:,2:3]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_x=kn_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz
        
        kn_reg=KNeighborsRegressor(n_neighbors=12)        
        kn_reg.fit(sat[:,0:1], sat[:,3:4]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_y=kn_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz
        
        kn_reg=KNeighborsRegressor(n_neighbors=12)         
        kn_reg.fit(sat[:,0:1], sat[:,4:5]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_z=kn_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz

        lin_reg=LinearRegression()
        lin_reg.fit(sat[:,0:1], sat[:,5:6]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_t=lin_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz
        
        sat1=np.c_[sat1_epok,i,sat1_x,sat1_y,sat1_z,sat1_t]  
   
        satKNN=np.append(satKNN,sat1)
    satKNN=satKNN.reshape(int(len(satKNN)/6),6)
    satKNN[:,2:5]=satKNN[:,2:5]*1000

####MLP Regression############################################################ 
satMLP=np.array([])     
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split        
from sklearn.linear_model import LinearRegression

for j in range(1,85500):
    for i in range(1,33):
        kk1=np.where(sp3_[:,0]==i)
        sat=np.c_[time,sp3_[kk1]]

        lin_reg=LinearRegression()
        lin_reg.fit(sat[:,0:1], sat[:,0:1]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_epok=lin_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz       
        
       
        sat_t_train,sat_t_test,sat_x_train,sat_x_test=train_test_split(sat[:,0:1],sat[:,2:3],test_size=0.33,random_state=0)
        mlp_reg=MLPRegressor(random_state=1,max_iter=500,)       
        mlp_reg.fit(sat_t_train, sat_x_train) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_x=mlp_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz
        
        sat_t_train,sat_t_test,sat_y_train,sat_y_test=train_test_split(sat[:,0:1],sat[:,3:4],test_size=0.33,random_state=0)
        mlp_reg=MLPRegressor(random_state=1,max_iter=500)       
        mlp_reg.fit(sat_t_train, sat_y_train) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_y=mlp_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz
        
        sat_t_train,sat_t_test,sat_z_train,sat_z_test=train_test_split(sat[:,0:1],sat[:,4:5],test_size=0.33,random_state=0)
        mlp_reg=MLPRegressor(random_state=1,max_iter=500)       
        mlp_reg.fit(sat_t_train, sat_z_train) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_z=mlp_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz

        lin_reg=LinearRegression()       
        lin_reg.fit(sat[:,0:1], sat[:,5:6]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_t=lin_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz
        
        sat1=np.c_[sat1_epok,i,sat1_x,sat1_y,sat1_z,sat1_t]  
   
        satMLP=np.append(satMLP,sat1)
    satMLP=satMLP.reshape(int(len(satMLP)/6),6)
    satMLP[:,2:5]=satMLP[:,2:5]*1000

######Lagrance interpolation#################################################

satLag=np.array([])    
from sklearn.linear_model import LinearRegression
from scipy.interpolate import lagrange
from sklearn.preprocessing import StandardScaler

for j in range(1,85500):
    for i in range(1,33):
        kk1=np.where(sp3_[:,0]==i)
        sat=np.c_[time,sp3_[kk1]]
      
        sc1=StandardScaler()
        x_olcekli=sc1.fit_transform(sat[:,2:3])
        sc2=StandardScaler()
        y_olcekli=sc2.fit_transform(sat[:,3:4])
        sc3=StandardScaler()
        z_olcekli=sc3.fit_transform(sat[:,4:5])
       
        sat_scaler=np.c_[x_olcekli,y_olcekli,z_olcekli]

        lin_reg=LinearRegression()       
        lin_reg.fit(sat[:,0:1], sat[:,0:1]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_epok=lin_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz        
        
        
        
        for k in time:
            if 0<=k-j<=900:
                uuu=np.where(time==k)[0][0]
                if uuu-10<0:              
                
                    lag=lagrange(sat[0:uuu+10,0],sat_scaler[0:uuu+10,0])
                    sat1_x=lag(j)
            
                    lag=lagrange(sat[0:uuu+10,0],sat_scaler[0:uuu+10,1])
                    sat1_y=lag(j)        
            
                    lag=lagrange(sat[0:uuu+10,0],sat_scaler[0:uuu+10,2])
                    sat1_z=lag(j)

                    break
                    # lag=lagrange(sat1[0:uuu+10,0],sat1[:,2])
                    # sat2_x=lag(j)
            
                    # lag=lagrange(sat1[0:uuu+10,0],sat1[:,3])
                    # sat2_y=lag(j)        
            
                    # lag=lagrange(sat1[0:uuu+10,0],sat1[:,4])
                    # sat2_z=lag(j)

                    # break
                
                    
                else:
               
                    lag=lagrange(sat[uuu-4:uuu+4,0],sat_scaler[uuu-4:uuu+4,0])
                    sat1_x=lag(j)
            
                    lag=lagrange(sat[uuu-4:uuu+4,0],sat_scaler[uuu-4:uuu+4,1])
                    sat1_y=lag(j)        
            
                    lag=lagrange(sat[uuu-4:uuu+4,0],sat_scaler[uuu-4:uuu+4,2])
                    sat1_z=lag(j)

                    break
                    # lag=lagrange(sat1[uuu-10:uuu+10,0],sat1[:,2])
                    # sat2_x=lag(j)
            
                    # lag=lagrange(sat1[uuu-10:uuu+10,0],sat1[:,3])
                    # sat2_y=lag(j)        
            
                    # lag=lagrange(sat1[uuu-10:uuu+10,0],sat1[:,4])
                    # sat2_z=lag(j)

                    # break
            
        lin_reg=LinearRegression()       
        lin_reg.fit(sat[:,0:1], sat[:,5:6]) #sat verisi içerisinden öğrenerek tahminleme yaptık
        sat1_t=lin_reg.predict([[j]]) #hangi epoğu tahmin edeceksek o saniyeyi giriyoruz

        sat1=np.c_[sat1_epok,i,sc1.inverse_transform(sat1_x.reshape(1,1)),sc2.inverse_transform(sat1_y.reshape(1,1)),sc3.inverse_transform(sat1_z.reshape(1,1)),sat1_t]
        # sat2=np.c_[sat2_epok,kk7[i],sat2_x,sat2_y,sat2_z,sat2_t]
   
        satLag=np.append(satLag,sat1)
    satLag=satLag.reshape(int(len(satLag)/6),6)
    satLag[:,2:5]=satLag[:,2:5]*1000