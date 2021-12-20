import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
#check the working directory
os.getcwd()
'C:\\Users\\U05458\\PycharmProjects'
#read dataset
data = pd.read_csv('train.csv')
data.head()


#Data PreProcessing-----------------------------------------------------------------------------------
#Check row and column count
print('Total data rows:',data.shape[0],'and total columns:',data.shape[1])
Total data rows: 15000 and total columns: 50
#check missing record
print('Missing record:',data.isnull().sum())
Missing record: TUKETIM_M1                   0
TUKETIM_M2                   0
TUKETIM_M3                   0
TUKETIM_M4                   0
TUKETIM_M5                   0
TUKETIM_M6                   0
TUKETIM_M7                   0
TUKETIM_M8                   0
TUKETIM_M9                   0
TUKETIM_M10                  0
TUKETIM_M11                  0
TUKETIM_M12                  0
TUKETIM_M13                 44
TUKETIM_M14                101
TUKETIM_M15                155
TUKETIM_M16                200
TUKETIM_M17                254
TUKETIM_M18                293
TUKETIM_M19                336
TUKETIM_M20                392
TUKETIM_M21                432
TUKETIM_M22                485
TUKETIM_M23                544
TUKETIM_M24                640
DEMAND_M1                   71
DEMAND_M2                 2388
DEMAND_M3                  783
DEMAND_M4                 2369
DEMAND_M5                  945
DEMAND_M6                 2582
DEMAND_M7                 1022
DEMAND_M8                 2757
DEMAND_M9                 1234
DEMAND_M10                2758
DEMAND_M11                1308
DEMAND_M12                3175
SOB_RISK_SKORU               0
SOKAK_RISK_SKORU             0
MAHALLE_RISK_SKORU           0
TESISAT_TIPI                 0
SAYAC_BASLANGIC_TARIHI      50
SAYAC_BITIS_TARIHI          50
SAYAC_MARKA                 50
SAYAC_MODEL                 50
SAYAC_MALZEME_ID            50
SAYAC_OLCUM_TURU            50
SAYAC_FAZ_N                 50
SAYAC_TAKILMA_TARIHI        50
SAYAC_YAPIM_YILI            50
NK_FLAG                      0
dtype: int64
print('Missing record ratio in set:',data.isnull().sum()/(data.shape[0]))
Missing record ratio in set: TUKETIM_M1                0.000000
TUKETIM_M2                0.000000
TUKETIM_M3                0.000000
TUKETIM_M4                0.000000
TUKETIM_M5                0.000000
TUKETIM_M6                0.000000
TUKETIM_M7                0.000000
TUKETIM_M8                0.000000
TUKETIM_M9                0.000000
TUKETIM_M10               0.000000
TUKETIM_M11               0.000000
TUKETIM_M12               0.000000
TUKETIM_M13               0.002933
TUKETIM_M14               0.006733
TUKETIM_M15               0.010333
TUKETIM_M16               0.013333
TUKETIM_M17               0.016933
TUKETIM_M18               0.019533
TUKETIM_M19               0.022400
TUKETIM_M20               0.026133
TUKETIM_M21               0.028800
TUKETIM_M22               0.032333
TUKETIM_M23               0.036267
TUKETIM_M24               0.042667
DEMAND_M1                 0.004733
DEMAND_M2                 0.159200
DEMAND_M3                 0.052200
DEMAND_M4                 0.157933
DEMAND_M5                 0.063000
DEMAND_M6                 0.172133
DEMAND_M7                 0.068133
DEMAND_M8                 0.183800
DEMAND_M9                 0.082267
DEMAND_M10                0.183867
DEMAND_M11                0.087200
DEMAND_M12                0.211667
SOB_RISK_SKORU            0.000000
SOKAK_RISK_SKORU          0.000000
MAHALLE_RISK_SKORU        0.000000
TESISAT_TIPI              0.000000
SAYAC_BASLANGIC_TARIHI    0.003333
SAYAC_BITIS_TARIHI        0.003333
SAYAC_MARKA               0.003333
SAYAC_MODEL               0.003333
SAYAC_MALZEME_ID          0.003333
SAYAC_OLCUM_TURU          0.003333
SAYAC_FAZ_N               0.003333
SAYAC_TAKILMA_TARIHI      0.003333
SAYAC_YAPIM_YILI          0.003333
NK_FLAG                   0.000000
dtype: float64
nulls=data.loc[:,['DEMAND_M2','DEMAND_M4','DEMAND_M6','DEMAND_M8','DEMAND_M10','DEMAND_M12']]
nulls


nulls.hist()
array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000001E70C920688>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x000001E70C93ADC8>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x000001E70CB89D88>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x000001E6C8996408>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x000001E6C89C8A48>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x000001E6C8A010C8>]],
      dtype=object)

#drop the columns which has got null values more than 15 percent
percentage = 15
min_count = int(((100-percentage)/100)*data.shape[0] + 1)
mod_df = data.dropna( axis = 1, thresh=min_count)

print(mod_df)
   
INDEX = pd.Series(np.arange(0,15000,1))
mod_df['INDEX'] = INDEX
C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
 
#fill the null values with 0

mod_df = mod_df.replace(np.nan, 0 )


print('Missing record:',mod_df.isnull().sum())



mod_df = mod_df.drop(columns=['SAYAC_BITIS_TARIHI'])
#specify datetime column
from datetime import datetime


mod_df['SAYAC_YAPIM_YILI'] = mod_df['SAYAC_YAPIM_YILI'].astype(int).round()

mod_df['SAYAC_YAPIM_YILI'][mod_df['SAYAC_YAPIM_YILI'] == 0] = 1970

mod_df['SAYAC_YAPIM_YILI'] = mod_df['SAYAC_YAPIM_YILI'].astype(str)

mod_df['SAYAC_YAPIM_YILI'] = pd.to_datetime(mod_df['SAYAC_YAPIM_YILI'])



mod_df['SAYAC_BASLANGIC_TARIHI'] = pd.to_datetime(mod_df['SAYAC_BASLANGIC_TARIHI'])
mod_df[mod_df['SAYAC_YAPIM_YILI'] == 1970]

mod_df
C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:7: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  import sys


#compute the meter occupancy attribute
from datetime import date
today = date.today()

system_dater = today.strftime("%Y-%m-%d")

system_dater

mod_df['system_dater'] = pd.to_datetime(system_dater)


delta = (mod_df['system_dater'] - mod_df['SAYAC_BASLANGIC_TARIHI']).dt.days
SAYAC_SURESI=delta


mod_df=mod_df.drop(columns=['SAYAC_BASLANGIC_TARIHI','system_dater'])

mod_df['SAYAC_SURESI'] = SAYAC_SURESI

mod_df


#since i will use the dataset in classificion models i will do one-hot-encoding on the categorical variables
mod_df= pd.concat([mod_df,pd.get_dummies(mod_df['SAYAC_MARKA'],prefix='SAYAC_MARKA',dummy_na = 'TRUE')],axis=1)
mod_df =mod_df.drop(columns = ['SAYAC_MARKA_nan','SAYAC_MARKA'])
mod_df= pd.concat([mod_df,pd.get_dummies(mod_df['TESISAT_TIPI'],prefix='TESISAT_TIPI',dummy_na = 'TRUE')],axis=1)
mod_df = mod_df.drop(columns = ['TESISAT_TIPI_nan','TESISAT_TIPI'])
mod_df= pd.concat([mod_df,pd.get_dummies(mod_df['SAYAC_MODEL'],prefix='SAYAC_MODEL',dummy_na = 'TRUE')],axis=1)
mod_df = mod_df.drop(columns = ['SAYAC_MODEL_nan','SAYAC_MODEL'])
mod_df= pd.concat([mod_df,pd.get_dummies(mod_df['SAYAC_MALZEME_ID'],prefix='SAYAC_MALZEME_ID',dummy_na = 'TRUE')],axis=1)
mod_df = mod_df.drop(columns = ['SAYAC_MALZEME_ID_nan','SAYAC_MALZEME_ID'])
mod_df= pd.concat([mod_df,pd.get_dummies(mod_df['SAYAC_OLCUM_TURU'],prefix='SAYAC_OLCUM_TURU',dummy_na = 'TRUE')],axis=1)
mod_df = mod_df.drop(columns = ['SAYAC_OLCUM_TURU_nan','SAYAC_OLCUM_TURU'])
mod_df= pd.concat([mod_df,pd.get_dummies(mod_df['SAYAC_FAZ_N'],prefix='SAYAC_FAZ_N',dummy_na = 'TRUE')],axis=1)
mod_df = mod_df.drop(columns = ['SAYAC_FAZ_N_nan','SAYAC_FAZ_N'])
mod_df_imbalanced = mod_df
#Scaling

mod_continuous = mod_df_imbalanced.iloc[:, 0:33]
mod_continuous


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

mod_scaled = pd.DataFrame(scaler.fit_transform(mod_continuous), columns=mod_continuous.columns)

print(mod_scaled)

mod_scaled.to_csv("mod_scaled.csv")
mod_df_imbalanced = pd.concat([mod_scaled,mod_df_imbalanced.iloc[:, 34:210],],axis=1)
mod_df_imbalanced


mod_df = mod_df_imbalanced
#correlation analysis
corr=mod_df.iloc[:,:].corr().sort_values(by='NK_FLAG',ascending=False).round(2)
corr

abscorr = corr[abs(corr['NK_FLAG']) > 0.05].index
abscorr
Index(['NK_FLAG', 'SOB_RISK_SKORU', 'SAYAC_MALZEME_ID_80001006.0',
       'SAYAC_MODEL_LUN10-B', 'SAYAC_MARKA_LUNA', 'SOKAK_RISK_SKORU',
       'SAYAC_MALZEME_ID_80000855.0', 'SAYAC_MODEL_LSM10-BUZ',
       'MAHALLE_RISK_SKORU', 'TUKETIM_M11', 'SAYAC_MODEL_LSM40-BUZ-KOM',
       'SAYAC_MALZEME_ID_80000857.0', 'SAYAC_MALZEME_ID_0.0',
       'SAYAC_OLCUM_TURU_0', 'SAYAC_FAZ_N_0', 'SAYAC_MARKA_0', 'SAYAC_MODEL_0',
       'TUKETIM_M6', 'TUKETIM_M12', 'TUKETIM_M7', 'TUKETIM_M5', 'TUKETIM_M10',
       'TUKETIM_M3', 'TUKETIM_M9', 'TESISAT_TIPI_Ticarethane-Sanayi',
       'TUKETIM_M14', 'TUKETIM_M13', 'TESISAT_TIPI_Mesken', 'SAYAC_MODEL_LUN1',
       'SAYAC_MALZEME_ID_80000107.0', 'SAYAC_MALZEME_ID_80000731.0',
       'SAYAC_MODEL_EC058MBW', 'SAYAC_OLCUM_TURU_Aktif', 'DEMAND_M3',
       'SAYAC_MARKA_ELEKTROMED'],
      dtype='object')
cm = np.corrcoef(mod_df[abscorr].values.T)
sns.set(font_scale=1)
plt.figure(figsize=(20,20))
hm = sns.heatmap(cm, annot=True,yticklabels=abscorr.values,xticklabels=abscorr.values,cmap='Blues')
plt.show()

#check the multicollinearity between important variables

from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import Parallel, delayed
import time

def calculate_vif_(X, thresh =5.0):
    variables = [X.columns[i] for i in range(X.shape[1])]
    dropped = True
    while dropped:
        dropped=False
        print(len(variables))
        vif = Parallel(n_jobs=-1,verbose=5)(delayed(variance_inflation_factor)(X[variables].values, ix) for ix in range(len(variables)))
       
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print(time.ctime() + ' dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables.pop(maxloc)
            dropped=True
    print('Remaining variables:')
    print([variables])
    return X[[i for i in variables]]

X = mod_df[['SOB_RISK_SKORU', 'SAYAC_MALZEME_ID_80001006.0',
       'SAYAC_MODEL_LUN10-B', 'SAYAC_MARKA_LUNA', 'SOKAK_RISK_SKORU',
       'SAYAC_MALZEME_ID_80000855.0', 'SAYAC_MODEL_LSM10-BUZ',
       'MAHALLE_RISK_SKORU', 'TUKETIM_M11', 'SAYAC_MODEL_LSM40-BUZ-KOM',
       'SAYAC_MALZEME_ID_80000857.0', 'SAYAC_MALZEME_ID_0.0',
       'SAYAC_OLCUM_TURU_0', 'SAYAC_FAZ_N_0', 'SAYAC_MARKA_0', 'SAYAC_MODEL_0',
       'TUKETIM_M6', 'TUKETIM_M12', 'TUKETIM_M7', 'TUKETIM_M5', 'TUKETIM_M10',
       'TUKETIM_M3', 'TUKETIM_M9', 'TESISAT_TIPI_Ticarethane-Sanayi',
       'TUKETIM_M14', 'TUKETIM_M13', 'TESISAT_TIPI_Mesken', 'SAYAC_MODEL_LUN1',
       'SAYAC_MALZEME_ID_80000107.0', 'SAYAC_MALZEME_ID_80000731.0',
       'SAYAC_MODEL_EC058MBW', 'SAYAC_OLCUM_TURU_Aktif', 'DEMAND_M3',
       'SAYAC_MARKA_ELEKTROMED']]

X2 = calculate_vif_(X,5)

Remaining variables:
[['SOB_RISK_SKORU', 'SAYAC_MODEL_LUN10-B', 'SAYAC_MARKA_LUNA', 'SOKAK_RISK_SKORU', 'SAYAC_MODEL_LSM10-BUZ', 'MAHALLE_RISK_SKORU', 'SAYAC_MALZEME_ID_80000857.0', 'SAYAC_MODEL_0', 'TUKETIM_M3', 'TUKETIM_M9', 'TESISAT_TIPI_Ticarethane-Sanayi', 'SAYAC_MALZEME_ID_80000107.0', 'SAYAC_MODEL_EC058MBW', 'DEMAND_M3', 'SAYAC_MARKA_ELEKTROMED']]
[Parallel(n_jobs=-1)]: Done   4 out of  15 | elapsed:    0.0s remaining:    0.1s
[Parallel(n_jobs=-1)]: Done   8 out of  15 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=-1)]: Done  12 out of  15 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=-1)]: Done  15 out of  15 | elapsed:    0.0s finished
mod_df = pd.concat([X2,mod_df['NK_FLAG']],axis=1)
mod_df


#check fraud records

mod_df['NK_FLAG'].value_counts()
0    13980
1     1020
Name: NK_FLAG, dtype: int64
#INDEX = pd.Series(np.arange(0,15000,1))
mod_df['INDEX'] = INDEX
#check the ratio of fraud records
(mod_df.INDEX[mod_df['NK_FLAG'] ==1].count() / mod_df.INDEX[mod_df['NK_FLAG'] == 0].count())*100
7.296137339055794
mod_df = mod_df.drop(columns=['INDEX'])
#Since there is an imbalanced class problem i will analyse the variable

print("fraud")
print(mod_df[mod_df['NK_FLAG']==1].describe())
fraud
       SOB_RISK_SKORU  SAYAC_MODEL_LUN10-B  SAYAC_MARKA_LUNA  \
count     1020.000000          1020.000000       1020.000000  
mean         1.060388             0.452941          0.681373  
std          2.109449             0.498025          0.466173  
min         -0.422024             0.000000          0.000000  
25%         -0.051827             0.000000          0.000000  
50%          0.445744             0.000000          1.000000  
75%          1.428851             1.000000          1.000000  
max         41.542282             1.000000          1.000000  

       SOKAK_RISK_SKORU  SAYAC_MODEL_LSM10-BUZ  MAHALLE_RISK_SKORU  \
count       1020.000000            1020.000000         1020.000000  
mean           0.440026               0.010784            0.360338  
std            0.864450               0.103337            1.518198  
min           -0.113628               0.000000           -0.247804  
25%            0.007250               0.000000           -0.083223  
50%            0.178457               0.000000            0.085945  
75%            0.557788               0.000000            0.524399  
max            8.438083               1.000000           42.684323  

       SAYAC_MALZEME_ID_80000857.0  SAYAC_MODEL_0   TUKETIM_M3   TUKETIM_M9  \
count                  1020.000000    1020.000000  1020.000000  1020.000000  
mean                      0.006863       0.018627     0.231505     0.221479  
std                       0.082597       0.135272     3.341438     3.675551  
min                       0.000000       0.000000    -0.398253    -0.209271  
25%                       0.000000       0.000000    -0.271702    -0.137601  
50%                       0.000000       0.000000    -0.150532    -0.076988  
75%                       0.000000       0.000000     0.062071     0.014997  
max                       1.000000       1.000000    84.599546    88.278407  

       TESISAT_TIPI_Ticarethane-Sanayi  SAYAC_MALZEME_ID_80000107.0  \
count                      1020.000000                  1020.000000  
mean                          0.175490                     0.032353  
std                           0.380572                     0.177022  
min                           0.000000                     0.000000  
25%                           0.000000                     0.000000  
50%                           0.000000                     0.000000  
75%                           0.000000                     0.000000  
max                           1.000000                     1.000000  

       SAYAC_MODEL_EC058MBW    DEMAND_M3  SAYAC_MARKA_ELEKTROMED  NK_FLAG  
count           1020.000000  1020.000000             1020.000000   1020.0  
mean               0.110784    -0.288705                0.198039      1.0  
std                0.314019     1.592151                0.398717      0.0  
min                0.000000    -1.119333                0.000000      1.0  
25%                0.000000    -0.917226                0.000000      1.0  
50%                0.000000    -0.585885                0.000000      1.0  
75%                0.000000    -0.033770                0.000000      1.0  
max                1.000000    33.607191                1.000000      1.0  
print("not fraud")
print(mod_df[mod_df['NK_FLAG'] ==0].describe())
not fraud
       SOB_RISK_SKORU  SAYAC_MODEL_LUN10-B  SAYAC_MARKA_LUNA  \
count    13980.000000         13980.000000      13980.000000  
mean        -0.077367             0.128684          0.421531  
std          0.812798             0.334862          0.493822  
min         -0.474596             0.000000          0.000000  
25%         -0.426992             0.000000          0.000000  
50%         -0.285629             0.000000          0.000000  
75%         -0.015772             0.000000          1.000000  
max         41.542282             1.000000          1.000000  

       SOKAK_RISK_SKORU  SAYAC_MODEL_LSM10-BUZ  MAHALLE_RISK_SKORU  \
count      13980.000000                13980.0        13980.000000  
mean          -0.032105                    0.0           -0.026291  
std            1.001702                    0.0            0.945968  
min           -0.113628                    0.0           -0.247804  
25%           -0.113628                    0.0           -0.208661  
50%           -0.113628                    0.0           -0.149833  
75%           -0.057249                    0.0           -0.053600  
max          105.239330                    0.0           67.373959  

       SAYAC_MALZEME_ID_80000857.0  SAYAC_MODEL_0    TUKETIM_M3    TUKETIM_M9  \
count                      13980.0   13980.000000  13980.000000  13980.000000  
mean                           0.0       0.002217     -0.016891     -0.016159  
std                            0.0       0.047039      0.504929      0.290533  
min                            0.0       0.000000     -0.398253     -0.209271  
25%                            0.0       0.000000     -0.269189     -0.139553  
50%                            0.0       0.000000     -0.117052     -0.067701  
75%                            0.0       0.000000      0.077377      0.026176  
max                            0.0       1.000000     22.932127     13.056099  

 
mod_df.sort_values(by=["NK_FLAG","MAHALLE_RISK_SKORU","SOKAK_RISK_SKORU"])


#explatory analysis
sns.countplot(x="SAYAC_MALZEME_ID_80000857.0",hue="NK_FLAG",data=mod_df)
<matplotlib.axes._subplots.AxesSubplot at 0x1e6e4996108>

sns.countplot(x="SAYAC_MARKA_LUNA",hue="NK_FLAG",data=mod_df)
<matplotlib.axes._subplots.AxesSubplot at 0x1e71b57ff08>

sns.countplot(x="SAYAC_MALZEME_ID_80000107.0",hue="NK_FLAG",data=mod_df)
<matplotlib.axes._subplots.AxesSubplot at 0x1e6c4a80148>

sns.countplot(x="SAYAC_MODEL_LUN10-B",hue="NK_FLAG",data=mod_df)
<matplotlib.axes._subplots.AxesSubplot at 0x1e6abf0aa88>

sns.countplot(x="SAYAC_MARKA_ELEKTROMED",hue="NK_FLAG",data=mod_df)
<matplotlib.axes._subplots.AxesSubplot at 0x1e6c222edc8>

sns.countplot(x="TESISAT_TIPI_Ticarethane-Sanayi",hue="NK_FLAG",data=mod_df)
<matplotlib.axes._subplots.AxesSubplot at 0x1e6c22b96c8>

#data preparation

Y_imbalance = mod_df.iloc[:,15]
Y_imbalance
X_imbalance = mod_df.iloc[:,0:15]
X_imbalance


X_imbalancetrain,X_imbalancetest,Y_imbalancetrain,Y_imbalancetest = train_test_split(X_imbalance,Y_imbalance,
                                                                                    test_size = 0.25,
                                                                                    random_state=123)
X_imbalancetrain

X_imbalancetest


mod_con=mod_df[['SOB_RISK_SKORU','SOKAK_RISK_SKORU','MAHALLE_RISK_SKORU','TUKETIM_M3','TUKETIM_M9','DEMAND_M3','NK_FLAG']]

Y_imbalance_con = mod_con.iloc[:,6]
Y_imbalance_con
X_imbalance_con = mod_con.iloc[:,0:6]
X_imbalance_con
X_imbalancetrain_con,X_imbalancetest_con,Y_imbalancetrain_con,Y_imbalancetest_con = train_test_split(X_imbalance_con,Y_imbalance_con,
                                                                                    test_size = 0.25,
                                                                                    random_state=123)
#machine learning models for imbalance dataset
#naive bayes model
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_imbalancetrain_con,Y_imbalancetrain_con)

train_model_est = model.predict(X_imbalancetrain_con)



print('the_train_accuracy_of_naive_bayes_model:',accuracy_score(Y_imbalancetrain_con, train_model_est))


y_model = model.predict(X_imbalancetest_con)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from matplotlib import pyplot


print('the_accuracy_of_naive_bayes_model:',accuracy_score(Y_imbalancetest_con,y_model))
print('the_precision_of_naive_bayes_model:',precision_score(Y_imbalancetest_con,y_model))
print('the_recall_of_naive_bayes_model:',recall_score(Y_imbalancetest_con,y_model))

print('the_F1_of_naive_bayes_model:',f1_score(Y_imbalancetest_con,y_model))

crosstab=pd.crosstab(Y_imbalancetest_con,y_model,rownames=['   Actual_theft'],colnames=['Estimated_theft'])
print(' ')
print(crosstab)

nb_probs = model.predict_proba(X_imbalancetest_con)

nb_probs
nb_probs = nb_probs[:,1]

nb_auc = roc_auc_score(Y_imbalancetest_con,nb_probs)
nb_auc

nb_fpr , nb_tpr , _ = roc_curve(Y_imbalancetest_con,nb_probs)

pyplot.plot(nb_fpr, nb_tpr, marker = '.' , label = 'naive_bayes')

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
the_train_accuracy_of_naive_bayes_model: 0.9207111111111111
the_accuracy_of_naive_bayes_model: 0.9226666666666666
the_precision_of_naive_bayes_model: 0.3211009174311927
the_recall_of_naive_bayes_model: 0.1394422310756972
the_F1_of_naive_bayes_model: 0.19444444444444445
 
Estimated_theft     0   1
   Actual_theft          
0                3425  74
1                 216  35
Text(0, 0.5, 'True Positive Rate')

#logistic regression model
from sklearn.linear_model import LogisticRegression

LogRmodel = LogisticRegression(penalty='l2',max_iter=50,solver='newton-cg')
LogRmodel.fit(X_imbalancetrain,Y_imbalancetrain)

train_model_est = LogRmodel.predict(X_imbalancetrain)



print('the_train_accuracy_of_logistic_regression_model:',accuracy_score(Y_imbalancetrain, train_model_est))


LogRpredict = LogRmodel.predict(X_imbalancetest)
print('the_accuracy_of_logistic_regression_model:',accuracy_score(Y_imbalancetest,LogRpredict))
print('the_precision_of_logitic_regression_model:',precision_score(Y_imbalancetest,LogRpredict))
print('the_recall_of_logistic_regression_model:',recall_score(Y_imbalancetest,LogRpredict))

print('the_F1_of_logistic_regression_model:',f1_score(Y_imbalancetest,LogRpredict))

crosstab=pd.crosstab(Y_imbalancetest,LogRpredict,rownames=['   Actual_theft'],colnames=['Estimated_theft'])
print(' ')
print(crosstab)

lr_probs = model.predict_proba(X_imbalancetest)

lr_probs
lr_probs = lr_probs[:,1]

lr_auc = roc_auc_score(Y_imbalancetest,lr_probs)
lr_auc

lr_fpr , lr_tpr , _ = roc_curve(Y_imbalancetest,lr_probs)

pyplot.plot(lr_fpr, lr_tpr, marker = '.' , label = 'logistic_regression')


pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
the_train_accuracy_of_logistic_regression_model: 0.9392888888888888
the_accuracy_of_logistic_regression_model: 0.9402666666666667
the_precision_of_logitic_regression_model: 0.7213114754098361
the_recall_of_logistic_regression_model: 0.1752988047808765
the_F1_of_logistic_regression_model: 0.28205128205128205
 
Estimated_theft     0   1
   Actual_theft          
0                3482  17
1                 207  44
Text(0, 0.5, 'True Positive Rate')

#support vector machine
from sklearn import svm
from sklearn.model_selection import GridSearchCV

parameter_grid = {'C': [0.01,0.1 , 1 , 10, 100, 1000],
        'gamma':[1,0.1,0.01, 0.001, 0.0001],
        'kernel' : ['rbf']}

SVMmodel = GridSearchCV(svm.SVC(probability=True), parameter_grid, refit=True)

SVMmodel.fit(X_imbalancetrain,Y_imbalancetrain)
SVMpredict = SVMmodel.predict(X_imbalancetest)

train_model_est = SVMmodel.predict(X_imbalancetrain)



print('the_train_accuracy_of_support_vector_machine_model:',accuracy_score(Y_imbalancetrain, train_model_est))



print('the_accuracy_of_support_vector_machine:',accuracy_score(Y_imbalancetest,SVMpredict))
print('the_precision_of_support_vector_machine:',precision_score(Y_imbalancetest,SVMpredict))
print('the_recall_of_support_vector_machine:',recall_score(Y_imbalancetest,SVMpredict))

print('the_F1_of_support_vector_machine:',f1_score(Y_imbalancetest,SVMpredict))

crosstab=pd.crosstab(Y_imbalancetest,SVMpredict,rownames=['   Actual_theft'],colnames=['Estimated_theft'])
print(' ')
print(crosstab)

svm_probs = SVMmodel.predict_proba(X_imbalancetest)

svm_probs
svm_probs = svm_probs[:,1]

svm_auc = roc_auc_score(Y_imbalancetest,svm_probs)
svm_auc

svm_fpr , svm_tpr , _ = roc_curve(Y_imbalancetest,svm_probs)

pyplot.plot(svm_fpr, svm_tpr, marker = '.' , label = 'support_vector_machine')


pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
  warnings.warn(CV_WARNING, FutureWarning)
the_train_accuracy_of_support_vector_machine_model: 0.9564444444444444
the_accuracy_of_support_vector_machine: 0.94
the_precision_of_support_vector_machine: 0.6181818181818182
the_recall_of_support_vector_machine: 0.27091633466135456
the_F1_of_support_vector_machine: 0.3767313019390581
 
Estimated_theft     0   1
   Actual_theft          
0                3457  42
1                 183  68
Text(0, 0.5, 'True Positive Rate')

#K nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
KNNmodel = KNeighborsClassifier(algorithm='brute',leaf_size=30,metric="minkowski",n_neighbors=9,n_jobs=5,weights='uniform')



KNNmodel.fit(X_imbalancetrain,Y_imbalancetrain)
KNNpredict = KNNmodel.predict(X_imbalancetest)

train_model_est = KNNmodel.predict(X_imbalancetrain)



print('the_train_accuracy_of_KNN_model:',accuracy_score(Y_imbalancetrain, train_model_est))


print('the accuracy of decision KNN:',accuracy_score(Y_imbalancetest,KNNpredict))
print('the precision of decision KNN:',precision_score(Y_imbalancetest,KNNpredict))
print('the recall of decision KNN:',recall_score(Y_imbalancetest,KNNpredict))

print('the F1 of decision KNN:',f1_score(Y_imbalancetest,KNNpredict))
crosstab=pd.crosstab(Y_imbalancetest,KNNpredict,rownames=['   Actual_theft'],colnames=['Estimated_theft'])
print(' ')
print(crosstab)

KNN_probs = KNNmodel.predict_proba(X_imbalancetest)

KNN_probs
KNN_probs = KNN_probs[:,1]

KNN_auc = roc_auc_score(Y_imbalancetest,KNN_probs)
KNN_auc

KNN_fpr , KNN_tpr , _ = roc_curve(Y_imbalancetest,KNN_probs)

pyplot.plot(KNN_fpr, KNN_tpr, marker = '.' , label = 'KNN')


pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
the_train_accuracy_of_KNN_model: 0.9496
the accuracy of decision KNN: 0.9429333333333333
the precision of decision KNN: 0.6728971962616822
the recall of decision KNN: 0.2868525896414343
the F1 of decision KNN: 0.40223463687150834
 
Estimated_theft     0   1
   Actual_theft          
0                3464  35
1                 179  72
Text(0, 0.5, 'True Positive Rate')

#multilayer neural network
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


parameter_grid = {'solver':['lbfgs'],'max_iter':[1000,1100,1200,1300,1400,1500],'alpha':10.0** -np.arange(1,10),
                 'hidden_layer_sizes':np.arange(10,15)}

clf = GridSearchCV(MLPClassifier(), parameter_grid, n_jobs = -1)

clf.fit(X_imbalancetrain,Y_imbalancetrain)

clfpredict = clf.predict(X_imbalancetest)

train_model_est = clf.predict(X_imbalancetrain)



print('the_train_accuracy_of_multilayer_neural_model:',accuracy_score(Y_imbalancetrain, train_model_est))



print('the_accuracy_of_multilayer_neural_model:',accuracy_score(Y_imbalancetest,clfpredict))
print('the_precision_of_multilayer_neural_model:',precision_score(Y_imbalancetest,clfpredict))
print('the_recall_of_multilayer_neural_model:',recall_score(Y_imbalancetest,clfpredict))

print('the_F1_of_multilayer_neural_model:',f1_score(Y_imbalancetest,clfpredict))

crosstab=pd.crosstab(Y_imbalancetest,clfpredict,rownames=['   Actual_theft'],colnames=['Estimated_theft'])
print(' ')
print(crosstab)

clf_probs = clf.predict_proba(X_imbalancetest)

clf_probs
clf_probs = clf_probs[:,1]

clf_auc = roc_auc_score(Y_imbalancetest,clf_probs)
clf_auc

clf_fpr , clf_tpr , _ = roc_curve(Y_imbalancetest,clf_probs)

pyplot.plot(clf_fpr, clf_tpr, marker = '.' , label = 'multilayer_neural')


pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
  warnings.warn(CV_WARNING, FutureWarning)
the_train_accuracy_of_multilayer_neural_model: 0.9548444444444445
the_accuracy_of_multilayer_neural_model: 0.9450666666666667
the_precision_of_multilayer_neural_model: 0.6380368098159509
the_recall_of_multilayer_neural_model: 0.41434262948207173
the_F1_of_multilayer_neural_model: 0.5024154589371982
 
Estimated_theft     0    1
   Actual_theft          
0                3440   59
1                 147  104
Text(0, 0.5, 'True Positive Rate')

#decision tree classifier
from sklearn import tree
DTmodel = tree.DecisionTreeClassifier(criterion="entropy",splitter="random",min_samples_split=30)
DTmodel.fit(X_imbalancetrain,Y_imbalancetrain)

train_model_est = DTmodel.predict(X_imbalancetrain)



print('the_train_accuracy_of_tree_model:',accuracy_score(Y_imbalancetrain, train_model_est))


DTpredict = DTmodel.predict(X_imbalancetest)
print('the accuracy of decision tree:',accuracy_score(Y_imbalancetest,DTpredict))
print('the precision of decision tree:',precision_score(Y_imbalancetest,DTpredict))
print('the recall of decision tree:',recall_score(Y_imbalancetest,DTpredict))

print('the F1 of decision tree:',f1_score(Y_imbalancetest,DTpredict))

crosstab=pd.crosstab(Y_imbalancetest,DTpredict,rownames=['   Actual_theft'],colnames=['Estimated_theft'])
print(' ')
print(crosstab)

dte_probs = DTmodel.predict_proba(X_imbalancetest)

dte_probs
dte_probs = dte_probs[:,1]

dte_auc = roc_auc_score(Y_imbalancetest,dte_probs)
dte_auc

dte_fpr , dte_tpr , _ = roc_curve(Y_imbalancetest,dte_probs)

pyplot.plot(dte_fpr, dte_tpr, marker = '.' , label = 'decision_trees_classifier')


pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
the_train_accuracy_of_tree_model: 0.9544888888888889
the accuracy of decision tree: 0.9301333333333334
the precision of decision tree: 0.4692737430167598
the recall of decision tree: 0.3346613545816733
the F1 of decision tree: 0.39069767441860465
 
Estimated_theft     0   1
   Actual_theft          
0                3404  95
1                 167  84
Text(0, 0.5, 'True Positive Rate')

 
 
 
#working with imbalance dataset (over_sampling)
class_count_0, class_count_1 = mod_df_imbalanced['NK_FLAG'].value_counts()
class_0 = mod_df_imbalanced[mod_df_imbalanced['NK_FLAG'] == 0]
class_1 = mod_df_imbalanced[mod_df_imbalanced['NK_FLAG'] == 1]

print('class_0:', class_0.shape)
print('class_1:', class_1.shape)
class_0: (13980, 209)
class_1: (1020, 209)
class_1_over = class_1.sample(class_count_0,replace=True)
df_over = pd.concat([class_1_over,class_0], axis=0)

df_over['NK_FLAG'].value_counts().plot(kind='bar',title='count(target)')
<matplotlib.axes._subplots.AxesSubplot at 0x1e7167aa708>

df_over['NK_FLAG'].value_counts()
1    13980
0    13980
Name: NK_FLAG, dtype: int64
df_over

#data analysis for balanced dataset
corr=df_over.iloc[:,:].corr().sort_values(by='NK_FLAG',ascending=False).round(2)
corr
abscorr = corr[abs(corr['NK_FLAG']) > 0.05].index
abscorr
cm = np.corrcoef(df_over[abscorr].values.T)
sns.set(font_scale=1)
plt.figure(figsize=(20,20))
hm = sns.heatmap(cm, annot=True,yticklabels=abscorr.values,xticklabels=abscorr.values,cmap='Blues')
plt.show()


abscorr

X = df_over[['SOB_RISK_SKORU', 'SAYAC_MALZEME_ID_80001006.0',
       'SAYAC_MODEL_LUN10-B', 'SAYAC_MARKA_LUNA', 'SOKAK_RISK_SKORU',
       'MAHALLE_RISK_SKORU', 'TESISAT_TIPI_Ticarethane-Sanayi',
       'SAYAC_FAZ_N_0', 'SAYAC_MALZEME_ID_0.0', 'SAYAC_MARKA_0',
       'SAYAC_MODEL_0', 'SAYAC_OLCUM_TURU_0', 'SAYAC_MODEL_LSM10-BUZ',
       'SAYAC_MALZEME_ID_80000855.0', 'TUKETIM_M11',
       'SAYAC_MALZEME_ID_80000857.0', 'SAYAC_MODEL_LSM40-BUZ-KOM',
       'SAYAC_MODEL_M550.2251', 'SAYAC_MALZEME_ID_80000755.0',
       'SAYAC_MODEL_AEL.MF.07', 'SAYAC_MALZEME_ID_80000526.0',
       'SAYAC_MODEL_AEL.TF.15', 'SAYAC_MALZEME_ID_80000089.0',
       'SAYAC_MARKA_KOHLER', 'DEMAND_M5', 'SAYAC_MODEL_EC15ATBW',
       'SAYAC_MALZEME_ID_80000732.0', 'SAYAC_MARKA_MAKEL',
       'SAYAC_OLCUM_TURU_Aktif', 'TESISAT_TIPI_Mesken',
       'SAYAC_MALZEME_ID_80000029.0', 'SAYAC_MODEL_EC018MBW', 'DEMAND_M3',
       'SAYAC_MALZEME_ID_80000107.0', 'SAYAC_MODEL_LUN1',
       'SAYAC_MODEL_EC058MBW', 'SAYAC_MALZEME_ID_80000731.0',
       'SAYAC_MARKA_ELEKTROMED']]

X2 = calculate_vif_(X,5)


Remaining variables:
[['SOB_RISK_SKORU', 'SAYAC_MODEL_LUN10-B', 'SAYAC_MARKA_LUNA', 'SOKAK_RISK_SKORU', 'MAHALLE_RISK_SKORU', 'TESISAT_TIPI_Ticarethane-Sanayi', 'SAYAC_OLCUM_TURU_0', 'SAYAC_MALZEME_ID_80000855.0', 'TUKETIM_M11', 'SAYAC_MODEL_LSM40-BUZ-KOM', 'SAYAC_MALZEME_ID_80000755.0', 'SAYAC_MALZEME_ID_80000526.0', 'SAYAC_MALZEME_ID_80000089.0', 'SAYAC_MARKA_KOHLER', 'DEMAND_M5', 'SAYAC_MALZEME_ID_80000732.0', 'SAYAC_MARKA_MAKEL', 'SAYAC_MODEL_EC018MBW', 'DEMAND_M3', 'SAYAC_MODEL_LUN1', 'SAYAC_MALZEME_ID_80000731.0', 'SAYAC_MARKA_ELEKTROMED']]
[Parallel(n_jobs=-1)]: Done  22 out of  22 | elapsed:    0.4s finished
df_over = pd.concat([X2,df_over['NK_FLAG']],axis=1)
#explatory analysis
sns.countplot(x="SAYAC_MARKA_LUNA",hue="NK_FLAG",data=df_over)
#data preparation
<matplotlib.axes._subplots.AxesSubplot at 0x1e6c033b788>

sns.countplot(x="TESISAT_TIPI_Ticarethane-Sanayi",hue="NK_FLAG",data=df_over)
<matplotlib.axes._subplots.AxesSubplot at 0x1e709207988>

sns.countplot(x="SAYAC_MODEL_LUN10-B",hue="NK_FLAG",data=df_over)
<matplotlib.axes._subplots.AxesSubplot at 0x1e6c032cec8>

df_over

Y_balance = df_over.iloc[:,22]
Y_balance

X_balance = df_over.iloc[:,0:22]
X_balance


X_balancetrain,X_balancetest,Y_balancetrain,Y_balancetest = train_test_split(X_balance,Y_balance,
                                                                                    test_size = 0.25,
                                                                                    random_state=123)
X_balancetrain
X_balancetest


df_over_con=df_over[['SOB_RISK_SKORU','SOKAK_RISK_SKORU','MAHALLE_RISK_SKORU','TUKETIM_M11','DEMAND_M5','DEMAND_M3','NK_FLAG']]

Y_balance_con = df_over_con.iloc[:,6]
Y_balance_con
X_balance_con = df_over_con.iloc[:,0:6]
X_balance_con
X_balancetrain_con,X_balancetest_con,Y_balancetrain_con,Y_balancetest_con = train_test_split(X_balance_con,Y_balance_con,
                                                                                    test_size = 0.25,
                                                                                    random_state=123)
#machine learning models for balance dataset
#naive bayes model
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_balancetrain_con,Y_balancetrain_con)

train_model_est = model.predict(X_balancetrain_con)



print('the_train_accuracy_of_naive_bayes_model:',accuracy_score(Y_balancetrain_con, train_model_est))


y_model = model.predict(X_balancetest_con)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from matplotlib import pyplot


print('the_accuracy_of_naive_bayes_model:',accuracy_score(Y_balancetest_con,y_model))
print('the_precision_of_naive_bayes_model:',precision_score(Y_balancetest_con,y_model))
print('the_recall_of_naive_bayes_model:',recall_score(Y_balancetest_con,y_model))

print('the_F1_of_naive_bayes_model:',f1_score(Y_balancetest_con,y_model))

crosstab=pd.crosstab(Y_balancetest_con,y_model,rownames=['   Actual_theft'],colnames=['Estimated_theft'])
print(' ')
print(crosstab)

nb_probs = model.predict_proba(X_balancetest_con)

nb_probs
nb_probs = nb_probs[:,1]

nb_auc = roc_auc_score(Y_balancetest_con,nb_probs)
nb_auc

nb_fpr , nb_tpr , _ = roc_curve(Y_balancetest_con,nb_probs)

pyplot.plot(nb_fpr, nb_tpr, marker = '.' , label = 'naive_bayes')

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
the_train_accuracy_of_naive_bayes_model: 0.597997138769671
the_accuracy_of_naive_bayes_model: 0.5967095851216023
the_precision_of_naive_bayes_model: 0.8752620545073375
the_recall_of_naive_bayes_model: 0.2362093352192362
the_F1_of_naive_bayes_model: 0.37202049454221425
 
Estimated_theft     0    1
   Actual_theft          
0                3336  119
1                2700  835
Text(0, 0.5, 'True Positive Rate')

#logistic regression model
from sklearn.linear_model import LogisticRegression


LogRmodel = LogisticRegression(penalty='l2',max_iter=50,solver='newton-cg')
LogRmodel.fit(X_balancetrain,Y_balancetrain)

train_model_est = LogRmodel.predict(X_balancetrain)



print('the_train_accuracy_of_logistic_regression_model:',accuracy_score(Y_balancetrain, train_model_est))


LogRpredict = LogRmodel.predict(X_balancetest)
print('the_accuracy_of_logistic_regression_model:',accuracy_score(Y_balancetest,LogRpredict))
print('the_precision_of_logitic_regression_model:',precision_score(Y_balancetest,LogRpredict))
print('the_recall_of_logistic_regression_model:',recall_score(Y_balancetest,LogRpredict))

print('the_F1_of_logistic_regression_model:',f1_score(Y_balancetest,LogRpredict))

crosstab=pd.crosstab(Y_balancetest,LogRpredict,rownames=['   Actual_theft'],colnames=['Estimated_theft'])
print(' ')
print(crosstab)

lr_probs = model.predict_proba(X_balancetest)

lr_probs
lr_probs = lr_probs[:,1]

lr_auc = roc_auc_score(Y_balancetest,lr_probs)
lr_auc

lr_fpr , lr_tpr , _ = roc_curve(Y_balancetest,lr_probs)

pyplot.plot(lr_fpr, lr_tpr, marker = '.' , label = 'logistic_regression')


pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
the_train_accuracy_of_logistic_regression_model: 0.8247019551740582
the_accuracy_of_logistic_regression_model: 0.8278969957081546
the_precision_of_logitic_regression_model: 0.8537621359223301
the_recall_of_logistic_regression_model: 0.7960396039603961
the_F1_of_logistic_regression_model: 0.8238910847606499
 
Estimated_theft     0     1
   Actual_theft            
0                2973   482
1                 721  2814
Text(0, 0.5, 'True Positive Rate')

#support vector machine
from sklearn import svm
from sklearn.model_selection import GridSearchCV

parameter_grid = {'C': [0.01,0.1 , 1 , 10, 100, 1000],
        'gamma':[1,0.1,0.01, 0.001, 0.0001],
        'kernel' : ['rbf']}

SVMmodel = GridSearchCV(svm.SVC(probability=True), parameter_grid, refit=True)

SVMmodel.fit(X_balancetrain,Y_balancetrain)
SVMpredict = SVMmodel.predict(X_balancetest)

train_model_est = SVMmodel.predict(X_balancetrain)



print('the_train_accuracy_of_support_vector_machine_model:',accuracy_score(Y_balancetrain, train_model_est))



print('the_accuracy_of_support_vector_machine:',accuracy_score(Y_balancetest,SVMpredict))
print('the_precision_of_support_vector_machine:',precision_score(Y_balancetest,SVMpredict))
print('the_recall_of_support_vector_machine:',recall_score(Y_balancetest,SVMpredict))

print('the_F1_of_support_vector_machine:',f1_score(Y_balancetest,SVMpredict))

crosstab=pd.crosstab(Y_balancetest,SVMpredict,rownames=['   Actual_theft'],colnames=['Estimated_theft'])
print(' ')
print(crosstab)



svm_probs = SVMmodel.predict_proba(X_balancetest)

svm_probs
svm_probs = svm_probs[:,1]

svm_auc = roc_auc_score(Y_balancetest,svm_probs)
svm_auc

svm_fpr , svm_tpr , _ = roc_curve(Y_balancetest,svm_probs)

pyplot.plot(svm_fpr, svm_tpr, marker = '.' , label = 'support_vector_machine')


pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
  warnings.warn(CV_WARNING, FutureWarning)
the_train_accuracy_of_support_vector_machine_model: 0.9933714830710539
the_accuracy_of_support_vector_machine: 0.9725321888412017
the_precision_of_support_vector_machine: 0.951634693326128
the_recall_of_support_vector_machine: 0.9963224893917964
the_F1_of_support_vector_machine: 0.9734660033167495
 
Estimated_theft     0     1
   Actual_theft            
0                3276   179
1                  13  3522
Text(0, 0.5, 'True Positive Rate')

SVMpredict_result = pd.DataFrame(SVMpredict)

Y_balancetest_result = pd.DataFrame(Y_balancetest)

Y_balancetest_result=Y_balancetest_result.iloc[:,:1].reset_index().drop(columns=['index'])

SVMpredict_result=SVMpredict_result.reset_index()
SVMpredict_result.reindex(range(20970,27960))

SVMpredict_result.columns=['row','NK_FLAG_estimate']
SVM_test_estimate=pd.concat([SVMpredict_result,Y_balancetest_result],axis=1)
SVM_test_estimate.to_csv("Electricity_theft_detection_test_estimation.csv")



#k nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
KNNmodel = KNeighborsClassifier(algorithm='brute',leaf_size=30,metric="minkowski",n_neighbors=9,n_jobs=5,weights='uniform')



KNNmodel.fit(X_balancetrain,Y_balancetrain)
KNNpredict = KNNmodel.predict(X_balancetest)

train_model_est = KNNmodel.predict(X_balancetrain)



print('the_train_accuracy_of_KNN_model:',accuracy_score(Y_balancetrain, train_model_est))


print('the accuracy of decision KNN:',accuracy_score(Y_balancetest,KNNpredict))
print('the precision of decision KNN:',precision_score(Y_balancetest,KNNpredict))
print('the recall of decision KNN:',recall_score(Y_balancetest,KNNpredict))

print('the F1 of decision KNN:',f1_score(Y_balancetest,KNNpredict))
crosstab=pd.crosstab(Y_balancetest,KNNpredict,rownames=['   Actual_theft'],colnames=['Estimated_theft'])
print(' ')
print(crosstab)

KNN_probs = KNNmodel.predict_proba(X_balancetest)

KNN_probs
KNN_probs = KNN_probs[:,1]

KNN_auc = roc_auc_score(Y_balancetest,KNN_probs)
KNN_auc

KNN_fpr , KNN_tpr , _ = roc_curve(Y_balancetest,KNN_probs)

pyplot.plot(KNN_fpr, KNN_tpr, marker = '.' , label = 'KNN')


pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
the_train_accuracy_of_KNN_model: 0.9360515021459227
the accuracy of decision KNN: 0.9238912732474964
the precision of decision KNN: 0.8747192413276765
the recall of decision KNN: 0.9915134370579916
the F1 of decision KNN: 0.9294616812516574
 
Estimated_theft     0     1
   Actual_theft            
0                2953   502
1                  30  3505
Text(0, 0.5, 'True Positive Rate')

#multilayer neural network
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


parameter_grid = {'solver':['lbfgs'],'max_iter':[1000,1100,1200,1300,1400,1500],'alpha':10.0** -np.arange(1,10),
                 'hidden_layer_sizes':np.arange(10,15)}

clf = GridSearchCV(MLPClassifier(), parameter_grid, n_jobs = -1)

clf.fit(X_balancetrain,Y_balancetrain)

clfpredict = clf.predict(X_balancetest)

train_model_est = clf.predict(X_balancetrain)



print('the_train_accuracy_of_multilayer_neural_model:',accuracy_score(Y_balancetrain, train_model_est))



print('the_accuracy_of_multilayer_neural_model:',accuracy_score(Y_balancetest,clfpredict))
print('the_precision_of_multilayer_neural_model:',precision_score(Y_balancetest,clfpredict))
print('the_recall_of_multilayer_neural_model:',recall_score(Y_balancetest,clfpredict))

print('the_F1_of_multilayer_neural_model:',f1_score(Y_balancetest,clfpredict))

crosstab=pd.crosstab(Y_balancetest,clfpredict,rownames=['   Actual_theft'],colnames=['Estimated_theft'])
print(' ')
print(crosstab)

clf_probs = clf.predict_proba(X_balancetest)

clf_probs
clf_probs = clf_probs[:,1]

clf_auc = roc_auc_score(Y_balancetest,clf_probs)
clf_auc

clf_fpr , clf_tpr , _ = roc_curve(Y_balancetest,clf_probs)

pyplot.plot(clf_fpr, clf_tpr, marker = '.' , label = 'multilayer_neural')


pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
  warnings.warn(CV_WARNING, FutureWarning)
the_train_accuracy_of_multilayer_neural_model: 0.7075345731998093
the_accuracy_of_multilayer_neural_model: 0.705865522174535
the_precision_of_multilayer_neural_model: 0.6587932145157827
the_recall_of_multilayer_neural_model: 0.8678925035360678
the_F1_of_multilayer_neural_model: 0.7490234374999999
 
Estimated_theft     0     1
   Actual_theft            
0                1866  1589
1                 467  3068
Text(0, 0.5, 'True Positive Rate')

#decision tree classifier
from sklearn import tree
DTmodel = tree.DecisionTreeClassifier(criterion="entropy",splitter="random",min_samples_split=30)
DTmodel.fit(X_balancetrain,Y_balancetrain)

train_model_est = DTmodel.predict(X_balancetrain)



print('the_train_accuracy_of_tree_model:',accuracy_score(Y_balancetrain, train_model_est))


DTpredict = DTmodel.predict(X_balancetest)
print('the accuracy of decision tree:',accuracy_score(Y_balancetest,DTpredict))
print('the precision of decision tree:',precision_score(Y_balancetest,DTpredict))
print('the recall of decision tree:',recall_score(Y_balancetest,DTpredict))

print('the F1 of decision tree:',f1_score(Y_balancetest,DTpredict))

crosstab=pd.crosstab(Y_balancetest,DTpredict,rownames=['   Actual_theft'],colnames=['Estimated_theft'])
print(' ')
print(crosstab)

dte_probs = DTmodel.predict_proba(X_balancetest)

dte_probs
dte_probs = dte_probs[:,1]

dte_auc = roc_auc_score(Y_balancetest,dte_probs)
dte_auc

dte_fpr , dte_tpr , _ = roc_curve(Y_balancetest,dte_probs)

pyplot.plot(dte_fpr, dte_tpr, marker = '.' , label = 'decision_trees_classifier')


pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
the_train_accuracy_of_tree_model: 0.952074391988555
the accuracy of decision tree: 0.934763948497854
the precision of decision tree: 0.9170956380384719
the recall of decision tree: 0.9575671852899575
the F1 of decision tree: 0.9368945474674786
 
Estimated_theft     0     1
   Actual_theft            
0                3149   306
1                 150  3385
Text(0, 0.5, 'True Positive Rate')

from sklearn.metrics import classification_report


print('Classification report:',classification_report(Y_balancetest,DTpredict))
Classification report:               precision    recall  f1-score   support

           0       0.95      0.91      0.93      3455
           1       0.92      0.96      0.94      3535

    accuracy                           0.93      6990
   macro avg       0.94      0.93      0.93      6990
weighted avg       0.94      0.93      0.93      6990

#read new set
new_fr = pd.read_csv('test.csv')
new_fr.head()

new = new_fr.drop(columns=['SAYAC_BITIS_TARIHI','INDEX'],axis=0)

new = new.replace(np.nan, 0 )


from datetime import datetime


new['SAYAC_YAPIM_YILI'] = new['SAYAC_YAPIM_YILI'].astype(int).round()

new['SAYAC_YAPIM_YILI'][new['SAYAC_YAPIM_YILI'] == 0] = 1970

new['SAYAC_YAPIM_YILI'] = new['SAYAC_YAPIM_YILI'].astype(str)

new['SAYAC_YAPIM_YILI'] = pd.to_datetime(new['SAYAC_YAPIM_YILI'])



new['SAYAC_TAKILMA_TARIHI'] = pd.to_datetime(new['SAYAC_TAKILMA_TARIHI'])
new['SAYAC_BASLANGIC_TARIHI'] = pd.to_datetime(new['SAYAC_BASLANGIC_TARIHI'])
new[new['SAYAC_YAPIM_YILI'] == 1970]



new= pd.concat([new,pd.get_dummies(new['SAYAC_MARKA'],prefix='SAYAC_MARKA',dummy_na = 'TRUE')],axis=1)



new= pd.concat([new,pd.get_dummies(new['TESISAT_TIPI'],prefix='TESISAT_TIPI',dummy_na = 'TRUE')],axis=1)
new = new.drop(columns = ['TESISAT_TIPI_nan','TESISAT_TIPI'])
new= pd.concat([new,pd.get_dummies(new['SAYAC_MODEL'],prefix='SAYAC_MODEL',dummy_na = 'TRUE')],axis=1)
new = new.drop(columns = ['SAYAC_MODEL_nan','SAYAC_MODEL'])
new= pd.concat([new,pd.get_dummies(new['SAYAC_MALZEME_ID'],prefix='SAYAC_MALZEME_ID',dummy_na = 'TRUE')],axis=1)
new = new.drop(columns = ['SAYAC_MALZEME_ID_nan','SAYAC_MALZEME_ID'])
new= pd.concat([new,pd.get_dummies(new['SAYAC_OLCUM_TURU'],prefix='SAYAC_OLCUM_TURU',dummy_na = 'TRUE')],axis=1)
new = new.drop(columns = ['SAYAC_OLCUM_TURU_nan','SAYAC_OLCUM_TURU'])
new= pd.concat([new,pd.get_dummies(new['SAYAC_FAZ_N'],prefix='SAYAC_FAZ_N',dummy_na = 'TRUE')],axis=1)
new = new.drop(columns = ['SAYAC_FAZ_N_nan','SAYAC_FAZ_N'])


new =new.drop(columns = ['SAYAC_MARKA_nan','SAYAC_MARKA','SAYAC_TAKILMA_TARIHI','SAYAC_BASLANGIC_TARIHI','SAYAC_YAPIM_YILI'])


new_continuous = new.iloc[:, 0:30]

mod_new = pd.DataFrame(scaler.fit_transform(new_continuous), columns=new_continuous.columns)


new = pd.concat([mod_new,new.iloc[:, 31:177],],axis=1)
new =pd.concat([new_fr['INDEX'],new],axis=1)


test_new=new[['SOB_RISK_SKORU', 'SAYAC_MODEL_LUN10-B', 'SAYAC_MARKA_LUNA', 'SOKAK_RISK_SKORU', 'MAHALLE_RISK_SKORU', 'TESISAT_TIPI_Ticarethane-Sanayi', 'SAYAC_OLCUM_TURU_0', 'SAYAC_MALZEME_ID_80000855.0', 'TUKETIM_M11', 'SAYAC_MODEL_LSM40-BUZ-KOM', 'SAYAC_MALZEME_ID_80000755.0', 'SAYAC_MALZEME_ID_80000526.0', 'SAYAC_MALZEME_ID_80000089.0', 'SAYAC_MARKA_KOHLER', 'DEMAND_M5', 'SAYAC_MALZEME_ID_80000732.0', 'SAYAC_MARKA_MAKEL', 'SAYAC_MODEL_EC018MBW', 'DEMAND_M3', 'SAYAC_MODEL_LUN1', 'SAYAC_MALZEME_ID_80000731.0', 'SAYAC_MARKA_ELEKTROMED']]
test_new


#model implementation
NK_FLAG = SVMmodel.predict(test_new)

new_fr['NK_FLAG'] = NK_FLAG



new_fr.to_csv("Electricity_Theft_Detection.csv") 

Get Outlook for Android
