from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd


startyear = 1979
endyear = 2019
cpi_baseyear = 2016

def get_precision_recall(data): 
    data_ml = _get_fortraining_data(data)
    data_full,data_train,data_test = data_ml[0],data_ml[1],data_ml[2]
    x_full,y_full = data_full.drop(['relMW_groups','agecat'],axis=1),data_full['relMW_groups']
    x_tr,y_tr = data_train.drop(['relMW_groups','agecat'],axis=1),data_train['relMW_groups']
    x_ts,y_ts = data_test.drop(['relMW_groups','agecat'],axis=1),data_test['relMW_groups']
    yhat_boost = _pred_boosted_tree(x_tr,y_tr,x_ts)
    yhat_tree = _pred_decision_tree(x_full,y_full,x_ts)
    yhat_rf = _pred_random_forest(x_tr,y_tr,x_ts)

    dfs = [data_train, data_test]
    for data in dfs:
        data['age2'] = np.power(data['age'],2)
        data['age3'] = np.power(data['age'],3)
        data['age4'] = np.power(data['age'],4)
        data['teen'] = np.where(data['age']<20,1,0)
        data['sex'] = data['sex'].astype(int)
        data['educcat'] = data['educcat'].astype(int)
        data['dmarried'] = data['dmarried'].astype(int)
        data['ruralstatus'] = data['ruralstatus'].astype(int)
        data['hispanic'] = data['hispanic'].astype(int)
        data['veteran'] = data['veteran'].astype(int)
        data['race'] = data['race'].astype(int)
        data['race_2'] = np.where(data['race']==1,1,0)
        data['young_adult'] = np.where(data['age'].isin(range(20,26)),1,0)
        data['race_sex_teen'] = data['race_2']*data['teen']*data['sex']
        data['race_sex_ya'] = data['race_2']*data['young_adult']*data['sex']
        data['relMW_groups2'] = data['relMW_groups'].astype(int)-1
        data['edu_sex_age'] = data['educcat']*data['sex']*data['age']

    yhat_lm =  _pred_linear_model(data_train,data_test)
    yhat_glm = _pred_basic_logit(data_train,data_test)
 # x_tr_vp = _get_polynomial_data(data_train)
 # x_ts_vp = _get_polynomial_data(data_test)
 # yhat_elnet = _pred_elastic_net(x_tr_vp,y_tr,x_ts_vp)

    var_names = ['rf', 'tree', 'lm','glm']

    precision_dict = {}
    recall_dict = {}
    f_interp = {}
    precision_df = {}

    precision_df["precision_boost"],precision_df["recall_boost"],_ = precision_recall_curve(y_ts,yhat_boost) 
    for name in var_names:
        yhat = vars()[f"yhat_{name}"]
        precision_dict[f"precision_{name}"],recall_dict[f"recall_{name}"],_ = precision_recall_curve(y_ts,yhat) 
        f_interp[f"f_interp_{name}"] = interp1d(recall_dict[f"recall_{name}"],precision_dict[f"precision_{name}"]) 
        precision_df[f"precision_interp_{name}"] = f_interp[f"f_interp_{name}"](precision_df["recall_boost"])
        precision_df[f"precision_diff_{name}"] = precision_df[f"precision_interp_{name}"] - precision_df["precision_boost"]
    
    return precision_df

def get_boost_income(data):
    data_ml = _get_fortraining_data(data)
    data_train = data_ml[1]
    x_tr,y_tr = data_train.drop(['relMW_groups','agecat'],axis=1),data_train['relMW_groups']
    boost_income = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.005, max_depth=6).fit(x_tr, y_tr)
    return boost_income

def _get_fortraining_data(data):
    data['relMW_groups'] = np.where(data['relMW_groups']!=3,1,0)
    data['relMW_groups'] = data['relMW_groups'].astype(bool)
    data['agecat'] = data['agecat'].astype("category")
    data['educcat'] = data['educcat'].astype("category")
    data['ruralstatus'] = data['ruralstatus'].astype("category")
    data['sex'] = data['sex'].astype("category")
    data['hispanic'] = data['hispanic'].astype("category")
    data['dmarried'] = data['dmarried'].astype("category")
    data['race'] = data['race'].astype("category")
    data['veteran'] = data['veteran'].astype("category")
    data_full = data.loc[(data['training']==1) & (~data['quarterdate'].isin(range(136,143)))]
    data_full = data_full[['race','sex','hispanic','agecat','age','dmarried','educcat','relMW_groups','ruralstatus','veteran']]

    np.random.seed(12345)
    smp_size = 15000
    train_ind = np.random.choice(data_full.index, size = smp_size, replace=False) 
    data_train = data_full.loc[train_ind]

    data_test = data_full.drop(train_ind,axis=0)

    return (data_full, data_train, data_test)

# def _get_polynomial_data(data):
# poly4 = PolynomialFeatures(degree=4,include_bias=False)
# poly2 = PolynomialFeatures(degree=2,include_bias=False)
# x_subset = data[['race','sex','hispanic','age','age2','dmarried','educcat','ruralstatus','veteran']]
# x_p4 = pd.DataFrame(poly4.fit_transform(x_subset.drop(['age2'],axis=1)))
# x_p2 = pd.DataFrame(poly2.fit_transform(x_subset.drop(['age'],axis=1)))
# x_vp = pd.concat([data[['age3','age4']],x_p4,x_p2],axis=1)      
# return x_vp

def _pred_boosted_tree(x_tr,y_tr,x_ts):
    boost_income = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.005, max_depth=6).fit(x_tr, y_tr)
    yhat_boost = boost_income.predict_proba(x_ts)[:,1]
    return yhat_boost

def _pred_decision_tree(x_full,y_full,x_ts):
    tree_income = DecisionTreeClassifier().fit(x_full,y_full)
    yhat_tree = tree_income.predict_proba(x_ts)[:,1]
    return yhat_tree

def _pred_linear_model(data_train,data_test):
    formula_basic = "relMW_groups2 ~ age3+age2+age+race_sex_teen+race_sex_ya+educcat+hispanic+race+sex+edu_sex_age"
    lm_fit = smf.ols(formula_basic, data=data_train).fit()
    yhat_lm = lm_fit.predict(data_test)
    return yhat_lm

def _pred_basic_logit(data_train,data_test):
    formula_basic = "relMW_groups ~ age+educcat"
    glm_fit = sm.formula.glm(formula_basic, family=sm.families.Binomial(),data=data_train).fit() #smf.logit
    yhat_glm = glm_fit.predict(data_test)
    return yhat_glm

def _pred_random_forest(x_tr,y_tr,x_ts):
    rf_income = RandomForestClassifier(n_estimators=2000,max_features=2).fit(x_tr,y_tr)
    yhat_rf = rf_income.predict_proba(x_ts)[:,1]
    return yhat_rf

# def _pred_elastic_net(x_tr_vp,y_tr,x_ts_vp): #Podemos exceptuar elastic net
# elnet_fit = LogisticRegressionCV(cv=10,penalty='elasticnet',l1_ratios=[0.5],cv=5,scoring='roc_auc').fit(x_tr_vp,y_tr)
# yhat_elnet = elnet_fit.predict_proba(x_ts_vp)[:,1]
# return yhat_elnet


