import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from scipy.interpolate import interp1d
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
#from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import plot_tree

def get_precision_recall(data):
    data_full, data_train, data_test = _get_fortraining_data(data)
    yhat_boost = _pred_boosted_tree(data_train, data_test)
    yhat_tree = _pred_decision_tree(data_full,data_test)
    yhat_rf = _pred_random_forest(data_train, data_test)

    dfs = [data_train, data_test]
    for data in dfs:
        data["teen"] = np.where(data["age"] < 20, 1, 0)
        data["race2"] = np.where(data["race"] == 1, 1, 0)
        data["young_adult"] = np.where(data["age"].isin(range(20, 26)), 1, 0)
        data["relMW_groups2"] = data["relMW_groups"].astype(int) - 1
            
    yhat_lm = _pred_linear_model(data_train, data_test)
    yhat_logit = _pred_basic_logit(data_train, data_test)

    precision_dict = {}
    recall_dict = {}
    f_interp = {}
    precision_df = {}

    (
        precision_df["precision_boost"],
        precision_df["recall_boost"],
        _,
    ) = precision_recall_curve(data_test["relMW_groups"], yhat_boost)

    var_names = ["rf", "tree", "lm", "logit"]
    ytest = data_test["relMW_groups"]

    for name in var_names:
        yhat = vars()[f"yhat_{name}"]
        (
            precision_dict[f"precision_{name}"],
            recall_dict[f"recall_{name}"],
            _,
        ) = precision_recall_curve(ytest, yhat)
        f_interp[f"f_interp_{name}"] = interp1d(
            recall_dict[f"recall_{name}"], precision_dict[f"precision_{name}"]
        )
        precision_df[f"precision_interp_{name}"] = f_interp[f"f_interp_{name}"](
            precision_df["recall_boost"]
        )
        precision_df[f"precision_diff_{name}"] = (
            precision_df[f"precision_interp_{name}"] - precision_df["precision_boost"]
        )

    return precision_df


def get_boost_income(data):
    data_ml = _get_fortraining_data(data)
    data_train = data_ml[1]
    formula = "relMW_groups~age+race+sex+hispanic+dmarried+ruralstatus+educcat+veteran"
    y_tr,x_tr = dmatrices(formula,data_train,return_type="dataframe")
    y_tr = y_tr["relMW_groups"]
    boost_income = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.005, max_depth=6, min_samples_leaf = 10).fit(x_tr, y_tr)
    return boost_income


def _get_fortraining_data(data):
    data["relMW_groups"] = np.where(data["relMW_groups"] != 3, 1, 0)
    data["relMW_groups"] = data["relMW_groups"].astype(bool)
    cat_cols = ["ruralstatus","sex","hispanic","dmarried","race","veteran","educcat","agecat"]
    data[cat_cols] = [data[col].astype("category") for col in cat_cols]

    data_full = data.loc[
        (data["training"] == 1) & (~data["quarterdate"].isin(range(136, 143)))
    ]
    data_full = data_full[
        [
            "race",
            "sex",
            "hispanic",
            "agecat",
            "age",
            "dmarried",
            "educcat",
            "relMW_groups",
            "ruralstatus",
            "veteran",
        ]
    ]

    x_train, x_test, y_train, y_test = train_test_split(data_full.drop(["relMW_groups"], axis=1),
        data_full["relMW_groups"], test_size=0.3, random_state=12345)

    data_train = pd.concat(y_train,x_train, axis=1)
    data_test = pd.concat(y_test,x_test, axis=1)
    return (data_full, data_train, data_test)


def _pred_boosted_tree(data_train, data_test):
    formula = "relMW_groups~age+race+sex+hispanic+dmarried+ruralstatus+educcat+veteran"
    y_tr,x_tr = dmatrices(formula,data_train,return_type="dataframe")
    _,x_ts = dmatrices(formula,data_test,return_type="dataframe") 
    y_tr = y_tr["relMW_groups"]

    boost_income = GradientBoostingClassifier(
        n_estimators=4000, learning_rate=0.005, max_depth=6, min_samples_leaf = 10
    ).fit(x_tr, y_tr)
    yhat_boost = boost_income.predict_proba(x_ts)[:, 1]
    return yhat_boost


def _pred_decision_tree(data_full,data_test):
    formula = "relMW_groups~age+race+sex+hispanic+dmarried+ruralstatus+educcat+veteran"
    y_full,x_full = dmatrices(formula,data_full,return_type="dataframe")
    _,x_ts = dmatrices(formula,data_test,return_type="dataframe") 
    y_full = y_full["relMW_groups"]

    tree_income = DecisionTreeClassifier().fit(x_full, y_full)
    yhat_tree = tree_income.predict_proba(x_ts)[:, 1]
    return yhat_tree

def _pred_linear_model(data_train, data_test):
    formula = "relMW_groups2 ~ age+hispanic+race+sex+educcat"
    formula = " + ".join([formula] + [f"I(age ** {i})" for i in range(2, 4)] + 
                         [f"I(race2 * sex * teen)"] + 
                         [f"I(race2 * sex * young_adult)"] + [f"I(age * sex * C(educcat, Treatment(reference='1')))"])
    y_x = dmatrices(formula,data_train,return_type="dataframe")
    data_train = pd.concat(y_x, axis=1)
    y_x = dmatrices(formula,data_test,return_type="dataframe") 
    data_test = pd.concat(y_x, axis=1)  
    lm_fit = smf.ols(formula, data=data_train).fit()
    yhat_lm = lm_fit.predict(data_test)
    return yhat_lm


def _pred_basic_logit(data_train, data_test):
    formula = "relMW_groups ~ age+educcat"
    y_x = dmatrices(formula,data_train,return_type="dataframe")
    data_train = pd.concat(y_x, axis=1)
    y_x = dmatrices(formula,data_test,return_type="dataframe") 
    data_test = pd.concat(y_x, axis=1)   
    logit_fit = smf.logit(
        formula, data=data_train
    ).fit()  
    yhat_logit = logit_fit.predict(data_test)
    return yhat_logit


def _pred_random_forest(data_train, data_test):
    formula = ("relMW_groups~age+race+sex+hispanic+dmarried+ruralstatus+educcat+veteran")
    y_tr,x_tr = dmatrices(formula,data_train,return_type="dataframe")
    y_ts,x_ts = dmatrices(formula,data_test,return_type="dataframe")
    y_tr = y_tr["relMW_groups"]
    y_ts = y_ts["relMW_groups"]

    rf_income = RandomForestClassifier(n_estimators=2000, max_features=2).fit(
        x_tr, y_tr
    )
    yhat_rf = rf_income.predict_proba(x_ts)[:, 1]
    return yhat_rf


