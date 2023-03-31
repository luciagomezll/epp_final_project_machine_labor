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
    """Computes precision, recall, and interpolated precision values for different classifiers
    using the input data.

    The function first splits the data into training and test sets,
    and then applies several classifiers to the training data to generate predictions for the
    test data. It then computes precision and recall values for each classifier and generates
    interpolated precision values using the precision and recall values for the boosted tree
    classifier. The function returns a dictionary of dataframes containing the precision,
    recall, interpolated precision, and precision difference values for each classifier.

    Args:
        data (pd.DataFrame): The full data set generated as a result of the data cleaning part.

    Returns:
        precision_df : dictionary of pandas DataFrames
        Dictionary containing precision, recall, interpolated precision, and precision
        difference values for each classifier. 

    """
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

    data_train = pd.get_dummies(data_train, columns=['educcat'])
    data_test = pd.get_dummies(data_test, columns=['educcat'])

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


def get_boost_basic_model(data):
    """Train the basic boosted tree model.

    Args:
        data (pd.DataFrame): The full data set generated as a result of the data cleaning part.

    Returns:
        model (sklearn.ensemble.GradientBoostingClassifier): The trained boosted tree model.

    """
    data_train = _get_fortraining_data(data)[1]
    xtr_basic,ytr_basic = data_train[["age","race","sex","hispanic","dmarried","ruralstatus","educcat","veteran"]],data_train['relMW_groups']
    boost_basic = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.005, max_depth=6, min_samples_leaf = 10).fit(xtr_basic,ytr_basic)
    return boost_basic

def get_boost_full_model(data):
    """Train the complete boosted tree model.

    Args:
        data (pd.DataFrame): The full data set generated as a result of the data cleaning part.

    Returns:
        model (sklearn.ensemble.GradientBoostingClassifier): The trained boosted tree model.

    """
    data_train = _get_fortraining_data(data)[1]
    formula = "relMW_groups~age+race+sex+hispanic+dmarried+ruralstatus+educcat+veteran"
    ytr_full,xtr_full = dmatrices(formula,data_train,return_type="dataframe")
    boost_full = GradientBoostingClassifier(
        n_estimators=4000, learning_rate=0.005, max_depth=6, min_samples_leaf = 10
    ).fit(xtr_full, ytr_full)
    return boost_full

def _get_fortraining_data(data):
    """Preprocesses and splits the input data into training and testing datasets.

    Args:
        data (pandas.DataFrame): The full data set generated as a result of the data cleaning part.

    Returns:
        A tuple containing three DataFrames: the full data, the training data, and the testing data.

    """
    data["relMW_groups"] = np.where(data["relMW_groups"] != 3, 1, 0)
    #data["relMW_groups"] = data["relMW_groups"].astype(int)
    cat_cols = ["ruralstatus","sex","hispanic","dmarried","race","veteran","educcat","agecat"]
    data[cat_cols] = data[cat_cols].astype("category")
    #data[cat_cols] = [data[col].astype("category") for col in cat_cols]

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

    data_train = pd.concat([y_train,x_train], axis=1)
    data_test = pd.concat([y_test,x_test], axis=1)
    return (data_full, data_train, data_test)

def _pred_boosted_tree(data_train, data_test):
    """Train the boosted tree model to predict individual's exposure to a minimum wage change (outcome variable).

    Args:
        data_train (pd.DataFrame): The training dataset used to train the boosted tree model.
        data_test (pd.DataFrame): The testing dataset used to predict individual's exposure to a minimum wage change.

    Returns:
        yhat_boost (np.array): A vector of predicted outcome variable values for the testing dataset.

    """
    formula = "relMW_groups~age+race+sex+hispanic+dmarried+ruralstatus+educcat+veteran"
    y_tr,x_tr = dmatrices(formula,data_train,return_type="dataframe")
    _,x_ts = dmatrices(formula,data_test,return_type="dataframe") 

    boost_income = GradientBoostingClassifier(
        n_estimators=4000, learning_rate=0.005, max_depth=6, min_samples_leaf = 10
    ).fit(x_tr, y_tr)
    yhat_boost = boost_income.predict_proba(x_ts)[:, 1]
    return yhat_boost


def _pred_decision_tree(data_train,data_test):
    """Train the tree model to predict individual's exposure to a minimum wage change (outcome variable).

    Args:
        data_train (pd.DataFrame): The training dataset used to train the tree model.
        data_test (pd.DataFrame): The testing dataset used to predict individual's exposure to a minimum wage change.

    Returns:
        yhat_tree (np.array): A vector of predicted outcome variable values for the testing dataset.

    """
    formula = "relMW_groups~age+race+sex+hispanic+dmarried+ruralstatus+educcat+veteran"
    y_tr,x_tr = dmatrices(formula,data_train,return_type="dataframe")
    _,x_ts = dmatrices(formula,data_test,return_type="dataframe") 

    tree_income = DecisionTreeClassifier().fit(x_tr, y_tr)
    yhat_tree = tree_income.predict_proba(x_ts)[:, 1]
    return yhat_tree

def _pred_linear_model(data_train, data_test):
    """Fit the linear Card and Krueger probability model to predict individual's exposure to a minimum wage change (outcome variable).

    Args:
        data_train (pd.DataFrame): The training dataset used to fit the linear Card and Krueger probability model.
        data_test (pd.DataFrame): The testing dataset used to predict individual's exposure to a minimum wage change.

    Returns:
        yhat_lm (np.array): A vector of predicted outcome variable values for the testing dataset.

    """
    formula = "relMW_groups2 ~ age+hispanic+race+sex+educcat_2+educcat_3+educcat_4"
    formula = " + ".join([formula] + [f"I(age ** {i})" for i in range(2, 4)] + ["race2:sex:teen"] + ["race2:sex:young_adult"] + [f"age:sex:educcat_{i}" for i in range(2,5)])
    lm_fit = smf.ols(formula, data=data_train).fit()
    yhat_lm = lm_fit.predict(data_test)
    return yhat_lm

def _pred_basic_logit(data_train, data_test):
    """Fit the basic logistic model to predict individual's exposure to a minimum wage change (outcome variable).

    Args:
        data_train (pd.DataFrame): The training dataset used to fit the logistic regression model.
        data_test (pd.DataFrame): The testing dataset used to predict individual's exposure to a minimum wage change.

    Returns:
        yhat_logit (np.array): A vector of predicted outcome variable values for the testing dataset.

    """
    formula = "relMW_groups ~ age+ educcat_2 + educcat_3 + educcat_4"
    logit_fit = smf.logit(
        formula, data=data_train
    ).fit() 
    log_odds = logit_fit.predict(data_test)
    yhat_logit = 1 / (1 + np.exp(-log_odds))
    return yhat_logit

def _pred_random_forest(data_train, data_test):
    """Train the random forest model to predict individual's exposure to a minimum wage change (outcome variable).

    Args:
        data_train (pd.DataFrame): The training dataset used to train the random forest model.
        data_test (pd.DataFrame): The testing dataset used to predict individual's exposure to a minimum wage change.

    Returns:
        yhat_rf (np.array): A vector of predicted outcome variable values for the testing dataset.

    """
    formula = ("relMW_groups~age+race+sex+hispanic+dmarried+ruralstatus+educcat+veteran")
    y_tr,x_tr = dmatrices(formula,data_train,return_type="dataframe")
    y_ts,x_ts = dmatrices(formula,data_test,return_type="dataframe")

    rf_income = RandomForestClassifier(n_estimators=2000, max_features=2).fit(
        x_tr, y_tr
    )
    yhat_rf = rf_income.predict_proba(x_ts)[:, 1]
    return yhat_rf


