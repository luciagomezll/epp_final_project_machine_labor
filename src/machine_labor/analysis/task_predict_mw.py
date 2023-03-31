import pytask
import pickle
from machine_labor.config import BLD

import pandas as pd
from machine_labor.analysis.predict_mw import get_precision_recall
from machine_labor.analysis.predict_mw import get_boost_basic_model 
from machine_labor.analysis.predict_mw import get_boost_full_model

@pytask.mark.depends_on(BLD / "python" / "data" / "fortraining_eventstudy_1979_2019.pkl")
@pytask.mark.produces(BLD / "python" / "data" / "precision_df.pkl")
def task_precision_recall_data(depends_on, produces):
    fortraining_eventstudy7919 = pd.read_pickle(depends_on)
    precision_df = get_precision_recall(fortraining_eventstudy7919)
    precision = pd.DataFrame.from_dict(precision_df)
    precision.to_pickle(produces)

@pytask.mark.depends_on(BLD / "python" / "data" / "fortraining_eventstudy_1979_2019.pkl")
@pytask.mark.produces(BLD / "python" / "models" / "boost_income_basic_model.pkl")
def task_boost_basic_model(depends_on, produces):
    fortraining_eventstudy7919 = pd.read_pickle(depends_on)
    boost_income_basic = get_boost_basic_model(fortraining_eventstudy7919)
    pickle.dump(boost_income_basic, open(produces, 'wb'))

@pytask.mark.depends_on(BLD / "python" / "data" / "fortraining_eventstudy_1979_2019.pkl")
@pytask.mark.produces(BLD / "python" / "models" / "boost_income_full_model.pkl")
def task_boost_full_model(depends_on, produces):
    fortraining_eventstudy7919 = pd.read_pickle(depends_on)
    boost_income_full = get_boost_full_model(fortraining_eventstudy7919)
    pickle.dump(boost_income_full, open(produces, 'wb'))
