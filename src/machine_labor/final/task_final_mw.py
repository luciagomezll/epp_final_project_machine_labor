import pytask
import pickle
from machine_labor.config import BLD

import pandas as pd
from machine_labor.final.plot_tables_mw import plot_precision_recall_curve
from machine_labor.final.plot_tables_mw import plot_precision_relative_boost
from machine_labor.final.plot_tables_mw import feature_importance

from statsmodels.iolib.smpickle import load_pickle

@pytask.mark.depends_on(BLD / "python" / "data" / "precision_df.pkl")
@pytask.mark.produces(BLD / "python" / "figures" / "precision_recall_curves.png")
def task_precision_recall_curves(depends_on, produces):
    precision_data = pd.read_pickle(depends_on)
    fig = plot_precision_recall_curve(precision_data)
    fig.write_image(produces)

@pytask.mark.depends_on(BLD / "python" / "data" / "precision_df.pkl")
@pytask.mark.produces(BLD / "python" / "figures" / "precision_relative_boost.png")
def task_precision_relative_boost_curves(depends_on, produces):
    precision_data = pd.read_pickle(depends_on)
    fig = plot_precision_relative_boost(precision_data)
    fig.write_image(produces)

@pytask.mark.depends_on(BLD / "python" / "models" / "boost_income_basic_model.pkl")
@pytask.mark.produces(BLD / "python" / "figures" / "feature_importance.png")
def task_feature_importance(depends_on, produces):
    boost_income = pickle.load(open(depends_on, 'rb'))
    fig = feature_importance(boost_income)
    fig.write_image(produces)
