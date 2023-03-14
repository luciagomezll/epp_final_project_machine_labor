import pytask
from machine_labor.config import BLD
from machine_labor.config import SRC

import pandas as pd
from machine_labor.data_management.clean_data_MW import get_forprediction_eventstudy_data
from machine_labor.data_management.clean_data_MW import get_fortraining_eventstudy_data

@pytask.mark.depends_on(
    {
        "forbalance": SRC / "data" / "VZmw_quarterly_lagsleads_1979_2019.dta",
        "eventclass": SRC / "data" / "eventclassification_2019.dta",
        "cpi": SRC / "data" / "cpiursai1977-2019.dta",
        "cps_morg": SRC / "data" / "cps_morg_2019_new.dta",
        "state_codes": BLD / "python" / "data" / "state_codes.pkl",
        "quarter_codes": BLD / "python" / "data" / "quarter_codes.pkl",
        "month_codes": BLD / "python" / "data" / "month_codes.pkl",
    }
)
@pytask.mark.produces(BLD / "python" / "data" / "forpredictionmorg_full_eventstudy_2019.pkl")
def task_forprediction_eventstudy_data(depends_on, produces):
    state_codes = pd.read_pickle(depends_on['state_codes'])
    quarter_codes = pd.read_pickle(depends_on['quarter_codes'])
    month_codes = pd.read_pickle(depends_on['month_codes'])
    data_forbalance = pd.read_stata(depends_on['forbalance'])
    data_eventclass = pd.read_stata(depends_on['eventclass'], convert_categoricals=False)
    data_cpi = pd.read_stata(depends_on['cpi'])
    data_cps_morg = pd.read_stata(depends_on['cps_morg'], convert_categoricals=False) 
    forprediction_eventstudy = get_forprediction_eventstudy_data(data_forbalance,data_eventclass,
                                                             data_cps_morg,data_cpi,quarter_codes,
                                                             state_codes,month_codes)
    forprediction_eventstudy.to_pickle(produces)

@pytask.mark.depends_on(BLD / "python" / "data" / "forpredictionmorg_full_eventstudy_2019.pkl")
@pytask.mark.produces(BLD / "python" / "data" / "fortraining_eventstudy_1979_2019.pkl")
def task_fortraining_eventstudy_data(depends_on, produces):
    forprediction = pd.read_pickle(depends_on)
    fortraining_eventstudy = get_fortraining_eventstudy_data(forprediction)
    fortraining_eventstudy.to_pickle(produces)
