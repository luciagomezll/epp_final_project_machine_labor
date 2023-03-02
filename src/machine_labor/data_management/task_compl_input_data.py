from machine_labor.config import BLD
from machine_labor.config import SRC
import pytask

import numpy as np
import pandas as pd
import datetime as dt

startyear = 1979

def quarter_codes(data):
    qdate = np.unique(data['quarterdate'])
    qnum = list(enumerate(qdate,76))
    quarter_codes = pd.DataFrame(qnum, columns=['quarternum','quarterdate'])
    quarter_codes['quarterdate'] = pd.PeriodIndex(quarter_codes['quarterdate'], freq='Q')
    quarter_codes['quarternum'] = quarter_codes['quarternum'].astype(int)
    return quarter_codes

def state_codes(data):
    nstate = np.unique(data['statenum'])
    statnum = list(enumerate(nstate,1))
    state_codes = pd.DataFrame(statnum, columns=['statenum','state_name'])
    state_codes['statenum'] = state_codes['statenum'].astype(int)
    return state_codes

def month_codes(data):
    data = data.drop_duplicates()
    nmonth = data.loc[data['year']==startyear,'month']
    nmonth = list(enumerate(nmonth,1))
    month_number = pd.DataFrame(nmonth, columns=['month_num','month'])
    month_number['monthnum'] = 'month' + month_number['month_num'].astype(str)
    data = pd.merge(data,month_number,how="left")
    return data

@pytask.mark.depends_on(SRC / "data" / "VZmw_quarterly_lagsleads_1979_2019.dta")
@pytask.mark.produces(BLD / "python" / "data" / "quarter_codes.pkl")
def task_quarter_codes(depends_on, produces):
    data = pd.read_stata(depends_on, columns=['quarterdate'])
    data = quarter_codes(data)
    data.to_pickle(produces)

@pytask.mark.depends_on(SRC / "data" / "cps_morg_2019_new.dta")
@pytask.mark.produces(BLD / "python" / "data" / "state_codes.pkl")
def task_state_codes(depends_on, produces):
    data = pd.read_stata(depends_on, columns=['statenum'])
    data = state_codes(data)
    data.to_pickle(produces)

@pytask.mark.depends_on(SRC / "data" / "cps_morg_2019_new.dta")
@pytask.mark.produces(BLD / "python" / "data" / "month_codes.pkl")
def task_month_codes(depends_on, produces):
    data = pd.read_stata(depends_on, columns=['year','month','monthdate'])
    data = month_codes(data)
    data.to_pickle(produces)

