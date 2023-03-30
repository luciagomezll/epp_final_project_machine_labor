"""Create separate datasets with basic information for quarter, month and state that
will be useful to merge datasets in the cleaning part."""
import numpy as np
import pandas as pd
import pytask
from machine_labor.config import BLD
from machine_labor.config import SRC

startyear = 1979


def quarter_codes(data):
    """Get the dataset with quarter numbers and quarte dates for the relevant period.

    Args:
        data (pandas.DataFrame): The data set.

    Returns:
        pd.DataFrame: It has all the quarter dates and the corresponding quarter numbers.

    """
    qdate = np.unique(data["quarterdate"])
    qnum = list(enumerate(qdate, 76))
    quarter_codes = pd.DataFrame(qnum, columns=["quarternum", "quarterdate"])
    quarter_codes["quarterdate"] = pd.PeriodIndex(
        quarter_codes["quarterdate"], freq="Q"
    )
    quarter_codes["quarternum"] = quarter_codes["quarternum"].astype(int)
    return quarter_codes


def state_codes(data):
    """Get the dataset with state names and state numbers.

    Args:
        data (pandas.DataFrame): The data set.

    Returns:
        pd.DataFrame: It has all the state names and the corresponding state numbers.

    """
    nstate = np.unique(data["statenum"])
    statnum = list(enumerate(nstate, 1))
    state_codes = pd.DataFrame(statnum, columns=["statenum", "state_name"])
    state_codes["statenum"] = state_codes["statenum"].astype(int)
    return state_codes


def month_codes(data):
    """Get the dataset with information about the months for the relevant period.

    Args:
        data (pandas.DataFrame): The data set.

    Returns:
        pd.DataFrame: It has all the month names and the corresponding numbers across years.

    """
    data = data.sort_values(by=["year", "month", "monthdate"])
    data = data.drop_duplicates()
    nmonth = data.loc[data["year"] == startyear, "month"]
    nmonth = list(enumerate(nmonth, 1))
    month_number = pd.DataFrame(nmonth, columns=["month_num", "month"])
    month_number["monthnum"] = "month" + month_number["month_num"].astype(str)
    data = pd.merge(data, month_number, how="left")
    return data


@pytask.mark.depends_on(
    BLD / "python" / "data" / "VZmw_quarterly_lagsleads_1979_2019.dta"
)
@pytask.mark.produces(BLD / "python" / "data" / "quarter_codes.pkl")
def task_quarter_codes(depends_on, produces):
    data = pd.read_stata(depends_on, columns=["quarterdate"])
    data = quarter_codes(data)
    data.to_pickle(produces)


@pytask.mark.depends_on(BLD / "python" / "data" / "cps_morg_2019_new.dta")
@pytask.mark.produces(BLD / "python" / "data" / "state_codes.pkl")
def task_state_codes(depends_on, produces):
    data = pd.read_stata(depends_on, columns=["statenum"])
    data = state_codes(data)
    data.to_pickle(produces)


@pytask.mark.depends_on(BLD / "python" / "data" / "cps_morg_2019_new.dta")
@pytask.mark.produces(BLD / "python" / "data" / "month_codes.pkl")
def task_month_codes(depends_on, produces):
    data = pd.read_stata(depends_on, columns=["year", "month", "monthdate"])
    data = month_codes(data)
    data.to_pickle(produces)
