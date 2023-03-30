import pytask
import requests
import pandas as pd

from zipfile import ZipFile
from machine_labor.config import BLD, SRC
from machine_labor.data_management.clean_data_MW import get_forprediction_eventstudy_data
from machine_labor.data_management.clean_data_MW import get_fortraining_eventstudy_data

@pytask.mark.produces(SRC / "data" / "epp_ml_mw_data.zip")
def task_download_df(produces):
    url = 'https://www.dropbox.com/scl/fo/npdgaje0u3ejd5o51ty01/h?dl=1&rlkey=1fra89j9ymk5lnf74lqaqd3bs'
    response = requests.get(url)
    with open(produces, "wb") as f:
        f.write(response.content)


data_files = {
    "VZmw_quarterly_lagsleads_1979_2019.dta": BLD / "python" / "data" / "VZmw_quarterly_lagsleads_1979_2019.dta",
    "eventclassification_2019.dta": BLD / "python" / "data" / "eventclassification_2019.dta",
    "cpiursai1977-2019.dta": BLD / "python" / "data" / "cpiursai1977-2019.dta",
    "cps_morg_2019_new.dta": BLD / "python" / "data" / "cps_morg_2019_new.dta",
}
for key, value in data_files.items():
    kwargs = {
        "key": key,
        "produces": value,
    }
    @pytask.mark.depends_on(SRC / "data" / "epp_ml_mw_data.zip")
    @pytask.mark.task(id=key, kwargs=kwargs)
    def task_unzip_df(depends_on, key):
        with ZipFile(depends_on, "r") as zip_ref:
            zip_ref.extract(key, path = BLD / "python" / "data")

@pytask.mark.depends_on(
    {
        "forbalance": BLD / "python" / "data" / "VZmw_quarterly_lagsleads_1979_2019.dta",
        "eventclass": BLD / "python" / "data" / "eventclassification_2019.dta",
        "cpi": BLD / "python" / "data" / "cpiursai1977-2019.dta",
        "cps_morg": BLD / "python" / "data" / "cps_morg_2019_new.dta",
        "state_codes": BLD / "python" / "data" / "state_codes.pkl",
        "quarter_codes": BLD / "python" / "data" / "quarter_codes.pkl",
        "month_codes": BLD / "python" / "data" / "month_codes.pkl",
    }
)
@pytask.mark.produces(BLD / "python" / "data" / "forpredictionmorg_full_eventstudy_2019.pkl")
def task_forprediction_eventstudy_data(depends_on,produces):
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
