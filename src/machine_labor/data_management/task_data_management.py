import pytask
from machine_labor.config import BLD
from machine_labor.config import SRC

import pandas as pd
from machine_labor.data_management import clean_data
from machine_labor.utilities import read_yaml


@pytask.mark.depends_on(
    {
        "data_info": SRC / "data_management" / "data_info.yaml",
        "data": SRC / "data" / "data.csv",
    }
)
@pytask.mark.produces(BLD / "python" / "data" / "data_clean.csv")
def task_clean_data_python(depends_on, produces):
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["data"])
    data = clean_data(data, data_info)
    data.to_csv(produces, index=False)



