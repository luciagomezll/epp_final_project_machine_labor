import numpy as np
import pandas as pd
import pytest
from machine_labor.config import BLD,TEST_DIR
from machine_labor.utilities import read_yaml
from machine_labor.data_management.clean_data_MW import get_forbalance_data
from machine_labor.data_management.clean_data_MW import get_prewindow_data
from machine_labor.data_management.clean_data_MW import _clean_cps_cpi_data
from machine_labor.data_management.clean_data_MW import get_totpop_data
from machine_labor.data_management.clean_data_MW import get_cps_cpi_data
from machine_labor.data_management.clean_data_MW import get_forprediction_eventstudy_data 
from machine_labor.data_management.clean_data_MW import get_fortraining_eventstudy_data

@pytest.fixture()
def input_data():
    data_forbalance =  pd.read_stata(BLD / "python" / "data" / "VZmw_quarterly_lagsleads_1979_2019.dta")
    data_eventclass = pd.read_stata(BLD / "python" / "data" / "eventclassification_2019.dta", convert_categoricals=False)
    data_cpi = pd.read_stata(BLD / "python" / "data" / "cpiursai1977-2019.dta")
    data_cps = pd.read_stata(BLD / "python" / "data" / "cps_morg_2019_new.dta", convert_categoricals=False)
    state_codes = pd.read_pickle(BLD / "python" / "data" / "state_codes.pkl")
    quarter_codes = pd.read_pickle(BLD / "python" / "data" / "quarter_codes.pkl")
    month_codes = pd.read_pickle(BLD / "python" / "data" / "month_codes.pkl")
    return (data_forbalance,data_eventclass,data_cpi,data_cps,state_codes,quarter_codes,month_codes)

@pytest.fixture()
def data_fortrain_original():
    data_fortrain = pd.read_stata(TEST_DIR / "data_management" / "fortraining_eventstudy_1979_2019.dta",convert_categoricals=False)
    return data_fortrain

@pytest.fixture()
def dim_intermediate_df():
    inputs ={
        "dim_forbalance_data": np.array([8364,4]),
        "dim_prewindow_data": np.array([8364,4]),
        "dim_clean_cps_cpi_data": np.array([659641,62]),
        "dim_totpop_data": np.array([8364,3]),
        "dim_merge_cps_cpi_data": np.array([659641,82]),
    }
    return inputs

@pytest.fixture()
def data_info():
    data_info = read_yaml(TEST_DIR / "data_management" / "data_info_fixture.yaml")
    return data_info

def test_compare_dimension_df_intermediate(input_data,dim_intermediate_df,data_info):
    data_forbalance,data_eventclass,data_cpi,data_cps,state_codes,quarter_codes,month_codes = input_data
    forbalance = get_forbalance_data(data_forbalance, quarter_codes)
    prewindow = get_prewindow_data(forbalance, data_eventclass, quarter_codes, state_codes)
    clean_cps_cpi = _clean_cps_cpi_data(data_cps, data_cpi, state_codes, quarter_codes, month_codes)
    clean_cps_cpi_temp = clean_cps_cpi.copy()
    totpop = get_totpop_data(clean_cps_cpi_temp)
    merge_cps_cpi = get_cps_cpi_data(clean_cps_cpi_temp)
    intermediate_df = data_info['intermediate_df']
    columns_to_drop = data_info['columns_to_drop']
    expected_dim = {}
    actual_dim = {}
    for df,name_df in zip((forbalance,prewindow,clean_cps_cpi,totpop,merge_cps_cpi),intermediate_df):
        expected_dim[f"expected_dim_{name_df}"] = dim_intermediate_df[f"dim_{name_df}_data"]
        drop_cols = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(drop_cols, axis=1)
        actual_dim[f"actual_dim_{name_df}"] = np.array([df.shape[0],df.shape[1]])
        np.testing.assert_array_almost_equal(expected_dim[f"expected_dim_{name_df}"],actual_dim[f"actual_dim_{name_df}"])

def test_compare_dimension_df_final(input_data,data_fortrain_original,data_info):
    data_forbalance,data_eventclass,data_cpi,data_cps,state_codes,quarter_codes,month_codes = input_data
    data_forprediction = get_forprediction_eventstudy_data(data_forbalance,
    data_eventclass,
    data_cps,
    data_cpi,
    quarter_codes,
    state_codes,
    month_codes)
    data_fortrain_replication = get_fortraining_eventstudy_data(data_forprediction)
    data_fortrain_replication = data_fortrain_replication.drop(data_info['columns_to_drop'], axis=1)
    assert set(data_fortrain_replication.columns).intersection(set(data_fortrain_original.columns))

def test_notna_columns_df_final(input_data, data_info):
    data_forbalance,data_eventclass,data_cpi,data_cps,state_codes,quarter_codes,month_codes = input_data
    data_forpred = get_forprediction_eventstudy_data(data_forbalance,
    data_eventclass,
    data_cps,
    data_cpi,
    quarter_codes,
    state_codes,
    month_codes)
    data_fortrain = get_fortraining_eventstudy_data(data_forpred)
    for var in data_info["important_variables"]:
        assert not data_fortrain[f'{var}'].isnull().all()

def test_compare_summary_stats(input_data, data_fortrain_original, data_info):
    data_forbalance,data_eventclass,data_cpi,data_cps,state_codes,quarter_codes,month_codes = input_data
    data_forprediction = get_forprediction_eventstudy_data(data_forbalance,
    data_eventclass,
    data_cps,
    data_cpi,
    quarter_codes,
    state_codes,
    month_codes)
    data_fortrain_replication = get_fortraining_eventstudy_data(data_forprediction)
    expected_stats = {}
    actual_stats = {}
    for var in data_info["important_variables"]:
        expected_stats[f"{var}"] = np.array([data_fortrain_original[f'{var}'].mean(),data_fortrain_original[f'{var}'].std()])
        actual_stats[f"{var}"] = np.array([data_fortrain_replication[f'{var}'].mean(),data_fortrain_replication[f'{var}'].std()])
        if var != "relMW_groups":
            assert np.allclose(expected_stats[f"{var}"],actual_stats[f"{var}"])
        else:
            assert not np.allclose(expected_stats[f"{var}"],actual_stats[f"{var}"])
