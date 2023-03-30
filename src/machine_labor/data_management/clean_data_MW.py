"""Download, clean and merge datasets and build variables for machine learning prediction of minimum wage worker classification.

"""
import numpy as np
import pandas as pd

def _setup():
    input = {
    "startyear" : 1979,
    "endyear" : 2019,
    "cpi_baseyear" : 2016,
    }
    return input

def get_fortraining_eventstudy_data(data):
    """Generates a dataset for prediction excluding self-employed.

    Args:
        data (pd.DataFrame): A cleaned dataset.

    Returns:
        pd.DataFrame: Dataset for prediction excluding self-employed.

    """
    fortraining = data[data["relMW_groups"].notna()]
    fortraining = fortraining[
        ~(fortraining["class"].isin([5, 6]) & (fortraining["year"] <= 1993))
    ]
    fortraining = fortraining[
        ~(fortraining["class94"].isin([6, 7]) & (fortraining["year"] >= 1994))
    ]
    return fortraining


def get_forprediction_eventstudy_data(
    data_forbalance,
    data_eventclass,
    data_cps,
    data_cpi,
    quarter_codes,
    state_codes,
    month_codes 
):
    """Generates a merged dataset for prediction purposes.

    Args:
        data_forbalance (pd.DataFrame): A raw dataset that 
            contains minimum wage data per state and quarter level.
        data_eventclass (pd.DataFrame): A raw dataset with information 
            to identify the relevant post and pre-period around prominent minimum wage changes. 
        data_cps (pd.DataFrame): The raw Current Population Survey (CPS)-ORG dataset containing hourly wages,
            weekly earnings, and a range of demographic variables. 
        data_cpi (pd.DataFrame): The raw Consumer Price Index (CPI) dataset per month.
        state_codes (pd.DataFrame): 
            A dataframe containing state information.
        quarter_codes (pd.DataFrame): 
            A dataframe containing quarter information.
        month_codes (pd.DataFrame): 
            A dataframe containing month information.

    Returns:
        pandas.DataFrame: A cleaned and merged dataset for prediction.

    """
    forbalance = get_forbalance_data(data_forbalance, quarter_codes)
    prewindow = get_prewindow_data(
        forbalance, data_eventclass, quarter_codes, state_codes
    )
    clean_cps_cpi = _clean_cps_cpi_data(
        data_cps, data_cpi, state_codes, quarter_codes, month_codes
    )
    totpop = get_totpop_data(clean_cps_cpi)
    data_cps_cpi = get_cps_cpi_data(clean_cps_cpi)
    merge_1 = pd.merge(
        data_cps_cpi, totpop, how="inner", on=["statenum", "quarterdate"]
    )
    merge_2 = pd.merge(
        merge_1,
        forbalance,
        how="inner",
        on=["statenum", "quarterdate", "year", "quarternum"],
    )
    merge_3 = pd.merge(merge_2, prewindow, how="inner", on=["statenum", "quarterdate","state_name"])
    merge_3["MW"] = np.exp(merge_3["logmw"])
    merge_3["ratio_mw"] = merge_3["origin_wage"] / merge_3["MW"]
    merge_3.loc[merge_3["MW"] == 0, "ratio_mw"] = 0
    merge_3.loc[
        (merge_3["ratio_mw"] < 1)
        & (merge_3["origin_wage"].notna())
        & (merge_3["origin_wage"] > 0),
        "relMW_groups",
    ] = 1
    merge_3.loc[
        (merge_3["ratio_mw"] >= 1)
        & (merge_3["ratio_mw"] < 1.25)
        & (merge_3["origin_wage"].notnull()),
        "relMW_groups",
    ] = 2
    merge_3.loc[
        (merge_3["ratio_mw"] >= 1.25) & (merge_3["origin_wage"].notna()), "relMW_groups"
    ] = 3
    merge_3["training"] = np.where(
        (merge_3["prewindow"] == 1) & (~merge_3["origin_wage"].isin([0, np.nan])), 1, 0
    )
    merge_3["validation"] = np.where(
        (merge_3["prewindow"] == 0)
        & (merge_3["postwindow"] == 0)
        & (~merge_3["origin_wage"].isin([0, np.nan])),
        1,
        0,
    )
    return merge_3

def get_prewindow_data(data_forbalance, data_eventclass, quarter_codes, state_codes):
    """Generate a data frame with the relevant post and pre-periods around a minimum wage change.

    Args:
        data_forbalance (pd.DataFrame): A dataframe with variables required to built post and pre-periods around a minimum wage change.
        data_eventclass (pd.DataFrame): A dataframe with information on events.
        quarter_codes (pd.DataFrame): A dataframe containing quarter information.
        state_codes (pd.DataFrame): A dataframe containing state information.
        
    Returns:
        (pd.DataFrame): A dataframe that identifies 20 quarters of a past minimum wage
            change and 12 quarters of a future minimum wage change by state and quarter level.

    """
    data_eventclass = data_eventclass.rename(columns={"quarterdate": "quarternum"})
    data_eventclass = _restrict_data(data_eventclass, quarter_codes)
    data_forb_event = pd.merge(data_forbalance, data_eventclass, how="left")
    data_forb_event["overallcountgroup"] = data_forb_event["overallcountgroup"].fillna(
        0
    )
    data_forb_event.loc[data_forb_event["fedincrease"] == 1, "overallcountgroup"] = 0
    data_forb_event.loc[
        data_forb_event["overallcountgroup"].notna()
        & data_forb_event["overallcountgroup"]
        > 1,
        "overallcountgroup",
    ] = 1
    data_forb_event["prewindow"] = 0
    data_forb_event["postwindow"] = data_forb_event["overallcountgroup"]

    for i in range(1, 13):
        data_forb_event[f"L{i}overallcountgroup"] = data_forb_event.groupby(
            ["statenum"]
        )["overallcountgroup"].shift(i, fill_value=0)
        data_forb_event[f"F{i}overallcountgroup"] = data_forb_event.groupby(
            ["statenum"]
        )["overallcountgroup"].shift(-i, fill_value=0)
        data_forb_event["prewindow"] = (
            data_forb_event["prewindow"] + data_forb_event[f"F{i}overallcountgroup"]
        )
        data_forb_event["postwindow"] = (
            data_forb_event["postwindow"] + data_forb_event[f"L{i}overallcountgroup"]
        )

    for i in range(13, 20):
        data_forb_event[f"L{i}overallcountgroup"] = data_forb_event.groupby(
            ["statenum"]
        )["overallcountgroup"].shift(i, fill_value=0)
        data_forb_event["postwindow"] = (
            data_forb_event["postwindow"] + data_forb_event[f"L{i}overallcountgroup"]
        )

    data_forb_event.loc[data_forb_event["postwindow"] >= 1, "postwindow"] = 1
    data_forb_event.loc[data_forb_event["prewindow"] >= 1, "prewindow"] = 1
    data_forb_event.loc[data_forb_event["postwindow"] == 1, "prewindow"] = 0
    prewindow = data_forb_event[["statenum", "quarterdate", "prewindow", "postwindow"]]
    prewindow = pd.merge(prewindow, state_codes, how="left", on=["statenum"])
    return prewindow

def get_forbalance_data(data_forbalance, quarter_codes):
    """Restrict the dataset 'forbalance' to the study period.

    Args:
        data_forbalance (pd.DataFrame): A dataset with variables required to built post and pre-periods around a minimum wage change.
        quarter_codes (pd.DataFrame): A dataframe containing quarter information.

    Returns:
        (pd.DataFrame): A dataset ready to be merged using the keys "quarterdate" and "year."

    """
    data_forbalance["quarterdate"] = pd.PeriodIndex(
        data_forbalance["quarterdate"], freq="Q"
    )
    data_forbalance = _restrict_data(data_forbalance, quarter_codes)
    return data_forbalance

def get_totpop_data(data_cps_cpi):
    """Generate the dataset containing total population variable (number of surveyed people) at state and quarter level.

    Args:
        data_cps_cpi (pandas.DataFrame): The dataset with merged CPS-ORG and CPI datasets.

    Returns:
        pd.DataFrame: A collapsed dataset at state and quarter level with total population variable.

    """
    list_var = ["monthdate", "quarterdate", "statenum", "earnwt"]
    totpop_temp = data_cps_cpi[list_var]
    totpop_temp["totalpopulation"] = totpop_temp.groupby(
        ["statenum", "monthdate"], as_index=False
    )["earnwt"].transform("sum")
    totpop_temp = totpop_temp[["statenum", "quarterdate", "totalpopulation"]]
    totpop = totpop_temp.groupby(["statenum", "quarterdate"], as_index=False).mean()
    return totpop


def get_cps_cpi_data(data_cps_cpi):
    """Clean the merged CPS-ORG and CPI dataset and build relevant variables.

    Args:
        data_cps_cpi (pandas.DataFrame): The dataset with merged CPS-ORG and CPI datasets.

    Returns:
        pandas.DataFrame: Contains the merged CPS-ORG and CPI dataset with additional created variables.

    """
    data_cps_cpi.loc[data_cps_cpi["paidhre"] == 1, "wage"] = (
        data_cps_cpi["earnhre"] / 100
    )
    data_cps_cpi.loc[data_cps_cpi["paidhre"] == 2, "wage"] = (
        data_cps_cpi["earnwke"] / data_cps_cpi["uhourse"]
    )
    data_cps_cpi.loc[
        (data_cps_cpi["paidhre"] == 2) & (data_cps_cpi["uhourse"] == 0), "wage"
    ] = np.nan
    data_cps_cpi["hoursimputed"] = np.where(
        (data_cps_cpi["I25a"].notna()) & (data_cps_cpi["I25a"] > 0), 1, 0
    )
    data_cps_cpi["wageimputed"] = np.where(
        (data_cps_cpi["I25c"].notna()) & (data_cps_cpi["I25c"] > 0), 1, 0
    )
    data_cps_cpi["earningsimputed"] = np.where(
        (data_cps_cpi["I25d"].notna()) & (data_cps_cpi["I25d"] > 0), 1, 0
    )

    varlist = ["hoursimputed", "earningsimputed", "wageimputed"]
    for column in data_cps_cpi[varlist]:
        data_cps_cpi.loc[data_cps_cpi["year"].isin(range(1989, 1994)), column] = 0
        data_cps_cpi.loc[
            (data_cps_cpi["year"] == 1994) |
            ((data_cps_cpi["year"] == 1995) & (data_cps_cpi["month_num"] <= 8)),
            column,
        ] = 0

    data_cps_cpi.loc[
        (data_cps_cpi["year"].isin(range(1989, 1994)))
        & (data_cps_cpi["earnhr"].isin([np.nan, 0]))
        & ((data_cps_cpi["earnhre"].notna()) & (data_cps_cpi["earnhre"] > 0)),
        "wageimputed",
    ] = 1

    data_cps_cpi.loc[
        (data_cps_cpi["year"].isin(range(1989, 1994)))
        & (data_cps_cpi["uhours"].isin([np.nan, 0]))
        & (data_cps_cpi["uhourse"].notna() & (data_cps_cpi["uhourse"] > 0)),
        "hoursimputed",
    ] = 1

    data_cps_cpi.loc[
        (data_cps_cpi["year"].isin(range(1989, 1994)))
        & (data_cps_cpi["uearnwk"].isin([np.nan, 0]))
        & (data_cps_cpi["earnwke"].notna() & (data_cps_cpi["earnwke"] > 0)),
        "earningsimputed",
    ] = 1

    data_cps_cpi["imputed"] = np.where(
        (data_cps_cpi["paidhre"] == 2) &
        (
            (data_cps_cpi["hoursimputed"] == 1) | (data_cps_cpi["earningsimputed"] == 1)
        ), 1, 0,
    )
    data_cps_cpi.loc[
        (data_cps_cpi["paidhre"] == 1) & (data_cps_cpi["wageimputed"] == 1), "imputed"
    ] = 1
    data_cps_cpi.loc[data_cps_cpi["imputed"] == 1, "wage"] = np.nan
    data_cps_cpi["logwage"] = np.where(data_cps_cpi["wage"].notna() & (data_cps_cpi["wage"] != 0),np.log(data_cps_cpi["wage"]),np.nan)
    data_cps_cpi["origin_wage"] = data_cps_cpi["wage"]
    data_cps_cpi["wage"] = (
        (data_cps_cpi["origin_wage"]) / (data_cps_cpi["cpi"] / 100)
    ) * 100
    data_cps_cpi.loc[data_cps_cpi["cpi"] == 0, "wage"] = np.nan

    #data_cps_cpi = data_cps_cpi.drop(["mlr","hispanic","black","dmarried","hgradecp","hsl","hsd","hs12"], axis=1)
    data_cps_cpi["mlr"] = np.where(
        data_cps_cpi["year"].isin(range(1979, 1989)), data_cps_cpi["esr"], np.nan
    )
    data_cps_cpi.loc[data_cps_cpi["year"].isin(range(1989, 1994)),"mlr"] = data_cps_cpi["lfsr89"]
    data_cps_cpi.loc[data_cps_cpi["year"].isin(range(1994, 2020)),"mlr"] = data_cps_cpi["lfsr94"]

    data_cps_cpi["hispanic"] = np.where(
        (data_cps_cpi["year"].isin(range(1976, 2003)))
        & (data_cps_cpi["ethnic"].isin(range(1, 8))),
        1, 0,
    )
    data_cps_cpi.loc[
    (data_cps_cpi["year"].isin(range(2003, 2014)))
    & (data_cps_cpi["ethnic"].isin(range(1, 6))),
    "hispanic"
    ] = 1
    data_cps_cpi.loc[
    (data_cps_cpi["year"].isin(range(2014, 2020)))
    & (data_cps_cpi["ethnic"].isin(range(1, 10))),
    "hispanic"
    ] = 1
    data_cps_cpi["black"] = np.where(
        (data_cps_cpi["race"] == 2) & (data_cps_cpi["hispanic"] == 0), 1, 0
    )
    data_cps_cpi.loc[data_cps_cpi["race"] >= 2,"race"] = 2
    data_cps_cpi["dmarried"] = np.where(data_cps_cpi["marital"] <= 2, 1, 0)
    data_cps_cpi.loc[data_cps_cpi["marital"].isna(),"dmarried"] = np.nan
    data_cps_cpi["sex"] = data_cps_cpi["sex"].replace(2, 0)
    data_cps_cpi["hgradecp"] = np.where(
        data_cps_cpi["gradecp"] == 1, data_cps_cpi["gradeat"], np.nan
    )
    data_cps_cpi["hgradecp"] = np.where(
        data_cps_cpi["gradecp"] == 2,
        data_cps_cpi["gradeat"] - 1,
        data_cps_cpi["hgradecp"],
    )
    data_cps_cpi.loc[data_cps_cpi["ihigrdc"].notna() & data_cps_cpi["hgradecp"].isna(),"hgradecp"] = data_cps_cpi["ihigrdc"]

    grade92code = list(range(31, 47))
    impute92code = (0, 2.5, 5.5, 7.5, 9, 10, 11, 12, 12, 13, 14, 14, 16, 18, 18, 18)

    for i, j in zip(grade92code, impute92code):
        a = i
        b = j
        data_cps_cpi.loc[data_cps_cpi["grade92"] == a, "hgradecp"] = b

    data_cps_cpi["hgradecp"] = data_cps_cpi["hgradecp"].replace(-1, 0)
    data_cps_cpi["hsl"] = np.where(data_cps_cpi["hgradecp"] <= 12, 1, 0)
    data_cps_cpi["hsd"] = np.where(
        (data_cps_cpi["hgradecp"] < 12) & (data_cps_cpi["year"] < 1992), 1, 0
    )
    data_cps_cpi.loc[(data_cps_cpi["grade92"] <= 38) & (data_cps_cpi["year"] >= 1992),"hsd"] = 1
    data_cps_cpi["hs12"] = np.where(
        (data_cps_cpi["hsl"] == 1) & (data_cps_cpi["hsd"] == 0), 1, 0
    )
    data_cps_cpi["conshours"] = np.where(
        data_cps_cpi["I25a"] == 0, data_cps_cpi["uhourse"], np.nan
    )
    data_cps_cpi["hsl40"] = np.where(
        (data_cps_cpi["hsl"] == 1) & (data_cps_cpi["age"] < 40), 1, 0
    )
    data_cps_cpi["hsd40"] = np.where(
        (data_cps_cpi["hsd"] == 1) & (data_cps_cpi["age"] < 40), 1, 0
    )
    data_cps_cpi["sc"] = np.where(
        data_cps_cpi["hgradecp"].isin([13, 14, 15]) & (data_cps_cpi["year"] < 1992),
        1,
        0,
    )
    data_cps_cpi.loc[data_cps_cpi["grade92"].isin(range(40, 43)) & (data_cps_cpi["year"] >= 1992),"sc"] = 1
    data_cps_cpi["coll"] = np.where(
        (data_cps_cpi["hgradecp"] > 15) & (data_cps_cpi["year"] < 1992), 1, 0
    )
    data_cps_cpi.loc[(data_cps_cpi["grade92"] > 42) & (data_cps_cpi["year"] >= 1992),"coll"] = 1
    data_cps_cpi["ruralstatus"] = np.where(data_cps_cpi["smsastat"] == 2, 1, 2)
    data_cps_cpi = data_cps_cpi.drop(['smsastat','smsa80','smsa93','smsa04'], axis=1)
    data_cps_cpi["veteran"] = np.where(
        data_cps_cpi["veteran_old"].notna()
        & (data_cps_cpi["veteran_old"] != 6),
        1, 0,
    )
    data_cps_cpi.loc[data_cps_cpi["vet1"].notna(),"veteran"] = 1
    data_cps_cpi["educcat"] = np.where(data_cps_cpi["hsd"] == 1, 1, 0)
    data_cps_cpi.loc[data_cps_cpi["hs12"] == 1,"educcat"] = 2
    data_cps_cpi.loc[data_cps_cpi["sc"] == 1,"educcat"] = 3
    data_cps_cpi.loc[data_cps_cpi["coll"] == 1,"educcat"] = 4
    data_cps_cpi["agecat"] = pd.cut(
        data_cps_cpi["age"], bins=[0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 100]
    )

    return data_cps_cpi



def _clean_cps_cpi_data(
    data_cps, data_cpi, state_codes, quarter_codes, month_codes
):
    """Clean and merge the current population survey (CPS)-ORG
        and CPI dataset.

    Args:
        data_cps (pd.DataFrame): The raw CPS-ORG dataset containing hourly wages,
            weekly earnings, and a range of demographic variables. 
        data_cpi (pd.DataFrame): The raw CPI dataset per month.
        state_codes (pd.DataFrame): 
            A dataframe containing state information.
        quarter_codes (pd.DataFrame): 
            A dataframe containing quarter information.
        month_codes (pd.DataFrame): 
            A dataframe containing month information.

    Returns:
        (pd.DataFrame): A dataset with merged CPS-ORG and CPI datasets.

    """
    data_cpi = data_cpi.melt(id_vars="year")
    data_cpi = data_cpi.rename(columns={"variable": "monthnum", "value": "cpi", "month": "monthnum"})
    data_cpi = data_cpi.loc[data_cpi["year"].between(_setup()["startyear"],_setup()["endyear"])]
    data_cpi["monthnum"] = data_cpi["monthnum"].astype("category")
    data_cpi = pd.merge(data_cpi, month_codes, how="left")
    cpibase = data_cpi.loc[data_cpi["year"] == _setup()["cpi_baseyear"], "cpi"].mean()
    data_cpi["cpi"] = 100 * (data_cpi["cpi"]/cpibase)

    data_cps.loc[:, "rowid"] = range(1, len(data_cps) + 1)
    data_cps = data_cps[['hhid','hhnum','lineno','minsamp','month','state','age','marital',
                         'race','sex','esr','ethnic','uhours','earnhr','uearnwk','earnwt',
                        'uhourse','paidhre','earnhre','earnwke','I25a','I25b','I25c','I25d',
                        'year','lfsr89','lfsr94','statenum','monthdate','quarterdate',
                        'quarter','division','gradeat','gradecp','ihigrdc','grade92','class',
                         'class94','smsa70','smsa80','smsa93','smsa04','smsastat',
                        'prcitshp','penatvty','veteran','pemntvty','pefntvty','vet1','rowid',
                        'ind70','ind80','ind02','occ70','occ80','occ802','occurnum','occ00',
                        'occ002','occ2011','occ2012']]
    data_cps = data_cps.rename(columns={
    "veteran": "veteran_old","quarterdate": "quarternum","month": "month_num"})
    data_cps = data_cps.loc[data_cps["year"] >= _setup()["startyear"]]

    data_merge_cps_cpi = pd.merge(
        data_cps, data_cpi, how="left", on=["year", "month_num", "monthdate"]
    )
    data_merge_cps_cpi = pd.merge(
        data_merge_cps_cpi, state_codes, how="left", on=["statenum"]
    )
    data_merge_cps_cpi = pd.merge(
        data_merge_cps_cpi, quarter_codes, how="left", on=["quarternum"]
    )
    data_merge_cps_cpi["quarterdate"] = data_merge_cps_cpi["quarterdate"].astype(str)
    return data_merge_cps_cpi

def _restrict_data(data, quarter_codes):
    """Generate the variable 'year' and restrict the dataset to the study period.

    Args:
        data (pd.DataFrame): The data set.
        quarter_codes (pd.DataFrame): 
            A dataframe containing quarter information.

    Returns:
        (pd.DataFrame): dataset ready to be merged using the keys "quarterdate" and "year."

    """
    column_list = ["quarterdate", "quarternum"]
    column = [x for x in data.columns if any(y in x for y in column_list)]
    data = pd.merge(data, quarter_codes, how="left", on=column)
    data["quarterdate"] = data["quarterdate"].astype(str)
    data["year"] = pd.to_datetime(data["quarterdate"]).dt.year
    data = data.loc[data["year"] >= _setup()["startyear"]]
    return data
