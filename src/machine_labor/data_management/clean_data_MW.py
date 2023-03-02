import numpy as np
import pandas as pd
import datetime as dt

startyear = 1979
endyear = 2019
cpi_baseyear = 2016
change_cpsmorg_ncol = {"veteran": "veteran_old","quarterdate": "quarternum",
                       "statenum": "state_name","month":"month_num"}

def get_fortraining_eventstudy_data(data_forbalance,data_eventclass,data_cps_morg,data_cpi,
                                    quarter_codes,state_codes,month_codes):
    forbalance = get_forbalance_data(data_forbalance,quarter_codes)
    prewindow = get_prewindow_data(forbalance,data_eventclass,quarter_codes,state_codes)
    totpop = get_totpop_data(data_cps_morg,data_cpi,state_codes,quarter_codes,month_codes)
    cps_morg_cpi = get_cps_morg_cpi_data(data_cps_morg,data_cpi,state_codes,quarter_codes,month_codes)
    merge_1 = pd.merge(cps_morg_cpi,totpop,how="inner",on=["statenum","quarterdate"])
    merge_2 = pd.merge(merge_1,forbalance,how="inner",on=["statenum","quarterdate"])
    merge_3 = pd.merge(merge_2,prewindow,how="inner",on=["statenum","quarterdate"])
    merge_3['MW'] = np.exp(merge_3['logmw'])
    merge_3['ratio_mw'] = merge_3['origin_wage']/merge_3['MW']
    merge_3['relMW_groups'] = np.where((merge_3['ratio_mw']<1) & (merge_3['origin_wage'].notna()) & (merge_3['origin_wage']>0),1,np.nan)
    merge_3['relMW_groups'] = np.where((merge_3['ratio_mw']>=1) & (merge_3['origin_wage']<1.25) & (merge_3['origin_wage'].notna()),2,merge_3['relMW_groups'])
    merge_3['relMW_groups'] = np.where((merge_3['ratio_mw']>=1.25) & (merge_3['origin_wage'].notna()),3,merge_3['relMW_groups'])
    merge_3['training'] = np.where((merge_3['prewindow']==1) & (~merge_3['origin_wage'].isin([0,np.nan])),1,0)
    merge_3['validation'] = np.where((merge_3['prewindow']==0) & (merge_3['postwindow']==0) & (~merge_3['origin_wage'].isin([0,np.nan])),1,0)
    return merge_3

def get_forbalance_data(data_forbalance,quarter_codes):
    data_forbalance['quarterdate'] = pd.PeriodIndex(data_forbalance['quarterdate'], freq='Q')
    forbalance = _restrict_data(data_forbalance,quarter_codes)
    return forbalance

def get_prewindow_data(forbalance,data_eventclass,quarter_codes,state_codes):
    #data_forbalance = get_forbalance_data(data_forbalance,quarter_codes)
    data_eventclass = data_eventclass.rename(columns={"quarterdate": "quarternum"})
    data_eventclass = _restrict_data(data_eventclass,quarter_codes)
    data_forb_event = pd.merge(forbalance,data_eventclass,how="left")
    data_forb_event["overallcountgroup"] = data_forb_event["overallcountgroup"].fillna(0)
    data_forb_event.loc[data_forb_event["fedincrease"] ==1,'overallcountgroup'] = 0
    data_forb_event.loc[data_forb_event["overallcountgroup"].notna() & data_forb_event["overallcountgroup"]>1,'overallcountgroup'] = 1 
    data_forb_event["prewindow"] = 0
    data_forb_event["postwindow"] = data_forb_event["overallcountgroup"]

    for i in range(1,13):
        data_forb_event[f"L{i}overallcountgroup"] =  data_forb_event.groupby(['statenum'])["overallcountgroup"].shift(i, fill_value=0)
        data_forb_event[f"F{i}overallcountgroup"] =  data_forb_event.groupby(['statenum'])["overallcountgroup"].shift(-i, fill_value=0)
        data_forb_event["prewindow"] = data_forb_event["prewindow"] + data_forb_event[f"F{i}overallcountgroup"]
        data_forb_event["postwindow"] = data_forb_event["postwindow"] + data_forb_event[f"L{i}overallcountgroup"]

    for i in range(13,20):
        data_forb_event[f"L{i}overallcountgroup"] =  data_forb_event.groupby(['statenum'])["overallcountgroup"].shift(i, fill_value=0)
        data_forb_event["postwindow"] = data_forb_event["postwindow"] + data_forb_event[f"L{i}overallcountgroup"]

    data_forb_event.loc[data_forb_event["postwindow"] >=1,'postwindow'] = 1
    data_forb_event.loc[data_forb_event["prewindow"] >=1,'prewindow'] = 1
    data_forb_event.loc[data_forb_event["postwindow"] ==1,'prewindow'] = 0
    prewindow = data_forb_event[['statenum','quarterdate','prewindow','postwindow']]
    prewindow = pd.merge(prewindow,state_codes,how="left",on=['statenum'])
    return prewindow

def get_totpop_data(data_cps_morg,data_cpi,state_codes,quarter_codes,month_codes):
    cps_morg_cpi = _clean_cps_morg_cpi_data(data_cps_morg,data_cpi,state_codes,quarter_codes,month_codes)
    list_var = ["monthdate", "quarterdate", "statenum", "earnwt"]
    totpop = cps_morg_cpi[list_var]
    totpop['totalpopulation'] = totpop.groupby(['statenum', 'monthdate'],as_index=False)['earnwt'].transform('sum')
    totpop = totpop.groupby(['statenum', 'quarterdate'],as_index=False).mean()[['totalpopulation']]
    return totpop

def get_cps_morg_cpi_data(data_cps_morg,data_cpi,state_codes,quarter_codes,month_codes):
    cps_morg_cpi = _clean_cps_morg_cpi_data(data_cps_morg,data_cpi,state_codes,quarter_codes,month_codes)
    cps_morg_cpi['wage'] = np.where(cps_morg_cpi["paidhre"]==1,cps_morg_cpi["earnhre"]/100,np.nan)
    cps_morg_cpi['wage'] = np.where(cps_morg_cpi["paidhre"]==2,cps_morg_cpi["earnwke"]/cps_morg_cpi["uhourse"],cps_morg_cpi['wage'])
    cps_morg_cpi['hoursimputed'] = np.where(cps_morg_cpi["I25a"].notna() & cps_morg_cpi["I25a"] > 0 ,1,0)
    cps_morg_cpi['wageimputed'] = np.where(cps_morg_cpi["I25c"].notna() & cps_morg_cpi["I25c"] > 0 ,1,0)
    cps_morg_cpi['earningsimputed'] = np.where(cps_morg_cpi["I25d"].notna() & cps_morg_cpi["I25d"] > 0 ,1,0) 
    varlist = ['hoursimputed','earningsimputed','wageimputed']

    for column in cps_morg_cpi[varlist]:
        cps_morg_cpi[column] = np.where(cps_morg_cpi['year'].isin(range(1989,1994)),0,cps_morg_cpi[column])
        cps_morg_cpi[column] = np.where((cps_morg_cpi['year']==1994) | (cps_morg_cpi['year']==1995 & (cps_morg_cpi['month_num']<=8)),0,cps_morg_cpi[column])

    cps_morg_cpi['wageimputed'] = np.where((cps_morg_cpi["year"].isin(range(1989,1994))) & (cps_morg_cpi["earnhr"].isna() & cps_morg_cpi["earnhr"]==0) & (cps_morg_cpi["earnhre"].notna() & cps_morg_cpi["earnhre"]>0),1,cps_morg_cpi['wageimputed'])
    cps_morg_cpi['hoursimputed'] = np.where((cps_morg_cpi["year"].isin(range(1989,1994))) & (cps_morg_cpi["uhours"].isna() & cps_morg_cpi["uhours"]==0) & (cps_morg_cpi["uhourse"].notna() & cps_morg_cpi["uhourse"]>0),1,cps_morg_cpi['hoursimputed'])
    cps_morg_cpi['earningsimputed'] = np.where((cps_morg_cpi["year"].isin(range(1989,1994))) & (cps_morg_cpi["uearnwk"].isna() & cps_morg_cpi["uearnwk"]==0) & (cps_morg_cpi["earnwke"].notna() & cps_morg_cpi["earnwke"]>0),1,cps_morg_cpi['earningsimputed'])
    cps_morg_cpi['imputed'] = np.where((cps_morg_cpi['paidhre']==2) & ((cps_morg_cpi['hoursimputed']==1) | (cps_morg_cpi['earningsimputed']==1)),1,0)
    cps_morg_cpi['imputed'] = np.where((cps_morg_cpi['paidhre']==1) & (cps_morg_cpi['wageimputed']==1),1,cps_morg_cpi['imputed'])
    cps_morg_cpi['wage'] = np.where(cps_morg_cpi['imputed']==1,np.nan,cps_morg_cpi['wage'])
    cps_morg_cpi['logwage'] = np.log(cps_morg_cpi['wage'])
    cps_morg_cpi['origin_wage'] = cps_morg_cpi['wage'] 
    cps_morg_cpi['wage'] = (cps_morg_cpi['origin_wage']/(cps_morg_cpi['cpi']/100))*100
    cps_morg_cpi['mlr'] = np.where(cps_morg_cpi['year'].isin(range(1979,1989)),cps_morg_cpi['esr'],np.nan) 
    cps_morg_cpi['mlr'] = np.where(cps_morg_cpi['year'].isin(range(1989,1994)),cps_morg_cpi['lfsr89'],cps_morg_cpi['mlr']) 
    cps_morg_cpi['mlr'] = np.where(cps_morg_cpi['year'].isin(range(1994,2020)),cps_morg_cpi['lfsr94'],cps_morg_cpi['mlr'])    
    cps_morg_cpi['hispanic'] = np.where((cps_morg_cpi['year'].isin(range(1976,2003))) & (cps_morg_cpi['ethnic'].isin(range(1,8))),1,0) 
    cps_morg_cpi['hispanic'] = np.where((cps_morg_cpi['year'].isin(range(2003,2014))) & (cps_morg_cpi['ethnic'].isin(range(1,6))),1,cps_morg_cpi['hispanic']) 
    cps_morg_cpi['hispanic'] = np.where((cps_morg_cpi['year'].isin(range(2014,2020))) & (cps_morg_cpi['ethnic'].isin(range(1,10))),1,cps_morg_cpi['hispanic'])
    cps_morg_cpi['black'] = np.where((cps_morg_cpi['race']==2) & (cps_morg_cpi['hispanic']==0),1,0) 
    cps_morg_cpi['race'] = np.where(cps_morg_cpi['race']>=2,2,cps_morg_cpi['race']) 
    cps_morg_cpi['dmarried'] = np.where(cps_morg_cpi["marital"].notna() & cps_morg_cpi["marital"] <= 2,1,0)
    cps_morg_cpi['sex'] = cps_morg_cpi['sex'].replace(2,0) 
    cps_morg_cpi['hgradecp'] = np.where(cps_morg_cpi['gradecp']==1,cps_morg_cpi['gradeat'],np.nan)
    cps_morg_cpi['hgradecp'] = np.where(cps_morg_cpi['gradecp']==2,cps_morg_cpi['gradeat']-1,cps_morg_cpi['hgradecp'])
    cps_morg_cpi['hgradecp'] = np.where(cps_morg_cpi['ihigrdc'].notna() & cps_morg_cpi['hgradecp'].isna(),cps_morg_cpi['ihigrdc'],cps_morg_cpi['hgradecp'])
    grade92code = list(range(31,46))
    impute92code = (0,2.5,5.5,7.5,9,10,11,12,12,13,14,14,16,18,18,18)

    for i,j in zip(grade92code,impute92code):
        a = grade92code[i]
        b = impute92code[j]
        cps_morg_cpi.loc[cps_morg_cpi['grade92']==a,'hgradecp'] = b
       
    cps_morg_cpi['hgradecp'] = cps_morg_cpi['hgradecp'].replace(-1,0)
    cps_morg_cpi['hsl'] = np.where(cps_morg_cpi['hgradecp']<=12,1,0)
    cps_morg_cpi['hsd'] = np.where((cps_morg_cpi['hgradecp']< 12) & (cps_morg_cpi['year']< 1992),1,0)
    cps_morg_cpi['hsd'] = np.where((cps_morg_cpi['grade92']<=38) & (cps_morg_cpi['year']>=1992),1,cps_morg_cpi['hsd'])
    cps_morg_cpi['hs12'] = np.where((cps_morg_cpi['hsl']==1) & (cps_morg_cpi['hsd']==0),1,0)
    cps_morg_cpi['conshours'] = np.where(cps_morg_cpi['I25a']==0,cps_morg_cpi['uhourse'],np.nan)
    cps_morg_cpi['hsl40'] = np.where((cps_morg_cpi['hsl']==1) & (cps_morg_cpi['age']<40),1,0)
    cps_morg_cpi['hsd40'] = np.where((cps_morg_cpi['hsd']==1) & (cps_morg_cpi['age']<40),1,0)
    cps_morg_cpi['sc'] = np.where(cps_morg_cpi['hgradecp'].isin([13,14,15]) & (cps_morg_cpi['year']<1992),1,0)
    cps_morg_cpi['sc'] = np.where(cps_morg_cpi['grade92'].isin(range(40,43)) & (cps_morg_cpi['year']>=1992),1,cps_morg_cpi['sc'])
    cps_morg_cpi['coll'] = np.where((cps_morg_cpi['hgradecp']>15) & (cps_morg_cpi['year']<1992),1,0)
    cps_morg_cpi['coll'] = np.where((cps_morg_cpi['grade92']>42) & (cps_morg_cpi['year']>=1992),1,cps_morg_cpi['coll'])
    cps_morg_cpi['ruralstatus'] = np.where(cps_morg_cpi['smsastat']==2,1,2)
    cps_morg_cpi['veteran'] = np.where(cps_morg_cpi['veteran_old'].notna() & (cps_morg_cpi['veteran_old']!='Nonveteran'),1,0)
    cps_morg_cpi['veteran'] = np.where(cps_morg_cpi['vet1'].notna(),1,cps_morg_cpi['veteran'])
    cps_morg_cpi['educcat'] = np.where(cps_morg_cpi['hsd']==1,1,0)
    cps_morg_cpi['educcat'] = np.where(cps_morg_cpi['hs12']==1,2,cps_morg_cpi['educcat'])
    cps_morg_cpi['educcat'] = np.where(cps_morg_cpi['sc']==1,3,cps_morg_cpi['educcat'])
    cps_morg_cpi['educcat'] = np.where(cps_morg_cpi['coll']==1,4,cps_morg_cpi['educcat'])
    cps_morg_cpi["agecat"] = pd.cut(cps_morg_cpi["age"], bins=[0, 20, 25, 30,35, 40, 45, 50, 55, 60,  100])
    return cps_morg_cpi

def _restrict_data(data,quarter_codes):
    column_list = ['quarterdate','quarternum']
    column = [x for x in data.columns if any(y in x for y in column_list)]
    data = pd.merge(data,quarter_codes,how='left',on=column)
    data['quarterdate'] = data['quarterdate'].astype(str)
    data['year'] = [x[:4] for x in data['quarterdate']]
    data['year'] = data['year'].astype(int)
    data = data.loc[data['year']>=startyear]
    return data

def _clean_cps_morg_cpi_data(cps_morg,data_cpi,state_codes,quarter_codes,month_codes):
    data_cpi = data_cpi.melt(id_vars="year")
    data_cpi = data_cpi.rename(columns={"variable": "monthnum", "value": "cpi"})
    data_cpi = data_cpi.loc[(data_cpi["year"] >= startyear) & (data_cpi["year"] <= endyear)]
    data_cpi["monthnum"] = data_cpi["monthnum"].astype("category")
    cpi = pd.merge(data_cpi,month_codes,how="left")
    cpi["cpibase"] = cpi.loc[cpi["year"] == cpi_baseyear,"cpi"].mean()
    cpi["cpi"] = 100*cpi["cpi"]/cpi["cpibase"]

    cps_morg = cps_morg.rename(columns=change_cpsmorg_ncol)
    cps_morg = cps_morg.loc[cps_morg['year']>=startyear]
    
    cps_morg_cpi = pd.merge(cps_morg,cpi,how="inner",on=["year","month","monthdate"])
    cps_morg_cpi = pd.merge(cps_morg_cpi,state_codes,how="left",on=['statenum'])
    cps_morg_cpi = pd.merge(cps_morg_cpi,quarter_codes,how="left",on=['quarternum'])
    return cps_morg_cpi
