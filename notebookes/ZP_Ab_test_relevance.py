#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import StructType

if "sparkContext" in dir():
    sparkContext.stop()
    
spark_conf = {
    "spark.dynamicAllocation.enabled": True,
    "spark.shuffle.service.enabled": True,
    "spark.dynamicAllocation.executorIdleTimeout": "60s",
    "spark.dynamicAllocation.cachedExecutorIdleTimeout": "10800s",
    "spark.scheduler.mode": "FAIR",
    "spark.dynamicAllocation.maxExecutors": 50,
    "spark.executor.memory": "6g",
    "spark.driver.memory": "6g",
    "spark.executor.cores": 1,
    "spark.port.maxRetries": 100,
    "spark.driver.maxResultSize": "15g",
    "spark.blacklist.enabled": True,
    "spark.sql.shuffle.partitions": 1000,
    "spark.yarn.max.executor.failures": 1000,
    "spark.submit.deployMode" : "client"
}
conf = SparkConf().setAll(spark_conf.items())
sqc = (SparkSession.builder
             .appName("AB test relevance")
             .config(conf=conf)
             .getOrCreate())


# In[1]:


import pathlib
import json

def date_path(date):
    return "{}/{}/{}/".format(date[0], "0" + str(date[1]) if date[1] < 10 else date[1], "0" + str(date[2]) if date[2] < 10 else date[2])

def hdfs_path(prefix, date, postfix=""):
    if prefix[0] != "/":
        prefix = "/" + prefix
    if prefix[-1] == "/":
        prefix = prefix[:-1]
    if postfix and postfix[0] == "/":
        postfix = postfix[1:]
    return "{}/{}{}".format(prefix, date_path(date), postfix)

def hdfs_paths(prefix, dates, postfix=""):
    return [hdfs_path(prefix, date, postfix) for date in dates]

def hdfs_statlogs_path(table_name, date):
    return hdfs_path("statlogs", date, table_name)

def hdfs_statlogs_paths(table_name, dates):
    return [hdfs_statlogs_path(table_name, date) for date in dates]

class SchemaCache:
    """ SchemaCache is useful when dealing with big data stored in json files. 
    This class caches schema to disk once it is loaded. 
    
    usage:
    >> schema_cache = Schema(backup_file="schema_cache.json", json_path="/path/to/json/on/hdfs")
    >> df = sqc.read.format("json").schema(schema_cache.schema).load(filenames)
    
    """
    def __init__(self, backup_file: str, json_path=""):
        self.path = pathlib.Path(backup_file)
        self.json_path = json_path
    
    def _save(self, df):
        with self.path.open('w') as f:
            f.write(df.schema.json())
            
    @property
    def schema(self):
        if not self.path.is_file():
            if self.json_path:
                df = sqc.read.json(self.json_path, samplingRatio=0.01)
            else:
                raise Exception("no source schema")
            self._save(df)
        
        with self.path.open('r') as f:
            self.json = f.read()
            struct = StructType.fromJson(json.loads(self.json))
            return struct

def get_statlog_df(name, dates):
    filenames = hdfs_statlogs_paths(name, dates)
    
    schema = SchemaCache(backup_file="schema_{}.json".format(name), json_path=hdfs_statlogs_path(name, dates[0]))
    
    df = sqc.read.format("json")        .schema(schema.schema)        .load(filenames)

    return df


# In[2]:


# Config
DATES = [(2020, 12, 2)]


# In[4]:


# context logs
cont_sdf = get_statlog_df('context', DATES)
cont_sdf.createOrReplaceTempView('context_sdf')

# context cpi logs
cont_cpi_sdf = get_statlog_df('context_cpi', DATES)
cont_cpi_sdf.createOrReplaceTempView('context_cpi_sdf')


# In[5]:


query = """
select 
    u.randomid as randomid,
    u.requestid as requestid, 
    u.impressionid as impressionid, 
    u.timestamp as timestamp,
    MAX(u.zoneid) AS zoneid,
    SUM(u.cpi) as cpi,
    MAX(testid) as testid
    
from (

select 
    requestid, 
    impressionid, 
    result.randomid as randomid,
    timestamp as timestamp,
    zone as zoneid,
    0 as cpi,
    testid
    
from context_sdf
    lateral view explode(results) res as result

where result.cpi IS null

UNION ALL

select 
    requestid,
    impressionid,
    randomid,
    timestamp,
    zone as zoneid,
    (cpi / 100) as cpi,
    testid
    
from context_cpi_sdf) as u

group by
    testid,
    u.randomid,
    u.requestid, 
    u.impressionid,
    u.timestamp
"""

sqc.sql(query).createOrReplaceTempView('context_sdf1')


# In[6]:


click_sdf = get_statlog_df('clean_click', DATES)
click_sdf.createOrReplaceTempView('click_sdf')


# In[7]:


query = """
select
    impression.requestid as requestid,
    impression.randomid as randomid,
    impression.impressionid as impressionid,
    impression.timestamp as timestamp,
    impression.zone,
    realcpc / 100 as realcpc,
    1 as clk
from click_sdf
where network='context'
"""

sqc.sql(query).createOrReplaceTempView("click_sdf1")


# In[8]:


query = """
select
    cl.requestid as requestid,
    cl.randomid as randomid,
    cl.impressionid as impressionid,
    cl.timestamp as timestamp,
    cl.zone,
    cl.realcpc / 100 as realcpc,
    cl.clk as clk,
    c.testid as testid
from context_sdf1 as c join click_sdf1 as cl on
    c.randomid = cl.randomid and
    c.timestamp = cl.timestamp
"""

sqc.sql(query).createOrReplaceTempView("sklik")


# In[9]:


auc_sdf = get_statlog_df('auction', DATES)
auc_sdf.createOrReplaceTempView('auc_sdf')


# In[10]:


query = """

  -- Tabulka vitezu pro zjisteni zda Sklik vyhral ci nikoliv
select
    a.requestid as requestid,
    imps.impressionid as impressionid,
    imps.stage as winner_stage, 
    imps.netprice as winner_netprice,
    if(imps.partner = 'SKLIK', 1, 0) as sklik_wins
    
from auc_sdf as a 
  
    lateral view explode(a.impressions) tab as imps
  
where
    -- Odstrani aukce kde vyhral urgent. Zde sklik nikdy nemuze vyhrat.
    imps.stage <> 'urgent'

"""

sqc.sql(query).createOrReplaceTempView('auctions')


# In[11]:


bids_sdf = get_statlog_df('bids', DATES)
bids_sdf.createOrReplaceTempView('bids_sdf')


# In[12]:


query = """
-- Tabulka Sklik bidu 
select

    b.requestid as requestid,
    bid.impressionid as impressionid,
    bid.dspimpressionid as dspimpressionid,
    
    if(bid.result = 'served', 1, 0) as winner_bid,
    -- Dalsi informace o bidu
    bid.netprice as netprice
    
from bids_sdf as b

  lateral view explode(b.responses) tab as resp
  lateral view explode(resp.bids) tab as bid
  
where 
    -- Eliminace nevalidnich odpovedi. Zajimaji nas pouze bidy pro sklik
    resp.result = 'OK' and
    bid.bidstatus = 'ok' and 
    resp.partner = 'SKLIK'
"""
sqc.sql(query).createOrReplaceTempView('bids')


# In[13]:


query = """
select 

  -- Id pro joinování s sklik logem
  b.requestid as requestid,
  b.dspimpressionid as impressionid,
  
  -- Informace o aukci kde se bid ucastnil
  a.winner_stage,     -- Vitezna stage
  if(a.sklik_wins == 1, 'sklik', 'other') as winner,       -- Pokud Sklik celkove vyhral aukci tak 'sklik' jinak 'other'
  a.winner_netprice,  -- Cena vitezne nabidky 
  
  -- Informace o konkrétním bidu
  b.winner_bid,       -- Pokud aktualni bid vyhral aukci tak 1 jinak 0
  b.netprice          -- Cena aktualniho bidu

from bids as b 
    
    join auctions as a on 
        b.requestid = a.requestid and
        b.impressionid = a.impressionid

"""

sqc.sql(query).createOrReplaceTempView('ssp')


# In[17]:


query = """
select 
    DISTINCT(ssp.winner_stage) AS winner_stage,
    ssp.winner AS winner,
    SUM(ssp.winner_netprice) AS winner_netprice,
    SUM(ssp.netprice) AS sum_netprice,
    sklik.testid as testid,
    COUNT(1) as cnt

from ssp

    left join sklik on 
        ssp.requestid=sklik.requestid and 
        ssp.impressionid=sklik.impressionid
    
group
    by ssp.winner, sklik.testid, ssp.winner_stage
"""

df = sqc.sql(query).toPandas()


# In[28]:


df1 = df.copy()
df1["cpt"] = df1.sum_netprice / df1.cnt
df1 = df1.sort_values(["winner_stage", "testid"])
df1[["winner_stage", "testid", "cpt", "cnt"]]

