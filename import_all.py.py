# Databricks notebook source


# COMMAND ----------



# COMMAND ----------

# -*- coding: utf-8 -*-


# Import packages
import os
from IPython import get_ipython
import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
# get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display

import copy
import datetime as dt
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import *

# Install Snowflake Connector
# !pip install snowflake-connector-python[pandas]
# conda install -n env_g1 -c conda-forge snowflake-connector-python[pandas]

import snowflake
import snowflake.connector

from numpy import mean
from numpy import std
from scipy import stats

import itertools
import csv

from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from sqlalchemy.dialects import registry  

# Install packages
# !pip install snowflake-sqlalchemy
# !pip install imbalanced-learn
# !pip install hyperopt
# !pip install lime
# !pip install dill




