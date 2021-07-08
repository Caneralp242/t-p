import numpy as np
import pandas as pd

df=pd.read_csv("/Users/caneralp/Downloads/country_vaccination_stats.csv")

df["daily_vaccinations"]=df["daily_vaccinations"].fillna(df.groupby("country")["daily_vaccinations"].transform("min"))

df

