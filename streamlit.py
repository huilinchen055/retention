import sys
import importlib
import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import create_engine, text
import sqlalchemy
import urllib
import seaborn as sns
import matplotlib.pyplot as plt
import xlwings as xw
# add utils to path for notebook to utilize things
sys.path.append("../")
# import custom modules
from src.utils import classes
from src.utils import functions
from src.utils import settings
# cell to reload any custom module, post updates
importlib.reload(settings)
importlib.reload(functions)
importlib.reload(classes)
# connect to M-SQL servers IXREPORT_COMMERCIAL to get all policies
connection_string = (
    f"mssql+pyodbc://{settings.SAGESURE_DB_AUTH_USER}:{urllib.parse.quote_plus(settings.SAGESURE_DB_AUTH_PASS)}"
    "@PRVAMDBRPT01.CLOUD.ICG360.NET:1433/IXREPORT_COMMERCIAL?"
    "driver=ODBC Driver 17 for SQL Server&TrustServerCertificate=yes&timeout=30"
)

# create an SQLAlchemy engine
engine = sqlalchemy.create_engine(connection_string)
query = text("""
SELECT *								
								
	FROM  							
		(Select distinct Product, InsightPolicyID, UpdateTimeStamp, PropertyState, QuoteNumber, PolicyNumber, EffectiveDatePolicyTerm, policyterm,
		DATEADD(YEAR, 1, EffectiveDatePolicyTerm) AS EffectiveDatePlusOneYear,
		CarrierName, TotalPremium, TIV						
								
		From IXREPORT_COMMERCIAL.dbo.QNPTerms						
			where Product = 'BOP' 					
										
		) t1						
								
	Left Join 							
		(select distinct InsightPolicyID, CancellationDate						
								
		from IXREPORT_COMMERCIAL.dbo.QuotesAndPoliciesAdditionalPolicyFields						
								
		)t2						
								
		on t1.InsightPolicyID = t2.InsightPolicyID						
								
	Left Join							
		(select distinct insightpolicyid, expirationdate						
								
		from IXREPORT_COMMERCIAL.dbo.QuotesAndPoliciesAdditionalPolicyFieldsI						
								
		)t3						
								
		on t1.insightpolicyid = t3.insightpolicyid						
								
	Left join 							
								
		(Select distinct InsightPolicyID, InsuranceScore, InsuranceScoreRange						
								
		From IXREPORT_COMMERCIAL.dbo.QuotesAndPoliciesAdditionalPolicyFieldsII						
								
		) t4						
								
		on t1.InsightPolicyID = t4.InsightPolicyID						
								
	Left join 							
								
		(Select distinct InsightPolicyID, LocationNumber, DistanceToCoastLocation, ConstructionYearRoofLocation, ConstructionYearLocation, CompetitivePremiumYearsWithCarrier						
								
		From IXREPORT_COMMERCIAL.dbo.QuotesandPoliciesCommercialAdditionalFieldI						
			where LocationNumber = 1					
								
		) t5						
								
		on t1.InsightPolicyID = t5.InsightPolicyID						
								
	Left join 							
								
		(Select distinct InsightPolicyID, LocationNumber, PropertyCountyLocation, OccupancyClassCodeLocation, OccupancyClassCodeLocationClass2, OccupancyClassCodeLocationClass3, OccupancyClassCodeLocationClass4,						
		PropertyOccupancyLocation						
								
		From IXREPORT_COMMERCIAL.dbo.QuotesandPoliciesCommercialAdditionalFieldIII						
			where LocationNumber = 1					
								
		) t6						
								
		on t1.InsightPolicyID = t6.InsightPolicyID	
	order by t1.insightpolicyid							
		;	

""")
# Execute the query using the parameters
try:
    df_sql = pd.read_sql_query(query, engine.connect())
    #print(df_sql.head())  # Display the result (or you can use df.info(), df.describe(), etc.)
except Exception as e:
    print("Error occurred:", e)
df_sql = df_sql.loc[:, ~df_sql.T.duplicated()]
df = df_sql.sort_values(by=['InsightPolicyID', 'EffectiveDatePolicyTerm'])
# Step 3: Shift to compare current term to the prior term’s expiration
df['nextEffective'] = df.groupby('InsightPolicyID')['EffectiveDatePolicyTerm'].shift(-1)

# Step 4: Define a renewal (current effective within 30 days after previous expiration)
df['IsRenewal'] = (
    abs(df['EffectiveDatePlusOneYear'] - df['nextEffective']).dt.days.between(0, 30)
)

# Step 5: Create time periods
df['Month'] = df['EffectiveDatePolicyTerm'].dt.to_period('M')
df['Quarter'] = df['EffectiveDatePolicyTerm'].dt.to_period('Q')
df['Year'] = df['EffectiveDatePolicyTerm'].dt.year
from datetime import datetime
today = pd.to_datetime(datetime.today().date())
cutoff_date = pd.to_datetime(datetime.today().date()) ##+ pd.DateOffset(months=2)
df=df[~((df['EffectiveDatePlusOneYear'] > cutoff_date))]
character = 'InsuranceScore'

# Convert to numeric, coercing errors to NaN
df[character] = pd.to_numeric(df[character], errors='coerce')

# Define bins and labels
bins = [-np.inf, 500, 600, 650, 750, np.inf]
labels = ['Poor', 'Below Average', 'Average', 'Good', 'Excellent']

# Create the grouped column using pd.cut
df[f"{character}Group"] = pd.cut(df[character], bins=bins, labels=labels, right=False)
yearly = (
    df.groupby(['Year', f"{character}Group"])
      .agg(Total=('InsightPolicyID', 'count'),
           Renewed=('IsRenewal', 'sum'))
      .assign(RenewalRate=lambda x: x['Renewed'] / x['Total'])
      .reset_index()
)
yearly
# Filter the data for the year 2022
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Sample placeholder for your data ---
# Replace this with your actual dataframe load
# Example dummy data:
yearly = pd.DataFrame({
    'Year': [2023]*5,
    'Group': ['A', 'B', 'C', 'D', 'E'],
    'RenewalRate': [0.85, 0.78, 0.92, 0.81, 0.88],
    'Total': [150, 120, 200, 175, 160]
})

# --- Parameters ---
character = 'Group'  # This should match the column name prefix
year = 2023

# --- Streamlit Title ---
st.title("📊 Renewal Rate by Group - 2023")

# --- Filter the Data ---
data = yearly[yearly['Year'] == year]

# --- Matplotlib / Seaborn Plot ---
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=character, y='RenewalRate', data=data, ci=None, ax=ax)

# Annotate top of each bar with the total
for bar, total in zip(ax.patches, data['Total']):
    height = bar.get_height()
    x_pos = bar.get_x() + bar.get_width() / 2
    ax.text(x_pos, height + 0.01, f'{int(total)}', ha='center', va='bottom', fontsize=9)

# Styling
ax.set_xlabel(character)
ax.set_ylabel('Renewal Rate')
ax.set_title(f'Renewal Rate by {character} - {year}')
plt.xticks(rotation=45)
plt.tight_layout()

# --- Display in Streamlit ---
st.pyplot(fig)
