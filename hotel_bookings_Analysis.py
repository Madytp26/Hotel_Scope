import pandas as pd
import numpy as np
from numpy import random
from numpy.linalg import inv
import matplotlib.pyplot as plt
import os
import scipy
from scipy import stats
from scipy.stats import skew,kurtosis,poisson,norm,chi2
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats import weightstats as ssw
from statsmodels.stats import anova
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chisquare
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer,mean_squared_error

from lttb import downsample

df=pd.read_csv('hotel_bookings.csv')
# print(df.head())
# print(df.shape)
############################################################################
# CHECKING FOR NULLS IN DATA
# print(df.isnull().sum())

df1=df.drop(columns='company')

############################################################################
# CHECKING FOR NULL IN 'AGENT'

# print(df1.isnull().sum())
# print(df1['agent'].median())

count_9=df1['agent'].value_counts().get(9,0)
# print(count_9)
# print((count_9/119390)*100)

df1['agent'] = df1['agent'].fillna('Missing')
# print(df1.isnull().sum())

############################################################################
# CHECKING FOR NULL IN 'CHILDREN' AND 'COUNTRY'

# print(df1[['children','country']])
# print(df1[['children','country']].isnull().sum())

# print(df1['country'].mode())

count_prt=df1['country'].value_counts().get('PRT',0)
# print(count_prt)
# print((count_prt/119390)*100)
# ~41%...also missing values are very less in number
# still better to put 'Missing'


# print(df1['children'].mode())

df1['country'] = df1['country'].fillna('Missing')
df1['children'] = df1['children'].fillna(0.0)
# print(df1[['children','country']].isnull().sum())
# print(df1.isnull().sum())


############################################################################
# FINDING DUPLICATES .... SINCE UNIQUE IS MISSING... LOOKING FOR FULLY DUPLICATED ROWS

# print(df1.duplicated().sum())               # 32001 fully duplicated rows
df1 = df1.drop_duplicates()
# print(df1.duplicated().sum())                 # now 0
# print(df1.shape)


############################################################################

#        EXPLORATIVE DATA ANALYSIS


df1.insert(0, 'Sr.No', range(1, len(df1) + 1))
print(df1)

###################  UNIVARIATES   ###################

# fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8))

# gbyc=df1.groupby('country')['Sr.No'].count()              #GROUPING COUNTRY BY COUNT
# gbymarket=df1.groupby('market_segment')['Sr.No'].count()              #GROUPING MARKET_SEGMENT BY COUNT


# gbyc.plot(kind='line',x='country',y='Sr.No.',ax=axes[0,0])
# df1['hotel'].value_counts().plot(kind='bar',xlabel='Hotel_type',ylabel='Count',ax=axes[0,1])
# gbymarket.plot(kind='pie',ax=axes[1,0])
# df1['meal'].value_counts().plot(kind='line',xlabel='Meal_Type',ylabel='Count',ax=axes[1,1])

######### other graphs if needed #########

# df1['country'].aggregate().plot(kind='barh',x='country',ylabel='count',ax=axes[0,0],xlim=10)
# df1['adr'].plot(kind='box',xlabel='ADR',ax=axes[1,1],ylim=(0,600))
# df1['reservation_status'].value_counts().plot(kind='area',ax=axes[1,1])

# plt.subplots_adjust(wspace=0.4, hspace=0.6)
# plt.show()


################### bivariates ###################

# fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8))



# months=['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December']

''' CHECKING SEASONALITY : PLOTTING TRAFFIC BY MONTHS '''

# df1['arrival_date_month']=pd.Categorical(df1['arrival_date_month'],categories=months,ordered=True)
# sns.countplot(data=df1,x='arrival_date_month',hue='hotel',ax=axes[0,0])
# axes[0, 0].tick_params(axis='x', rotation=45)

''' CHECKING TRAFFIC : PLOTTING SEGMENTS BY MONTHS '''

# sns.countplot(data=df1,x='market_segment',hue='hotel',ax=axes[0,1])
# axes[0, 1].tick_params(axis='x', rotation=45)


# gbyadr=df1.groupby('customer_type')['adr'].count()        #GROUPING CUSTOMER TYPE BY DAILY TRAFFIC
# # print(gbyadr)
# gbyadr.plot(kind='pie',ax=axes[1,0])
# axes[1, 0].tick_params(axis='x', rotation=45)


# gbydis=df1.groupby('distribution_channel')['Sr.No'].count()        #GROUPING BOOKING TYPE BY COUNT OF CUSTOMERS
# # print(gbydis)
# gbydis.plot(kind='area',ax=axes[1,1])

# plt.subplots_adjust(wspace=0.4, hspace=0.6)
# plt.show()


################### multivariates ###################


#   GROUPING HOTEL TYPE BY NUMBER OF STAYS 

# gbyh=df1.groupby('hotel')[['stays_in_weekend_nights','stays_in_week_nights']].mean().reset_index()

# # print(gbyh)
# gbyh.set_index('hotel').plot(kind='bar')

# sns.catplot(data=df1,x='hotel',hue='reserved_room_type',col='customer_type',kind='count')

# plt.show()


################### CO-RELATION ###################

####  PEARSONS ####

# fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(20,16))

# # SELECTING ONLY NUMERIC DATA BCZ PEARSON/SPEARMAN REQUIRE ONLY NUMERIC DATA

# df2 = df1.select_dtypes(include='number')
# # print(df2)
# pearson=df2.corr(method='pearson')
# sns.heatmap(data=pearson,cmap='coolwarm',ax=axes[0])

#### SPEARMAN ####

# spearman=df2.corr(method='spearman')
# sns.heatmap(data=spearman,cmap='BuPu',ax=axes[1])
# plt.subplots_adjust(wspace=0.4, hspace=0.6)
# plt.show()


################### HYPOTHESES TESTING ###################
# '''
# hypothesis 1:
# Ho:no difference in average daily rate (ADR) between bookings made through Online TA and Direct channels.
# Ha:is a difference in ADR between bookings made through Online TA and Direct channels.

# '''
''' LOCATING ONLY ROW VALUES IN MARKET SEGMENT HAVING ONLINE BOOKING '''
# online = df1.loc[df1['market_segment'] == 'Online TA', 'adr'].values
# print(online)
''' LOCATING ONLY ROW VALUES IN MARKET SEGMENT HAVING DIRECT BOOKING '''
# direct = df1.loc[df1['market_segment'] == 'Direct', 'adr'].values
# print(direct)
# # Means, stds, sizes
# mean_online = np.mean(online)
# mean_direct = np.mean(direct)
# std_online = np.std(online, ddof=1)
# std_direct = np.std(direct, ddof=1)
# no = len(online)
# nd = len(direct)

# # Standard error & z
# se_diff = np.sqrt(std_online**2/no + std_direct**2/nd)
# z_stat = (mean_online - mean_direct) / se_diff

# # Two-tailed p-value
# p_value = 2 * (1 - norm.cdf(abs(z_stat)))

# print('z_stat:',z_stat)
# print('p_value:',p_value)
# if p_value < 0.05:
#     print("Reject H0: ADR differs between Online TA and Direct")
# else:
#     print("Fail to reject H0: No significant difference in ADR")




# '''
# Hypothesis 2 :
# Ho:Room upgrades are independent of lead time 
# Ha:Room upgrades are dependent on lead time 

# '''



# from scipy.stats import chi2_contingency

# # Create indicators
''' CHECKING WHERE DIFFERENT ROOM WAS GIVEN INSTEAD OF RESERVED '''
# df1['room_upgrade'] = (df1['reserved_room_type'] != df1['assigned_room_type']).astype(int)
# df1['lead_bin'] = pd.cut(df1['lead_time'], bins=[-1, 7, 30, 365], labels=['Short', 'Medium', 'Long'])

# # Contingency table and test
# table = pd.crosstab(df1['room_upgrade'], df1['lead_bin'])
# chi2, p, dof, expected = chi2_contingency(table)

# print("Chi² =", round(chi2, 2), ", p =", round(p, 5))


# if p < 0.05:
#     print("Reject H0: Room upgrades are dependent on lead time")
# else:
#     print("Fail to reject H0: Room upgrades are independent of lead time")



# '''
# Hypothesis 3 :
# Ho:duration is the same across all customer types
# Ha:duration is the not same across all customer types

# '''


# from scipy.stats import f_oneway

# df1['stay'] = df1['stays_in_weekend_nights'] + df1['stays_in_week_nights']
# df1['stay'] = pd.to_numeric(df1['stay'], errors='coerce')
# df1 = df1.dropna(subset=['stay'])

# groups = [group['stay'].values for _, group in df1.groupby('customer_type')]
# groups = [np.array(g).astype(float) for g in groups if len(g) > 0]

# for i, g in enumerate(groups):
#     print(f"Group {i}: length={len(g)}")

# f_stat, p = f_oneway(*groups)
# print("F =", round(f_stat, 2), ", p =", round(p, 5))

# if p < 0.05:
#     print("Reject H₀: Stay duration differs by customer type")
# else:
#     print("Fail to reject H₀: No significant difference in stay duration")




################### KEY QUESTIONS ###################

#1
# corr = df1[['adr', 'lead_time', 'total_of_special_requests', 'booking_changes', 'stays_in_weekend_nights', 'stays_in_week_nights']].corr()['adr'].sort_values()
# # print("Correlation with ADR:\n", corr)


# #2

# plt.scatter(df1['lead_time'], df1['booking_changes'], alpha=0.3)
# plt.xlabel('Lead Time (days)')
# plt.ylabel('Booking Changes')
# plt.title('Booking Changes vs Lead Time')
# # plt.show()


# #3
# top_countries = df1['country'].value_counts().head(10).index
# avg_adr = df1[df1['country'].isin(top_countries)].groupby('country')['adr'].mean()
# avg_adr.plot(kind='bar')
# plt.ylabel('Average ADR')
# plt.title('Average ADR for Top 10 Countries')
# # plt.show()



#4
# df1['room_upgrade'] = (df1['reserved_room_type'] != df1['assigned_room_type']).astype(int)
# upgrade_rate = df1['room_upgrade'].mean()
# # print(upgrade_rate)
# plt.bar(['No Upgrade', 'Upgrade'], [1 - upgrade_rate, upgrade_rate])
# plt.title('Room Upgrade Rate')
# plt.show()


#5
# match_rate = (df1['reserved_room_type'] == df1['assigned_room_type']).mean()
# # print(match_rate)
# plt.bar(['Match', 'Mismatch'], [match_rate, 1 - match_rate])
# plt.title('Match Rate of Reserved and Assigned Room Types')
# # plt.show()


#6
# df1['group_size'] = df1['adults'] + df1['children'] + df1['babies']
# df1['group_size'].value_counts().sort_index().plot(kind='bar')
# plt.xlabel('Group Size')
# plt.ylabel('Number of Bookings')
# plt.title('Distribution of Group Sizes')
# # plt.show()


#7
# avg_adr_by_customer = df1.groupby('customer_type')['adr'].mean()
# avg_adr_by_customer.plot(kind='bar')
# plt.ylabel('Average ADR')
# plt.title('ADR by Customer Type')
# # plt.show()


#8
# df1.boxplot(column='lead_time', by='customer_type')
# plt.title('Lead Time by Customer Type')
# # plt.suptitle('')
# plt.xlabel('Customer Type')
# plt.ylabel('Lead Time (days)')
# # plt.show()


#9
# plt.scatter(df1['lead_time'], df1['booking_changes'], alpha=0.3)
# plt.xlabel('Lead Time')
# plt.ylabel('Booking Changes')
# plt.title('Booking Changes vs Lead Time')
# # plt.show()


#10
# df1['total_stay'] = df1['stays_in_weekend_nights'] + df1['stays_in_week_nights']
# avg_stay = df1.groupby('customer_type')['total_stay'].mean()
# avg_stay.plot(kind='bar')
# plt.ylabel('Average Stay Duration')
# plt.title('Average Stay Duration by Customer Type')
# # plt.show()


#11
# plt.bar(['Upgraded', 'Not Upgraded'], [upgrade_rate, 1-upgrade_rate])
# # print(1-upgrade_rate)
# plt.title('Room Upgrade Frequency')
# plt.show()


# # #12
# mean_changes = df1.groupby('total_of_special_requests')['booking_changes'].mean()
# mean_stays = df1.groupby('total_of_special_requests')['total_stay'].mean()
# mean_changes.plot(label='Booking Changes')
# mean_stays.plot(label='Stay Duration')
# # plt.legend()
# plt.title('Booking Changes and Stay Duration vs Special Requests')
# # plt.show()


# # #13
# segment_stats = df1.groupby('market_segment').agg({'adr':'mean', 'is_canceled':'mean'})
# segment_stats.plot.scatter(x='is_canceled', y='adr')
# plt.title('ADR vs Cancellation Rate by Market Segment')
# plt.show()


# #14
# corr = df1[['adr', 'lead_time', 'total_of_special_requests', 'booking_changes', 'stays_in_weekend_nights', 'stays_in_week_nights']].corr()['adr'].sort_values()
# # print("Correlation with ADR:\n", corr)


# #15
# df1.groupby('customer_type')['adr'].mean().sort_values().plot(kind='bar')
# plt.title('Average ADR by Customer Type')
# # plt.show()


# # #16
# top_countries = df1['country'].value_counts().head(10).index
# df1[df1['country'].isin(top_countries)].groupby('country')[['lead_time','adr']].mean().plot(kind='bar', subplots=True)
# plt.suptitle('Lead Time and ADR by Country')
# plt.show()



# #17
# df1['high_adr'] = df1['adr'] > df1['adr'].median()
# df1.groupby('high_adr')[['total_of_special_requests','booking_changes']].mean().plot(kind='bar')
# plt.title('Special Requests and Booking Changes by ADR Level')
# plt.show()


# #18
# top_countries = df1['country'].value_counts().head(5).index
# df1[df1['country'].isin(top_countries)].groupby('country')[['lead_time','total_stay']].mean().plot(kind='bar')
# plt.title('Lead Time and Stay Length by Country')
# plt.show()


#19
# df1.groupby(df1['booking_changes'] > 0)[['total_of_special_requests','is_canceled']].mean().plot(kind='bar')
# plt.title('Special Requests and Cancellation by Booking Changes')
# plt.show()







