import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import math
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

store_details = pd.read_csv('store.csv')

train_data = pd.read_csv('train.csv')

combined_data = pd.merge(store_details, train_data, on='Store')

# columns = list(combined_data.columns)
# columns.remove('Date')
# columns.remove('CompetitionDistance')
# for col in columns:
#     print(col,"---------->", combined_data[col].unique())

combined_data['Year'] = combined_data['Date'].apply(lambda x: int(str(x)[:4]))
combined_data['Month'] = combined_data['Date'].apply(lambda x: int(str(x)[5:7]))

# plt.subplot(3, 3, 1)
# sns.barplot(x='Promo', y='Sales', data=combined_data)
#
# plt.subplot(3, 3, 2)
# sns.barplot(x='DayOfWeek', y='Sales', data=combined_data)
#
# plt.subplot(3, 3, 3)
# sns.barplot(x='SchoolHoliday', y='Sales', data=combined_data)
#
# plt.subplot(3, 3, 4)
# sns.barplot(x='StateHoliday', y='Sales', data=combined_data)
#
#
#
# plt.show()


store_details.update(store_details[['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']].fillna(0))

mean_competition_distance = store_details['CompetitionDistance'].mean()
store_details['CompetitionDistance'].fillna(mean_competition_distance, inplace=True)

mode_competition_open_month = store_details['CompetitionOpenSinceMonth'].mode()[0]
mode_competion_open_year = store_details['CompetitionOpenSinceYear'].mode()[0]

store_details['CompetitionOpenSinceMonth'].fillna(mode_competition_open_month, inplace=True)
store_details['CompetitionOpenSinceYear'].fillna(mode_competion_open_year, inplace=True)

combined_data = pd.merge(store_details, train_data, on='Store')

mean_of_sales = np.mean(combined_data['Sales'])
std_of_sales = np.std(combined_data['Sales'])
# print("Mean is: ", mean_of_sales)
# print("std is: ", std_of_sales)

threshold = 3
outlier = []
for i in combined_data['Sales']:
    z = (i - mean_of_sales) / std_of_sales
    if z > threshold:
        outlier.append(i)
print('Total outlier in dataset are: ', len(outlier))
print('Maximum Sales Outlier: ', max(outlier))
print('Minimum Sales Outlier: ', min(outlier))

sales_zero = combined_data.loc[combined_data['Sales'] == 0]
sales_greater_than_30 = combined_data.loc[combined_data['Sales'] > 30000]

combined_data.drop(combined_data.loc[combined_data['Sales'] > 30000].index, inplace=True)
print("Length of Total Data", len(combined_data))
print("Length of data where sales is zero: ", len(sales_zero),
      "which is ", len(sales_zero)/len(combined_data)*100, "% of the whole data")
print("Length of data where sales greater than 30k: ", len(sales_greater_than_30),
      "which is ", len(sales_greater_than_30)/len(combined_data)*100, "% of the whole data")
print(combined_data.shape)
no_holiday_zero_sales = combined_data.loc[(combined_data['Sales'] == 0) & (combined_data['Open'] == 1) &
                                          (combined_data['StateHoliday'] == 0) & (combined_data['SchoolHoliday'] == 0)]
# print("Size of the data where sales were zero even when stores were open", len(no_holiday_zero_sales))

combined_data.drop(combined_data.loc[(combined_data['Sales'] == 0) & (combined_data['Open'] == 1) &
                                     (combined_data['StateHoliday'] == 0) &
                                     (combined_data['SchoolHoliday'] == 0)].index, inplace=True)
combined_data['Year'] = combined_data['Date'].apply(lambda x: int(str(x)[:4]))
combined_data['Month'] = combined_data['Date'].apply(lambda x: int(str(x)[5:7]))
combined_data.drop(['Date'], axis=1, inplace=True)
# encoding all categorical variable to numeric values

# linearregression Model
label_encoder = preprocessing.LabelEncoder()

combined_data['StoreType'] = label_encoder.fit_transform(combined_data['StoreType'])
combined_data['Assortment'] = label_encoder.fit_transform(combined_data['Assortment'])

# for promo interval
combined_data["PromoInterval"].loc[combined_data["PromoInterval"] == "Jan,Apr,Jul,Oct"] = 1
combined_data["PromoInterval"].loc[combined_data["PromoInterval"] == "Feb,May,Aug,Nov"] = 2
combined_data["PromoInterval"].loc[combined_data["PromoInterval"] == "Mar,Jun,Sept,Dec"] = 3
'''
new_promo_interval = []
for i in range(len(combined_data)):
    if combined_data['PromoInterval'][i] == 'Jan,Apr,Jul,Oct':
        new_promo_interval.append(1)
    elif combined_data['PromoInterval'][i] == 'Feb,May,Aug,Nov':
        new_promo_interval.append(2)
    elif combined_data['PromoInterval'][i] == 'Mar,Jun,Sept,Dec':
        new_promo_interval.append(3)
    else:
        new_promo_interval.append(0)

combined_data['PromoInterval'] = new_promo_interval        
'''

# for State Holiday
combined_data["StateHoliday"].loc[combined_data["StateHoliday"] == "a"] = 1
combined_data["StateHoliday"].loc[combined_data["StateHoliday"] == "b"] = 2
combined_data["StateHoliday"].loc[combined_data["StateHoliday"] == "c"] = 3

'''
state_holiday_list = []
for i in range(len(combined_data)):
    if combined_data['StateHoliday'][i] == 'a':
        state_holiday_list.append(1)
    elif combined_data['StateHoliday'][i] == 'b':
        state_holiday_list.append(2)
    elif combined_data['StateHoliday'][i] == 'c':
        state_holiday_list.append(3)
    else:
        state_holiday_list.append(0)


combined_data['StateHoliday'] = state_holiday_list
'''

combined_data['StateHoliday'] = pd.to_numeric(combined_data['StateHoliday'])
combined_data['PromoInterval'] = pd.to_numeric(combined_data['PromoInterval'])

# implementing model


combined_data_subset = combined_data[combined_data['Open'] == 1]
combined_data_subset_closed = combined_data[combined_data['Open'] == 0]
x_train, x_test, y_train, y_test_open = train_test_split(
    combined_data_subset.drop(['Sales', 'Customers', 'Open'], axis=1),
    combined_data_subset['Sales'], test_size=0.20)
epsilon = 1e-10

# random forest regressor
rf_regr = RandomForestRegressor()
rf_regr.fit(x_train, y_train)

feature_importance = rf_regr.feature_importances_
# print("feature importance is ::", feature_importance)

prediction_open = rf_regr.predict(x_test)
prediction_close = np.zeros(combined_data_subset_closed.shape[0])
prediction = np.append(prediction_open, prediction_close)
y_test = np.append(y_test_open, np.zeros(combined_data_subset_closed.shape[0]))

columns = list(x_train.columns)

feature_importance_value = []
for i in range(len(feature_importance)):
    feature_importance_value.append(round(feature_importance[i], 5))

feature_importance_df = pd.DataFrame({"Features": columns,
                                      "Values": feature_importance_value})
print(feature_importance_df)

feature_importance_df.sort_values(by=["Values"], inplace=True, ascending=False)

plt.figure(figsize=(15, 6))

sns.barplot(x=feature_importance_df["Features"], y=feature_importance_df["Values"], data=feature_importance_df)
plt.show()














# print("r2_score is :L ", r2_score(y_test, prediction))
# print("Mean absolute Error is: %.2f"% mean_absolute_error(y_test, prediction))
# print("Root mean squared error: ", math.sqrt(mean_squared_error(y_test, prediction)))

# plt.figure(figsize=(10, 10))
# plt.scatter(y_test, prediction, c='crimson')
#
#
# p1 = max(max(prediction), max(y_test))
# p2 = min(min(prediction), min(y_test))
#
# plt.plot([p1, p2], [p1, p2], 'b-')
# plt.xlabel('True Values', fontsize=14)
# plt.ylabel('Predicted Value', fontsize=14)
# plt.axis('equal')
# plt.show()
