#taxi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px
#for buiding models
from scipy import stats
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn import cluster
from sklearn import feature_selection

taxi_data = pd.read_csv("train.csv")
print('Data shape: {}'.format(taxi_data.shape))
#data has 11 features and almost 1,5 million observations

taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S') 
print(taxi_data['pickup_datetime'].min())
print(taxi_data['pickup_datetime'].max())

print(taxi_data.isnull().sum())
#there is no missing data
print("Unique vendor's amount", taxi_data['vendor_id'].nunique())
print('Max of passenger count', taxi_data['passenger_count'].max())

print('Trip duration mean', taxi_data['trip_duration'].mean())
print('Trip duration median', taxi_data['trip_duration'].median())
print('Min trip duration', taxi_data['trip_duration'].min())
print('Max trip duration', taxi_data['trip_duration'].max())

print("there are outliers; we can't use mean as central measure of trip duration, better use those measures that are indepent of outliers, such as log mean or median")

def add_datetime_features(df):
    #this function converts date and time data to new features, returns dataframe with new features
    df['pickup_date'] = df['pickup_datetime'].dt.date
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
    return df
taxi_data = add_datetime_features(taxi_data)

print(taxi_data.info())
print(taxi_data[taxi_data['pickup_day_of_week'] == 5]['id'].count())
print(taxi_data.groupby('pickup_date').count().mean()[0])

holidays = pd.read_csv('holiday_data.csv')
print(holidays.info())

def only_date(feature):
    #this function takes date from string
    feature = feature[feature.index(';') + 1: feature.index(';', feature.index(';') + 1)]
    return feature

holidays['date'] = holidays['day;date;holiday'].apply(only_date)
holidays['date'] = pd.to_datetime(holidays['date']).dt.date


def add_holiday_features(df, h_days):
    #function checks if this day is a holiday and creates new feature with 1's in holidays
    df['pickup_holiday'] = df['pickup_date'].apply(lambda x: 1 if x in list(h_days['date']) else 0)
    return df

taxi_data = add_holiday_features(taxi_data, holidays)

print(taxi_data[taxi_data['pickup_holiday'] == 1]['trip_duration'].median())

osrm = pd.read_csv('osrm_data_train.csv') # this file contains data about shortest route between two points


def add_osrm_features(df1, df2):
    #function adds needed features from table with shortest data to main dataframe
    df2 = df2[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
    df = pd.merge(df1, df2, how='left', on='id')
    return df

taxi_data = add_osrm_features(taxi_data, osrm)
print(taxi_data['trip_duration'].median() - taxi_data['total_travel_time'].median())
print(taxi_data.isna().sum().max())

def get_haversine_distance(lat1, lng1, lat2, lng2):
    #function for finding  distance using the haversine formula
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    EARTH_RADIUS = 6371 
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1
    d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
    h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def get_angle_direction(lat1, lng1, lat2, lng2):
    #function for calculation the angle of direction of movement
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    lng_delta_rad = lng2 - lng1
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    alpha = np.degrees(np.arctan2(y, x))
    return alpha

def add_geographical_features(df):
    #function adds new geographical features to main dataset
    lat1, lat2 = df['pickup_latitude'].values, df['dropoff_latitude'].values
    lng1, lng2 = df['pickup_longitude'].values, df['dropoff_longitude'].values
    df['haversine_distance'] = get_haversine_distance(lat1, lng1, lat2, lng2)
    df['direction'] = get_angle_direction(lat1, lng1, lat2, lng2)
    return df

taxi_data = add_geographical_features(taxi_data)
print(taxi_data['haversine_distance'].median())
    
coords = np.hstack((taxi_data[['pickup_latitude', 'pickup_longitude']], taxi_data[['dropoff_latitude', 'dropoff_longitude']])) 
#create a training sample with geographic coordinates of all points 
kmeans = cluster.KMeans(n_clusters=10, random_state=42, n_init=10) #create a clustering algorithm
kmeans.fit(coords)
    
def add_cluster_features(df, model):
    #function adds new feature with data from clustering model
    df['geo_cluster'] = model.predict(df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']].values)
    return df

taxi_data = add_cluster_features(taxi_data, kmeans)
print(taxi_data['geo_cluster'].value_counts().min())

weather = pd.read_csv('weather_data.csv') #file with weather datas

def add_weather_features(df, weather):
    #function adds new feature with weather data
    weather['time'] = pd.to_datetime(weather['time'])
    weather['date'] = weather['time'].dt.date
    weather['hour'] = weather['time'].dt.hour
    
    weather_cols = ['date', 'hour', 'temperature', 'visibility', 'wind speed', 'precip', 'events']
    
    df = df.merge(weather[weather_cols], left_on=['pickup_date', 'pickup_hour'], right_on=['date', 'hour'], how='left')
    df = df.drop(['date', 'hour'], axis=1)
    
    return df

taxi_data = add_weather_features(taxi_data, weather)
print('In snowy weather were', end=' ')
print(taxi_data['events'].value_counts()['Snow'], 'trips')
print('Raws with missing data', end=' ')
print(taxi_data.isnull().sum())
#there is some missing data

def fill_null_weather_data(df):
    #function fills missing data in 'events' column with 'None', and in all other columns with median data
    weather_cols = ['temperature', 'visibility', 'wind speed', 'precip', 'events']
    osrm_cols = ['total_distance', 'total_travel_time', 'number_of_steps'] 
    for col in weather_cols:
        if col == 'events':
            df['events'] = df['events'].fillna('None')
        else:
            df[col] = df[col].fillna(df.groupby('pickup_date')[col].transform('median'))
    for col in osrm_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

taxi_data = fill_null_weather_data(taxi_data)
print(round(taxi_data['temperature'].median(), 1))

avg_speed = taxi_data['total_distance'] / taxi_data['trip_duration'] * 3.6
#fig, ax = plt.subplots(figsize=(10, 5))
#sns.scatterplot(x=avg_speed.index, y=avg_speed, ax=ax)
#ax.set_xlabel('Index')
#ax.set_ylabel('Average speed')

outliers_duration = list(taxi_data[taxi_data['trip_duration'] > 3600*12].index)
outliers_speed = list(taxi_data[avg_speed > 250].index)

print(f'Outliers in trip duration (>12 hours): {len(outliers_duration)}')

print(f'Outliers in average speed (>250 km/h): {len(outliers_speed)}')

taxi_data = taxi_data.drop(outliers_duration+outliers_speed)
#delete outliers where trip duration is over 12 hours or average speed is more than 250 km/h
print(f'Cleaned data shape: {taxi_data.shape[0]}')


taxi_data['trip_duration_log'] = np.log(taxi_data['trip_duration']+1)
# we will check RMSLE metric, so convert target logarithmly

#fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#sns.histplot(data=taxi_data, x='trip_duration_log', ax=ax[0], bins=50, kde=True, palette='husl')
#ax[0].set_title('Histogram trip duration (log)')

#sns.boxplot(data=taxi_data, x='trip_duration_log', ax=ax[1], palette='husl')
#ax[1].set_title('Boxplot trip duration (log)')

H0 = "It's normal distribution."
H1 = 'It is not normal distribution.'
alpha = 0.05
#test hypothesis of normal distribution
_, p_value = stats.normaltest(taxi_data['trip_duration_log'])
print(f'p-value: {round(p_value, 3)}')
print(H0) if p_value > alpha/2 else print(H1)
#It is not normal distribution

train_data = taxi_data.copy()
drop_columns = ['id', 'dropoff_datetime', 'pickup_datetime', 'pickup_date']

train_data = train_data.drop(drop_columns, axis=1)
print(f'Shape of data: {train_data.shape}')
train_data['vendor_id'] = train_data['vendor_id'].replace({1:0,2:1})
train_data['store_and_fwd_flag'] = train_data['store_and_fwd_flag'].replace({'N':0,'Y':1})
columns_to_change = ['pickup_day_of_week', 'geo_cluster', 'events']
one_hot_encoder = preprocessing.OneHotEncoder(drop='first', sparse_output=False)

data_onehot = one_hot_encoder.fit_transform(train_data[columns_to_change])
column_names = one_hot_encoder.get_feature_names_out(columns_to_change)
data_onehot = pd.DataFrame(data_onehot, columns=column_names)

train_data = pd.concat([train_data.reset_index(drop=True).drop(columns_to_change, axis=1), data_onehot], axis=1)
print('Shape of data: {}'.format(train_data.shape))

X = train_data.drop(['trip_duration', 'trip_duration_log'], axis=1)
y = train_data['trip_duration']
y_log = train_data['trip_duration_log']

X_train, X_valid, y_train_log, y_valid_log = model_selection.train_test_split(X, y_log, test_size=0.33, random_state=42) #split set to train and valid sets

selector = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=35) #select 35 features
selector.fit(X_train, y_train_log)
best_features = selector.get_feature_names_out()
X_train = X_train[best_features]
X_valid = X_valid[best_features]

scaler = preprocessing.MinMaxScaler()
#scale data
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

lr = linear_model.LinearRegression()
#build linear regression model
lr.fit(X_train_scaled, y_train_log)
y_train_pred_log = lr.predict(X_train_scaled)
y_valid_pred_log = lr.predict(X_valid_scaled)
rmsle_train = np.sqrt(metrics.mean_squared_error(y_train_log, y_train_pred_log))
rmsle_valid = np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_pred_log))
print(f"RMSLE on train set: {rmsle_train:.4f}")
print(f"RMSLE on valid set: {rmsle_valid:.4f}")

poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
#create polynomial features
X_train_scaled_poly = poly.fit_transform(X_train_scaled)
X_valid_scaled_poly = poly.transform(X_valid_scaled)
lr_poly = linear_model.LinearRegression()
#build linear regression model with polynomial features
lr_poly.fit(X_train_scaled_poly, y_train_log)
y_train_log_poly_pred = lr_poly.predict(X_train_scaled_poly)
y_valid_log_poly_pred = lr_poly.predict(X_valid_scaled_poly)
poly_rmsle_train = np.sqrt(metrics.mean_squared_error(y_train_log, y_train_log_poly_pred))
poly_rmsle_valid = np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_log_poly_pred))
print(f"RMSLE on train set with polynomial features: {poly_rmsle_train:.4f}")
print(f"RMSLE on valid set with polynomial features: {poly_rmsle_valid:.4f}")

ridge = linear_model.Ridge(alpha=1)
#add l2 regulation
ridge.fit(X_train_scaled_poly, y_train_log)
y_train_log_ridge_pred = ridge.predict(X_train_scaled_poly)
y_valid_log_ridge_pred = ridge.predict(X_valid_scaled_poly)
ridge_rmsle_train = np.sqrt(metrics.mean_squared_error(y_train_log, y_train_log_ridge_pred))
ridge_rmsle_valid = np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_log_ridge_pred))
print(f"RMSLE on train set with L2 regulation: {ridge_rmsle_train:.4f}")
print(f"RMSLE on valid set with L2 regulation: {ridge_rmsle_valid:.4f}")

dt = tree.DecisionTreeRegressor()
#build decision tree model
dt.fit(X_train_scaled, y_train_log)
y_train_log_dt_pred = dt.predict(X_train_scaled)
y_valid_log_dt_pred = dt.predict(X_valid_scaled)
dt_rmsle_train = np.sqrt(metrics.mean_squared_error(y_train_log, y_train_log_dt_pred))
dt_rmsle_valid = np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_log_dt_pred))
print(f"RMSLE on train set decision tree: {dt_rmsle_train:.4f}")
print(f"RMSLE on valid set decision tree: {dt_rmsle_valid:.4f}")

param_grid = [{'max_depth': range(7, 21)}]
#search for bestfitting parameters
grid_search_dt = model_selection.GridSearchCV(
    estimator=tree.DecisionTreeRegressor(), 
    param_grid=param_grid, 
    cv=5,     
    n_jobs=-1, 
    scoring='neg_root_mean_squared_error' 
)
grid_search_dt.fit(X_train_scaled, y_train_log)
print(f'Best Model: {grid_search_dt.best_estimator_}')
y_train_log_dt_gs = grid_search_dt.predict(X_train_scaled)
y_valid_log_dt_gs = grid_search_dt.predict(X_valid_scaled)
dtgs_rmsle_train = np.sqrt(metrics.mean_squared_error(y_train_log, y_train_log_dt_gs))
dtgs_rmsle_valid = np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_log_dt_gs))
print(f"RMSLE on train set decision tree with grid search: {dtgs_rmsle_train:.4f}")
print(f"RMSLE on valid set decision tree with grid search: {dtgs_rmsle_valid:.4f}")

rf = ensemble.RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    criterion='squared_error',
    min_samples_split=20,
    random_state=42,
    n_jobs=-1
)
#build random forest model
rf.fit(X_train_scaled, y_train_log)
y_train_rf_pred = rf.predict(X_train_scaled)
y_valid_rf_pred = rf.predict(X_valid_scaled)
rf_rmsle_train = np.sqrt(metrics.mean_squared_error(y_train_log, y_train_rf_pred))
rf_rmsle_valid = np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_rf_pred))
print(f"RMSLE on train set random forest: {rf_rmsle_train:.4f}")
print(f"RMSLE on valid set random forest: {rf_rmsle_valid:.4f}")

gb = ensemble.GradientBoostingRegressor(
    learning_rate=0.5,
    n_estimators=100,
    max_depth=6,
    min_samples_split=30,
    random_state=42,
    verbose=True
)
#build gradient boosting model
gb.fit(X_train_scaled, y_train_log)
y_train_gb_pred = gb.predict(X_train_scaled)
y_valid_gb_pred = gb.predict(X_valid_scaled)
gb_rmsle_train = np.sqrt(metrics.mean_squared_error(y_train_log, y_train_gb_pred))
gb_rmsle_valid = np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_gb_pred))
print(f"RMSLE on train set gradient boosting: {gb_rmsle_train:.4f}")
print(f"RMSLE on valid set gradient boosting: {gb_rmsle_valid:.4f}")

results_df = pd.DataFrame(
    data=[
        np.round([
            rmsle_train, poly_rmsle_train,
            ridge_rmsle_train, dt_rmsle_train,
            dtgs_rmsle_train, rf_rmsle_train,
            gb_rmsle_train
        ], 4),
        np.round([
            rmsle_valid, poly_rmsle_valid,
            ridge_rmsle_valid, dt_rmsle_valid,
            dtgs_rmsle_valid, rf_rmsle_valid,
            gb_rmsle_valid
        ], 4)
    ],
    columns=[
        'LR', 'LR-poly', 'Ridge',
        'DTR', 'DTR-GS', 'RFR',
        'GBR'
    ],
    index=['RMSLE train', 'RMSLE valid']
)

print(results_df)