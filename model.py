import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

def load_data():
	data = pd.read_csv('../Capstone/Real Estate ROI/Data/RDC_Inventory_Core_Metrics_County_History.csv')
	df = data.drop(len(data)-1)
	Null_fips = df.loc[df['county_name'].isnull(),'county_fips'].unique() # fix missing county name related to Valdez-Cordova, AK
	df.fillna({'county_name':'valdez-cordova, ak'}, inplace=True)
	df.rename(columns={'month_date_yyyymm':'date'}, inplace=True)
	df['date'] = pd.to_datetime(df['date'], format='%Y%m')
	df = df.drop_duplicates(subset=['date','county_name'])
	df = df.sort_values(['county_name', 'date']).reset_index(drop=True)

	return df

def calculate_roi(month):
	df = load_data()

	# remove counties with few listings
	df = df.loc[(df['total_listing_count'] >= 2)]

	# make sure all possible dates are present
	date_correct = (
		df
		.groupby('county_name')['date']
		.apply(lambda x: (min(x), max(x)))
		.apply(lambda x: pd.date_range(x[0], x[1], freq='MS'))
		.reset_index()
		.explode('date')
		.reset_index(drop=True)
	)

	# calculate roi
	df_ = date_correct.merge(df, left_on=['county_name','date'], right_on=['county_name','date'], how='left')
	df_['roi'] = ((df_.groupby('county_name')['median_listing_price'].shift(-month) - df_['median_listing_price'])
					/ df_['median_listing_price'] * 100)
	df_ = df_.dropna(subset='roi').reset_index(drop=True)
	df_['class'] = pd.cut(df_['roi'], 
							bins=[min(df_['roi']), 5, max(df_['roi'])], 
							labels=[-1, 1], 
							include_lowest=True)

	# remove outliers
	for i in range(3,df_.shape[1]-1):
		z = np.abs(stats.zscore(df_.iloc[:,i]))
		outliers = np.where(z > 1.5)[0]
		df_.iloc[outliers][['roi']] # check outlier roi values
		df_ = df_.drop(outliers).reset_index(drop=True)

	return df_

class ROIModel(BaseEstimator, ClassifierMixin):
	def fit(self, X, y):
		self.estimator = HistGradientBoostingClassifier( # these best parameter values found by using GridSearchCV based on highest scores from sklearn.metrics.classification_report(y_test, y_pred)
					        l2_regularization=0.001, 
					        learning_rate=0.1,
					        max_depth=300, 
					        max_iter=200, 
					        max_leaf_nodes=100,
					        min_samples_leaf=30
					        )
		self.estimator.fit(X, y)
		return self

	def predict(self, X):
		y_pred = self.estimator.predict(X)
		return y_pred

	def prep_train_data(self, month:int=12):
		df = calculate_roi(month)
		bad_features = ['roi','county_name','date','county_fips','quality_flag', 'class']
		X_data = df.drop(bad_features, axis=1)
		y_data = df['class']

		for train_idx, test_idx in ShuffleSplit(1, test_size=0.2, random_state=42).split(range(len(df))):
			X_train = X_data.iloc[train_idx]
			y_train = y_data.iloc[train_idx]
			X_test = X_data.iloc[test_idx]
			y_test = y_data.iloc[test_idx]
		
		return X_train, y_train, X_test, y_test

def model_input():
	df = load_data()
	d_last_date = df.groupby('county_name')['date'].max().reset_index()
	d_last_date['last_date_flag'] = 1
	df_ = d_last_date.merge(df, left_on=['county_name','date'], right_on=['county_name','date'], how='right')
	df_ = df_.dropna(subset='last_date_flag').reset_index(drop=True)

	return df_

def model_output(month):
	# fit model
	classifier = ROIModel()
	X_train, y_train, X_test, y_test = classifier.prep_train_data(month)
	classifier.fit(X_train, y_train)

	# predict
	df = model_input()
	bad_features = ['county_name','date','county_fips','quality_flag', 'last_date_flag']
	X = df.drop(bad_features, axis=1)
	y = classifier.predict(X)

	prediction = pd.DataFrame(data = y, 
	              index = df['county_name'], 
	              columns = ['y_pred'])

	return prediction

m = 6 # months
model_output(m).to_parquet(path='../Capstone/Real Estate ROI/Data/prediction_'+str(m), index=True)

