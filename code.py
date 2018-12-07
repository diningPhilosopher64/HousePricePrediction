import pandas as pd
import numpy as np
import numpy.random as rand
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import kurtosis, skew
import warnings

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')


sns.set(style="ticks", color_codes=True)




class HousePrices(object):
	seq2 = pd.Series(np.arange(2))

	#Static class models.
	lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
	ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
	model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,learning_rate=0.05, max_depth=3,min_child_weight=1.7817, n_estimators=2200,reg_alpha=0.4640, reg_lambda=0.8571,subsample=0.5213, silent=1,random_state =7, nthread = -1)
	GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10,loss='huber', random_state =5)
	model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,learning_rate=0.05, n_estimators=720,max_bin = 55, bagging_fraction = 0.8,bagging_freq = 5, feature_fraction = 0.2319,feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
	KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

	#Constructor
	def __init__(self,trainData,testData):
		self.trainData = trainData
		self.testData = testData

	def dataImport(self):
		self.train = pd.read_csv(self.trainData)
		self.test = pd.read_csv(self.testData)
		self.train_Id = self.train['Id']
		self.test_Id = self.test['Id']
		self.train.drop("Id", axis = 1, inplace = True)
		self.test.drop("Id", axis = 1, inplace = True)

	def display(self):
		print(len(self.train.columns))
		fig, ax = plt.subplots()
		ax.scatter(x = self.train['GrLivArea'], y = self.train['SalePrice'])
		plt.ylabel('SalePrice', fontsize=13)
		plt.xlabel('GrLivArea', fontsize=13)
		#plt.show()


		# corrmat = self.train.corr()
		# f, ax = plt.subplots(figsize=(12, 9))
		# sns.heatmap(self.corrmat, vmax=.8, square=True);
		plt.show()

		# sns.distplot(self.train['SalePrice'] , fit=norm);

		# # Get the fitted parameters used by the function
		# (mu, sigma) = norm.fit(self.train['SalePrice'])
		# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

		# #Now plot the distribution
		# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
		# plt.ylabel('Frequency')
		# plt.title('SalePrice distribution')

		# #Get also the QQ-plot
		# fig = plt.figure()
		# res = stats.probplot(self.train['SalePrice'], plot=plt)
		# plt.show()

		# f, ax = plt.subplots(figsize=(15, 12))
		# plt.xticks(rotation='90')
		# sns.barplot(x=self.all_data_na.index, y=self.all_data_na)
		# plt.xlabel('Features', fontsize=15)
		# plt.ylabel('Percent of missing values', fontsize=15)
		# plt.title('Percent missing data by feature', fontsize=15)

		#plt.show()

	def removeOutliers(self):
		self.train = self.train.drop(self.train[(self.train['GrLivArea']>4000) & (self.train['SalePrice']<300000)].index)

	def preProcess(self):
		self.removeOutliers()

		self.train['SalePrice'] = np.log1p(self.train['SalePrice'])
		self.ntrain = self.train.shape[0]
		self.ntest = self.test.shape[0]
		self.y_train = self.train.SalePrice.values
		self.all_data = pd.concat((self.train, self.test)).reset_index(drop=True)
		self.all_data.drop(['SalePrice'], axis=1, inplace=True)
		print("all_data size is : {}".format(self.all_data.shape))


		self.all_data_na = (self.all_data.isnull().sum() / len(self.all_data)) * 100
		self.all_data_na = self.all_data_na.drop(self.all_data_na[self.all_data_na == 0].index).sort_values(ascending=False)[:30]
		self.missing_data = pd.DataFrame({'Missing Ratio' :self.all_data_na})	


		self.preprocessCategoricalColumns()
		self.preProcessNumericalColumns()

	def preprocessCategoricalColumns(self):
		#Converting PoolQC column to categorical and then using a probability distribution to fill the None values.

		print("Total Number of values ", self.all_data['PoolQC'].shape[0])
		print("Number of Null Values",self.all_data['PoolQC'].isna().sum())


			#
			#				PoolQC
			#			
			#

		#Filling NaN with None because if you convert to categorical without filling out NaN values, pandas does not consider NaN 
		# as one of the values  in the categorical column. 

		# (1) Filling NaN with None values and make  the column categorical
		self.all_data["PoolQC"] = self.all_data.PoolQC.fillna("None")
		self.all_data['PoolQC'] = pd.Categorical(self.all_data.PoolQC)

		# (2) Finding probabilities of each occurance 

		print("Before filling :")
		print(self.all_data['PoolQC'].value_counts())

		self.poolQC_probabilities = [0.98,0.006666667,0.006666667,0.006666667]
		self.poolQC_Values = ['None','Gd','Fa','Ex']
		#We need to replace only the 'None' type. Generating a sample from probability distribution
		self.indices = self.all_data[self.all_data['PoolQC'] == 'None'].index


		# (3) Use a distribution to fill out "None" values now.
		self.all_data.iloc[self.indices,65] = np.random.choice(self.poolQC_Values,len(self.indices),p=self.poolQC_probabilities)



		print("After filling :")
		print(self.all_data.PoolQC.value_counts())


		############################################################################################



			#
			#				MiscFeature
			#			
			#
				#Number of Missing values in MiscFeature
		self.all_data.MiscFeature.isna().sum()  #  1404 Null values in this column 



		#Filling NaN with None because if you convert to categorical without filling out NaN values, pandas does not consider NaN 
		# as one of the values  in the categorical column. 

		# (1) Filling NaN with None values and make  the column categorical
		self.all_data["MiscFeature"] = self.all_data['MiscFeature'].fillna("None")
		self.all_data['MiscFeature'] = pd.Categorical(self.all_data['MiscFeature'])
		self.all_data.MiscFeature = self.all_data.MiscFeature.astype('category')



		# print("Before Filling :")
		# print(self.all_data['MiscFeature'].value_counts())




		# (2) Finding probabilities of each occurance 
		print(self.all_data['MiscFeature'].value_counts())
		self.MiscFeature_probabilities = [0.962962963,0.033607682,0.001371742,0.001371742,0.000685871]
		self.MiscFeature_Values = ['None','Shed','Othr','Gar2','TenC']



		#We need to replace only the 'None' type. Generating a sample from probability distribution
		self.indices = self.all_data[self.all_data['MiscFeature'] == 'None'].index
		#Find the column index so as to use 'iloc'   . 56 is the col
		np.argwhere(self.all_data.columns == 'MiscFeature')



		# (3) Use a distribution to fill out "None" values now.
		self.all_data.iloc[self.indices,56] = np.random.choice(self.MiscFeature_Values,len(self.indices),p=self.MiscFeature_probabilities)

		# print("After filling")
		# print(self.all_data["MiscFeature"].value_counts())


		############################################################################################



			#
			#				Alley
			#			
			#


		#Number of Missing values in Alley
		self.all_data['Alley'].isna().sum()  #  1367 Null values in this column 



		#Filling NaN with None because if you convert to categorical without filling out NaN values, pandas does not consider NaN 
		# as one of the values  in the categorical column. 

		# (1) Filling NaN with None values and make  the column categorical
		self.all_data["Alley"] = self.all_data['Alley'].fillna("None")
		self.all_data['Alley'] = pd.Categorical(self.all_data['Alley'])



		# (2) Finding probabilities of each occurance 

		print("Before filling :")
		print(self.all_data['Alley'].value_counts())


		# Count of 'None' : 1367
		# Count of 'Grvl' : 50
		# Count of 'Pave' : 41

		self.Alley_probabilities = [0.937585734,0.034293553,0.028120713]
		self.Alleyy_Values = ['None','Grvl','Pave']


		#We need to replace only the 'None' type. Generating a sample from probability distribution
		self.indices = self.all_data[self.all_data['Alley'] == 'None'].index
		#Find the column index so as to use 'iloc'   . 3 is the col
		np.argwhere(self.all_data.columns == 'Alley')


		# (3) Use a distribution to fill out "None" values now.
		self.all_data.iloc[self.indices,3] = np.random.choice(self.Alleyy_Values,len(self.indices),p=self.Alley_probabilities)
		print("gg")
		self.all_data['Alley'].value_counts()



		print("After filling :")
		print(self.all_data['Alley'].value_counts())



		###########################################################################################



			#
			#				Fence
			#			
			#



		#Number of Missing values in Alley
		self.all_data['Fence'].isna().sum()  #  1177 Null values in this column 



		#Filling NaN with None because if you convert to categorical without filling out NaN values, pandas does not consider NaN 
		# as one of the values  in the categorical column. 

		# (1) Filling NaN with None values and make  the column categorical
		self.all_data["Fence"] = self.all_data['Fence'].fillna("None")
		self.all_data['Fence'] = pd.Categorical(self.all_data['Fence'])



		# (2) Finding probabilities of each occurance 

		print("Before filling :")
		print(self.all_data['Fence'].value_counts())


		# Count of 'None' : 1177
		# Count of 'MnPrv' : 157
		# Count of 'GdPrv' : 59
		# Count of 'GdWo' : 54
		# Count of 'MnWw' : 11

		self.Fence_probabilities = [0.807270233,0.107681756,0.040466392,0.037037037,0.007544582]
		self.Fence_Values = ['None','MnPrv','GdPrv','GdWo','MnWw']
		#We need to replace only the 'None' type. Generating a sample from probability distribution
		self.indices = self.all_data[self.all_data['Fence'] == 'None'].index
		#Find the column index so as to use 'iloc'   . 25 is the col
		np.argwhere(self.all_data.columns == 'Fence')



		# (3) Use a distribution to fill out "None" values now.
		self.all_data.iloc[self.indices,25] = np.random.choice(self.Fence_Values,len(self.indices),p=self.Fence_probabilities)

	


		print("After filling :")
		print(self.all_data['Fence'].value_counts())

		


		#########################################################################################



			#
			#				FirePlaceQu
			#			
			#

		#Number of Missing values in FireplaceQu
		self.all_data['FireplaceQu'].isna().sum()  #  690 Null values in this column 



		#Filling NaN with None because if you convert to categorical without filling out NaN values, pandas does not consider NaN 
		# as one of the values  in the categorical column. 

		# (1) Filling NaN with None values and make  the column categorical
		self.all_data["FireplaceQu"] = self.all_data['FireplaceQu'].fillna("None")
		self.all_data['FireplaceQu'] = pd.Categorical(self.all_data['FireplaceQu'])



		# (2) Finding probabilities of each occurance 
		print("Before filling :")
		print(self.all_data['FireplaceQu'].value_counts())


		# Count of 'None' : 690
		# Count of 'Gd' : 378
		# Count of 'TA' : 313
		# Count of 'Fa' : 33
		# Count of 'Ex' : 24
		# Count of 'Po' : 20




		self.FireplaceQu_probabilities = [0.473251029,0.259259259,0.214677641,0.022633745,0.016460905,0.013717421]
		self.FireplaceQu_Values = ['None','Gd','TA','Fa','Ex','Po']


		#We need to replace only the 'None' type. Generating a sample from probability distribution
		self.indices = self.all_data[self.all_data['FireplaceQu'] == 'None'].index


		#Find the column index so as to use 'iloc'   . 26 is the col
		np.argwhere(self.all_data.columns == 'FireplaceQu')


		# (3) Use a distribution to fill out "None" values now.
		self.all_data.iloc[self.indices,26] = np.random.choice(self.FireplaceQu_Values,len(self.indices),p=self.FireplaceQu_probabilities)


		print("After filling :")
		print(self.all_data['FireplaceQu'].value_counts())


		###########################################################################################



			#
			#				LotFrontage
			#			
			#

		'''
		Assuming houses belonging to the same Neighborhood will have similar LotFrontage, we groupby Neighborhood
		and then take mean for each locality. Then we substitute the missing values of a particular Neighborhood with
		the mean of that Neighborhood
		'''


		self.lotFrontage_df = self.all_data[['Neighborhood','LotFrontage']].copy()
		self.groupby_Neighborhood = self.lotFrontage_df.groupby('Neighborhood')

		self.indices = self.all_data[self.all_data['LotFrontage'].isna()].index


		self.mean_Neighborhood = self.groupby_Neighborhood.mean()
		self.mean_Neighborhood.head()


		for i in self.indices:    
		    self.locality = self.all_data.iloc[i,59]      
		    self.value = self.mean_Neighborhood.get_value(self.locality,'LotFrontage')     
		    self.all_data.iloc[i,49] = self.value
		   




		###########################################################################################



			#
			#				
			#	 (6)GarageYrBlt (7) GarageArea (8) GarageCar
			#
			#   (9)GarageType (10) GarageFinish (11) GarageQual (12)GarageCond


		for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):			
			self.all_data[col] = self.all_data[col].fillna(0)

		
		self.all_data['GarageType'] = self.all_data['GarageType'].fillna('None')
		self.all_data['GarageFinish'] = self.all_data['GarageFinish'].fillna('None')
		self.all_data['GarageQual'] = self.all_data['GarageQual'].fillna('None')
		self.all_data['GarageCond'] = self.all_data['GarageCond'].fillna('None')




		


		for col in ('BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath'):
			self.all_data[col] = self.all_data[col].fillna(0)


		for col in ('BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual'):
			self.all_data[col] = self.all_data[col].fillna('None')




		#############################################################################################



			#
			#				
			#	 Electrical , Exterior1st,Exterior2nd,SaleType,KitchenQual
			#
			#   


		#Electrical has only 1 Null value , hence replacing by most frequently occuring value i.e. mode of the column


		self.all_data['Electrical'] = self.all_data['Electrical'].fillna(self.all_data['Electrical'].mode()[0])


		#Similarly for Exterior1st, Exterior2nd,SaleType and KitchenQual
		self.all_data['Exterior1st'] = self.all_data['Exterior1st'].fillna(self.all_data['Exterior1st'].mode()[0])
		self.all_data['Exterior2nd'] = self.all_data['Exterior2nd'].fillna(self.all_data['Exterior2nd'].mode()[0])
		self.all_data['KitchenQual'] =self.all_data['KitchenQual'].fillna(self.all_data['KitchenQual'].mode()[0])
		self.all_data['SaleType'] = self.all_data['SaleType'].fillna(self.all_data['SaleType'].mode()[0])







		##############################################################################################



			#
			#				
			#	 
			#    'MasVnrArea','MasVnrType' and other columns
			#
			#   

		self.indices = self.all_data[self.all_data['MasVnrArea'] == 0].index
		

		self.all_data['MasVnrArea'] = self.all_data['MasVnrArea'].fillna(0)
		self.all_data['MasVnrType'] = self.all_data['MasVnrType'].fillna('None')
		self.all_data = self.all_data.drop(['Utilities'], axis=1)

		self.all_data["Functional"] = self.all_data["Functional"].fillna("Typ")
		self.all_data['MSSubClass'] = self.all_data['MSSubClass'].fillna("None")








		##############################################################################################


		# Hence no remaining Columns with missing values.

		# MSSubClass is categorical as only a certain set of numbers are appearing. Hence converting it to categorical

		# OverallCond is categorical as only a certain set of numbers are appearing. Hence converting it to categorical




		self.all_data['MSSubClass'].unique()
		#array([ 20, 180,  60,  80,  50,  75,  30,  70,  90, 120,  45, 190,  85,  160,  40])

		self.all_data['MSSubClass'] = self.all_data['MSSubClass'].apply(str)


		self.all_data['OverallCond'].unique()
		#array([6, 5, 7, 8, 3, 4, 9, 2, 1])

		self.all_data['OverallCond'] = self.all_data['OverallCond'].apply(str)


		#Unlike Yrbuilt , YrSold is taking only a set of numbers converting it to categorical.
		self.all_data['YrSold'].unique()
		#array([2008, 2006, 2010, 2007, 2009])

		self.all_data['YrSold'] = self.all_data['YrSold'].astype(str)

		#Similarly for MonthSold ie MoSold
		self.all_data['MoSold'].unique()
		#array([ 5,  6,  3,  4, 12,  7,  8, 11,  1, 10,  2,  9])

		self.all_data['MoSold'] = self.all_data['MoSold'].astype(str)




		#	 Linear regression works only on columns with numeric values , Using labelEncoder to convert 
		#	the categorical colums to a numeric values

		#Set of columns which have categorical values:

		self.columns = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond','ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1','BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond','YrSold', 'MoSold')


		for column in self.columns:
		    self.lbl = LabelEncoder() 
		    self.lbl.fit(list(self.all_data[column].values)) 
		    self.all_data[column] = self.lbl.transform(list(self.all_data[column].values))


		# skewness = skewness[abs(skewness) > 0.75]
		# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

		# from scipy.special import boxcox1p
		# self.skewed_features = skewness.index
		# lam = 0.15
		# for feat in self.skewed_features:
		#     #all_data[feat] += 1
		#     self.all_data[feat] = boxcox1p(self.all_data[feat], self.lam)




		# This will map the labels of categorical data to 0,1,2,3 etc.
		self.all_data = pd.get_dummies(self.all_data)

	def preProcessNumericalColumns(self):
		#These features are positively correlated with the salePrice hence creating new features by 
		#taking 3 polynomials square, cube and square root 

		# Taking the top 10 correlated valuse.

		# OverallQual    0.817315
		# GrLivArea      0.715624
		# GarageCars     0.687771
		# GarageArea     0.662332
		# TotalBsmtSF    0.637558
		# 1stFlrSF       0.608198
		# FullBath       0.582020
		# YearBuilt      0.572574

		# As total square feet is important. Adding total sqfootage feature 
		self.all_data['TotalSF'] = self.all_data['TotalBsmtSF'] + self.all_data['1stFlrSF'] + self.all_data['2ndFlrSF']

		self.all_data["OverallQual-s2"] = self.all_data["OverallQual"] ** 2
		self.all_data["OverallQual-s3"] = self.all_data["OverallQual"] ** 3
		self.all_data["OverallQual-Sq"] = np.sqrt(self.all_data["OverallQual"])


		self.all_data["GrLivArea-s2"] = self.all_data["GrLivArea"] ** 2
		self.all_data["GrLivArea-s3"] = self.all_data["GrLivArea"] ** 3
		self.all_data["GrLivArea-Sq"] = np.sqrt(self.all_data["GrLivArea"])



		self.all_data["GarageCars-s2"] = self.all_data["GarageCars"] ** 2
		self.all_data["GarageCars-s3"] = self.all_data["GarageCars"] ** 3
		self.all_data["GarageCars-Sq"] = np.sqrt(self.all_data["GarageCars"])



		self.all_data["GarageArea-s2"] = self.all_data["GarageArea"] ** 2
		self.all_data["GarageArea-s3"] = self.all_data["GarageArea"] ** 3
		self.all_data["GarageArea-Sq"] = np.sqrt(self.all_data["GarageArea"])



		self.all_data["TotalBsmtSF-s2"] = self.all_data["TotalBsmtSF"] ** 2
		self.all_data["TotalBsmtSF-s3"] = self.all_data["TotalBsmtSF"] ** 3
		self.all_data["TotalBsmtSF-Sq"] = np.sqrt(self.all_data["TotalBsmtSF"])


		self.all_data["1stFlrSF-s2"] = self.all_data["1stFlrSF"] ** 2
		self.all_data["1stFlrSF-s3"] = self.all_data["1stFlrSF"] ** 3
		self.all_data["1stFlrSF-Sq"] = np.sqrt(self.all_data["1stFlrSF"])


		self.all_data["FullBath-s2"] = self.all_data["FullBath"] ** 2
		self.all_data["FullBath-s3"] = self.all_data["FullBath"] ** 3
		self.all_data["FullBath-Sq"] = np.sqrt(self.all_data["FullBath"])


		self.all_data["YearBuilt-s2"] = self.all_data["YearBuilt"] ** 2
		self.all_data["YearBuilt-s3"] = self.all_data["YearBuilt"] ** 3
		self.all_data["YearBuilt-Sq"] = np.sqrt(self.all_data["YearBuilt"])

		self.all_data["TotalSF-s2"] = self.all_data["TotalSF"] ** 2
		self.all_data["TotalSF-s3"] = self.all_data["TotalSF"] ** 3
		self.all_data["TotalSF-Sq"] = np.sqrt(self.all_data["TotalSF"])


		
		self.train = self.all_data[:1020]
		self.test = self.all_data[1020:]


		self.all_data.to_csv('./all.csv')
	#Validation function
	

	def rmsle_cv(self,model):
		#self.n_folds = 5
		self.kf = KFold(5, shuffle=True, random_state=42).get_n_splits(self.train.values)
		self.rmse= np.sqrt(-cross_val_score(model, self.train.values, self.y_train, scoring="neg_mean_squared_error", cv = self.kf))
		return(self.rmse)

	#Lasso
	def lasso_model(self):
		#self.lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
		#self.score = self.rmsle_cv(self.lasso)
		self.score = self.rmsle_cv(HousePrices.lasso)
		print("\nLasso score: {:.4f} ({:.4f})\n".format(self.score.mean(), self.score.std()))

	# ElasticNet.
	def elasticNet(self):
		#self.ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
		self.score = self.rmsle_cv(HousePrices.ENet)
		print("ElasticNet score: {:.4f} ({:.4f})\n".format(self.score.mean(), self.score.std()))

	#Kernel Ridge regression.
	def kernelRegression(self):
		#self.KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
		self.score = self.rmsle_cv(HousePrices.KRR)	
		print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(self.score.mean(), self.score.std()))

	#GradientBoosting.
	def gradientBoosting(self):
		#self.GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10,loss='huber', random_state =5)
		self.score = self.rmsle_cv(HousePrices.GBoost)
		print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(self.score.mean(), self.score.std()))

	# XgbRegressor.
	def xgbRegressor(self):
		#self.model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,learning_rate=0.05, max_depth=3,min_child_weight=1.7817, n_estimators=2200,reg_alpha=0.4640, reg_lambda=0.8571,subsample=0.5213, silent=1,random_state =7, nthread = -1)
		self.score = self.rmsle_cv(HousePrices.model_xgb)
		print("Xgboost score: {:.4f} ({:.4f})\n".format(self.score.mean(), self.score.std()))

	# LgbRegressor.
	def lgbRegressor(self):
		#model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,learning_rate=0.05, n_estimators=720,max_bin = 55, bagging_fraction = 0.8,bagging_freq = 5, feature_fraction = 0.2319,feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
		self.score = self.rmsle_cv(HousePrices.model_lgb)
		print("LgbRegressor score: {:.4f} ({:.4f})\n".format(self.score.mean(), self.score.std()))






if __name__ == "__main__":
	train_data_name = sys.argv[1]
	test_data_name = sys.argv[2]

	model = HousePrices(train_data_name,test_data_name)
	model.dataImport()
	model.display()

	model.preProcess()

	#model.lasso_model()
	#model.elasticNet()

	# model.gradientBoosting()
	# model.xgbRegressor()

	# model.lgbRegressor()
	

	

	
