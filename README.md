# readme
# Google Analytics Customer Revenue Prediction EDA 
# 竞赛数据预处理
## requirement
  os
  pandas
  json
  ast
  warnings
  sklearn
  lightgbm
## 1、使用pandas读取csv，观察数据
### 训练集行数：1708345
### 训练集列数：13
### 测试集行数：401589
### 测试集列数：13
### 列名：

 channelGrouping  | customDimensions  | date | device | fullVisitorId | geoNetwork hits | socialEngagementType | totals | trafficSource | visitId | visitNumber | visitStartTime
 ----  | ----  | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
 
 ## 2、因为数据量过大，所以将数据集分块处理，8G及以下内存电脑无法直接读取数据集

 ## 3、处理JSON列和类JSON列
 ### 需要处理的列的描述：
 #### JSON列：
 ##### 1、decice列
 ##### 2、geoNetwork列
 ##### 3、totals列
 ##### 4、trafficSource列
 #### 类JSON列：
 ##### 1、customDimensions列
 
 ## 4、处理时间数据
 ### 生成date、week、month列
 
  ## 5、删除常数列 删除缺失值太多的列
  ### 值是常数的列去掉，剩38列
  ### 缺失大于90%的字段删掉，剩31列
## 6、数据集缺省值填充
  ### 'bounces', 'transactions', 'totalTransactionRevenue', 'index', 'newVisits', 'timeOnSite', 'sessionQualityDim' 填充0
  ### 'referralPath', 'isTrueDirect', 'keyword', 'value' 填充Nan的值
