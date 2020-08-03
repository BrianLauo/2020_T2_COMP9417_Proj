# 使用pandas读取csv，观察数据，尝试解析JSON列
# 数据量过大，应拆分数据集（20000条1组）
# 分别解析JSON，通过json_normalize方法解析json列，非标准json格式需要先去掉最外层中括号（literal_eval）
# 合并数据集，检查总行数
# 缺失值分析和数据预处理
# 创建组合特征，计算出下次购买时间，区分新增购买和重复购买
# 数据透视和描述统计，按年月汇总，估计答案大致范围
# 尝试建模求解

import os
import pandas as pd
import json
from pandas import json_normalize
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')


data_path = './dataset/'

""" 1、显示数据集基本信息 """
# # 内存低于16G无法直接读取集数据
# # 建议分块处理数据集
# train_set = pd.read_csv(data_path + 'train_v2.csv')
# test_set = pd.read_csv(data_path + 'test_v2.csv')

# # 训练集行数：1708345
# # 训练集列数：13
# # 测试集行数：401589
# # 测试集列数：13
# print("[*] 行数：", train_set.shape[0])
# print("[*] 列数：", train_set.shape[1])

# # 列名：['channelGrouping' 'customDimensions' 'date' 'device' 'fullVisitorId'
# #  'geoNetwork' 'hits' 'socialEngagementType' 'totals' 'trafficSource'
# #  'visitId' 'visitNumber' 'visitStartTime']
# 需要处理的列的描述：
# 1、decice列：
# {"browser": "Firefox",
# "browserVersion": "not available in demo dataset",
# "browserSize": "not available in demo dataset",
# "operatingSystem": "Windows",
# "operatingSystemVersion": "not available in demo dataset",
# "isMobile": false,
# "mobileDeviceBranding": "not available in demo dataset",
# "mobileDeviceModel": "not available in demo dataset",
# "mobileInputSelector": "not available in demo dataset",
# "mobileDeviceInfo": "not available in demo dataset",
# "mobileDeviceMarketingName": "not available in demo dataset",
# "flashVersion": "not available in demo dataset",
# "language": "not available in demo dataset",
# "screenColors": "not available in demo dataset",
# "screenResolution": "not available in demo dataset",
# "deviceCategory": "desktop"}
# 2、geoNetwork列：
# {"continent": "Americas",
# "subContinent": "Northern America",
# "country": "United States",
# "region": "not available in demo dataset",
# "metro": "not available in demo dataset",
# "city": "not available in demo dataset",
# "cityId": "not available in demo dataset",
# "networkDomain": "level3.net",
# "latitude": "not available in demo dataset",
# "longitude": "not available in demo dataset",
# "networkLocation": "not available in demo dataset"}
# 3、totals列：
# {"visits": "1",
# "hits": "1",
# "pageviews": "1",
# "bounces": "1",
# "newVisits": "1"}
# 4、trafficSource列：
# {"referralPath": "/pagead/render_post_ads.html",
# "campaign": "(not set)",
# "source": "googleads.g.doubleclick.net",
# "medium": "referral",
# "adwordsClickInfo": {"criteriaParameters": "not available in demo dataset"}}
# print("[*] 列数：", train_set.columns.values)


""" 2、因为数据量过大，将数据集分块处理，8G及以下内存电脑建议使用 """


def read_df(path, file_name, nrows=10000):
    os.chdir(path)
    df = pd.read_csv(file_name, dtype={'fullVisitorId': 'str', 'visitId': 'str'}, chunksize=nrows)
    return df


def split_df():
    """
    切分数据集 并存储
    训练集被切分 并存放在了 /train_split/文件夹下
    :return:
    """
    chunk_all = read_df(data_path, 'test_v2.csv', nrows=10000)

    i = 0
    for chunk in chunk_all:
        # print(type(chunk))
        chunk.to_csv(str(i) + '.csv', index=False)
        print('No. %s is done.' % i)
        i += 1


# # 切分数据集 并存储
# split_df()

""" 3、处理JSON列和类JSON列 """
def JSON_df():
    # 拆分文件夹目录
    path_split = "./dataset/test_split/"
    for csv_name in os.listdir(path_split):
        # 需要处理的JSON列
        json_col = ['device', 'geoNetwork', 'totals', 'trafficSource']
        print(path_split+csv_name)
        data = pd.read_csv(path_split+csv_name, sep=',', header=0, converters={column: json.loads for column in json_col})
        data['customDimensions'][data['customDimensions'] == "[]"] = "[{}]"
        # data['customDimensions'] = demjson.decode(data['customDimensions'].apply(literal_eval).str[0])
        data['customDimensions'] = data['customDimensions'].apply(literal_eval).str[0]
        # 每个需处理的JSON列处理为多列的值
        json_col.append('customDimensions')
        for col in json_col:
            data_col = json_normalize(data[col])
            data_col.columns = [f"{sub_col}" for sub_col in data_col.columns]
            data = data.drop(col, axis=1).merge(data_col, right_index=True, left_index=True)
        os.remove(path_split+csv_name)
        data.to_csv(path_split+csv_name, index=False)

# # 处理JSON列
# JSON_df()


""" 4、处理时间数据 """
def date_df():
    """
    处理时间数据 生成 date month week
    :return:
    """
    # 拆分文件夹目录
    path_split = "./dataset/test_split/"
    for csv_name in os.listdir(path_split):
        print(path_split + csv_name)
        data = pd.read_csv(path_split+csv_name, sep=',', header=0)
        # 处理时间 strp对象是字符串 所以不采用这个方法选择年月了
        data['date'] = data['date'].astype(str)
        data['date'] = data['date'].apply(lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:8])
        data['date'] = pd.to_datetime(data['date'])  # 时间戳
        data['month'] = data['date'].apply(lambda x: x.strftime('%Y-%m'))
        data['week'] = data['date'].dt.weekday
        data['transactionRevenue'] = data['transactionRevenue'].astype(float).fillna(0)
        os.remove(path_split + csv_name)
        data.to_csv(path_split + csv_name, index=False)

# # 处理时间数据
# date_df()

""" 5、删除多余的列 """
def drop_df():
    """
    删除常数列 删除缺失值太多的列
    :return:
    """
    # 拆分文件夹目录
    path_split = "./dataset/test_split/"
    for csv_name in os.listdir(path_split):
        print(path_split + csv_name)
        data = pd.read_csv(path_split + csv_name, sep=',', header=0)
        for r in data.columns:
            a = data[r].value_counts()
            if len(a) < 2:
                print(r)
                print(a)
                print('----')
        # 值是常数的列去掉，剩38列
        data = data.drop(['socialEngagementType', 'browserSize', 'browserVersion', 'flashVersion',
                          'language', 'mobileDeviceBranding', 'mobileDeviceInfo', 'mobileDeviceMarketingName',
                          'mobileInputSelector', 'operatingSystemVersion', 'screenResolution', 'screenColors', 'cityId',
                          'latitude', 'longitude', 'networkLocation', 'visits', 'adwordsClickInfo.criteriaParameters'], axis=1)

        # 缺失值个数
        for r in data.columns:
            a = len(data[r][pd.isnull(data[r])]) / len(data)
            if a > 0.8:
                print(r)
                print(a)
                print('----')
        # 缺失大于90%的字段删掉，剩31列  不要把revenue删掉了
        data = data.drop(
            ['adContent', 'adwordsClickInfo.adNetworkType', 'adwordsClickInfo.gclId', 'adwordsClickInfo.isVideoAd',
             'adwordsClickInfo.page', 'adwordsClickInfo.slot', 'hits_x'], axis=1)
        os.remove(path_split + csv_name)
        data.to_csv(path_split + csv_name, index=False)

# # 删除多余的列
# drop_df()

def fill_df():
    """
    对于有缺省的列进行填充
    bounces 列填充0
    transactions 列填充0
    totalTransactionRevenue 列填充0
    :return:
    """
    # 拆分文件夹目录
    path_split = "./dataset/test_split/"
    for csv_name in os.listdir(path_split):
        print(path_split + csv_name)
        data = pd.read_csv(path_split + csv_name, sep=',', header=0)
        fill_0_col = ['bounces', 'transactions', 'totalTransactionRevenue', 'index', 'newVisits', 'timeOnSite', 'sessionQualityDim']
        for col in fill_0_col:
            data[col] = data[col].astype(float).fillna(0)
        fill_None_col = ['referralPath', 'isTrueDirect', 'keyword', 'value']
        for col in fill_None_col:
            data[col] = data[col].astype(str).fillna("Nan")
        if 'hits_x' in  data.columns.values:
            data = data.drop(['hits_x'], axis=1)
        os.remove(path_split + csv_name)
        data.to_csv(path_split + csv_name, index=False)
# # 缺省值填充
# fill_df()



""" 数据集合并 """
def concat_df():
    df_list = []

    # 拆分文件夹目录
    path_split = "./dataset/test_split/"
    for csv_name in os.listdir(path_split):
        print(path_split + csv_name)
        data = pd.read_csv(path_split + csv_name, sep=',', header=0)
        df_list.append(data)
    df = pd.concat(df_list, ignore_index=True)
    fill_None_col = ['referralPath', 'isTrueDirect', 'keyword', 'value']
    for col in fill_None_col:
        df[col] = df[col].astype(str).fillna("Nan")
    df.to_csv("test.csv", index=False)

# 数据集合并
concat_df()