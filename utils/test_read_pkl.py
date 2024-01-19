import os

import pandas as pd
import pickle
PROJECT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

path1 = PROJECT_PATH + os.sep + 'data' + os.sep + "csi300.pkl"

pd.set_option('display.max_columns',1000)   # 设置最大显示列数的多少
pd.set_option('display.width',1000)         # 设置宽度,就是说不换行,比较好看数据
pd.set_option('display.max_rows',500)
df = pd.read_pickle(path1)
# print(df)


path2 = PROJECT_PATH+ os.sep + 'data' + os.sep +"data_train_predict"+ os.sep + "2022-12-30.pkl"
df2 = pd.read_pickle(path2)
# print(df2)
for key,value in df2.items():
    print(key+": "+str(value))