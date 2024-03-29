import os
import time
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

feature_cols = ['high', 'low', 'close', 'open', 'to', 'vol']


def cal_pccs(x, y, n):
    sum_xy = np.sum(np.sum(x * y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x * x))
    sum_y2 = np.sum(np.sum(y * y))
    pcc = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    return pcc


def calculate_pccs(xs, yss, n):
    result = []
    for name in yss:
        ys = yss[name] # 取出一个股票代码的所有 6*20
        tmp_res = []
        for pos, x in enumerate(xs):
            y = ys[pos]
            tc = cal_pccs(x, y, n)
            tmp_res.append(cal_pccs(x, y, n))
        result.append(tmp_res)
    tn = np.mean(result, axis=1)
    return np.mean(result, axis=1)


def stock_cor_matrix(ref_dict, codes, n, processes=1):
    if processes > 1:
        pool = mp.Pool(processes=processes)
        args_all = [(ref_dict[code], ref_dict, n) for code in codes]
        results = [pool.apply_async(calculate_pccs, args=args) for args in args_all]
        output = [o.get() for o in results]
        data = np.stack(output)
        return pd.DataFrame(data=data, index=codes, columns=codes)
    data = np.zeros([len(codes), len(codes)])
    for i in tqdm(range(len(codes))):
        data[i, :] = calculate_pccs(ref_dict[codes[i]], ref_dict, n)
    print(data)
    return pd.DataFrame(data=data, index=codes, columns=codes)


PROJECT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

path1 = "/home/THGNN-main/data/csi300.pkl"
path1 = PROJECT_PATH + os.sep + 'data' + os.sep + "csi300.pkl"
df1 = pickle.load(open(path1, 'rb'), encoding='utf-8')  # Index(['dt', 'code', 'close', 'high', 'low', 'open', 'to', 'vol', 'label'], dtype='object')
# prev_date_num Indicates the number of days in which stock correlation is calculated
prev_date_num = 20  # 可以理解为一个滑动窗口
date_unique = df1['dt'].unique()
stock_trade_data = date_unique.tolist()
stock_trade_data.sort()
stock_num = df1.code.unique().shape[0]
# dt is the last trading day of each month
dt = ['2022-11-30', '2022-12-30']
# for i in ['2020','2021','2022']:
#     for j in ['01','02','03','04','05','06','07','08','09','10','11','12']:
#         stock_m=[k for k in stock_trade_data if k>i+'-'+j and k<i+'-'+j+'-32']
#         dt.append(stock_m[-1])
df1['dt'] = df1['dt'].astype('datetime64')

for i in range(len(dt)):
    df2 = df1.copy()
    end_data = dt[i]
    start_data = stock_trade_data[stock_trade_data.index(end_data) - (prev_date_num - 1)]
    df2 = df2.loc[df2['dt'] <= end_data]
    df2 = df2.loc[df2['dt'] >= start_data]
    code = sorted(list(set(df2['code'].values.tolist())))
    test_tmp = {}
    for j in tqdm(range(len(code))):

        t1 = df2['code']
        t2 = code[j]
        df3 = df2.loc[df2['code'] == code[j]]
        y = df3[feature_cols].values  # 取出每个股票代码的所有列  0001、0002 00063 00069  一共四个
        if y.T.shape[1] == prev_date_num:
            test_tmp[code[j]] = y.T
    t1 = time.time()
    result = stock_cor_matrix(test_tmp, list(test_tmp.keys()), prev_date_num, processes=1)
    result = result.fillna(0) # nan置为0
    for i in range(0, stock_num): # 对角线置1
        result.iloc[i, i] = 1
    t2 = time.time()
    print('time cost', t2 - t1, 's')
    result_relation_path = path1 + os.sep + "relation" + os.sep + str(end_data) + ".csv"
    result.to_csv(result_relation_path)
