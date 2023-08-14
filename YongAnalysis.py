import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats
from pprint import pprint
import sys


class DataExploratioin:
    '''
    데이터 탐색 시 사용 가능한 Class 

    기존 존재하는 프레임워크들을 이용하여 자주 이용하는 프레임워크들을 활용하여 나만의 분석 툴을 만들려고 함 

    데이터 요약, 결측값 처리 등의 내용이 담겨있는 class 
    '''

    def __init__(self, data):
        self.data = data

    def summarize(self):
        '''
        데이터를 초창기에 요약해주는 method
        '''

        cols = self.data.columns

        size = round(sys.getsizeof(self.data) / 1024 ** 2, 2)

        print(f'data size : {size}MB')

        self.result = pd.DataFrame()

        self.result['Dtype'] = self.data.dtypes.values
        self.result['Count'] = self.data.count().values
        self.result['Nunique'] = self.data.nunique().values
        self.result['Missing value'] = self.data.isna().sum().values
        self.result['Missing %'] = [str(round(
            missing / len(self.data), 2) * 100) + '%' for missing in self.result['Missing value']]
        self.result['Most Freq Value'] = self.data.mode().iloc[0].values

        freq_prop = []

        for i, col in enumerate(cols):

            raw_data = self.data.loc[~self.data[col].isna(), col]
            freq_value = self.result['Most Freq Value'].iloc[i]

            prop = np.mean(
                np.array(raw_data == freq_value)
            )

            prop_str = str(round(np.mean(prop) * 100, 1)) + '%'

            if prop_str == 'nan%':
                freq_prop.append(self.result['Missing %'].iloc[i])
            else:
                freq_prop.append(prop_str)

        self.result['Most Freq Value %'] = freq_prop

        self.result['Min'] = self.data.describe(include='all').T['min'].values
        self.result['Max'] = self.data.describe(include='all').T['max'].values
        self.result['Mean'] = self.data.describe(
            include='all').T['mean'].values
        self.result['Median'] = self.data.describe(
            include='all').T['50%'].values

        self.result = self.result.set_index(cols)

        self.result = self.result.fillna('-')

        return self.result
