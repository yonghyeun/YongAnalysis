import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats
from pprint import pprint
import sys
import time 
import warnings 


class DataExploratioin:
    '''
    데이터 탐색 시 사용 가능한 Class 

    기존 존재하는 프레임워크들을 이용하여 자주 이용하는 프레임워크들을 활용하여 나만의 분석 툴을 만들려고 함 

    데이터 요약, 결측값 처리 등의 내용이 담겨있는 class 
    '''

    def __init__(self, data):
        self.data = data
        warnings.filterwarnings(action = 'ignore')
                

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
        
        memory = (self.data.memory_usage(deep = True) // 1024 **2).values[1:] # index 의 usage 는 제외하고 보자 

        
        self.result['MB'] = [str(m) + ' mb' for m in memory]
        self.result = self.result.set_index(cols)

        self.result = self.result.fillna('-')

        display(self.result)
    
    
    def progress_bar(self,iterable, total_blocks = 10):
        
        total_items = len(iterable)
        block_size = total_items // total_blocks
        
        for i, item in enumerate(iterable, start=1):
            if i % block_size == 0 or i == total_items:
                progress = (i / total_items) * 100
                blocks = int(progress / (100 / total_blocks))
                empty_blocks = total_blocks - blocks
                progress_bar = '■' * blocks + '▢' * empty_blocks
                print(f"\rProgress: [{progress_bar}] {progress:.2f}%", end='', flush=True)
            yield item
            time.sleep(0.0000001)
    
    def reduce_size(self):
                
        original_size = round(sys.getsizeof(self.data) / 1024 ** 2,2)
        
        df = self.data.copy()
        
        for col in self.progress_bar(df.columns):
            
            dtp = df[col].dtype
            
            if dtp == 'object':
                df[col] = df[col].astype('category')
            else: # numeric type이면 
                
                if min(df[col]) >= 0 : # 부호가 없다면 unit 으로 변경해줘도 된다.
                    max_value = max(df[col])
                    
                    bits = [8,16,32,64]
                    
                    for bit in bits: # 최소한의 비트로 표현 될 수 있게 dtype 변경 
                        if max_value < 2 ** bit:
                            # 결측치가 있는 경우 astype 으로 변경하지 못하니 결측치를 채워준 후 변경하고 다시 결측치를 채우자 
                            df[col] = df[col].fillna(2 ** bit - 1)
                            df[col] = df[col].astype(f'uint{bit}')
                            df[col] = df[col].replace(2 ** bit - 1, np.NaN)
                            break
                        
                else: # 부호가 있다면 int type 으로 바꿔주자 
                    
                    max_value = max(abs(min(df[col])), max(df[col]))
                    
                    bits = [8,16,32,64]
                    
                    for bit in bits:
                        if max_value < 2 ** bit:
                            df[col] = df[col].fillna(2 ** bit - 1)
                            df[col] = df[col].astype(f'int{bit}')
                            df[col] = df[col].replace(2 ** bit - 1, np.NaN)
                            break
                        
        print('\n')
                        
        after_size = round(sys.getsizeof(df) / 1024 ** 2,2)
        
        # 바꾼 후 결과 보여주기 
        after = DataExploratioin(df)
        after.summarize()
        
        print(f'\n {original_size}MB -> {after_size}MB')
            
        return df
                        
            
                