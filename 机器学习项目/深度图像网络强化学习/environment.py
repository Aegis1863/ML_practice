import pandas as pd
from datetime import datetime, timedelta
import mplfinance as mpf
import numpy as np
import PIL
from io import BytesIO

def get_data(self, code, start, long):
    
    '''
    获取数据的函数, 数据已经下载了, 一般无需调用
    
    code: 'sh.600000'
    
    start: '2017-12-31'
    
    long: 持续时间
    '''
    import baostock as bs
    year, month, day = map(int, start.split('-'))
    date = datetime(year, month, day)
    end = (date + timedelta(days=long)).strftime('%Y-%m-%d')
    try:
        bs.login()

        rs = bs.query_history_k_data_plus(code,
            "date, code, open, high, low, close",
            start_date=start, end_date=end, frequency="d", adjustflag="2")
        print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        data = pd.DataFrame(data_list, columns=rs.fields)
        data.to_csv("k_data.csv", index=False)
        print('已保存为k_data.csv')
        bs.logout()
    except:
        print('数据获取失败')

class stock:
    def __init__(self):
        print('========================')
        print('v-2.1. 修改作图函数')
        print('========================')

        self.my_color = mpf.make_marketcolors(
            up="red",
            down="green",
            edge="black",
            wick="black"
        )
        self.my_style = mpf.make_mpf_style(
            marketcolors=self.my_color,
        )

    def make_img(self, df, figure_show=0):
        '''
        传入数据，返回k线图
        df: 数据, open, high, low, close, (volumn)
        '''
        fig, ax = mpf.plot(df.iloc[self.order - self.k_num:self.order], type='candle', 
                           style=self.my_style, returnfig=True, closefig=True, axisoff=True)
        self.order += 1 # 做一次图表示交易日加一
        
        buffer_ = BytesIO()
        fig.savefig(buffer_, format='png')
        buffer_.seek(0)
        dataPIL = PIL.Image.open(buffer_)
        fig_array = np.asarray(dataPIL)
        buffer_.close()
        
        if figure_show:
            return fig
        else:
            return fig_array[..., 0:3] # 只返回RGB
        
    def reset(self, offline=1, k_num=60, info_return=0, figure_show=0):
        '''
        重置环境状态
        
        offline: 默认用离线数据
        
        k_num: k线数量

        info_return: 返回信息，默认不返回
        
        figure_show: 1显示，默认不显示
        '''
        
        if offline == 1: # 离线数据
            self.data = pd.read_csv('k_data.csv')
        else: # 已经有数据了, 则无需此操作
            code= input('输入股票代码 如: sh.600000')
            start = input('输入开始时间, 如: 2020-12-01')
            long = int(input('输入持续时间, 如: 100'))
            self.data = get_data(str(code), str(start), long)
            
        self.amount = 0 # 持仓状态
        self.order = np.random.randint(k_num, int(self.data.shape[0] * 0.9)) #### 随机确定开始时间, order用于记录过了多少交易日
        self.k_num = k_num # 给模型看的k线数量
        self.start_place = self.order
        
        self.data.index = pd.DatetimeIndex(self.data['date']) # 索引设为日期, 作图需要
        self.data = self.data[['open', 'close', 'high', 'low']] # 作图需要的数据
        obs = self.make_img(self.data, figure_show) # 作图, 作为obs返回
        info = None # 无其他信息
        if info_return == 1:
            return obs, info
        else:
            return obs
    
    def step(self, action):
        '''
        采取动作
        
        0:买入1手 | 1: 满仓10手 | 2:不变 | 3:卖出1手 | 4:卖空-10手
        
        return next_state, reward, done
        '''
        
        # 采取动作调整持仓
        if self.amount < 10 and action == 0: # 买入1
            self.amount += 1
        elif action == 1:
            self.amount = 10
        elif self.amount > -10 and action == 3:
            self.amount -= 1
        elif action == 4:
            self.amount = -10
        elif action == 2:
            pass
        
        # 计算盈亏，持仓乘以涨跌率乘以10
        reward = ((self.data['close'][self.order] - self.data['close'][self.order-1]) \
                  / self.data['close'][self.order-1]) * self.amount * 10
        
        #### 若交易了60天则结束
        if self.order - self.start_place >= 60:
            done = 1
        else:
            done = 0
        
        return self.make_img(self.data, figure_show=0), reward, done

