import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.pylab import date2num
import mplfinance as mpf
import numpy as np


def fig2data(fig):
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombuffer("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    return image[:, :, :3]

def get_data(self, code, start, long):
    '''
    code: 'sh.600000'
    
    start: '2017-12-31'
    
    long: 持续时间
    '''
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
        print('v-2.0. 去掉了历史记录')
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
        df: 数据, open, high, low, close, (volumn)
        '''
        fig, ax = mpf.plot(df.iloc[self.order - self.k_num:self.order], type='candle', style=self.my_style, 
                           returnfig=True, closefig=True, axisoff=True)
        self.order += 1
        if figure_show:
            return fig
        else:
            return fig2data(fig)
        
    def reset(self, offline=1, k_num=40, info_return=0, figure_show=0):
        '''
        offline: 默认用离线数据
        
        k_num: k线数量
        
        info_return: 返回信息，默认不返回
        
        figure_show: 1显示，默认不显示
        '''
        
        if offline == 1: # 离线数据
            self.data = pd.read_csv('k_data.csv')
        else:
            code= input('输入股票代码 如: sh.600000')
            start = input('输入开始时间, 如: 2020-12-01')
            long = int(input('输入持续时间, 如: 100'))
            self.data = get_data(str(code), str(start), long)
            
        self.total_reward = 0 # 初始化总奖励
        self.amount = 0 # 持仓状态
        self.order = np.random.randint(k_num, int(self.data.shape[0] * 0.9)) #### 随机取开始时间
        self.k_num = k_num # 给模型看的k线数量
        self.start_place = self.order
        
        self.data.index = pd.DatetimeIndex(self.data['date'])
        self.data = self.data[['open', 'close', 'high', 'low']]
        obs = self.make_img(self.data, figure_show)
        info = None
        if info_return == 1:
            return obs, info
        else:
            return obs
    
    def step(self, action):
        '''
        0:买入1; 1: 满仓10; 2:不变0; 3:卖出1; 4:卖空-10
        
        return next_state, reward, done
        '''
        
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
        
        reward = ((self.data['close'][self.order] - self.data['close'][self.order-1]) \
                  / self.data['close'][self.order-1]) * self.amount * 10 # 计算盈亏，持仓乘以涨跌率乘以10
        
        if self.total_reward < -20 or self.order - self.start_place >= 60: #### 亏损-20或者盈利达到60结束
            done = 1
        else:
            done = 0
        
        return self.make_img(self.data, figure_show=0), reward, done

'''
import psutil
import matplotlib.pyplot as plt
i = 0
his = []
env = stock()
while i < 2000:
    i += 1
    env.reset()
    if i % 2 == 0: # 偶数卖出
        env.step(0)
    else:
        env.step(3)
    print(f'已交易{i+1}次    ', end='\r')
'''