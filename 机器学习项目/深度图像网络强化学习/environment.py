'''
get_data(code, start, long), 在线下载数据, 输入例如 ('sh.600000', '2017-12-31', '100')

make_img(), 返回k线图, 如无调试需要可以不管, 执行reset()方法自动调用

reset()), 初始化一个环境, 给出obs(, info), 即一个初始k线图的数组形式(和一个信息备注)

step(action), 输入一个动作, 返回 next_state, reward, done
'''

from click import style
import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.pylab import date2num
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np

def fig2data(fig):
    '''
    输入图片, 输出ndarray数组
    '''
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

class stock:
    def __init__(self):
        self.data = 0
        self.reward_list = []
        print('get_data(code, start, long)在线下载数据')
        print('reset()返回一个初始k线图')
        print('step(action)采取动作action')

        self.my_color = mpf.make_marketcolors(
            up="red",  # 上涨K线的颜色
            down="green",  # 下跌K线的颜色
            edge="black",  # 蜡烛图箱体的颜色
            wick="black"  # 蜡烛图影线的颜色
        )

        # 自定义风格
        self.my_style = mpf.make_mpf_style(
            marketcolors=self.my_color,
        )
        
        self.total_reward = 0
        self.buy_amount = 0
        
    def get_data(self, code, start, long):
        '''
        无初始数据, 需要在线下载时使用该方法
        
        code: 'sh.600000'
        
        start: '2017-12-31'
        
        long: 整数
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

            #### 打印结果集 ####
            data_list = []
            while (rs.error_code == '0') & rs.next():
                # 获取一条记录，将记录合并在一起
                data_list.append(rs.get_row_data())
            data = pd.DataFrame(data_list, columns=rs.fields)
            data.to_csv("k_data.csv", index=False)
            print('数据已下载到同目录下 k_data.csv 文件')
            #### 登出系统 ####
            bs.logout()
        except:
            print('在线获取数据时发生错误')

    def make_img(self, df, time, figure_show=0):
        '''
        df: 数据, 以时间为索引, open, high, low, close, (volumn)
        figure_show: 是否返回图片, 默认0, 返回数组
        '''

        fig, ax = mpf.plot(df.iloc[self.order-time:self.order], type='candle', style=self.my_style, 
                           returnfig=True, closefig=True, axisoff=True)
        self.order += 1
        if figure_show:
            return fig
        else:
            return fig2data(fig)
        
    def reset(self, offline=1, time=40, info_return=0, figure_show=0):
        '''
        flag: 有本地数据传1, 没有本地数据传0
        
        time: 时间周期
        
        info_return: 额外信息, 默认不显示
        
        figure_show: 返回图像而不是数组, 默认0, 返回数组
        '''
        self.order = time # 时间周期 (几根k线)
        self.time = time # 固定的周期
        
        if offline == 1: # 有本地数据
            self.data = pd.read_csv('k_data.csv')
        else:
            code= input('输入代码: 如 sh.600000')
            start = input('起始时间: 如 2020-12-01')
            long = int(input('持续天数: 如 100'))
            self.data = self.get_data(str(code), str(start), long)
        self.data.index = pd.DatetimeIndex(self.data['date'])
        self.data = self.data[['open', 'close', 'high', 'low']]
        obs = self.make_img(self.data, self.time, figure_show)
        info = None
        if info_return == 1:
            return obs, info
        else:
            return obs
    
    def step(self, action):
        '''
        买入1, 卖出-1, 全买 10, 全卖 -10, 不变0
        
        return next_state, reward, done
        '''
        if self.buy_amount <= 10 and action > 0: # 只让最多有10个买入
            self.buy_amount += action
        elif self.buy_amount > -action and action < 0: # 有买入才能卖出
            self.buy_amount += action
        reward = ((self.data['close'][-1] - self.data['open'][-1]) / self.data['open'][-1])\
            * self.buy_amount # 持仓量乘以涨跌比例
        self.total_reward += reward
        
        self.reward_list.append(self.total_reward) # 记录历史总奖励
        
        if self.total_reward < -20 or self.order - self.time >= 50: # 分数小于-20或者次数大于50结束
            done = 1
        else:
            done = 0
        return self.make_img(self.data, self.time, figure_show=0), reward, done
        

