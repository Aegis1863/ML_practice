'''
get_data(code, start, long), ??????, ???? ('sh.600000', '2017-12-31', '100')

make_img(), ??k??, ??????????, ??reset()??????

reset()), ???????, ??obs(, info), ?????k???????(???????)

step(action), ??????, ?? next_state, reward, done
'''


import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.pylab import date2num
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np

def fig2data(fig):
    '''
    ????, ??ndarray??
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

def get_data(self, code, start, long):
    '''
    ?????, ????????????
    
    code: 'sh.600000'
    
    start: '2017-12-31'
    
    long: ??
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

        #### ????? ####
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # ??????,????????
            data_list.append(rs.get_row_data())
        data = pd.DataFrame(data_list, columns=rs.fields)
        data.to_csv("k_data.csv", index=False)
        print('?????????? k_data.csv ??')
        #### ???? ####
        bs.logout()
    except:
        print('???????????')

class stock:
    def __init__(self):
        
        print('v-1.9')
        print('get_data(code, start, long) ??????')
        print('reset() ??????k??')
        print('step(action) ????action')

        self.my_color = mpf.make_marketcolors(
            up="red",  # ??K????
            down="green",  # ??K????
            edge="black",  # ????????
            wick="black"  # ????????
        )

        # ?????
        self.my_style = mpf.make_mpf_style(
            marketcolors=self.my_color,
        )

    def make_img(self, df, figure_show=0):
        '''
        df: ??, ??????, open, high, low, close, (volumn)
        figure_show: ??????, ??0, ????
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
        flag: ??????1, ???????0
        
        k_num: ????, ???????k?
        
        episods: ?????
        
        info_return: ????, ?????
        
        figure_show: ?????????, ??0, ????
        '''
        
        if offline == 1: # ?????
            self.data = pd.read_csv('k_data.csv')
        else:
            code= input('????: ? sh.600000')
            start = input('????: ? 2020-12-01')
            long = int(input('????: ? 100'))
            self.data = get_data(str(code), str(start), long)
            
        self.reward_list = [] # ????
        self.total_reward = 0 # ???
        self.amount = 0 # ???
        self.amount_history = [] # ????
        self.order = np.random.randint(k_num, int(self.data.shape[0] * 0.9)) #### ????, ?????
        self.k_num = k_num # ????? (??k?)
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
        0:??1, 1:???10, 2:??0, 3:??1, 4:???-10
        
        return next_state, reward, done
        '''
        if self.amount < 10 and action == 0: # ???
            self.amount += 1
        elif action == 1: # ??
            self.amount = 10
        elif self.amount > -10 and action == 3: # ????10?
            self.amount -= 1
        elif action == 4: # ???
            self.amount = -10
        elif action == 2:
            pass
        
        self.amount_history.append(self.amount) # ????
        
        reward = ((self.data['close'][self.order] - self.data['close'][self.order-1]) \
                  / self.data['close'][self.order-1]) * self.amount * 10 # ?????????*10
        
        self.reward_list.append(self.total_reward) # ???????
        
        if self.total_reward < -20 or self.order - self.start_place >= 100: #### ????-20??????100??
            done = 1
        else:
            done = 0
            
        return self.make_img(self.data, figure_show=0), reward, done
        

