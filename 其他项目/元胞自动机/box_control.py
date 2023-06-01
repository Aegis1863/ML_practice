from random import randint

def get_life_coordinate(box_num, life_num): # 随机设定每一个元胞的初始坐标
    ret = []
    for n in range(0, life_num):
        ret.append((randint(0, box_num), randint(0, box_num)))
    return ret

def init_boxs(box_num, win_width, win_height, life_num):
    # 初始化一个方块容器
    life_coordinate = get_life_coordinate(box_num, life_num) # 获得初始坐标
    boxs = []
    for x in range(0, box_num):
        tmp = []
        for y in range(0, box_num):
            if (x, y) in life_coordinate:
                value = 1
            else:
                value = 0
            tmp.append({
                'value': value,
                'x': x * (win_width / box_num),
                'y': y * (win_height / box_num),
                'width': win_width / box_num,
                'height': win_height / box_num
            }) # 每个格子的状态、位置和宽高，都是pygame绘图必须的参数
        boxs.append(tmp) # 又套了一层列表
    return boxs # 一共160个行，但每一行中又有160个列（160*160个格子数据）


def change_boxs(boxs):
    boxs_copy = boxs.copy() # 建立一个副本
    # 进行一次细胞变换
    # 如果一个细胞周围有3个细胞为生，则将细胞转换为生
    # 如果一个细胞周围有2个细胞为生，则状态保持不变
    # 在其他情况下，细胞都会转为死
    x_len = len(boxs_copy)
    for x, v1 in enumerate(boxs_copy):
        y_len = len(v1)
        for y, v2 in enumerate(v1):
            count = 0
            # 超过上限
            if x == x_len-1 and y == y_len-1: 
                # 计算到最右下角网格的情况，x是0到box_num-1因此y_len-1
                if boxs_copy[x-1][y-1]['value'] == 1: # 左上
                    count += 1
                if boxs_copy[x-1][y]['value'] == 1: # 左
                    count += 1
                if boxs_copy[x][y-1]['value'] == 1: # 上
                    count += 1
                if boxs_copy[0][y]['value'] == 1: # 相当于下，后同（延拓图）
                    count += 1
                if boxs_copy[0][y-1]['value'] == 1: ## 左下，y改为y-1，原来的错了
                    count += 1
                if boxs_copy[0][0]['value'] == 1: # 右下为
                    count += 1
                if boxs_copy[x][0]['value'] == 1: # 右
                    count += 1
                if boxs_copy[x-1][0]['value'] == 1: # 右上
                    count += 1
            elif x == x_len-1:
                # 计算到下边界的情况，但不会包括右下角，因为第一个if已经包括了
                if boxs_copy[x-1][y-1]['value'] == 1: # 左上
                    count += 1
                if boxs_copy[x-1][y]['value'] == 1: # 上
                    count += 1
                if boxs_copy[x-1][y+1]['value'] == 1: # 右上
                    count += 1
                if boxs_copy[x][y-1]['value'] == 1: # 左
                    count += 1
                if boxs_copy[x][y+1]['value'] == 1: # 右
                    count += 1
                if boxs_copy[0][y-1]['value'] == 1: # 相当于左下，后同（延拓图）
                    count += 1
                if boxs_copy[0][y]['value'] == 1: # 下
                    count += 1
                if boxs_copy[0][y+1]['value'] == 1: # 右下
                    count += 1
            elif y == y_len-1: # 右边界的情况，但是不会包括右下角
                if boxs_copy[x-1][y-1]['value'] == 1:  # 左上
                    count += 1
                if boxs_copy[x][y-1]['value'] == 1: # 左
                    count += 1
                if boxs_copy[x+1][y-1]['value'] == 1: # 左下
                    count += 1
                if boxs_copy[x-1][y]['value'] == 1: # 上
                    count += 1
                if boxs_copy[x+1][y]['value'] == 1: # 下
                    count += 1
                if boxs_copy[x-1][0]['value'] == 1: # 相当于右上
                    count += 1
                if boxs_copy[x][0]['value'] == 1: # 右
                    count += 1
                if boxs_copy[x+1][0]['value'] == 1: # 右下
                    count += 1
            # 注意：这里不考虑左边和上边，因为0-1=-1，索引-1就等于倒数第一个，相当于自动转换了
            else: # 其他情况，即非右或下边界的网格
                if boxs_copy[x-1][y-1]['value'] == 1: # 左上
                    count += 1
                if boxs_copy[x-1][y]['value'] == 1: # 上
                    count += 1
                if boxs_copy[x-1][y+1]['value'] == 1: # 右上
                    count += 1
                if boxs_copy[x][y-1]['value'] == 1: # 左
                    count += 1
                if boxs_copy[x][y+1]['value'] == 1: # 右
                    count += 1
                if boxs_copy[x+1][y-1]['value'] == 1: # 左下
                    count += 1
                if boxs_copy[x+1][y]['value'] == 1: # 下
                    count += 1
                if boxs_copy[x+1][y+1]['value'] == 1: # 右下
                    count += 1

            if count == 3:
                boxs[x][y]['value'] = 1
            elif count == 2:
                pass
            else:
                boxs[x][y]['value'] = 0

if __name__ == '__main__':
    boxs = init_boxs(20, 400, 400, 10)
    change_boxs(boxs)

