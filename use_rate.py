import math
from constellation import *
from population import create_each_user
from base import *
from link import *
"""
单颗卫星利用率等参数统计
"""
def one_node_rate(h, t):
    '''
    单颗卫星的移动过程中利用率变化
    '''
    create_onenode(h, t)
    users = create_each_user(10000)
    for user in users:
        user.append(1)
    sate_locat={}
    name = 'data/one_node%d'%h+ '/60.txt'
    with open(name, 'r') as f:
        data = f.readlines()
        count = 0
        for line in data:
            count += 1
            line = line.strip().split(',')
            sate = [[0, float(line[1]), float(line[2])]]
            sate_locat[line[0]] = minlen(users, sate, h)
    # print(sate_locat)
    c = 0
    for k,v in sate_locat.items():
        if v[0] > 0:
            c += 1
    print(c/t)
    f = open('data/tmp.txt%d'%h,'w') 
    s = str(sate_locat)
    f.write(s)
    f.close()

# one_node_rate(550, 1200)
def use_rate_of_diff_h():
    '''
    不同高度的卫星，在运行7天过程中的利用率的比较。
    '''
    # import numpy as np
    h_all = [200, 300,  400,  500,  600,  700,  800,  900, 1000,
       1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    time = 1440 * 7
    used_total = {}
    for h in h_all:
        print(h)
        sate_locat = one_node_rate(h, time)
        used = 0
        for k, v in sate_locat.items():
            if v[0] > 0:
                used += 1
        used_total[h] = used
        print(used)
    print(used_total)

def CountUsers(shell, h, elevation):
    '''
    计算一个卫星在一个回归周期内的星下点经过节点的差别
    如：经过的总的海陆比、人口变化等
    '''
    import csv
    beta = elevation / 180 * math.pi
    R = 6371
    alpha = math.asin(R / (R+h) * math.sin(beta + math.pi/2))
    theta = math.asin((R+h) / R * math.sin(alpha)) - alpha
    r = theta * R#覆盖半径
    ran  = r / 111
    d = (R + h) * math.cos(alpha) -math.sqrt((R + h)**2 * math.cos(alpha)**2 - 2 * R * h -h**2) 
    population=[[0]*360 for i in range(180)]
    with open('data/population.csv','r') as f:
        reader = csv.reader(f)
        count=0
        for row in reader:
            for index in range(len(row)):
                if row[index]=='-9999':
                    population[count][index]=0
                else:
                    population[count][index]=float(row[index])
            count=count+1
    
    cycle=60*24
    popu_stat = []
    # popu_stat一维是不同星下点轨迹,二维是时间
    for i in range(int(360/shell) - 1):
        tmp = [0] * 480
        popu_stat.append(tmp)
        

    for cur_min in range(0,cycle,3):
        floder = 'shell%d/'%shell
        file = floder + '%d.txt' % cur_min # 输出的文件
        with open(file, 'r') as f:
            # 读每一行，查看当前line[0]号卫星目前所在的经纬度，转化为xyz坐标，找到所有能看见当前这个卫星的
            # 用户节点，累加。
            for line in f.readlines():
                line = line.strip('\n').split(',')
                lati=float(line[2])
                longi=float(line[3])
                xyz = lonlat2xyz(lati, longi, h + 6371)
                
                popu_zeros = 0
                popu_one = 0
                # 统计经过美国本土的差异
                for i in range(115,139):
                    for j in range(110,160):
                        if lati - ran - 2 < i - 90 and lati + ran +2 > i - 90 and longi - ran*2 - 4 < j - 180 and longi + ran*2 +4 > j - 180:
                                xyz2 = lonlat2xyz (i - 90, j - 180, R)
                                dis = math.sqrt((xyz[0] - xyz2[0])**2 + (xyz[1] - xyz2[1])**2 +(xyz[2] - xyz2[2])**2)
                                if dis < d:
                                	# 统计海陆比
                                	# if population[i][j] == 0: popu_stat[int(line[0])][0] += 1
                                	# else: popu_stat[int(line[0])][1] += 1
                                	# 统计总的人口
                                	popu_stat[int(line[0])][int(cur_min/3)] += population[i][j]
    

    # print(popu_stat)
    import matplotlib.pyplot as plt
    x = []
    y = []
    for i in range(len(popu_stat)):
        for j in range(len(popu_stat[i])):
            if popu_stat[i][j]>0:
                x.append(i)
                y.append(popu_stat[i][j])
    plt.scatter(x, y,marker=".")
    plt.xlabel('angle of deflection')
    plt.ylabel('users')
    plt.show()
    
    
    # s1 = ''
    # s2 = ''
    # for i in range(len(popu_stat)):
    # 	s1 += str(popu_stat[i][0]) + ','
    # for i in range(len(popu_stat)):
    # 	s2 += str(popu_stat[i][1])+ ','
    # print(s1)
    # print(s2)
    # print(k for i in )
# CountUsers(15, 570, 40)

def CountUsers2(shell, h, elevation):
    '''
    计算一个卫星在一个回归周期内的负载变化情况
    '''
    import csv
    beta = elevation / 180 * math.pi
    R = 6371
    alpha = math.asin(R / (R+h) * math.sin(beta + math.pi/2))
    theta = math.asin((R+h) / R * math.sin(alpha)) - alpha
    r = theta * R#覆盖半径
    ran  = r / 111
    d = (R + h) * math.cos(alpha) -math.sqrt((R + h)**2 * math.cos(alpha)**2 - 2 * R * h -h**2) 
    population=[[0]*360 for i in range(180)]
    with open('population.csv','r') as f:
        reader = csv.reader(f)
        count=0
        for row in reader:
            for index in range(len(row)):
                if row[index]=='-9999':
                    population[count][index]=0
                else:
                    population[count][index]=float(row[index])
            count=count+1
    
    cycle=60*24
    popu_stat = [0] * math.ceil(cycle / 3 + 1)
    # print(popu_stat)
    for cur_min in range(0,cycle,3):
        floder = 'shell%d/'%shell
        file = floder + '%d.txt' % cur_min # 输出的文件
        # 统计第一颗卫星的覆盖用户变化
        with open(file, 'r') as f:
            line = f.readline().strip('\n').split(',')
            lati=float(line[2])
            longi=float(line[3])
            xyz = lonlat2xyz(lati, longi, h + 6371)
            for i in range(115,139):
                for j in range(110,160):
                    if lati - ran - 2 < i - 90 and lati + ran +2 > i - 90 and longi - ran*2 - 4 < j - 180 and longi + ran*2 +4 > j - 180:
                            xyz2 = lonlat2xyz (i - 90, j - 180, R)
                            dis = math.sqrt((xyz[0] - xyz2[0])**2 + (xyz[1] - xyz2[1])**2 +(xyz[2] - xyz2[2])**2)
                            if dis < d:
                                popu_stat[int(cur_min / 3)] += population[i][j]
    # print(popu_stat)
# CountUsers2(15, 570, 40)

"""
普通walker星座利用率
"""

def minlen_add_sate():
    '''
    规定卫星的参数和用户数，使用最短距离分配获取卫星的连接数
    不断增加卫星数量尝试满足所有用户的接入
    '''
    mx = 100
    orbit = 30
    num = 25
    for i in range(300, 40000, 300):
        user_list = create_each_user(i)
        create_conste([orbit], [num], [550], [60], 'data/fengwo', 100)
        flag = True
        _user = 0
        for t in range(0, 100, 3):
            sate_list = get_sat_loca('data/fengwo/%d.txt'%t)
            sate_num = minlin_each(user_list, sate_list, 550, mx)
            if len(sate_num) == 0:
                flag = False
                break
            _user += sum(sate_num)
        if flag:
            print(i, orbit, num, _user/(len(sate_num)*mx*34), sum(sate_num))
        else:
            if orbit > num : num += 1
            else: orbit += 1
            while True:
                create_conste([orbit], [num], [550], [60], 'data/fengwo', 100)
                flag = True
                for t in range(0, 100, 3):
                    sate_list = get_sat_loca('data/fengwo/%d.txt'%t)
                    sate_num = minlin_each(user_list, sate_list, 550, mx)
                    if len(sate_num) == 0:
                        flag = False
                        break
                if flag:break
                else: 
                    if orbit > num : num += 1
                    else: orbit += 1
# minlen_add_sate()

def minlen_add_user():
    '''
    规定卫星的参数，使用最短距离分配获取卫星的连接数
    不断增加用户数直到不能满足需求，获得最大的利用率
    '''
    mx = 100
    # telesat
    # orbit = [27, 40]
    # num = [13, 33]
    # h = [1015, 1325]
    # incli = [98.98, 50.88]
    # ele = 10
    
    # OneWeb
    # orbit = [36, 32, 32]
    # num = [49, 72, 72]
    # h = [1200, 1200, 1200]
    # incli = [87.9, 55, 40]
    # ele = 25

    # Starlink
    orbit = [72, 72, 6, 4, 36]
    num = [22, 22, 58, 43, 20]
    h = [540, 550, 560, 560, 570]
    incli = [53.2, 53, 97.6, 97.6, 70]
    ele = 25

    # Kuiper
    # orbit = [28, 36, 34]
    # num = [28, 36, 34]
    # h = [590, 610, 630]
    # incli = [33, 42, 51.9]
    # ele = 35


    create_conste(orbit, num, h, incli, 'data/fengwo', 4)
    total_capacity = 0
    for i in range(len(orbit)):
        total_capacity += orbit[i] * num[i]
    total_capacity *= mx
    most_user = 0
    for i in range(int(total_capacity/100*3), 40000, 300):
        user_list = create_each_user(i)
        sate_list = get_sat_loca('data/fengwo/0.txt')
        sate_num, user_link = minlin_each(user_list, sate_list, 550, mx, ele)
        if len(sate_num) == 0:
            print(total_capacity,most_user,'done')
            break
        else:
            most_user = sum(sate_num)
            print(most_user)
# minlen_add_user()


def minlen_add_capa():
    '''
    不断增加卫星容量直到满足所有用户接入
    '''
    mx = 100
    orbit = 15
    num = 25
    for i in range(1100,1200, 400):
        user_list = create_each_user(i)
        create_conste([orbit], [num], [550], [60], 'data/fengwo', 3)
        sate_list = get_sat_loca('data/fengwo/0.txt')
        sate_num, u = minlin_each(user_list, sate_list, 550, mx)
        c=0
        for s in sate_num:
            if s > 0:
                c+= 1
        print(c)
        if len(sate_num) > 0:
            print(i, mx, sum(sate_num)/(len(sate_num)*mx), sum(sate_num))
        else:
            while True:
                mx += 10
                sate_num, u = minlin_each(user_list, sate_list, 550, mx)
                if len(sate_num) > 0: break
# minlen_add_capa()

def minlen_change_incli():
    '''
    不同倾角下卫星的利用率和block比例
    '''
    mx = 100
    orbit = 30
    num = 25
    user = 8000
    for incli in range(40,50):
        user_list = create_each_user(user)
        create_conste([orbit], [num], [550], [incli], 'data/fengwo', 3)
        sate_list = get_sat_loca('data/fengwo/0.txt')
        sate_num, block_user = minlin_block_user(user_list, sate_list, 550, mx)
        print(incli, sum(sate_num)/(len(sate_num)*mx), (user-sum(sate_num))/user,len(block_user)/user)
# minlen_change_incli()

def minlen_sta():
    '''
    计算一段时间内所有卫星的利用率并存入文件中
    '''
    mx = 100
    orbit = 15
    num = 25
    i = 4000
    user_list = create_each_user(i)
    create_conste([orbit], [num], [550], [60], 'data/fengwo', 100)
    for t in range(100):
        sate_list = get_sat_loca('data/fengwo/%d.txt'%t)
        sate_num = minlin_each(user_list, sate_list, 550, mx)
        print(t, sum(sate_num))
        if len(sate_num) > 0:
            re = sate_list
            for i in range(len(sate_list)):
                re[i].append(sate_num[i])
            file = 'data/sate_num/%d.txt' % t # 输出的文件
            with open(file, 'w') as f:
                for ree in re:
                    f.writelines('%d,%f,%f,%d\n' % (ree[0], ree[1], ree[2], ree[3]))
# minlen_sta()

def draw_rate():
    '''
    读取一段时间的卫星利用率数据并统计到经纬度粒度，进行可视化展示
    '''
    from matplotlib import pyplot as plt
    import seaborn as sns
    re = [[list() for i in range(360)] for j in range(180)]
    for t in range(100):
        file = 'data/sate_num/%d.txt' % t
        with open(file, 'r') as f:
            data = f.readlines()
            for row in data:
                line = row.split(',')
                lat = float(line[1])
                lon = float(line[2])
                user = int(line[3])
                re[(math.floor(lat))+90][math.floor(lon) + 180].append(user)
    for i in range(len(re)):
        for j in range(len(re[i])):
            if len(re[i][j]) == 0:
                re[i][j] = 0
            else:
                re[i][j] = sum(re[i][j])/len(re[i][j])/2

    from mpl_toolkits.basemap import Basemap
    import numpy as np

    x = np.linspace(-180, 180, 360)
    y = np.linspace(-90, 90, 180)
    X, Y = np.meshgrid(x, y)
    Z = re
    map = Basemap()
    map.drawcoastlines()
    # map.fillcontinents(color='gray', zorder=1)
    map.pcolormesh(x, y, Z, cmap='hot_r', zorder=0)
    map.colorbar()
    
    plt.show()            
# draw_rate()

def user_rate():
    '''
    直方图展示卫星利用率分布图
    '''
    from matplotlib import pyplot as plt
    re = []
    for t in range(100):
        file = 'data/sate_num/%d.txt' % t
        with open(file, 'r') as f:
            data = f.readlines()
            for row in data:
                line = row.split(',')
                if int(line[3]) != 0:
                    re.append(int(line[3]))
    re.sort()
    plt.hist(re,bins=20)
    plt.show()
# user_rate()

def draw_block_user():
    '''
    画出block的用户的点分布
    '''
    from matplotlib import pyplot as plt
    h = 569
    users = create_each_user(8000)
    create_conste([30], [15], [h], [60], 'data/fengwo', 5)
    sate_list = get_sat_loca('data/fengwo/0.txt')
    sate_num, block_users = minlin_block_user(users, sate_list, h, 100)
    print(len(block_users))
    x = []
    y = []
    for i in block_users:
        x.append(i[1])
        y.append(i[0])
    from mpl_toolkits.basemap import Basemap
    map = Basemap()
    map.drawcoastlines()
    plt.scatter(x, y, marker='.', color='r')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()
# for i in range(10):
# draw_block_user()

def link_use_rate(capa, t=''):
    '''
    卫星ISL和GSL利用率的复合
    '''
    link = []
    sate = []
    used_dic = set()
    cc = []

    with open('data/fengwo/cap_link%s.txt'%str(t), 'r') as f:
        data = f.readlines()
        for row in data:
            b, e, c = map(int, row.split(','))
            link.append([b, e, c])
            cc.append(c)
    
    with open('data/fengwo/cap_sate%s.txt'%str(t), 'r') as f:
        data = f.readlines()
        for row in data:
            u = int(row)
            sate.append(u)
    
    for s in sate:
        if s > 0: used_dic.add(s)
    for d in link:
        if d[2] < capa:
            used_dic.add(d[1])
            used_dic.add(d[0])
    

    cap = 0
    lin = 0
    for s in sate:
        cap += s
    for d in link:
        lin += capa-d[2]
    
    # print('单星：',len(used_dic)/len(sate))
    # print("容量：", cap/len(sate)/400, lin/len(sate)/2/cap)
    print(len(used_dic)/len(sate), cap/len(sate)/400, lin/len(sate)/4/capa)
    # logging.info('利用率：%f,%f,%f'%(len(used_dic)/len(sate), cap/len(sate)/400, lin/len(sate)/2/capa))
    return '%f,%f,%f'%(len(used_dic)/len(sate), cap/len(sate)/400, lin/len(sate)/4/capa)
    # from matplotlib import pyplot as plt
    # plt.hist(cc,bins=20)
    # plt.show()

def per_link_use_rate(typ, capa, sateID=0):
    '''
    针对不同时间不同卫星的利用率进行分别分析
    1. 给出某一颗卫星利用率随时间的变化 
    2. 给出不同卫星的差异化利用率
    3. 给出卫星在不同区域时的利用率变化
    '''
    n = 3309
    link_list = [] # 链路剩余容量
    sate_list = [] # 单颗卫星接入量
    used_dic_list = [] # 使用过的卫星

    for i in range(n):
        t = i * 3
        link = [0] * 1584
        sate = []
        used_dic = set()
        with open('data/fengwo/cap_link%d.txt'%t, 'r') as f:
            data = f.readlines()
            for row in data:
                b, e, c = map(int, row.split(','))
                link[b-1] += (capa - c)/2
                link[e-1] += (capa - c)/2
                # link.append([b, e, c])
                if c < capa:
                    used_dic.add(b)
                    used_dic.add(e)
        
        with open('data/fengwo/cap_sate%d.txt'%t, 'r') as f:
            data = f.readlines()
            for row in data:
                u = int(row)
                sate.append(u)
        
        for s in sate:
            if s > 0: used_dic.add(s)
            
        link_list.append(link)
        sate_list.append(sate)
        used_dic_list.append(used_dic)
    
    # print(sate_list)
    def check_sate_i(sateID):
        '''
        查看一颗卫星一天内的利用率变化
        '''
        link_t = []
        sate_t = []
        used_t = []
        # print(sate_list[3])
        for t in range(len(link_list)):
            # print(sate_list[t][0])
            link_t.append(link_list[t][sateID])
            sate_t.append(sate_list[t][sateID])
            if i in used_dic_list[t]:
                used_t.append(1)
            else:
                used_t.append(0)
        return link_t, sate_t, used_t
    
    def check_diff():
        '''
        查看不同卫星一天内的利用率差异（统计值）
        '''
        link_add = [0] * 1584
        sate_add = [0] * 1584
        used_add = [0] * 1584
        for t in range(len(link_list)):
            for i in range(len(link_list[t])):
                link_add[i]+=link_list[t][i]
                sate_add[i]+=sate_list[t][i]
                if i in used_dic_list[t]:
                    used_add[i]+=1
        return link_add, sate_add, used_add

    def diff_region():
        '''
        查看不同区域的卫星利用率的差异性
        '''
        re_link = [[list() for i in range(360)] for j in range(180)]
        re_sate = [[list() for i in range(360)] for j in range(180)]
        re_used = [[list() for i in range(360)] for j in range(180)]
        for t in range(len(link_list)):
            sate = get_sat_loca('data/fengwo/%d.txt'%(t*3))
            for i in range(len(link_list[t])):
                lat = sate[i][1]
                lon = sate[i][2]
                re_link[(math.floor(lat))+90][math.floor(lon) + 180].append(link_list[t][i])
                re_sate[(math.floor(lat))+90][math.floor(lon) + 180].append(sate_list[t][i])
                if i in used_dic_list[t]:
                    re_used[(math.floor(lat))+90][math.floor(lon) + 180].append(1)
                else:
                    re_used[(math.floor(lat))+90][math.floor(lon) + 180].append(0)
        for i in range(len(re_link)):
            for j in range(len(re_link[i])):
                if len(re_link[i][j]) == 0: re_link[i][j] = 0
                else: re_link[i][j] = sum(re_link[i][j])/len(re_link[i][j])

                if len(re_sate[i][j]) == 0: re_sate[i][j] = 0
                else: re_sate[i][j] = sum(re_sate[i][j])/len(re_sate[i][j])
                
                if len(re_used[i][j]) == 0: re_used[i][j] = 0
                else: re_used[i][j] = sum(re_used[i][j])/len(re_used[i][j])
        return re_link, re_sate, re_used

    if typ == 1: return check_sate_i(i)
    elif typ == 2: return check_diff()
    else: return diff_region()
    



# link,sate,used=per_link_use_rate(2, 50, sateID=0)
# sate = [i/480 for i in sate]
# link = [i/480 for i in link]

# # link = [i/480/200*100 for i in link]
# capi = [link[i]+sate[i] for i in range(len(sate))]



# sate = [i/4 for i in sate]
# link = [i/2 for i in link]
# capi = [i/6 for i in capi]
# plt.plot(capi)
# capi.sort()
# plt.plot(capi)

# plt.xlabel('satellite')
# plt.ylabel('capacity utilization/%')
# plt.show()
# from mpl_toolkits.basemap import Basemap
# import numpy as np
# x = np.linspace(-180, 180, 360)
# y = np.linspace(-90, 90, 180)
# X, Y = np.meshgrid(x, y)
# Z = link
# map = Basemap()
# map.drawcoastlines()
# # map.fillcontinents(color='gray', zorder=1)
# # print(Z)
# map.pcolormesh(x, y, Z, cmap='hot_r', shading='auto')
# map.colorbar()
# plt.show()            


# link_use_rate(10)
"""
星下点轨迹固定的卫星利用率
"""


def cal_least_sate():
    '''
    在基础覆盖的容量外，计算需要多少共星下点轨迹卫星才能满足block节点的需求
    '''
    users = create_each_user(8000)
    h = 550
    create_conste([30], [15], [h], [60], 'data/fengwo', 1441)
    for rainit in range(2, 24, 2):
        n = 21
        sate_loca = create_same_point(15, rainit, n)
        for t in range(0, len(sate_loca[0]), 3):
            
            sate_list = get_sat_loca('data/fengwo/%d.txt'%t)
            sate_num, block_users = minlin_block_user(users, sate_list, h)
            # print(len(block_users))
            flag = True
            while flag:
                print(t,n)
                sate_loca = create_same_point(15, rainit, n)
                sate_list = []
                for i in range(len(sate_loca)):
                    sate_list.append([i, sate_loca[i][0][0], sate_loca[i][0][1]])
                a, b = minlin_block_user(block_users, sate_list, 570)
                # print(n,len(b))
                if len(b) > 0:
                    flag = True
                    n+=1
                else:
                    flag = False
        print(rainit, n)
# cal_least_sate()


