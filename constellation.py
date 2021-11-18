from matplotlib import colors, pyplot as plt
from base import get_same_point_h
import os, datetime, math
from population import *
# from link import *

def create_shell(orbit, num, h, incli, fold, cycle, snap = 3):
    '''
    生成卫星cycle时间段里面snap为时间片的单层卫星的轨迹，每个时间片存储一个文件
    '''
    import ephem
    R=6371
    INCLINATION = incli
    ECCENTRICITY = 0.001 # 离心率
    ARG_OF_PERIGEE = 0.0 # 近地点
    ORBIT_CYCLE = math.sqrt(4 * math.pi**2 * (R + h)**3 / 397865.5)  # 卫星运行周期s
    EARTH_CYCLE = 86400
    MEAN_MOTION = EARTH_CYCLE / ORBIT_CYCLE# 每天转多少圈
    floder = '%s/'%fold
    if not os.path.exists(floder):
            os.mkdir(floder)
    else:
        os.system("rm -rf%s*"%floder)
    cnt = 0 # 卫星编号
    now = datetime.datetime(2021, 6, 7, 0, 0, 0)
    for cur_min in range(0, cycle, snap):
        cur_time = (now + datetime.timedelta(minutes=cur_min)).strftime("%Y-%m-%d %H:%M:%S")
        cnt = 0
        file = floder + '/%d.txt' % cur_min # 输出的文件
        with open(file, 'w') as f:
            for cur_orbit in range(1, orbit + 1):
                raan =  (cur_orbit - 1) * 360 / orbit
                for cur_num in range(1, num + 1):
                    if cur_orbit % 2 == 0:
                        meanAnomaly = 360 / num / 2
                    else: meanAnomaly = 0
                    meanAnomaly += (cur_num - 1) * (360 / num)
                    sat = ephem.EarthSatellite()
                    sat._epoch = now
                    sat._inc = INCLINATION
                    sat._raan = raan
                    sat._M = meanAnomaly
                    sat._n = MEAN_MOTION
                    sat._e = ECCENTRICITY  # 偏心率
                    sat._ap = ARG_OF_PERIGEE  # 圆
                    sat.compute(cur_time)
                    f.writelines('%d,%d,%s,%s,%s\n' % (
                    cnt, cur_orbit, math.degrees(sat.sublat), math.degrees(sat.sublong), sat.elevation / 1000))
                    cnt += 1
            # print('%d分钟完成' % (cur_min + snap))
    # drawMap(fold)
# create(23, 25, 550, 53, "fengwo")

def create_shell_init(raan_init, maan_init, h, incli, cycle, fold='data/fengwo', snap=3):
    '''
    根据星座中每个卫星的初始raan和maan获得星座位置
    '''
    import ephem
    R=6371
    INCLINATION = incli
    ECCENTRICITY = 0.001 # 离心率
    ARG_OF_PERIGEE = 0.0 # 近地点
    ORBIT_CYCLE = math.sqrt(4 * math.pi**2 * (R + h)**3 / 397865.5)  # 卫星运行周期s
    EARTH_CYCLE = 86400
    MEAN_MOTION = EARTH_CYCLE / ORBIT_CYCLE# 每天转多少圈
    floder = '%s/'%fold
    if not os.path.exists(floder):
        os.mkdir(floder)

    now = datetime.datetime(2021, 6, 7, 0, 0, 0)
    for cur_min in range(0, cycle, snap):
        cur_time = (now + datetime.timedelta(minutes=cur_min)).strftime("%Y-%m-%d %H:%M:%S")
        file = floder + '/%d.txt' % cur_min # 输出的文件
        
        with open(file, 'w') as f:
            for i in range(len(raan_init)):
                raan = raan_init[i]
                meanAnomaly = maan_init[i]
                sat = ephem.EarthSatellite()
                sat._epoch = now
                sat._inc = INCLINATION
                sat._raan = raan
                sat._M = meanAnomaly
                sat._n = MEAN_MOTION
                sat._e = ECCENTRICITY  # 偏心率
                sat._ap = ARG_OF_PERIGEE  # 圆
                sat.compute(cur_time)
                f.writelines('%d,%d,%s,%s,%s\n' % (
                i, i, math.degrees(sat.sublat), math.degrees(sat.sublong), sat.elevation / 1000))

def create_conste(orbit_list, num_list, h_list, incli_list, fold, t):
    '''
    生成卫星t时间段里面snap为时间片的卫星星座轨迹
    '''
    import ephem
    R = 6371
    ECCENTRICITY = 0.001 # 离心率
    ARG_OF_PERIGEE = 0.0 # 近地点
    EARTH_CYCLE = 86400
    cycle = t + 1
    snap = 1
    floder = '%s'%fold
    if not os.path.exists(floder):
        os.mkdir(floder)
    # else:
    #     os.system("del %s\\* /Q"%floder)
    
    for cur_min in range(0, cycle, snap):
        now = datetime.datetime(2021, 6, 7, 0, 0, 0)
        cur_time = (now + datetime.timedelta(minutes=cur_min)).strftime("%Y-%m-%d %H:%M:%S")
        file = floder + '/%d.txt' % cur_min # 输出的文件
        cnt = 0 # 卫星编号
        with open(file, 'w') as f:
            for shell in range(len(orbit_list)):
                INCLINATION = incli_list[shell]
                ORBIT_CYCLE = math.sqrt(4 * math.pi**2 * (R + h_list[shell])**3 / 397865.5)  # 卫星运行周期s
                MEAN_MOTION = EARTH_CYCLE / ORBIT_CYCLE# 每天转多少圈
                
                for cur_orbit in range(1, orbit_list[shell] + 1):
                    raan =  (cur_orbit - 1) * 360 / orbit_list[shell]
                    for cur_num in range(1, num_list[shell] + 1):
                        if cur_orbit % 2 == 0:
                            meanAnomaly = 360 / num_list[shell] / 2
                        else: meanAnomaly = 0
                        meanAnomaly += (cur_num - 1) * (360 / num_list[shell])
                        sat = ephem.EarthSatellite()
                        sat._epoch = now
                        sat._inc = INCLINATION
                        sat._raan = raan
                        sat._M = meanAnomaly
                        sat._n = MEAN_MOTION
                        sat._e = ECCENTRICITY  # 偏心率
                        sat._ap = ARG_OF_PERIGEE  # 圆
                        sat.compute(cur_time)
                        f.writelines('%d,%d,%s,%s,%s\n' % (
                        cnt, cur_orbit, math.degrees(sat.sublat), math.degrees(sat.sublong), sat.elevation / 1000))
                        cnt += 1

def create_onenode(h, cycle, incli = 60, snap = 1, raan = 0, maan = 0, file = 60):
    '''
    生成一个高度为h的卫星cycle时间的轨迹，时间间隔默认为1min，卫星的倾角默认为60°
    结果存储在one_node/里
    '''
    import ephem
    R = 6371
    INCLINATION = incli
    ECCENTRICITY = 0.001 # 离心率
    ARG_OF_PERIGEE = 0.0 # 近地点
    ORBIT_CYCLE = math.sqrt(4 * math.pi**2 * (R + h)**3 / 397865.5)  # 卫星运行周期s
    # EARTH_CYCLE = 86164
    EARTH_CYCLE = 86400
    MEAN_MOTION = EARTH_CYCLE / ORBIT_CYCLE # 每天转多少圈
    floder = 'data/one_node%d'%h
    if not os.path.exists(floder):
        os.mkdir(floder)
    # cnt = 0 # 卫星编号
    now = datetime.datetime(2021, 6, 7, 0, 0, 0)
    file = floder + '/%d.txt'%file # 输出的文件
    with open(file, 'w') as f:
        for cur_min in range(0, cycle, snap):
            cur_time = (now + datetime.timedelta(minutes=cur_min)).strftime("%Y-%m-%d %H:%M:%S")
            sat = ephem.EarthSatellite()
            sat._epoch = now
            sat._inc = INCLINATION
            sat._raan = raan
            sat._M = maan
            sat._n = MEAN_MOTION
            sat._e = ECCENTRICITY  # 偏心率
            sat._ap = ARG_OF_PERIGEE  # 圆
            sat.compute(cur_time)
            f.writelines('%d,%d,%s,%s,%s\n' % (
            0, 1, math.degrees(sat.sublat), math.degrees(sat.sublong), sat.elevation / 1000))

def draw_onenode(h,incli,raan=-20,color='b'):
    '''
    画一个回归周期内卫星的轨迹以及对不同区域的覆盖浓度
    '''
    create_onenode(h, 1426, incli = incli, snap = 1, raan =raan, maan = 0, file = 60)
    sate_list = get_sat_loca('data/one_node%d/60.txt'%h)
    lat = [sate_list[0][1]]
    lon = [sate_list[0][2]]
    for sate in sate_list:
        if abs(sate[2]-lon[-1])>100:
            plt.plot(lon,lat,color=color)
            lat=[]
            lon=[]
        lat.append(sate[1])
        lon.append(sate[2])
    plt.plot(lon,lat,color=color)
    re = [[0 for i in range(360)] for i in range(180)]
    r,d = beta2r(h,25)
    for lon in range(-180, 180):
        print(lon)
        for lat in range(-90, 90):
            for sate in sate_list:
                if abs(lat - sate[1]) < r / 111 and abs(lon - sate[2]) < r/111*2:
                    d2 = latlon2d(lat,sate[1],lon,sate[2],6371, 6371+h)
                    if d2 < d:
                        re[lat+90][lon+180] += 1
    m = max([max(i) for i in re])
    re = [[i/m for i in y] for y in re]
    import numpy as np 
    from mpl_toolkits.basemap import Basemap          
    x = np.linspace(-180, 180, 360)
    y = np.linspace(-90, 90, 180)
    map = Basemap()
    map.drawcoastlines()
    map.pcolormesh(x, y, re, cmap='hot_r', shading='auto')
    map.colorbar()
# h=get_same_point_h(13)
# h=h-20
# h=523
# draw_onenode(h,60)
# draw_onenode(h,60,-10,'r')

# plt.show()

def create_same_point(cycle, rainit = 0, n = 20):
    '''
    生成回归周期是cycle圈的共星下点轨迹的卫星,每圈有n颗卫星
    '''
    h = get_same_point_h(cycle)
    T = 24*60 / cycle
    for i in range(n * cycle):
        # print(int(i*360/n/cycle))
        create_onenode(h, int(T/n), maan=-int(i*360/n), raan=360/n*T*i/1440+rainit, file=int(i*360/n/cycle))
    sate_loca = [list() for i in range(n*cycle)]
    for i in range(n*cycle):
        with open('data/one_node%d/%d.txt'%(h, int(i*360/n/cycle)), 'r') as f:
            data = f.readlines()
            for row in data:
                row = row.split(',')
                sate_loca[i].append([float(row[2]), float(row[3])])
    return sate_loca
# create_same_point(15)

def create_same_point_t(cycle, rainit = 0, n = 3, t = 1440):
    '''
    生成回归周期是cycle圈的共星下点轨迹的卫星,每圈有n颗卫星。返回所有卫星t时间内的轨迹
    '''
    h = get_same_point_h(cycle)
    T = 24*60 / cycle
    for i in range(n * cycle):
        # print(int(i*360/n/cycle))
        create_onenode(h, t, maan=-int(i*360/n), raan=360/n*T*i/1440+rainit, file=int(i*360/n/cycle))
        print(int(i*360/n/cycle))
    sate_loca = [list() for i in range(n*cycle)]
    for i in range(n*cycle):
        with open('data/one_node%d/%d.txt'%(h, int(i*360/n/cycle)), 'r') as f:
            data = f.readlines()
            for row in data:
                row = row.split(',')
                sate_loca[i].append([float(row[2]), float(row[3])])
    return sate_loca

def draw_same_point(cycle, rainit = 0, n = 20):
    '''
    读取共星下点轨迹卫星的星下点并scatter出来
    '''
    # users = create_each_user(8000)
    sate_loca = create_same_point(cycle, rainit, n)
    # sate_list = get_sat_loca('data/fengwo/%d.txt'%0)
    # sate_num, block_users = minlin_block_user(users, sate_list, 550)
    # ulat = []
    # ulon = []
    # # print(block_users)
    # for user in block_users:
    #     ulat.append(user[0])
    #     ulon.append(user[1])

    lat = []
    lon = []
    for sat in sate_loca:
        for loc in sat:
            # print(loc)
            lat.append(loc[0])
            lon.append(loc[1])
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap
    map = Basemap()
    map.drawcoastlines()
    plt.scatter(lon, lat, marker='.')
    # plt.scatter(ulon, ulat, color='r', marker='.')
    plt.show()
# draw_same_point(2, 0, 2)

def sate_circle(d, dr, i):
    '''
    获取卫星的覆盖区域
    '''
    u = [i / 180 * math.pi for i in range(-90,91)]
    a1 = math.sin(i) * math.cos(d)
    b1 = math.cos(i)
    a2 = math.cos(i) * math.sin(dr)
    b2 = math.tan(d) * math.sin(i)

    lat = []
    lon = []
    for ui in u:
        lat.append(math.asin(a1*math.sin(ui)-a2))
        lon.append(math.atan(b1*math.tan(ui)+b2/math.cos(ui)))

        lat.append(math.asin(a1*math.sin(ui)+a2))
        lon.append(math.atan(b1*math.tan(ui)-b2/math.cos(ui)))
    
    lat = [i*180/math.pi for i in lat]
    lon = [i*180/math.pi-90 for i in lon]
    
    
    u = [i / 180 * math.pi for i in range(91,180)]
    u.extend([i / 180 * math.pi for i in range(-180,-90)])
    for ui in u:
        lat.append(math.asin(a1*math.sin(ui)-a2)*180/math.pi)
        lon.append(math.atan(b1*math.tan(ui)+b2/math.cos(ui))*180/math.pi+90)

        lat.append(math.asin(a1*math.sin(ui)+a2)*180/math.pi)
        lon.append(math.atan(b1*math.tan(ui)-b2/math.cos(ui))*180/math.pi+90)
    return lat, lon
    from matplotlib import pyplot as plt
    plt.plot(lon, lat)
    plt.show()
# sate_circle(8/180*math.pi, 8/180*math.pi, 60/180*math.pi)
def sate_circle_same(i,n):
    '''
    获取倾角为i，回归圈数为n的卫星的覆盖
    '''
    line = 4
    h = get_same_point_h(n)
    d = beta2theta(h,70)
    dr = 0.6*d
    alpha = 1/math.log(n) if 0<1/math.log(n) <1 else 1
    alpha = alpha/2
    lat, lon = sate_circle(d, dr, i)
    lonlat = [(lon[i],lat[i]) for i in range(len(lon))]
    lonlat.sort()
    lat = [i[1] for i in lonlat]
    lon = [i[0] for i in lonlat]
    plt.plot(lon, lat,alpha=alpha,color='b',linewidth=line)
    
    for j in range(1,n):
        latt = []
        lonn = []
        for i in range(len(lat)):
            if lon[i] < 360/n*j-180:
                lonn.append(lon[i]+360-360/n*j)
                latt.append(lat[i])
        plt.plot(lonn,latt,alpha=alpha,color='b',linewidth=line)

        latt = []
        lonn = []
        for i in range(len(lat)):
            if lon[i] > 360/n*j-180:
                lonn.append(lon[i]-360/n*j-0.9)
                latt.append(lat[i])
        plt.plot(lonn,latt,alpha=alpha,color='b',linewidth=line)

# sate_circle_same(60/180*math.pi,15)
# sate_circle_same(60/180*math.pi,4)
# plt.show()

def draw_lat_sum():
    '''
    读取一段时间的卫星数据并统计到纬度粒度，进行可视化展示
    '''
    from matplotlib import pyplot as plt
    re = [0 for i in range(180)]
    create_conste([32], [22], [550], [30], 'data/fengwo', 500)
    for t in range(500):
        sate_list = get_sat_loca('data/fengwo/%d.txt'%t)
        for sate in sate_list:
            re[math.floor(sate[1]) + 90] += 1
    plt.plot([i for i in range(-90,90)],re)
    plt.xlabel('latitude')
    plt.ylabel('number of satellites')
    plt.show() 
# draw_lat_sum()

def draw_lat_cap():
    '''
    读取一段时间的卫星数据并统计每个纬度的可见卫星，进行可视化展示
    '''
    from matplotlib import pyplot as plt
    re = [0 for i in range(180)]
    create_conste([30], [22], [550], [50], 'data/fengwo', 500)
    theta = beta2theta(550, 25)
    theta = theta * 180 / math.pi
    theta_add = 0
    for i in range(int(-1*theta), int(theta)):
        theta_add += math.sqrt(theta**2 - i**2)
    cap = 10
    for t in range(500):
        # print(t)
        sate_list = get_sat_loca('data/fengwo/%d.txt'%t)
        for sate in sate_list:
            mid = math.floor(sate[1]) + 90
            for i in range(int(-1*theta), int(theta)):
                re[mid + i] += cap / theta_add * math.sqrt(theta**2 - i**2)
    
    re = [i / 500 for i in re]
    print(re)
    plt.plot([i for i in range(-90,90)],re)
    plt.xlabel('latitude')
    plt.ylabel('capacity')
    plt.show()
     
# draw_lat_cap()

def draw_lat_lon_cap():
    '''
    未完成
    读取一段时间的卫星数据并统计每个经纬度的可见卫星，进行可视化展示
    '''
    from matplotlib import pyplot as plt
    # re = [[0 for i in range(360)] for j in range(180)]
    # # create_conste([30], [22], [550], [50], 'data/fengwo', 500)
    # theta = beta2theta(550, 25)
    # r, d = beta2r(550, 25)
    # theta = theta * 180 / math.pi
    # theta_add = 0
    # for i in range(int(-1*theta), int(theta)):
    #     theta_add += math.sqrt(theta**2 - i**2)
    # cap = 10
    # for t in range(500):
    #     # print(t)
    #     sate_list = get_sat_loca('data/fengwo/%d.txt'%t)
    #     for sate in sate_list:
    #         mid_lat = math.floor(sate[1]) + 90
    #         mid_lon = math.floor(sate[2]) + 180
    #         for i in range(int(-1*theta), int(theta)):
    #             lat = mid_lat + i
    #             for j in range(int(-1*2*theta), int(2*theta)):
    #                 lon = lon_add([mid_lon], j)
    #                 if latlon2d(lat, mid_lat, lon, mid_lon, 6371, 6371+550) < d:
    #                     re[lat][lon] += 1
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    '''创建3D轴对象'''
    ax = Axes3D(fig)
    re = [[0 for i in range(360)] for j in range(180)]
    def f(x,y):
        return re[x][y]
    # ax.plot_surface(X=[i for i in range(180)], Y=[i for i in range(360)], f(X,Y))
    # plt.xlabel('latitude')
    # plt.ylabel('capacity')
    plt.show()
# draw_lat_lon_cap()


def draw_change_time(lat_a):
    '''
    给出0~24h经过最低纬度的24颗卫星所有经过lat_a纬度的点对应的本地时间
    '''
    from math import pi
    t = [i for i in range(96)]
    tt = []
    x=[]
    y1=[]
    y2=[]
    lat=[]
    lon=[]
    for j in [i for i in range(0,24)]:
        for i in t:
            tt.append(change_local_time_with(j, i))
            l1,l2=lonca(i, 0, 0, 53, 550)
            lat.append(l1*180/pi)
            lon.append(l2)
        re = []
        for i in range(1, len(t)):
            if lat[i] > lat_a and lat[i-1]<lat_a:
                re.append(tt[i])
            if lat[i] < lat_a and lat[i-1]>lat_a:
                re.append(tt[i])
        y1.append(re[0])
        y2.append(re[1])
        x.append(j)
        tt = []
        lat=[]
        lon=[]
    from matplotlib import pyplot as plt
    plt.plot(x,y1, x,y2)
    plt.show()
# draw_change_time(40)

def sta_time_over_lat(lat):
    '''
    对一个星座的所有卫星，统计经过纬度lat的时间分布
    '''
    count_t = [0 for i in range(24)]
    create_conste([72], [66], [550], [53], 'data/fengwo', 100)
    sate_last = get_sat_loca('data/fengwo/0.txt')
    for t in range(1, 96):
        sate_this = get_sat_loca('data/fengwo/%d.txt'%t)
        for i in range(len(sate_this)):
            if sate_last[i][1] < lat and sate_this[i][1] >= lat:
                lon = sate_this[i][2]
                t_local = int(local_time_after_t(lon, t))
                count_t[t_local]+=1
            if sate_last[i][1] >= lat and sate_this[i][1] < lat:
                lon = sate_this[i][2]
                t_local = int(local_time_after_t(lon, t))
                count_t[t_local]+=1
    print(count_t)

# sta_time_over_lat(30)
