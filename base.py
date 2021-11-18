import math

from numpy import left_shift

def lonlat2xyz(lat,lon,r):
    '''
    将经纬度转换为xyz坐标
    '''
    pi = math.pi
    Theta = pi / 2 - lat * pi / 180
    Phi = 2 * pi + lon * pi / 180
    X=r*math.sin(Theta)*math.cos(Phi)
    Y=r*math.sin(Theta)*math.sin(Phi)
    Z=r*math.cos(Theta)
    xyz=[X,Y,Z]
    return xyz
# print(lonlat2xyz(41, -87,6371))
def beta2r(h, beta):
    '''
    将仰角转化为地面覆盖半径和最远通信距离
    '''
    beta = beta / 180 * math.pi
    R = 6371
    theta = math.acos(R*math.cos(beta)/(R+h)) - beta
    r = theta * R #覆盖半径
    d = math.sqrt(R**2 + (R + h)**2 - 2 * R * (R + h) * math.cos(theta))
    # print(theta)
    return r, d

# print(beta2r(550,25))
def beta2theta(h, beta):
    beta = beta / 180 * math.pi
    R = 6371
    theta = math.acos(R*math.cos(beta)/(R+h)) - beta
    return theta
# print(beta2theta(550,25))
def get_sat_loca(filename):
    '''
    获取文件中每个卫星的sate_ID, lat, long
    '''
    sate_list = []
    with open(filename, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip().split(',')
            sate = [int(line[0]), float(line[2]), float(line[3])]
            sate_list.append(sate)
    return sate_list

def lon_add(lon_list, add):
    '''
    进行左右偏移
    '''
    lonn = []
    for l in lon_list:
        if l + add <= 180: lonn.append(l + add)
        else:
            l = (l + add + 180) % 360
            l = l - 180
            lonn.append(l) 
    return lonn
# print(lon_add([1],-30))
# print(lon_add([0], 360/86400*3600))
def lon_relative(lon1, lon2):
    '''
    lon1与lon2之间的经度间隔,即从lon1出发逆时针找到lon2需要经过的经度
    lon1-lon2
    '''
    if lon1 >= 0:
        if lon2 >= 0:
            if lon2 >= lon1: return lon2 - lon1
            else: return 360-lon1 + lon2
        else:
            return 360 - lon1 + lon2
    else:
        if lon2 >= 0:
            return lon2 - lon1
        else:
            if lon2 >= lon1: return lon2 - lon1
            else: return 360 - lon1 + lon2

# print(lon_relative(-160,-170))

# for i in range(-180,180, 10):
#     print(lon_relative(0,i))
def num2lon(num):
    '''
    给定一个数字如果不在经度范围内则调整为正常的经度
    '''
    if -math.pi<num<=math.pi: return num
    while num <= -2*math.pi: num += 2*math.pi
    if num <=-1.5*math.pi: num += math.pi
    else:
        num+=2*math.pi
    while num > math.pi: num -=2*math.pi
    return num

def latlon2d(lat1, lat2, lon1, lon2, h1, h2):
    xyz1 = lonlat2xyz(lat1, lon1, h1)
    xyz2 = lonlat2xyz(lat2, lon2, h2)
    d = math.sqrt((xyz1[0] - xyz2[0])**2 + (xyz1[1] - xyz2[1])**2 + (xyz1[2] - xyz2[2])**2)
    return d

def sample_spherical(npoints, ndim=3):
    '''
    在球表面随机均匀撒npoints个点
    '''
    import numpy as np
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def in_range(x, left, right):
    '''
    给定左右两边的经度值返回x是否在此范围内
    ''' 
    if left < right:
        if x >=left and x<=right:
            return True
        else:
            return False
    else:
        if x>=left and x<=180 or x<=-180 and x<=right:
            return True
        else:
            return False

def get_same_point_h(cycle):
    '''
    获取回归周期为cycle圈的卫星的高度
    '''
    T = ((23*60+56)*60+4)/cycle
    r = (397865.5 * T ** 2 / 4 / math.pi ** 2) ** (1 / 3)
    R = 6371
    h = r - R
    # print(r,h)
    return h
# get_same_point_h(12)

def coverage_rate(beta, h, n):
    '''
    计算给定beta、h、n的卫星星座的覆盖利用率
    '''
    beta = beta / 180 * math.pi
    R = 6371
    theta = math.acos(R*math.cos(beta)/(R+h)) - beta
    S = 2*math.pi*R**2*(1-math.cos(theta))
    U = 4*math.pi*R**2/(S*n)
    return U

def get_dir(s1, s2, num=22):
    '''
    s1在s2的上下左右（1234），其中保证s1和s2一定是连着的
    '''
    s1_orbit = int(s1/num)
    s2_orbit = int(s2/num)
    s1_num = s1%num
    s2_num = s2%num
    if s1_orbit < s2_orbit:
        if s2_orbit - s1_orbit == 1: return 3
        else: return 4
    elif s1_orbit > s2_orbit:
        if s1_orbit - s2_orbit == 1: return 4
        else: return 3
    else:
        if s1_num > s2_num:
            if s1_num - s2_num == 1: return 1
            else: return 2
        else:
            if s2_num- s1_num == 1: return 2
            else: return 1


def next_sate(i, num):
    '''
    获取index=i的卫星的同一圈下一颗卫星编号
    如index = 5， num=6，则返回0
    '''
    if (i+1)%num == 0:
        return i-num+1
    return i + 1

def lonca(t, l0, theta0, incli, h):
    '''
    根据卫星的初始RAAN：l0，倾角incli，轨道高度h, 时间t min计算卫星在tmin时间的经纬度
    todo：还存在问题，详见1100~1300的错误
    '''
    t = t * 60
    incli = incli / 180 * math.pi
    theta0 = theta0 / 180 * math.pi
    we = 7.2722 * 10 ** (-5) #rad/s
    R = 6371
    T = math.sqrt(4 * math.pi ** 2 * (R + h) ** 3 / 397865.5)
    ws = 2 * math.pi / T
    theta = theta0 + ws * t
    theta = theta % (2 * math.pi)
    lon = l0 + math.atan(math.cos(incli) * math.tan(theta)) - we * t
    if math.pi / 2 <= theta <= 1.5*math.pi:
        lon += math.pi
    elif 1.5*math.pi<theta<=2*math.pi:
        lon += math.pi*2
    lon = num2lon(lon)
    lat = math.asin(math.sin(incli) * math.sin(theta))
    return lat, lon

def change_local_time_with(t0, t, raan=0,theta=0,incli=53,h=550):
    '''
    计算随时间变化卫星所在区域的时间变化
    出发区域时间为t0，绕行时间为t，求绕行后所在区域的时间
    '''
    import math
    lat, lon = lonca(t, raan, theta, incli, h)
    lon_change = lon_relative(-120, lon*180/math.pi)
    t = t0 + lon_change / 360* 24 + t / 60
    t = t % 24
    # print(lat*180/math.pi, t)
    return lat*180/math.pi,t
    # return lat,lon

def change_local_time_with2(t0, t, raan=0,theta=0,incli=53,h=550):
    '''
    计算随时间变化卫星所在区域的时间变化
    出发区域时间为t0，绕行时间为t，求绕行后所在区域的时间
    '''
    import math
    lat, lon = lonca(t, raan, theta, incli, h)
    lon_change = lon_relative(-120, lon*180/math.pi)
    t = t0 + lon_change / 360* 24 + t / 60
    t = t % 24
    # print(lat*180/math.pi, t)
    return lat*180/math.pi,lon*180/math.pi,t

def local_time_after_t(lon, t):
    '''
    lon-180的初始时间为0，计算运行t时间后，lon对应经度的本地时间
    '''
    relative_lon = lon + 180
    t_180 = t
    t_relative = relative_lon/360*24
    return (t_180+t_relative)%24
# print(local_time_after_t(120, 1))
# for i in range(0,100):
#     change_local_time_with(0,i,raan=-120,incli=53)
    
def get_lon():
    i = 53/180*math.pi
    x=[]
    y=[]
    for theta in range(-179, 180):
        x.append(theta)
        t = theta / 180 * math.pi
        lam = math.atan(math.cos(i)*math.tan(t))
        if -math.pi<=t<=-math.pi/2:
            lam-=math.pi
        elif math.pi/2<=t<=math.pi:
            lam+=math.pi
        y.append(lam*180/math.pi)
    from matplotlib import pyplot as plt
    plt.plot(x,y)
    plt.show()
# get_lon()
# 4*6371**2*math.atan((1-math.cos(math.pi/90))*(1-math.cos(math.pi/90))/2/(math.cos(math.pi/90)+math.cos(math.pi/90)))

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename='data/log.txt', format=LOG_FORMAT, datefmt=DATE_FORMAT, level=logging.DEBUG)

def dijkstra(graph, s):
    '''
    graph使用dict存储，每个key表示一个节点，里面有到其他节点的距离d
    '''
    passed = [s] 
    nopassed = [x for x in range(len(graph)) if x != s]
    inf = len(graph) * len(graph)
    dis = [inf] * len(graph)
    pare = [inf] * len(graph)
    idx = s # current node
    dis[s] = 0
    pare[s] = -1

    while len(nopassed):
        for i, d in graph[idx]:
            if dis[i] > dis[idx] + d:
                dis[i] = dis[idx] + d
                pare[i] = idx

        idx = nopassed[0]
        for i in nopassed:
            if dis[i] < dis[idx]: idx = i

        nopassed.remove(idx)
        passed.append(idx)

    return dis, pare