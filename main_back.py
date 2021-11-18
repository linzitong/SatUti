import os
import ephem
import datetime
import math
from draw import drawMap
def create(orbit, num, h, incli, fold):
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
        os.system("rm %s*"%floder)
    cycle = 4
    snap = 3
    cnt = 0 # 卫星编号
    now = datetime.datetime.now()
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
            print('%d分钟完成' % (cur_min + snap))
    # drawMap(fold)
# create(23, 25, 550, 53, "fengwo")

def create_onenode(h):
	R=6371
	INCLINATION = 60
	ECCENTRICITY = 0.001 # 离心率
	ARG_OF_PERIGEE = 0.0 # 近地点
	ORBIT_CYCLE = math.sqrt(4 * math.pi**2 * (R + h)**3 / 397865.5)  # 卫星运行周期s
	EARTH_CYCLE = 86164
	MEAN_MOTION = EARTH_CYCLE / ORBIT_CYCLE# 每天转多少圈
	floder = 'one_node%d'%h
	if not os.path.exists(floder):
	        os.mkdir(floder)
	else:
	    os.system("rm %s*"%floder)
	cycle = 60*24*6
	snap = 1
	cnt = 0 # 卫星编号
	now = datetime.datetime.now()
	file = floder + '/%d.txt'%cycle # 输出的文件
	with open(file, 'w') as f:
		for cur_min in range(0, cycle, snap):
			cur_time = (now + datetime.timedelta(minutes=cur_min)).strftime("%Y-%m-%d %H:%M:%S")
			raan =  0
			meanAnomaly =0
			sat = ephem.EarthSatellite()
			sat._epoch = now
			sat._inc = INCLINATION
			sat._raan = raan
			sat._M = meanAnomaly
			sat._n = MEAN_MOTION
			sat._e = ECCENTRICITY  # 偏心率
			sat._ap = ARG_OF_PERIGEE  # 圆
			sat.compute(cur_time)
			f.writelines('%d,%s,%s,%s\n' % (
			cur_min, math.degrees(sat.sublat), math.degrees(sat.sublong), sat.elevation / 1000))


def lonlat2xyz(lat,lon,r):
	pi = math.pi
	Theta = pi / 2 - lat * pi / 180
	Phi = 2 * pi + lon * pi / 180
	X=r*math.sin(Theta)*math.cos(Phi)
	Y=r*math.sin(Theta)*math.sin(Phi)
	Z=r*math.cos(Theta)
	xyz=[X,Y,Z]
	return xyz



def beta2r(h, beta):
    beta = beta / 180 * math.pi
    R = 6371
    alpha = math.asin(R / (R+h) * math.sin(beta + math.pi/2))
    theta = math.asin((R+h) / R * math.sin(alpha)) - alpha
    r = theta * R #覆盖半径
    d = math.sqrt(R**2 + (R + h)**2 - 2 * R * (R + h) * math.cos(theta))
    # print(r, d)
    return r, d

# beta2r(570,25)


import csv
# 不考虑时区的变化，仅计算区间内经过区域的人口的总数
def CountUsers(shell, h, elevation):
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
    popu_stat = []
    for i in range(int(360/shell)):
    	popu_stat.append([0, 0]) # 每个type的卫星所能服务的用户的数量
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
                for i in range(0,180):
                    for j in range(0,360):
                        if lati - ran - 2 < i - 90 and lati + ran +2 > i - 90 and longi - ran*2 - 4 < j - 180 and longi + ran*2 +4 > j - 180:
                                xyz2 = lonlat2xyz (i - 90, j - 180, R)
                                dis = math.sqrt((xyz[0] - xyz2[0])**2 + (xyz[1] - xyz2[1])**2 +(xyz[2] - xyz2[2])**2)
                                if dis < d:
                                	# 统计海陆比
                                	# if population[i][j] == 0: popu_stat[int(line[0])][0] += 1
                                	# else: popu_stat[int(line[0])][1] += 1
                                	# 统计总的人口
                                	popu_stat[int(line[0])] = population[i][j]
    
    print(popu_stat)
    s1 = ''
    s2 = ''
    for i in range(len(popu_stat)):
    	s1 += str(popu_stat[i][0]) + ','
    for i in range(len(popu_stat)):
    	s2 += str(popu_stat[i][1])+ ','
    print(s1)
    print(s2)
    # print(k for i in )
    
# CountUsers(15, 570, 40)

# 计算第一个卫星在一个回归周期内的负载变化情况
def CountUsers2(shell, h, elevation):
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
            for i in range(0,180):
                for j in range(0,360):
                    if lati - ran - 2 < i - 90 and lati + ran +2 > i - 90 and longi - ran*2 - 4 < j - 180 and longi + ran*2 +4 > j - 180:
                            xyz2 = lonlat2xyz (i - 90, j - 180, R)
                            dis = math.sqrt((xyz[0] - xyz2[0])**2 + (xyz[1] - xyz2[1])**2 +(xyz[2] - xyz2[2])**2)
                            if dis < d:
                                popu_stat[int(cur_min / 3)] +=  population[i][j]
    print(popu_stat)
    
CountUsers2(15, 570, 40)

def main():
	h = [20223, 13924, 10382, 6414, 4183, 2724, 1683, 570, 277]
	cycle = [2, 3, 4, 6, 8, 10, 12, 15, 16]
	for i in [0]:
		print(cycle[i])
		# create(cycle[i], h[i])
		CountUsers(cycle[i], h[i], 40)


def fun(x, a, b):
    return x**a +b

sate={}
sate_num = 0
def create_link(orbit, num, h, fold):
    sate_num = orbit*num
    start_nodes= []
    end_nodes = []
    capacity = []
    cost = []
    cur_min = 3
    satPosFile = '%s/%d.txt'%(fold, cur_min)
    print(satPosFile)
    with open(satPosFile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            nums = line.split(',')
            sate[int(nums[0]) + 1] = lonlat2xyz(float(nums[2]), float(nums[3]), 6371 + h)
    # print(sate)
    # 左右
    for i in range(0, orbit - 1):
        for j in range(1, num + 1):
            # print(i*num +j, (i+1)*num +j)
            d = int(math.sqrt((sate[i*num +j][0] + sate[(i+1)*num +j][0])**2 + (sate[i*num +j][1] + sate[(i+1)*num +j][1])**2 + (sate[i*num +j][2] + sate[(i+1)*num +j][2])**2))
            start_nodes.append(i * num + j)
            end_nodes.append((i+1)*num + j)
            cost.append(d)
            end_nodes.append(i * num + j)
            start_nodes.append((i+1)*num + j)
            cost.append(d)
    for j in range(1, num + 1):
        d = int(math.sqrt((sate[j][0] - sate[(orbit-1)*num +j][1])**2 + (sate[j][1] - sate[(orbit-1)*num +j][1])**2 + (sate[j][2] - sate[(orbit-1)*num +j][2])**2))
        start_nodes.append(j)
        end_nodes.append((orbit - 1)*num +j)
        end_nodes.append(j)
        start_nodes.append((orbit - 1)*num +j)
        cost.append(d)
        cost.append(d)
    
    # 上下
    for i in range(orbit):
        for j in range(1, num):
            d = int(math.sqrt((sate[i*num +j][0] + sate[i*num + j + 1][0])**2 + (sate[i*num +j][1] + sate[i*num + j + 1][1])**2 + (sate[i*num +j][2] + sate[i*num + j + 1][2])**2))
            start_nodes.append(i*num +j)
            end_nodes.append(i*num + j + 1)
            end_nodes.append(i*num +j)
            start_nodes.append(i*num + j + 1)
            cost.append(d)
            cost.append(d)
        d = int(math.sqrt((sate[i*num +j][0] + sate[i*num + num][0])**2 + (sate[i*num +j][1] + sate[i*num + num][1])**2 + (sate[i*num +j][2] + sate[i*num + num][2])**2))
        start_nodes.append(i*num + 1)
        end_nodes.append(i*num + num)
        end_nodes.append(i*num + 1)
        start_nodes.append(i*num + num)
        cost.append(d)
        cost.append(d)
    capacity = [100] * len(cost)
    # print(len(capacity))
    tmp = [0] * orbit * num
    for  i in start_nodes:
        tmp[i - 1] += 1
    return start_nodes, end_nodes, capacity, cost

def add_ground_link(sate_num, start_nodes, end_nodes, capacity, cost, fold = "fengwo"):
    ground_latlon=[
        [22, 114],
        [51, -10],
        [35, -100],
        [31, 34],
        [-15, -47]
    ]
    for i in range(sate_num + 1, sate_num + 6):
        j = ground_latlon[i - sate_num - 1]
        sate[i] = lonlat2xyz(j[0],j[1], 6371)
    
    l, d = beta2r(550, 0)
    print(d)
    d = int(d)
    for i in range(1, sate_num + 1):
        for j in range(sate_num + 1, sate_num +6):
            dis = math.sqrt((sate[i][0] - sate[j][0])** 2 +(sate[i][1] - sate[j][1])** 2 +(sate[i][2] - sate[j][2])** 2)
            # print(dis, d)
            if dis < d:
                # add link
                start_nodes.append(i)
                end_nodes.append(j)
                start_nodes.append(j)
                end_nodes.append(i)
                capacity.append(100)
                capacity.append(100)
                cost.append(d)
                cost.append(d)
    
    

# 博客：https://developers.google.com/optimization/flow/mincostflow
from ortools.graph import pywrapgraph
def MFMC(orbit, num, h):
    create(orbit, num, h, 53, "fengwo")
    start_nodes, end_nodes, capacity, cost = create_link(orbit, num, h, "fengwo")
    
    sate_num = orbit * num
    add_ground_link(sate_num, start_nodes, end_nodes, capacity, cost)
    for i in range(len(start_nodes)):
        start_nodes[i] -= 1
        end_nodes[i] -= 1
    
    supply = [0] * (sate_num + 5)
    print(len(start_nodes),len(end_nodes),len(capacity),len(cost))
    supply_change=[
        [1,2,13889],
        [1,3,21508],
        [2,3,21442],
        [2,4,20757],
        [3,5,36848]
    ]
    # print(cost)
    # 修改：对每一对流，仅允许对应两个节点连接其他节点是不能连接的
    for change in supply_change:     
        supply = [0] * (sate_num + 5)
        supply[sate_num - 1 + change[0]] = int(change[2]*0.1)
        supply[sate_num - 1 + change[1]] = int(-1 * change[2]*0.1)

        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        for i in range(len(start_nodes)):
            min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i],end_nodes[i],capacity[i],cost[i])
        for i in range(len(supply)):
            min_cost_flow.SetNodeSupply(i,supply[i])
        
        if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
            print('Maximum flow:',min_cost_flow.MaximumFlow())
            print('Minimum cost:', min_cost_flow.OptimalCost())
            print('')
            # print('  Arc    Flow / Capacity  Cost')
            for i in range(min_cost_flow.NumArcs()):
                # cost_here = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
                if min_cost_flow.Flow(i) > 0:
                    # if min_cost_flow.Tail(i) >= sate_num:# or min_cost_flow.Head(i)>=sate_num:
                        # print('%1s -> %1s   %3s  / %3s       %3s' % (
                        #     min_cost_flow.Tail(i),
                        #     min_cost_flow.Head(i),
                        #     min_cost_flow.Flow(i),
                        #     min_cost_flow.Capacity(i),
                        #     cost_here))
                    for j in range(len(start_nodes)):
                        if start_nodes[j] == min_cost_flow.Tail(i) and end_nodes[j] == min_cost_flow.Head(i):
                            capacity[j] -= min_cost_flow.Flow(i)
                            break
        else:
            print('There was an issue with the min cost flow input.')
            break
    

    # 在startnode和endnode中存储的节点号是从0 开始的
    drawMap(sate_num, "fengwo",start_nodes,end_nodes,capacity)
# MFMC(72, 22, 550)
# beta2r(550,40)