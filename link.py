import math
from constellation import *
from population import *
from base import *


"""
星地链路
"""
def minlin_each(users, sate, h, mx = 100, ele=35):
    '''
    根据用户和卫星经纬度，以最短距离计算每个卫星的接入量
    如果接入量超过mx则不满足要求返回空
    users[[lat, longi]...]
    sate[[sate_ID, lat, longi]]
    '''
    R = 6371
    r = R + h
    user_xyz=[]
    sate_xyz=[] 
    user_link = [-1] * len(users)
    sate_num = [0]* len(sate)
    for user in users:
        xyz = lonlat2xyz(user[0], user[1], R)
        user_xyz.append(list(xyz))
    for s in sate:
        xyz = lonlat2xyz(s[1], s[2], r)
        xyz.append(s[0])
        sate_xyz.append(list(xyz))
    r, d = beta2r(h, ele)
    ran = r / 111

    for i in range(len(user_xyz)):
        u = user_xyz[i]
        min_now = d
        has_sat = False
        for j in range(len(sate_xyz)):
            s = sate_xyz[j]
            if abs(users[i][0] - sate[j][1]) > ran or abs(users[i][1] - sate[j][2]) > ran * 2 or 360 - abs(users[i][1]) - abs(sate[j][2]) <= ran * 2: continue
            dd = math.sqrt((u[0] - s[0])**2 + (u[1] - s[1])**2 + (u[2] - s[2])**2)
            if dd < min_now :
                has_sat = True
                if sate_num[s[3]] < mx:
                    min_now = dd
                    user_link[i] = s[3]
        if user_link[i] != -1:
            sate_num[user_link[i]] += 1
        elif has_sat:
            # print("not enough satellites", users[i])
            return [], []

    return sate_num, user_link

def minlin_same_dir(users, sate, h, num, mx = 100):
    '''
    根据用户和卫星经纬度，以最短距离计算每个卫星的接入量
    如果接入量超过mx则不满足要求返回空
    users[[lat, longi]...]
    sate[[sate_ID, lat, longi]]
    '''
    R = 6371
    r = R + h
    user_xyz=[]
    sate_xyz=[] 
    user_link = [-1] * len(users)
    sate_num = [0]* len(sate)
    for user in users:
        xyz = lonlat2xyz(user[0], user[1], R)
        user_xyz.append(list(xyz))
    for s in sate:
        xyz = lonlat2xyz(s[1], s[2], r)
        xyz.append(s[0])
        sate_xyz.append(list(xyz))
    r, d = beta2r(h, 25)
    ran = r / 111

    for i in range(len(user_xyz)):
        u = user_xyz[i]
        min_now = d
        has_sat = False
        for j in range(len(sate_xyz)):
            s = sate_xyz[j]
            if sate[j][1] - sate[next_sate(j,num)][1] > 0: continue
            if abs(users[i][0] - sate[j][1]) > ran or abs(users[i][1] - sate[j][2]) > ran * 2 or 360 - abs(users[i][1]) - abs(sate[j][2]) <= ran * 2: continue
            dd = math.sqrt((u[0] - s[0])**2 + (u[1] - s[1])**2 + (u[2] - s[2])**2)
            if dd < min_now :
                has_sat = True
                if sate_num[s[3]] < mx:
                    min_now = dd
                    user_link[i] = s[3]
        if user_link[i] != -1:
            sate_num[user_link[i]] += 1
        elif has_sat:
            # print("not enough satellites", users[i])
            return []

    return sate_num, user_link

def minlen(users, sate, h):
    '''
    根据用户和卫星经纬度，以最短距离计算每个卫星的接入量
    不考虑卫星的最大接入量
    users[[lat, longi, user_num]...]
    sate[[sate_ID, lat, longi]]
    '''
    r = 6371 + h
    user_xyz=[]
    sate_xyz=[] 
    sate_num = [0]* len(sate)
    c = 0
    for user in users:
        xyz = lonlat2xyz(user[0], user[1], 6371)
        xyz.append(user[2])
        c += 1
        user_xyz.append(list(xyz))

    for s in sate:
        xyz = lonlat2xyz(s[1], s[2], r)
        xyz.append(s[0])
        sate_xyz.append(list(xyz))
    r, d = beta2r(h, 25)
    for u in user_xyz:
        min_now = d
        link = -1
        for s in sate_xyz:
            dd = math.sqrt((u[0] - s[0])**2 + (u[1] - s[1])**2 + (u[2] - s[2])**2)
            if dd < min_now:
                min_now = dd
                link = s[3]      
        if link != -1:
            sate_num[link] += u[3]
    return sate_num

def visible(fold):
    '''
    在地面生成均匀的随机点，计算指定星座对随机点的可见度来估算覆盖度
    '''
    # 生成随机点计算
    r = 6371
    x, y, z = sample_spherical(1000)
    x = [i*r for i in x]
    y = [i*r for i in y]
    z = [i*r for i in z]
    orbit = [28,36,34]
    beta=35
    create_conste(orbit_list=orbit, 
        num_list=[28,36,34],
        h_list=[590,610,630], 
        incli_list=[33,42,51.9], 
        fold=fold,
        t=3)
    re = 0
    count = 0
    for cur_min in range(0, 90, 3):
        count += 1
        print(cur_min)
        sate_list = get_sat_loca(fold+'/%d.txt'%cur_min)
        c = 0
        for i in range(len(x)):
            for s in sate_list:
                xyz = lonlat2xyz(s[0], s[1], s[2] + 6371)
                if math.sqrt((x[i]-xyz[0])**2 + (y[i]-xyz[1])**2 + (z[i]-xyz[2])**2) < beta2r(s[2],beta)[1]:
                    c += 1
                    break
        re += c
    print(re/count)
# visible("data/fengwo")
def dis(a1_lonat, a2_lonlat, a1, a2, r, d):
    if abs(a1_lonat[0]-a2_lonlat[0]) *111 < r and abs(a1_lonat[1]-a2_lonlat[1])*2*111 < r:
        if math.sqrt((a1[0]-a2[0])**2 + (a1[1]-a2[1])**2 + (a1[2]-a2[2])**2) < d:
            return 1
    return 0

def draw_hot_map():
    '''
    对指定时间卫星的分布，测算不同经纬度块可见卫星数
    '''
    h=550
    R=6371
    re = [[0]*360 for _ in range(180)]
    sate_list = get_sat_loca('data/fengwo/1428.txt')
    r,d = beta2r(h,25)
    sate_xyz = []
    for sate in sate_list:
        sate_xyz.append(lonlat2xyz(sate[1],sate[2],R+h))
    for i in range(len(re)):
        # print(i)
        for j in range(len(re[i])):
            lat = i - 90
            lon = j - 180
            xyz = lonlat2xyz(lat, lon, R)
            for z in range(len(sate_xyz)):
                re[i][j] += dis([lat,lon], sate_list[z][1:], xyz, sate_xyz[z],r,d)
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.heatmap(re, cmap='GnBu')
    plt.show()

# draw_hot_map()

def minlin_block_user(users, sate, h, mx=100):
    '''
    根据用户和卫星经纬度，以最短距离计算每个卫星的接入量,返回不能连接卫星的用户的位置
    users[[lat, longi]...]
    sate[[sate_ID, lat, longi]]
    '''
    R = 6371
    r = R + h
    user_xyz=[]
    sate_xyz=[] 
    block_user = []
    sate_num = [0]* len(sate)
    for user in users:
        xyz = lonlat2xyz(user[0], user[1], R)
        user_xyz.append(list(xyz))
    for s in sate:
        xyz = lonlat2xyz(s[1], s[2], r)
        xyz.append(s[0])
        sate_xyz.append(list(xyz))
    r, d = beta2r(h, 25)
    ran = r / 111

    for i in range(len(user_xyz)):
        u = user_xyz[i]
        min_now = d
        link = -1
        has_sat = False
        for j in range(len(sate_xyz)):
            s = sate_xyz[j]
            if abs(users[i][0] - sate[j][1]) > ran or abs(users[i][1] - sate[j][2]) > ran * 2 or 360 - abs(users[i][1]) - abs(sate[j][2]) <= ran * 2: continue
            dd = math.sqrt((u[0] - s[0])**2 + (u[1] - s[1])**2 + (u[2] - s[2])**2)
            if dd < min_now :
                has_sat = True
                if sate_num[s[3]] < mx:
                    min_now = dd
                    link = s[3]      
        if link != -1:
            sate_num[link] += 1
        elif has_sat:
            block_user.append([users[i][0], users[i][1]])
    return sate_num, block_user

"""
星间链路
"""
def create_link(orbit, num, h, fold, link_cap=100):
    ''''
    每个卫星生成上下左右的4条链路，其容量为link_cap
    可以认为，卫星ab之间分别存在一条a->b  b->a的容量为link_cap的链路
    todo:其中的cost会随着时间变，这里存在bug但不影响目前的结果只会影响cost
    '''
    start_nodes= []
    end_nodes = []
    capacity = []
    cost = []
    sate = {}
    cur_min = 3
    satPosFile = '%s/%d.txt'%(fold, cur_min)
    # print(satPosFile)
    with open(satPosFile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            nums = line.split(',')
            sate[int(nums[0]) + 1] = lonlat2xyz(float(nums[2]), float(nums[3]), 6371 + h)
    # 左右
    for i in range(0, orbit - 1):
        for j in range(1, num + 1):
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
    capacity = [link_cap] * len(cost)
    # print(len(capacity))
    tmp = [0] * orbit * num
    for  i in start_nodes:
        tmp[i - 1] += 1
    return start_nodes, end_nodes, capacity, cost, sate

def add_ground_link(sate_num, start_nodes, end_nodes, capacity, cost, sate, ground_latlon):
    '''
    根据ground_latlon的经纬度计算地站与卫星可形成的链接
    '''
    for i in range(sate_num + 1, sate_num + len(ground_latlon) + 1):
        j = ground_latlon[i - sate_num - 1]
        sate[i] = lonlat2xyz(j[0],j[1], 6371)
    
    l, d = beta2r(550, 25)
    d = int(d)
    for i in range(1, sate_num + 1):
        for j in range(sate_num + 1, sate_num +len(ground_latlon) + 1):
            dis = math.sqrt((sate[i][0] - sate[j][0])** 2 +(sate[i][1] - sate[j][1])** 2 +(sate[i][2] - sate[j][2])** 2)
            if dis < d*4:
                # add link
                start_nodes.append(i)
                end_nodes.append(j)
                start_nodes.append(j)
                end_nodes.append(i)
                capacity.append(100)
                capacity.append(100)
                cost.append(d)
                cost.append(d)
    return sate

def MFMC(start_nodes, end_nodes, capacity, cost, supply_change, sate_num, capacity_ground, ground_latlon, star_link_count, fault):
    '''
    使用最大流算法获得系统的最大容量（按照最后一跳也计入最短路径的方式）
    '''
    c = 0
    from ortools.graph import pywrapgraph
    node_num = int(len(ground_latlon)/2)
    cost_all = 0

    for change in supply_change:
        c += 1
        for i in range(len(capacity)):
            if start_nodes[i] == sate_num - 1 + change[0] or end_nodes[i] == sate_num - 1 + change[1]:
                pass
            elif start_nodes[i] in [j for j in range(sate_num,sate_num+node_num)] or end_nodes[i] in [j for j in range(sate_num,sate_num+node_num)]:
                capacity[i] = 0
        
        supply = [0] * (sate_num + len(ground_latlon))
        supply[change[0]] = change[2]
        supply[change[1]] = -1 * change[2]

        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        for i in range(len(start_nodes)):
            min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i],end_nodes[i],capacity[i],cost[i])
        for i in range(len(supply)):
            min_cost_flow.SetNodeSupply(i,supply[i])
        if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
            # print('%dok'%c)
            for i in range(min_cost_flow.NumArcs()):
                cost_here = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
                cost_all += cost_here
                if min_cost_flow.Flow(i) > 0:
                    for j in range(len(start_nodes)):
                        if start_nodes[j] == min_cost_flow.Tail(i) and end_nodes[j] == min_cost_flow.Head(i):
                            capacity[j] -= min_cost_flow.Flow(i)
                            break
        else:
            print('There was an issue with the min cost flow input.')
            fault.append(change)
            # break
    
        # 处理：所有地站相关的连接中，除了change关联的两类路径，其余全部由capacity_ground替换
        for i in range(len(capacity)):
            if start_nodes[i] == sate_num - 1 + change[0] or end_nodes[i] == sate_num - 1 + change[1]:
                pass
            elif start_nodes[i] in [j for j in range(sate_num,sate_num+node_num)] or end_nodes[i] in [j for j in range(sate_num,sate_num+node_num)]:
                capacity[i] = capacity_ground[i - star_link_count]

def al_mfmc(orbit, num, h, n):
    '''
    使用最大流算法获得系统的最大容量（按照最后一跳也计入的方式）
    '''
    # 博客：https://developers.google.com/optimization/flow/mincostflow
    
    create_shell(orbit, num, h, 53,"data/fengwo",4)
    start_nodes, end_nodes, capacity, cost, sate = create_link(orbit, num, h, "data/fengwo")
    
    # 记录有多少连接是星间链接
    star_link_count = len(start_nodes)
    sate_num = orbit * num
    users = []
    total_fault = []
    for j in range(int(n/10)):
        print(j)
        ground_latlon= create_each_user(10)
        users.extend(ground_latlon)
        sate = add_ground_link(sate_num, start_nodes, end_nodes, capacity, cost, sate, ground_latlon)
        for i in range(len(start_nodes)):
            start_nodes[i] -= 1
            end_nodes[i] -= 1
        node_num = int(len(ground_latlon)/2)
        supply_change = [[i, i+node_num, 1] for i in range(node_num)]
        capacity_ground = capacity[star_link_count:len(capacity)]
        fault = []
        MFMC(start_nodes, end_nodes, capacity, cost, supply_change, sate_num, capacity_ground, ground_latlon, star_link_count, fault)
        if len(fault): 
            for f in fault:
                total_fault.append([ground_latlon[f[0]], ground_latlon[f[1]]])

        start_nodes = start_nodes[:star_link_count]
        end_nodes = end_nodes[:star_link_count]
        capacity = capacity[:star_link_count]
        cost = cost[:star_link_count]
        for i in range(sate_num + 1, sate_num + len(ground_latlon) + 1):
            del sate[i]

        for i in range(len(start_nodes)):
            start_nodes[i] += 1
            end_nodes[i] += 1
    for i in range(len(start_nodes)):
            start_nodes[i] -= 1
            end_nodes[i] -= 1
    
    print(total_fault)
    with open('data/fengwo/cap_link.txt', 'w') as f:
        for i in range(len(capacity)):
            f.writelines('%d,%d,%d\n'%(start_nodes[i], end_nodes[i],capacity[i]))
    # with open('data/fengwo/cap_sate.txt', 'w') as f:
    #     for i in range(len(sate_num)):
    #         f.writelines('%d\n'%sate_num[i])
    from draw import drawMap
    drawMap(sate_num, "data/fengwo",start_nodes,end_nodes,capacity,users)
# al_mfmc(72, 22, 550, 5000)

def MFMC_topo(orbit, num, h, user_num, link_cap, trans_per,ele=25,sate_fold='data/fengwo/9.txt'):
    '''
    使用最大流算法获得系统的最大容量(按照用户选择最近的卫星连接后运行最大流算法)
    运行时间：1000用户23s,5000用户99s，6000用户116s；接入量最大在6000左右。
    '''
    # 博客：https://developers.google.com/optimization/flow/mincostflow
    # import time
    # time_begin=time.time()
    # create_shell(orbit, num, h, 53,"data/fengwo",4)
    start_nodes, end_nodes, capacity, cost, sate = create_link(orbit, num, h, "data/fengwo", link_cap)
    
    # 记录有多少连接是星间链接
    sate_num = orbit * num
    users = []
    link_dic = {}

    users= create_each_user(user_num)
    sate = get_sat_loca(sate_fold)
    for i in range(len(start_nodes)):
        start_nodes[i] -= 1
        end_nodes[i] -= 1
    
    # 获得每个用户连接到哪颗卫星上。
    sate_num, user_link = minlin_each(users, sate, h, mx=400,ele=ele)
    if len(sate_num) == 0: 
        print("%d, %d, %f,超出接入容量"%(user_num, link_cap, trans_per))
        logging.info("%d, %d, %f,超出接入容量"%(user_num, link_cap, trans_per))
        return -1
    node_num = int(len(users)/2)
    node_num = int(node_num * trans_per)
    # print('h', user_num, node_num)

    # 将所有用户进行配对，认为这些用户互相通信，记录每一对卫星之间通信对数
    for i in range(node_num):
        link1 = user_link[i]
        link2 = user_link[i+node_num]
        if link1 != -2 and link2 != -1:
            if (link1, link2) in link_dic:
                link_dic[(link1, link2)] += 1
            else:
                link_dic[(link1, link2)] = 1    

    c = 0
    from ortools.graph import pywrapgraph
    node_num = int(len(users)/2)
    cost_all = 0
    # print(len(link_dic))
    supply = [0] * len(sate_num)
    fault = []

    # 对每一对通信的卫星，使用最大流的方式计算得到每个链路的占用
    for k, v in link_dic.items():
        if k[0] == k[1]: continue
        c += 1
        # print(c)
        supply[k[0]] = v
        supply[k[1]] = -1 * v

        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        for i in range(len(start_nodes)):
            min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i],end_nodes[i],capacity[i],cost[i])
        for i in range(len(supply)):
            min_cost_flow.SetNodeSupply(i,supply[i])
        if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
            # print('%dok'%c)
            for i in range(min_cost_flow.NumArcs()):
                cost_here = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
                cost_all += cost_here
                if min_cost_flow.Flow(i) > 0:
                    for j in range(len(start_nodes)):
                        if start_nodes[j] == min_cost_flow.Tail(i) and end_nodes[j] == min_cost_flow.Head(i):
                            capacity[j] -= min_cost_flow.Flow(i)
                            break
        else:
            # print('There was an issue with the min cost flow input.%d&%d'%(k[0],k[1]))
            logging.info("%d, %d, %f,超出转交容量"%(user_num, link_cap, trans_per))
            print("%d, %d, %f,超出转交容量"%(user_num, link_cap, trans_per))
            return -2
            # break
        supply[k[0]] = 0
        supply[k[1]] = 0
    
    # 记录每一对链路的实际占用以及每一个卫星接入量的实际占用
    with open('data/fengwo/cap_link.txt', 'w') as f:
        for i in range(len(capacity)):
            f.writelines('%d,%d,%d\n'%(start_nodes[i], end_nodes[i],capacity[i]))
    with open('data/fengwo/cap_sate.txt', 'w') as f:
        for i in range(len(sate_num)):
            f.writelines('%d\n'%sate_num[i])
    print(user_num, link_cap, trans_per)
    logging.info('%d,%d,%f'%(user_num, link_cap, trans_per))
    # from draw import drawMap
    # drawMap(len(sate_num), "data/fengwo",start_nodes,end_nodes,capacity,users)
    return 1
    # print(time.time()-time_begin)

# MFMC_topo(72, 22, 550, 15000)
def MFMC_topo_t(orbit, num, h, users, link_cap, trans_per, t):
    '''
    使用最大流算法获得系统的最大容量(按照用户选择最近的卫星连接后运行最大流算法)
    运行时间：1000用户23s,5000用户99s，6000用户116s；接入量最大在6000左右。
    '''
    start_nodes, end_nodes, capacity, cost, sate = create_link(orbit, num, h, "data/fengwo", link_cap)
    
    # 记录有多少连接是星间链接
    sate_num = orbit * num
    link_dic = {}
    user_num = len(users)
    # change_day_night(t*60)
    # users= create_each_user(user_num, folder='data/population180.360.3.csv')
    # users = create_user
    sate = get_sat_loca('data/fengwo/%d.txt'%t)
    for i in range(len(start_nodes)):
        start_nodes[i] -= 1
        end_nodes[i] -= 1
    
    # 获得每个用户连接到哪颗卫星上。
    sate_num, user_link = minlin_each(users, sate, h, mx=400)
    if len(sate_num) == 0: 
        print("%d, %d, %f,超出接入容量"%(user_num, link_cap, trans_per))
        logging.info("%d, %d, %f,超出接入容量"%(user_num, link_cap, trans_per))
        return -1
    node_num = int(len(users)/2)
    node_num = int(node_num * trans_per)
    # print('h', user_num, node_num)

    # 将所有用户进行配对，认为这些用户互相通信，记录每一对卫星之间通信对数
    for i in range(node_num):
        link1 = user_link[i]
        link2 = user_link[i+node_num]
        if link1 != -2 and link2 != -1:
            if (link1, link2) in link_dic:
                link_dic[(link1, link2)] += 1
            else:
                link_dic[(link1, link2)] = 1    

    c = 0
    from ortools.graph import pywrapgraph
    node_num = int(len(users)/2)
    cost_all = 0
    # print(len(link_dic))
    supply = [0] * len(sate_num)
    fault = []

    # 对每一对通信的卫星，使用最大流的方式计算得到每个链路的占用
    for k, v in link_dic.items():
        if k[0] == k[1]: continue
        c += 1
        # print(c)
        supply[k[0]] = v
        supply[k[1]] = -1 * v

        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        for i in range(len(start_nodes)):
            min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i],end_nodes[i],capacity[i],cost[i])
        for i in range(len(supply)):
            min_cost_flow.SetNodeSupply(i,supply[i])
        if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
            # print('%dok'%c)
            for i in range(min_cost_flow.NumArcs()):
                cost_here = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
                cost_all += cost_here
                if min_cost_flow.Flow(i) > 0:
                    for j in range(len(start_nodes)):
                        if start_nodes[j] == min_cost_flow.Tail(i) and end_nodes[j] == min_cost_flow.Head(i):
                            capacity[j] -= min_cost_flow.Flow(i)
                            break
        else:
            # print('There was an issue with the min cost flow input.%d&%d'%(k[0],k[1]))
            logging.info("%d, %d, %f,超出转交容量"%(user_num, link_cap, trans_per))
            print("%d, %d, %f,超出转交容量"%(user_num, link_cap, trans_per))
            return -2
            # break
        supply[k[0]] = 0
        supply[k[1]] = 0
    
    # 记录每一对链路的实际占用以及每一个卫星接入量的实际占用
    with open('data/fengwo/cap_link.txt', 'w') as f:
        for i in range(len(capacity)):
            f.writelines('%d,%d,%d\n'%(start_nodes[i], end_nodes[i],capacity[i]))
    with open('data/fengwo/cap_sate.txt', 'w') as f:
        for i in range(len(sate_num)):
            f.writelines('%d\n'%sate_num[i])

    # from draw import drawMap
    # drawMap(len(sate_num), "data/fengwo",start_nodes,end_nodes,capacity,users)
    print(user_num, link_cap, trans_per)
    logging.info('%d,%d,%f'%(user_num, link_cap, trans_per))
    return 1
# create_shell(72, 22, 550, 53,"data/fengwo",4)
# users = create_each_user(2800)
# MFMC_topo_t(72, 22, 550, users, 50, 0.5,t=3)

def MFMC_topo_t_store(orbit, num, h, users, link_cap, trans_per, t):
    '''
    使用最大流算法获得系统的最大容量(按照用户选择最近的卫星连接后运行最大流算法)
    运行时间：1000用户23s,5000用户99s，6000用户116s；接入量最大在6000左右。
    '''
    start_nodes, end_nodes, capacity, cost, sate = create_link(orbit, num, h, "data/kuiper", link_cap)
    
    # 记录有多少连接是星间链接
    sate_num = orbit * num
    link_dic = {}
    user_num = len(users)
    # change_day_night(t*60)
    # users= create_each_user(user_num, folder='data/population180.360.3.csv')
    # users = create_user
    sate = get_sat_loca('data/kuiper/%d.txt'%t)
    for i in range(len(start_nodes)):
        start_nodes[i] -= 1
        end_nodes[i] -= 1
    
    # 获得每个用户连接到哪颗卫星上。
    sate_num, user_link = minlin_each(users, sate, h, mx=400)
    if len(sate_num) == 0: 
        print("%d, %d, %f,超出接入容量"%(user_num, link_cap, trans_per))
        logging.info("%d, %d, %f,超出接入容量"%(user_num, link_cap, trans_per))
        return -1
    node_num = int(len(users)/2)
    node_num = int(node_num * trans_per)
    # print('h', user_num, node_num)

    # 将所有用户进行配对，认为这些用户互相通信，记录每一对卫星之间通信对数
    for i in range(node_num):
        link1 = user_link[i]
        link2 = user_link[i+node_num]
        if link1 != -2 and link2 != -1:
            if (link1, link2) in link_dic:
                link_dic[(link1, link2)] += 1
            else:
                link_dic[(link1, link2)] = 1    

    c = 0
    from ortools.graph import pywrapgraph
    node_num = int(len(users)/2)
    cost_all = 0
    # print(len(link_dic))
    supply = [0] * len(sate_num)
    fault = []

    # 对每一对通信的卫星，使用最大流的方式计算得到每个链路的占用
    for k, v in link_dic.items():
        if k[0] == k[1]: continue
        c += 1
        # print(c)
        supply[k[0]] = v
        supply[k[1]] = -1 * v

        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        for i in range(len(start_nodes)):
            min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i],end_nodes[i],capacity[i],cost[i])
        for i in range(len(supply)):
            min_cost_flow.SetNodeSupply(i,supply[i])
        if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
            # print('%dok'%c)
            for i in range(min_cost_flow.NumArcs()):
                cost_here = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
                cost_all += cost_here
                if min_cost_flow.Flow(i) > 0:
                    for j in range(len(start_nodes)):
                        if start_nodes[j] == min_cost_flow.Tail(i) and end_nodes[j] == min_cost_flow.Head(i):
                            capacity[j] -= min_cost_flow.Flow(i)
                            break
        else:
            # print('There was an issue with the min cost flow input.%d&%d'%(k[0],k[1]))
            logging.info("%d, %d, %f,超出转交容量"%(user_num, link_cap, trans_per))
            print("%d, %d, %f,超出转交容量"%(user_num, link_cap, trans_per))
            return -2
            # break
        supply[k[0]] = 0
        supply[k[1]] = 0
    
    # 记录每一对链路的实际占用以及每一个卫星接入量的实际占用
    # with open('data/kuiper/cap_link%d.txt'%t, 'w') as f:
    #     for i in range(len(capacity)):
    #         f.writelines('%d,%d,%d\n'%(start_nodes[i], end_nodes[i],capacity[i]))
    # with open('data/kuiper/cap_sate%d.txt'%t, 'w') as f:
    #     for i in range(len(sate_num)):
    #         f.writelines('%d\n'%sate_num[i])
    # print(user_num, link_cap, trans_per)
    # logging.info('%d,%d,%f'%(user_num, link_cap, trans_per))
    from draw import drawMap
    drawMap(len(sate_num), "data/kuiper",start_nodes,end_nodes,capacity,users,cur_min=t)
    return 1

def hops(orbit, num):
    '''
    计算节点之间的跳数
    '''
    h = 550
    incli = 53
    fold = "ceshi"
    sate_num = orbit * num
    create_shell(orbit, num, h, incli, fold)     
    r, d = beta2r(550, 35)
    start_nodes, end_nodes, capacity, cost = create_link(orbit, num, h, fold)
    add_ground_link(sate_num, start_nodes, end_nodes, capacity, cost)
    graph={}
    for i in range(0,sate_num+6):
        graph[i] = []
    for s in range(len(start_nodes)):
        graph[start_nodes[s]].append([end_nodes[s],1])
    # print(graph)

    dis, pare = dijkstra(graph, sate_num+5)
    print(dis[sate_num+1:sate_num+6])
    # 1585    1586 1587   1588  1589
    # 开普敦  柏林  芝加哥  东京  华盛顿
# orbit = [10,15,20,25,30,35,40,45,50]
# num = [10,15,20,25,30,35,40,45,50]
# for i in range(len(orbit)):
#     hops(orbit[i],num[i])
