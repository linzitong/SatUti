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

def create_each_user(num,folder='data/population180.360.2.csv'):
    '''
    根据用户分布的权重随机生成num个用户
    '''
    import csv, random
    csv_reader = open(folder,'r')
    reader = csv.reader(csv_reader)
    lat = -1
    population = []
    for row in reader:
        popu = []
        for i in row:
            popu.append(float(i))
        population.append(popu)
    total_num = sum(map(sum, population))
    csv_reader.close()

    summ = 0
    for x in range(len(population)):
        for y in range(len(population[x])):
            summ = summ + population[x][y]
            population[x][y] = summ

    user = []
    for i in range(num):
        a = random.random() * total_num
        for x in range(len(population)):
            flag = 0
            if a > population[x][-1]:
                continue
            else:
                for y in range(len(population[x])):
                    if a > population[x][y]: continue
                    else:
                        flag = 1
                        break
            if flag: break
        lat = 90 - (x - random.random()) * len(population) / 180
        lon = (y - random.random()) * len(population) / 180 -180
        user.append([lat, lon])
    return user

def change_day_night_each(users, t):
    '''
    对用户进行时区的变化，获取tmin时间的用户，用户在本地时间基本遵循相同的活跃度
    认为在时间0的时候，0经线时间为0时,输入t单位为s
    '''
    import random
    zero_zone = lon_add([0], 360/1440*t) #  时间为0的经线
    active = [0.442622951,0.327868852,0.262295082,0.229508197,0.237704918,0.295081967,
        0.442622951,0.737704918,0.852459016,0.836065574,0.819672131,0.836065574,0.836065574,
        0.836065574,0.868852459,0.885245902,0.901639344,0.983606557,0.983606557,1,
        1,0.983606557,0.852459016,0.655737705] # 不同时间的用户的活跃度
    user_re = []
    for user in users:
        longi = user[1]
        rela = lon_relative(zero_zone[0], longi)
        rela_time = 24/360*rela
        t = int(rela_time)
        r = random.random()
        if r <= active[t]:
            user_re.append(user)
    return user_re

def main():
    users = create_each_user(1000)
    t = 0 #分钟
    users_t = change_day_night_each(users, t)
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap
    lon=[]
    lat=[]
    for u in users_t:
        lon.append(u[1])
        lat.append(u[0])

    map = Basemap()
    map.drawcoastlines()
    plt.scatter(lon,lat,marker='.')
    plt.show()

main()