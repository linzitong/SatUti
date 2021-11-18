from matplotlib import pyplot as plt
from base import *
def get_population():
    '''
    将原始population文件的数据进行处理存储到文件中
    '''
    import csv
    csv_write = open('data/population180.360.2.csv',"w",newline='')
    writer = csv.writer(csv_write)
    rows=[]
    with open('data/population180.360.csv','r') as f:
        reader = csv.reader(f)
        count=0
        for row in reader:
            for i in range(len(row)):
                if row[i] == '-9999':
                    row[i] = '0'
            rows.append(row)
    writer.writerows(rows)
    csv_write.close()

def change_day_night(t):
    '''
    对用户进行时区的变化，获取ts时间的用户，用户在本地时间基本遵循相同的活跃度
    认为在时间0的时候，0经线时间为0时,输入t单位为s
    '''
    import csv
    zero_zone = lon_add([0], -360/86400*t) #  时间为0的经线
    print(zero_zone)
    active = [0.442622951,0.327868852,0.262295082,0.229508197,0.237704918,0.295081967,
        0.442622951,0.737704918,0.852459016,0.836065574,0.819672131,0.836065574,0.836065574,
        0.836065574,0.868852459,0.885245902,0.901639344,0.983606557,0.983606557,1,
        1,0.983606557,0.852459016,0.655737705] # 不同时间(t=0~24)的用户的活跃度
    csv_write = open('data/population180.360.3.csv',"w",newline='')
    writer = csv.writer(csv_write)
    rows=[]
    with open('data/population180.360.2.csv','r') as f:
        reader = csv.reader(f)
        for row in reader:
            for i in range(len(row)):
                longi = i - 180
                rela = lon_relative(zero_zone[0],longi)
                rela_time = 24/360*rela
                t = int(rela_time%24)
                # print(t)
                # print(t,rela,rela_time,zero_zone[0])
                row[i] = str(float(row[i])*active[t])
            rows.append(row)
    writer.writerows(rows)
    csv_write.close()
# change_day_night(46400)

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

def create_user(num, t, fold='data/population180.360.3.csv'):
    '''
    根据每个经纬度块的比例给出每个经纬度块在总数为num时该有的用户数
    '''
    import csv
    # change_day_night(t)
    total_people = 7969436034
    ratio = num/total_people
    csv_reader = open(fold,'r')
    reader = csv.reader(csv_reader)
    users = []
    lat = -1
    for row in reader:
        lat += 1
        for longi in range(len(row)):
            if row[longi] != '0':
                # print(row[longi])
                user = float(row[longi]) * ratio
                x = longi - 180
                y = -1 * lat + 90
                users.append([y, x, user])
    csv_reader.close()
    return users

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


def stat_user(type="lat", folder='data/population180.360.2.csv'):
    '''
    根据经纬度统计人口分布
    '''
    import csv
    csv_reader = open(folder,'r')
    reader = csv.reader(csv_reader)
    lat = -1
    population = []
    for row in reader:
        popu = []
        for i in row:
            popu.append(float(i))
        population.append(popu)
    csv_reader.close()
    re = []
    if type == "lon":
        re = population[0]
        for i in range(1, len(population)):
            for j in range(len(population[i])):
                re[j] += population[i][j]
        # plt.plot([i for i in range(180,-180,-1)], re)
        # plt.xlabel('longitude')
    else:
        re = [sum(i)/10**6/2.3 for i in population]
        # print(re)
        # plt.plot([i for i in range(90,-90,-1)], re)
        # plt.xlabel('latitude')
    # plt.ylabel('population')
        re.reverse()
    return re
    # plt.show()


def stat_user_each_time():
    '''
    统计在不同时间内不同经度的人口情况
    '''
    x = [i for i in range(-180, 180)]
    a = [i*3600 for i in [0,15]]
    for i in a:
        change_day_night(i)
        re = stat_user("lon",'data/population180.360.3.csv')
        print(sum(re))
        plt.plot(x, re, linewidth=2,label='t=%d'%(int(i/3600)))
    plt.xlabel('latitude')

    plt.ylabel('population')
    re = stat_user("lon",'data/population180.360.2.csv')
    print(sum(re))
    plt.plot(x, re, linewidth=1,label='population',linestyle='--')
    plt.legend()
    plt.show()
# stat_user_each_time()



