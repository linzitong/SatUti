from math import pi

from matplotlib.pyplot import xlabel
from link import *
from use_rate import *
import matplotlib
font_xy={'size':23}
font_legend={'size':15}
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def draw_base_conste(orbit_list, num_list, h_list, incli_list, ele_list, cap_list):
    left = 0
    per = 0.5
    for i in range(len(orbit_list)):
        cap = cap_list[i]
        orbit = orbit_list[i]
        num = num_list[i]
        h = h_list[i]
        incli = incli_list[i]
        ele = ele_list[i]
        print("##### constellation %d, %d %d #######"%(orbit, num, h))
        logging.info("##### constellation %d, %d %d #######"%(orbit, num, h))
        create_shell(orbit, num, h, incli,"data/fengwo",10)
        right_begin = int(orbit*num*400*0.03)
        right = right_begin
        user_num = right
        re = MFMC_topo(orbit, num, h, user_num, cap, per, ele)
        while re == -1: # 接入容量受限
            user_num = int(right / 2)
            right = user_num
            re = MFMC_topo(orbit, num, h, user_num, cap, per, ele)
            # 当找到下界的时候离开循环

        while re == -2: # 链路受限
            user_num = int(right / 2)
            right = user_num
            re = MFMC_topo(orbit, num, h, user_num, cap, per, ele)
            # 当找到下界的时候离开循环

        if re == 1:
            left = right
            right *= 2
        # 二分查找寻找上界
        while re == 1:
            user_num = right
            re = MFMC_topo(orbit, num, h, user_num, cap, per, ele)
            if re == 1:
                left = user_num
                right = left * 2
        
        # 极值就在left和right 之间
        while (right - left) / left > 0.05:
            mid = int((right + left) / 2)
            user_num = mid
            re = MFMC_topo(orbit, num, h, user_num, cap, per, ele)
            if re == 1:
                left = mid
            else:
                right = mid
        re = MFMC_topo(orbit, num, h, right, cap, per, ele)
        non = "转交不足" if re == -2 else "接入不足"
        log = link_use_rate(cap)
        print("result: "+non+"%d,%d,%d,%d"%(user_num, orbit, num, h)+log)
        logging.info("result: "+non+"%d,%d,%d,%d"%(user_num, orbit, num, h)+log)

def draw_base_conste_with_t(orbit_list, num_list, h_list, incli_list, ele_list, cap_list, user_base):
    per = 0.5
    for i in range(len(orbit_list)):
        cap = cap_list[i]
        orbit = orbit_list[i]
        num = num_list[i]
        h = h_list[i]
        incli = incli_list[i]
        ele = ele_list[i]
        print("##### constellation %d, %d %d #######"%(orbit, num, h))
        logging.info("##### constellation %d, %d %d #######"%(orbit, num, h))
        create_shell(orbit, num, h, incli,"data/fengwo",100,snap=1)
        result = []
        user_num = user_base[i]
        for t in range(0,100,10):
            sate = 'data/fengwo/%d.txt'%t
            re = MFMC_topo(orbit, num, h, user_num, cap, per, ele,sate)
            if re !=1:
                while re != 1:
                    user_num = int(user_num*0.9)
                    re = MFMC_topo(orbit, num, h, user_num, cap, per, ele,sate)
            else:
                while re == 1:
                    user_num = int(user_num*1.1)
                    re = MFMC_topo(orbit, num, h, user_num, cap, per, ele,sate)
                user_num = int(user_num/1.1)
            result.append(user_num)
            log = link_use_rate(cap)
            print("result: %d, %d"%(h, user_num)+log)
            logging.info("result: %d, %d"%(h, user_num)+log)
        user_num = int(sum(result)/len(result))
        logging.info("result: %d, %d"%(h, user_num))
        
def cal_p7():
    '''
    对不同规模的卫星，不断寻找其容量上界。
    '''
    orbit_list = [25+i for i in range(14) ]
    num_list = orbit_list
    h_list = [550 for i in range(14)]
    incli_list = [53 for i in range(14)]
    ele_list = [25 for i in range(14)]
    cap_list = [100 for i in range(14)]
    draw_base_conste(orbit_list, num_list, h_list, incli_list, ele_list, cap_list)
# cal_p7()

def cal_p8():
    '''
    对不同倾角的卫星星座，不断寻找其容量上界。
    '''
    orbit_list = [72 for i in range(8)]
    num_list = [22 for i in range(8)]
    h_list = [550 for i in range(8)]
    incli_list = [i*10-5 for i in range(2, 10)]
    ele_list = [25 for i in range(8)]
    cap_list = [100 for i in range(8)]
    draw_base_conste(orbit_list, num_list, h_list, incli_list, ele_list, cap_list)
# draw_p8()

def cal_p8b():
    '''
    对不同高度的卫星星座。
    '''
    orbit_list = [72 for i in range(16)]
    num_list = [22 for i in range(16)]
    h_list = [2000]*4
    incli_list = [53 for i in range(16)]
    ele_list = [25 for i in range(16)]
    cap_list = [100 for i in range(16)]
    draw_base_conste(orbit_list, num_list, h_list, incli_list, ele_list, cap_list)
# cal_p8b()

def tmp():
    orbit=72
    num=22
    h=2000
    incli=53
    ele=25
    cap=100
    per=0.2
    user_num=66528
    create_shell(orbit, num, h, incli,"data/fengwo",10)
    re = MFMC_topo(orbit, num, h, user_num, cap, per, ele)
    log = link_use_rate(cap)
    print("result: %d,%d,%d,%d"%(user_num, orbit, num, h)+log)
# tmp()

def cal_p8b2():
    '''
    对不同高度的卫星星座。
    '''
    orbit_list = [72 for i in range(8)]
    num_list = [22 for i in range(8)]
    h_list = [i*200 for i in range(3,11)]
    incli_list = [53 for i in range(8)]
    ele_list = [25 for i in range(8)]
    cap_list = [100 for i in range(180)]
    user_base=[22517,35640,40392,52272,61776,52272,61776,71280]
    draw_base_conste_with_t(orbit_list, num_list, h_list, incli_list, ele_list, cap_list, user_base)
# cal_p8b2()

def cal_p8c():
    '''
    对不同仰角的卫星星座。
    '''
    orbit_list = [72 for i in range(7)]
    num_list = [22 for i in range(7)]
    h_list = [550 for i in range(7)]
    incli_list = [53 for i in range(7)]
    ele_list = [5*i for i in range(4,12)]
    cap_list = [100 for i in range(7)]
    draw_base_conste(orbit_list, num_list, h_list, incli_list, ele_list, cap_list)
# cal_p8c()

def draw_p2b2():
    '''
    获取oneweb的接入容量
    '''
    orbit_list = [72]
    num_list = [22]
    incli_list = [53]
    for i in range(len(orbit_list)):
        orbit = orbit_list[i]
        num = num_list[i]
        incli = incli_list[i]
        create_shell(orbit, num, 550, incli,"data/fengwo",4)
        right_begin = int(orbit*num*400*0.03)
        left = 0
        right = right_begin
        user_num = right_begin
        sate = get_sat_loca('data/fengwo/3.txt')
        users= create_each_user(user_num)
        sate_num, user_link = minlin_each(users, sate, 550, mx=400,ele=25)
        while len(sate_num) > 0:
            user_total = right
            users= create_each_user(user_total)
            sate_num, user_link = minlin_each(users, sate, 550, mx=400,ele=25)
            print(user_total, len(sate_num))
            if len(sate_num) > 0:
                left = user_total
                right = left * 2
        
        while (right-left)/left > 0.1:
            user_total = int((left + right)/2)
            users= create_each_user(user_total)
            sate_num, user_link = minlin_each(users, sate, 550, mx=400,ele=25)
            print(user_total, len(sate_num))
            if len(sate_num) == 0: 
                right = user_total
            else:
                left = user_total
        print(orbit, num, left, sum(sate_num)/(orbit*num*400))
# draw_p2b2()

def draw_base(link_cap, trans_per):
    left = 0
    right_begin = int(72*22*400*0.03)
    
    create_shell(72, 22, 550, 53,"data/fengwo",4)
    for cap in link_cap:
        for per in trans_per:
            print('####### %d,%f ######'%(cap,per))
            logging.info('####### %d,%f ######'%(cap,per))
            right = right_begin
            user_num = right
            re = MFMC_topo(72, 22, 550, user_num, cap, per)
            while re == -1: # 接入容量受限
                user_num = int(right / 2)
                right = user_num
                re = MFMC_topo(72, 22, 550, user_num, cap, per)
                # 当找到下界的时候离开循环

            while re == -2: # 链路受限
                user_num = int(right / 2)
                right = user_num
                re = MFMC_topo(72, 22, 550, user_num, cap, per)
                # 当找到下界的时候离开循环

            if re == 1:
                left = right
                right *= 2
            # 二分查找寻找上界
            while re == 1:
                user_num = right
                re = MFMC_topo(72, 22, 550, user_num, cap, per)
                if re == 1:
                    left = user_num
                    right = left * 2
            
            # 极值就在left和right 之间
            while (right - left) / left > 0.1:
                mid = int((right + left) / 2)
                user_num = mid
                re = MFMC_topo(72, 22, 550, user_num, cap, per)
                if re == 1:
                    left = mid
                else:
                    right = mid
            re = MFMC_topo(72, 22, 550, right, cap, per)
            non = "转交不足" if re == -2 else "接入不足"
            print("result: "+non+"%d,%d,%f"%(user_num, cap, per))
            logging.info("result: "+non+"%d,%d,%f"%(user_num, cap, per))
            link_use_rate(cap)

def draw_p9a():
    '''
    随着转交比例的改变，统计整体容量的变化情况
    '''
    link_cap = [50]
    trans_per = [0.1*i for i in range(1,10)]
    draw_base(link_cap, trans_per)

def draw_p9b():
    '''
    修改转交接口的容量，计算总吞吐和利用率
    '''
    link_cap = [i*10 for i in range(10)]
    trans_per = [0.5]
    draw_base(link_cap, trans_per)

def draw_p3():
    '''
    随时间变化的卫星的利用率和吞吐的变化
    '''
    users = create_each_user(10000)
    # create_shell(72, 22, 550, 53,"data/fengwo", 1440, snap = 1)
    for t in [i*60 for i in range(24)]:
        users_t = change_day_night_each(users, t)
        u = len(users_t)  
        cap = 50
        per = 0.2
        re = MFMC_topo_t(72, 22, 550, users_t, cap, per, t)
        if re == 1:
	        print("result: %d,%d"%(t, u), end = ' ')
	        logging.info("result: %d,%d,%f"%(u, cap, per))
	        link_use_rate(cap)

def cal_p4():
    '''
    同一颗卫星长时间的容量利用率变化
    '''
    from time import time
    users = create_each_user(8000)
    # create_shell(34, 34, 630, 51.9,"data/kuiper", 1440*7, snap = 3)
    users_t = users
    for t in range(108,1440*7,3):
        users_t = change_day_night_each(users, t)
        u = len(users_t)      
        cap = 50
        per = 0.2
        re = MFMC_topo_t_store(34, 34, 630, users_t, cap, per, t)
        if re == 1:
            print(time(), "result: %d,%d"%(t, u), end = ' ')
            s = link_use_rate(cap,t)
            logging.info("%f, result:%d,%d,%s"%(time(),t, u, s))
# cal_p4()

def draw_p4(capa=50):
    n = 480
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
    # 备选：86/85
    link1, sate1, used1 = check_sate_i(86)
    capi1 = [(link1[i]+sate1[i]) for i in range(len(link1))]
    link2, sate2, used2 = check_sate_i(85)
    capi2 = [(link2[i]+sate2[i]) for i in range(len(link2))]
    x = [i*3/60 for i in range(n)]
    # print(sum(sate1),sum(sate2))
    # print(sum(link1),sum(link2))
    plt.legend(prop=font_legend)
    plt.tick_params(labelsize=15)
    plt.plot(x, [i/2 for i in link2], label = '$\mathregular{sat_{86}}$',marker='.')
    plt.plot(x, [i/2 for i in link1], label = '$\mathregular{sat_{87}}$',marker='.')
    
    
    plt.xlabel('t/hours', font_xy)
    plt.ylabel('utilization/%', font_xy)
    plt.legend(prop=font_legend)
    plt.show()
# draw_p4()

def draw_p4d_single(capa=50):
    n = 480
    link_list = [] # 链路剩余容量
    sate_list = [] # 单颗卫星接入量
    used_dic_list = [] # 使用过的卫星

    for i in range(480,480+n):
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

    x = [i*3/60 for i in range(n)]
    link1, sate1, used1 = check_sate_i(0)


    # c_link=0
    # c_sate=0
    # for i in link1:
    #     if i >=160 :c_link+=1
    # for i in sate1:
    #     if i >=320:c_sate+=1
    # print(c_link/len(link1),c_sate/len(sate1))
    
    plt.plot(x, [i/4 for i in sate1], label = 'GSL')
    plt.plot(x, [i/2 for i in link1], label = 'ISL')
    plt.xlabel('t/hours',font_xy)
    plt.ylabel('utilization/%',font_xy)
    plt.legend(prop={'size':20})
    plt.tick_params(labelsize=15)
    plt.show()
# draw_p4d_single(capa=50)

def draw_p4d_diff(capa=50):
    '''
    多颗连续卫星一段时间内的容量变化
    '''
    n = 40
    link_list = [] # 链路剩余容量
    sate_list = [] # 单颗卫星接入量
    used_dic_list = [] # 使用过的卫星

    for i in range(700,700+n):
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

    x = [i*3 for i in range(n)]
    link_add=[0] * n
    count=0
    for i in range(120,124):
        count+=1
        link1, sate1, used1 = check_sate_i(i)
        link_add = [link_add[i]+link1[i]/2 for i in range(len(link_add))]
        # capi1 = [(link1[i]+sate1[i]) for i in range(len(link1))]
        # print([i/2 for i in link1])
        plt.plot([i for i in x], [i/2 for i in link1], label = '$\mathregular{sat_{%d}}$'%(i+1),marker='.')
    print(max(link_add))
    # plt.plot(x, [i/2 for i in link2], label = '897')
    # print(link1)
    # print(link2)
    
    
    plt.xlabel('t/minutes',font_xy)
    plt.ylabel('utilization/%',font_xy)
    plt.legend(prop=font_legend)
    plt.tick_params(labelsize=15)
    plt.show()
draw_p4d_diff(capa=50)

def cal_p6():
    '''
    同一星座不同卫星的利用率差异
    '''
    link,sate,used=per_link_use_rate(2, 50)
    capi = [link[i]+sate[i] for i in range(len(sate))]
    used = [i/3309*100 for i in used]

    sate = [i/3309/7 for i in sate]
    link = [i/3309/7 for i in link]
    capi = [i/3309/7 for i in capi]
    x=[i for i in range(1584)]
    print(sate)
    print(link)
    print(capi)
    print(used)
    # plt.plot(x,sate,label='GSL',linewidth=3)
    # plt.plot(x,capi,label='capacity',linewidth=3)
    
    y = []
    for i in range(22):
        for j in range(72):
            y.append(link[j*22+i])
    plt.plot(x,y)
    # plt.legend()
    plt.xlabel('satellite id')
    plt.ylabel('GSL utilization/%')
    plt.show()
# cal_p6()

def draw_p6():
    '''
    画同一个星座中不同卫星的利用率
    '''
    use_list = []
    with open('data/tongji/used.txt', 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip().split(',')
            sate = [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
            use_list.append(sate)
    gsl = [s[0] for s in use_list]
    isl = [s[1] for s in use_list]
    capi = [s[2] for s in use_list]
    used = [s[3] for s in use_list]
    
    # fig, ax1 = plt.subplots()
    plt.tick_params(labelsize=15)
    # ax2 = ax1.twinx()
    plt.scatter([i for i in range(len(gsl))],gsl,marker='.',label='utilization of each sat_ID')
    gsl.sort()
    plt.scatter([i for i in range(len(gsl))],gsl,color='orange',label='utilization distribution',marker='.')
    plt.ylabel('GSL utilization/%',font_xy)
    plt.xlabel('satellite ID/sequence',font_xy)
    plt.legend()
    # ax2.set_ylabel('average utilization/%',font_xy)
    # ax1.set_xlabel('satellite index',font_xy)


    # link = [50]*6336
    # sate = [0]*1584
    # for t in range(480,480*7,3):
    #     with open('data/fengwo/cap_link%s.txt'%str(t), 'r') as f:
    #         data = f.readlines()
    #         for i in range(len(data)):
    #             row = data[i]
    #             b, e, c = map(int, row.split(','))
    #             if c < link[i]: link[i] = c
    #     with open('data/fengwo/cap_sate%s.txt'%str(t), 'r') as f:
    #         data = f.readlines()
    #         for i in range(len(data)):
    #             row = data[i]
    #             c =int(row)
    #             if c > sate[i]: sate[i] = c
    # link = [50-i for i in link]
    
    # # print(link)
    # link.sort()
    # link=[i*2 for i in link]
    # ax1.plot([i/4 for i in range(len(link))],link,label='sorted maximum utilization',color='purple',linewidth=3)
    # ax1.set_ylabel("maximum utilization/%")
    
    # handles1, labels1 = ax1.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    # plt.legend(handles1+handles2, labels1+labels2, loc='upper left')
    plt.tick_params(labelsize=15)
    plt.show()
# draw_p6()

def max_use():
    '''
    画同一个星座中不同卫星的利用率
    '''
    link = [50]*6336
    sate = [0]*1584
    for t in range(480,480*7,3):
        with open('data/fengwo/cap_link%s.txt'%str(t), 'r') as f:
            data = f.readlines()
            for i in range(len(data)):
                row = data[i]
                b, e, c = map(int, row.split(','))
                if c < link[i]: link[i] = c
        with open('data/fengwo/cap_sate%s.txt'%str(t), 'r') as f:
            data = f.readlines()
            for i in range(len(data)):
                row = data[i]
                c =int(row)
                if c > sate[i]: sate[i] = c
    link = [50-i for i in link]
    
    link.sort()
    link=[i*2 for i in link]
    fig=plt.figure(figsize=(8,4))
    plt.plot([i/4 for i in range(len(link))],link,label='ISL',linewidth=3)
    plt.ylabel("maximum utilization/%")
    plt.xlabel('index')
    sate.sort()
    sate=[i/4 for i in sate]
    plt.plot([i for i in range(len(sate))],sate,label='GSL',linewidth=3)
    plt.legend()
    plt.show()
# max_use()

def draw_p6i():
    '''
    画同一个星座中不同卫星的利用率,卫星index方式进行修改
    '''
    use_list = []
    with open('data/tongji/used.txt', 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip().split(',')
            sate = [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
            use_list.append(sate)
    # gsl = [s[0] for s in use_list]
    isl = [s[1] for s in use_list]
    # capi = [s[2] for s in use_list]
    # used = [s[3] for s in use_list]
    y = []
    for i in range(22):
        for j in range(72):
            y.append(isl[j*22+i])
    print(y)
    plt.plot([i for i in range(len(isl))],y,marker='.')
    # used.sort()
    # plt.plot(used,linewidth=3,color='orange')
    plt.xlabel('satellite id')
    plt.ylabel('ISL utilization/%')
    plt.show()
# draw_p6i()

def draw_p6e():
    '''
    不同区域之间的利用率差异性
    '''
    link, sate,used = per_link_use_rate(3, 50)
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    x = np.linspace(-180, 180, 360)
    y = np.linspace(-90, 90, 180)
    print(max([max(i) for i in link]),sum([sum(i) for i in link])/len(link)/len(link))
    print(max([max(i) for i in sate]),sum([sum(i) for i in sate])/len(link)/len(link))
    # print(max(max(used)),sum([sum(i) for i in used])/len(link)/len(link))
    Z = [[sate[i][j]/2 for j in range(len(used[i]))] for i in range(len(used))]
    map = Basemap()
    map.drawcoastlines()
    map.pcolormesh(x, y, Z, cmap='hot_r', shading='auto')
    map.colorbar()

    # plt.show()
# draw_p6e()

def draw_p7():
    '''
    卫星规模变化
    '''
    import numpy as np
    sate_scale = [str(i) for i in [575,675,783,899,1023,1155,1295,1443,1599,1763,1935,2115,2303,2499,2703,2915]]
    throughput = [560.6,556.85,557.85,640.5,843.95,952.85,1068.35,1190.45,1319.15,1454.45,1596.35,1506.9,1640.85,1593.1,1723.15,1639.65]
    GSL = [4.4791,4.1056,3.7344,3.5523,3.741,4.1095,4.1114,3.7379,3.7376,4.11287578,3.737726098,3.740425532,3.36658706,3.179671869,2.991305956,2.806603774]
    ISL = [12.03955,11.6737,11.1523,11.6894,13.15495,15.3829,16.2979,15.73875,16.5336,19.35564379,17.91098191,19.10756501,17.87364307,17.47679072,17.03079911,16.87349914]
    capacity = [8.259325,7.88965,7.44335,7.62085,8.447975,9.7462,10.20465,9.738325,10.1356,11.73425978,10.82435401,11.42399527,10.62011507,10.32823129,10.01105253,9.840051458]
    fig, ax1 = plt.subplots(figsize=(8,4))
    plt.tick_params(labelsize=14)
    ax2 = ax1.twinx()
    plt.tick_params(labelsize=14)
    ax2.plot(sate_scale, throughput, label='throughput',marker='o',color='darkviolet',linewidth=3)
    ax2.set_ylabel("throuput/Gbps",font_xy)
    ax1.set_ylabel("utilization/%",font_xy)
    bar_width = 0.3
    x = np.arange(len(sate_scale))
    ax1.bar(x-bar_width, GSL, bar_width,label='GSL')
    ax1.bar(x, ISL,bar_width, label='ISL')
    ax1.bar(x+bar_width, capacity,bar_width, label='link')
    ax1.set_xlabel('scale of constellations', font_xy)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    plt.legend(handles1+handles2, labels1+labels2, loc='upper left',prop=font_legend)
    plt.show()
# draw_p7()

def draw_p8():
    '''
    卫星倾角比例变化
    '''
    import numpy as np
    sate_scale = [str(i) for i in [15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]]
    throughput = [504.9,504.9,326.7,445.5,653.4,1306.8,1544.4,1306.8,1128.6,1306.8,1009.8,1128.6,1128.6,1009.8,891,891]
    single = [76.51515152,77.0202,67.61363636,76.8939,75.82070707,85.41666667,84.97474747,83.33333333,77.96717172,76.83080808,76.07323232,73.48484848,72.34848485,71.71717172,66.85606061,71.1489899]
    GSL = [0.563604798,0.8193,0.638888889,1.0631,1.830176768,3.836963384,4.366950758,4.101325758,3.558869949,3.748895202,3.374684343,3.561079545,3.374368687,2.999526515,2.811395202,2.811079545]
    ISL = [1.429924242,1.77825,1.278882576,2.18765,3.800662879,8.031881313,9.333491162,8.925031566,7.706755051,8.298768939,7.495580808,8.11489899,7.555397727,6.866003788,6.274463384,6.406565657]
    capacity = [0.99676452,1.298775,0.958885732,1.625375,2.815419823,5.934422348,6.85022096,6.513178662,5.6328125,6.023832071,5.435132576,5.837989268,5.464883207,4.932765152,4.542929293,4.608822601]
    fig, ax1 = plt.subplots()
    plt.tick_params(labelsize=15)
    ax2 = ax1.twinx()
    ax2.plot(sate_scale, throughput, label='throughput',marker='o',color='darkviolet',linewidth=3)
    ax2.set_ylabel("throuput/Gbps",font_xy)
    ax1.set_ylabel("utilization/%",font_xy)
    bar_width = 0.3
    x = np.arange(len(sate_scale))
    ax1.bar(x-bar_width, GSL, bar_width,label='GSL')
    ax1.bar(x, ISL,bar_width, label='ISL')
    ax1.bar(x+bar_width, capacity,bar_width, label='link')
    ax1.set_xlabel('inclination',font_xy)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.tick_params(labelsize=15)
    plt.legend(handles1+handles2, labels1+labels2, loc='upper left',prop=font_legend)
    plt.show()
# draw_p8()

def draw_p8b():
    '''
    卫星高度的变化
    '''
    import numpy as np
    sate_scale = [str(int(i/100)) for i in [200,400,600,800,1000,1200,1400,1600,1800,2000]]
    throughput = [772.2,1128.6,1544.4,1782,2019.6,2613.6,3088.8,3088.8,3088.8,3564]
    single = [71.78030303,77.27272727,79.86111111,81.94444444,83.01767677,87.12121212,87.18434343,83.14393939,85.16414141,89.07828283]
    GSL = [1.844065657,3.355587121,4.866477273,5.245265152,6.37042298,8.24542298,9.74542298,9.744002525,9.743844697,10.49479167]
    ISL = [4.112847222,7.229324495,10.55965909,11.37452652,13.97048611,17.86174242,21.20438763,21.55003157,21.70407197,22.81613005]
    capacity = [2.978456439,5.292455808,7.713068182,8.309895833,10.17045455,13.0535827,15.4749053,15.64701705,15.72395833,16.65546086]
    fig, ax1 = plt.subplots()
    plt.tick_params(labelsize=15)
    ax2 = ax1.twinx()
    plt.tick_params(labelsize=15)
    ax2.plot(sate_scale, throughput, label='throughput',marker='o',color='darkviolet',linewidth=3)
    ax2.set_ylabel("throuput/Gbps",font_xy)
    ax1.set_ylabel("utilization/%",font_xy)
    bar_width = 0.3
    x = np.arange(len(sate_scale))
    ax1.bar(x-bar_width, GSL, bar_width,label='GSL')
    ax1.bar(x, ISL,bar_width, label='ISL')
    ax1.bar(x+bar_width, capacity,bar_width, label='link')
    ax1.set_xlabel('altitude/x100km',font_xy)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1+handles2, labels1+labels2, loc='upper left',prop=font_legend)
    # plt.savefig('fig/p8b.png')
    # plt.savefig('fig/p8b.eps')
    plt.show()
# draw_p8b()

def draw_p8c():
    '''
    用户仰角变化
    '''
    import numpy as np
    sate_scale = [str(i) for i in [0,5,10,15,20,25,30,35,40,45,50,55]]
    throughput = [3564,3564,2613.6,2019.6,1544.4,1306.8,1306.8,1009.8,772.2,653.4,653.4,504.9]
    single = [90.4040404,89.52020202,83.77525253,77.71464646,83.33333333,80.61868687,77.02020202,78.78787879,79.29292929,74.55808081,71.96969697,65.46717172]
    GSL = [10.49400253,10.4938447,7.494633838,5.99447601,4.869002525,4.115372475,3.735164141,3.356376263,2.228377525,1.945391414,1.781407828,1.158933081]
    ISL = [22.91272096,23.21875,16.44839015,13.00236742,10.50820707,9.038983586,8.037405303,7.275883838,4.871843434,4.252051768,3.905934343,2.586332071]
    capacity = [16.70336174,16.85629735,11.97151199,9.498421717,7.688604798,6.57717803,5.886284722,5.316130051,3.55011048,3.098721591,2.843671086,1.872632576]
    fig, ax1 = plt.subplots()
    plt.tick_params(labelsize=15)
    ax2 = ax1.twinx()
    plt.tick_params(labelsize=15)
    ax2.plot(sate_scale, throughput, label='throughput',marker='o',color='darkviolet',linewidth=3)
    ax2.set_ylabel("throuput/Gbps",font_xy)
    ax1.set_ylabel("utilization/%",font_xy)
    bar_width = 0.3
    x = np.arange(len(sate_scale))
    ax1.bar(x-bar_width, GSL, bar_width,label='GSL')
    ax1.bar(x, ISL,bar_width, label='ISL')
    ax1.bar(x+bar_width, capacity,bar_width, label='link')
    ax1.set_xlabel('elevation',font_xy)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1+handles2, labels1+labels2, loc='upper right',prop=font_legend)
    plt.show()
# draw_p8c()

def draw_p9():
    '''
    卫星转交比例的变化
    '''
    import numpy as np
    sate_scale = [str(i) for i in [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]]
    throughput = [1306.8,1306.8,1306.8,1009.8,772.2,653.4,653.4,504.9,445.5,386.1,282.15,326.7,282.15,282.15,252.45,222.75,193.05,193.05,163.35,163.35,163.35]
    single = [0,71.40151515,75.56818182,77.27272727,75.56818182,79.22979798,79.10353535,81.69191919,82.57575758,78.03030303,73.86363636,78.21969697,76.19949495,79.04040404,76.89393939,74.24242424,78.40909091,76.95707071,74.81060606,70.89646465,78.3459596]
    GSL = [4.113005051,4.113005051,4.117108586,3.365372475,2.61852904,2.05792298,1.870265152,1.590909091,1.401357323,1.122001263,0.935921717,0.935921717,0.888415404,0.841382576,0.748106061,0.702020202,0.654829545,0.560132576,0.468118687,0.561079545,0.514046717]
    ISL = [0,8.633838384,17.72348485,21.61805556,23.0719697,22.13825758,24.69823232,24.39267677,25.34343434,22.30239899,20.24116162,22.31755051,23.00378788,23.95770202,22.56691919,23.07575758,23.31881313,20.50441919,18.89962121,23.51704545,22.41098485]
    capacity = [4.113005051,5.017171717,6.838383838,7.015909091,6.709217172,6.073989899,6.435858586,6.151262626,6.189772727,5.358080808,4.796969697,5.212247475,5.311489899,5.464646465,5.111868687,5.176767677,5.187626263,4.548989899,4.154419192,5.152272727,4.893434343]
    fig, ax1 = plt.subplots()
    
    ax2 = ax1.twinx()
    ax2.plot(sate_scale, throughput, label='throughput',marker='o',color='darkviolet',linewidth=3)
    ax2.set_ylabel("throuput/Gbps")
    ax1.set_ylabel("utilization/%")
    bar_width = 0.3
    x = np.arange(len(sate_scale))
    ax1.bar(x-bar_width, GSL, bar_width,label='GSL')
    ax1.bar(x, ISL,bar_width, label='ISL')
    ax1.bar(x+bar_width, capacity,bar_width, label='link')
    ax1.set_xlabel('Proportion of remote referrals')
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1+handles2, labels1+labels2, loc='upper right')
    plt.show()
# draw_p9()

def draw_p9b():
    '''
    卫星转交容量变化
    '''
    import numpy as np
    sate_scale = [str(i) for i in [0.5,1.25,2.5,5,10,20,40]]
    throughput = [81.65,252.45,386.1,891,1306.8,1306.8,1306.8]
    single = [65.1515,77.9672,82.2601,86.8687,90.5934,89.7727,88.5101]
    GSL = [0.2339,0.7486,1.3089,2.6185,3.7397,3.7405,3.7405]
    ISL = [19.4176,25.2784,21.61395,21.4582,15.20645,7.5947,3.7864]
    capacity = [1.977872727,5.65456,8.07725,12.03835,11.3842,6.82386,3.7813]
    fig, ax1 = plt.subplots()
    
    ax2 = ax1.twinx()
    ax2.plot(sate_scale, throughput, label='throughput',marker='o',color='darkviolet',linewidth=3)
    ax2.set_ylabel("throuput/Gbps")
    ax1.set_ylabel("utilization/%")
    bar_width = 0.3
    x = np.arange(len(sate_scale))
    ax1.bar(x-bar_width, GSL, bar_width,label='GSL')
    ax1.bar(x, ISL,bar_width, label='ISL')
    ax1.bar(x+bar_width, capacity,bar_width, label='link')
    ax1.set_xlabel('ISL capacity/Gbps')
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1+handles2, labels1+labels2, loc='upper right')
    plt.show()
# draw_p9b()

def draw_time():
    '''
    画出星座在0时刻所有卫星的local time，以及所有卫星在赤道升交点的时候的local time
    '''
    a = get_sat_loca('data/tmp/0.txt')
    zero_lon = 0
    re = []
    for sateid in range(len(a)):
        lon = a[sateid][2]
        rela = lon_relative(0, lon)
        rela_time = 24/360*rela
        re.append(rela_time)
        # # 下面是统计前12个小时的时间
        # orbit = int(sateid/22)
        # if orbit % 2 == 0:
        #     begin_time = rela_time - 24/22*(sateid%22)
        #     if begin_time < 0:
        #         begin_time+=24
        # else:
        #     begin_time = rela_time - 24/22*(sateid%22+0.5)
        #     if begin_time < 0:
        #         begin_time+=24
        # re.append(begin_time)
    
    # plt.scatter([i for i in range(len(re))], re)

    # 下面是统计每一组的时间
    y = []
    for i in range(22):
        for j in range(72):
            y.append(re[j*22+i])

    plt.plot([x/72 for x in range(len(y))],y, marker='.')
    # re.sort()
    # plt.plot(re)
    plt.xlabel('group')
    plt.ylabel('initial local time')
    plt.show()
# draw_time()

def draw_active():
    '''
    画出不同时间的活跃度，以及不同时间进入北半球的话在北半球的平均活跃度
    '''
    active = [0.442622951,0.327868852,0.262295082,0.229508197,0.237704918,0.295081967,
        0.442622951,0.737704918,0.852459016,0.836065574,0.819672131,0.836065574,0.836065574,
        0.836065574,0.868852459,0.885245902,0.901639344,0.983606557,0.983606557,1,
        1,0.983606557,0.852459016,0.655737705]
    p=[]
    for i in range(24):
        re = 0
        for j in range(i, i-12, -1):
            re += active[j]
        p.append(re/12)
    plt.bar([i for i in range(24)], [i*100 for i in p])
    plt.xlabel('t/h')
    plt.ylabel('activity/%')
    plt.show()
# draw_active()

def draw_time_one_sate():
    '''
    计算一个轨道周期内卫星经过区域的本地时间
    '''
    print(local_time_after_t(120, 1))
    t=[]
    lat=[]
    time1=[]
    time2=[]
    for i in range(0,180):
        t.append(i)
        lat1,t1=change_local_time_with(0,i,raan=29.35,incli=53)
        lat2,t2=change_local_time_with(0,i,raan=64.95,incli=53)
        lat.append(lat1)
        time1.append(t1)
        time2.append(t2)
    print(time1[0],time2[0])
    fig, ax1 = plt.subplots()
    plt.tick_params(labelsize=15)
    ax2 = ax1.twinx()
    plt.tick_params(labelsize=15)
    ax2.plot(t, lat, label='latitude',linestyle=':',linewidth=3,color='g')
    ax2.set_ylabel("latitude",font_xy)
    ax1.set_ylabel("local time",font_xy)

    ax1.plot(t, time1, label='s1',linewidth=3)
    ax1.plot(t, time2, label='s2',linewidth=3)
    ax1.set_xlabel('run time/minutes',font_xy)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1+handles2, labels1+labels2, loc='upper left',prop=font_legend)
    # plt.savefig('fig/time_one_sate.png')
    # plt.savefig('fig/time_one_sate.eps')
    plt.show()
# draw_time_one_sate()

def draw_use_change():
    '''
    卫星星座在7天内随时间变化的卫星利用率
    '''
    use_list = []
    with open('data/tongji/use_change.txt', 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip().split(',')
            sate = [float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])]
            use_list.append(sate)
    t = [s[0] for s in use_list]
    tp = [s[1] for s in use_list]
    used = [s[2] for s in use_list]
    gsl = [s[3] for s in use_list]
    isl = [s[4] for s in use_list]
    capi = [s[5] for s in use_list]
    fig, ax1 = plt.subplots()
    plt.tick_params(labelsize=15)
    ax2 = ax1.twinx()
    plt.tick_params(labelsize=15)
    ax2.plot(t, tp, label='throughput',linewidth=1,color='r')
    ax2.set_ylabel("throughput/Gbps",font_xy)
    ax2.set_ylim([0,500])
    ax1.set_ylabel("utilization/%",font_xy)
    ax1.set_ylim([0,10])
    ax1.plot(t, gsl, label='GSL',linewidth=1)
    ax1.plot(t, isl, label='ISL',linewidth=1)
    ax1.plot(t, capi, label='link',linewidth=1)
    ax1.set_xlabel('t/day',font_xy)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1+handles2, labels1+labels2, loc='upper left',prop={'size':17})
    plt.show()
# draw_use_change()

def draw_traffic(n):
    '''
    画出不同经度下的人口和卫星容量
    '''
    use_list = []
    with open('data/tongji/traffic%d.txt'%n, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip().split(',')
            if n == 1:
                sate = [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
            else:
                sate = [float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])]
            use_list.append(sate)
    lat = [s[0] for s in use_list]
    capi = [s[1] for s in use_list]
    traffic = [s[2] for s in use_list]
    idle = [s[3] for s in use_list]
    if n != 1:
        lost = [s[4] for s in use_list]
        plt.fill_between(lat,lost,label='block traffic')
    plt.fill_between(lat,capi,label='satellite capacity')
    plt.fill_between(lat,traffic, label='user traffic')
    plt.fill_between(lat,idle,label='idle capacity')
    plt.tick_params(labelsize=15)
    plt.xlabel('latitude',font_xy)
    plt.ylabel('traffic/Gbps',font_xy)
    plt.legend(loc='lower right',prop=font_legend)
    plt.show()
# draw_traffic(1)

def draw_p3():
    '''
    画不同星座的利用率情况
    '''
    import numpy as np
    conste = ['Telesat', 'OneWeb', 'Starlink', 'Kuiper']
    # scale = [1698,6372,3808,3236]
    tp = [7088,7605,3072,4575]
    single = [85.71622724,31.46578782,80.12663399,79.91347342]
    gsl = [11.49045257,5.968220339,3.95847193,2.633266378]
    isl = [34.44919362,0,16.1713352,7.573393078]
    capi = [19.14336625,5.968220339,8.029426354,4.279975278]
    cove = [1.315789474,0.884955752,4.166666667,9.259259259]
    
    fig, ax1 = plt.subplots(figsize=(10,3))
    plt.tick_params(labelsize=15)
    # ax2 = ax1.twinx()
    plt.tick_params(labelsize=15)
    # ax2.plot(conste, scale, label = 'scale',linewidth=2,marker='.')
    
    # ax2.set_ylabel("throuput/Gbps",font_xy)
    ax1.set_ylabel("utilization/%",font_xy)
    bar_width = 0.15
    x = np.arange(len(conste))
    ax1.bar(x-2.5*bar_width, single, bar_width,label='used')
    ax1.bar(x-1.5*bar_width, gsl,bar_width, label='GSL')
    ax1.bar(x-0.5*bar_width, isl,bar_width, label='ISL')
    ax1.bar(x+0.5*bar_width, capi,bar_width, label='link')
    ax1.bar(x+1.5*bar_width, cove,bar_width, label='coverage')
    # ax2.bar(x+2.5*bar_width, tp, bar_width, label='throughput',color='w',edgecolor='black')
    # ax1.set_xlabel('ISL capacity/Gbps')
    handles1, labels1 = ax1.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    plt.xticks(x,conste)
    plt.legend(handles1, labels1, loc='upper right',prop={'size':13})
    plt.show()
# draw_p3()

def draw_same_access():
    '''
    对同shell的卫星进行接入统计，获得不同shell卫星的接入差异。
    '''
    cycle = 15
    h = get_same_point_h(cycle)
    # h = 523
    incli = 53
    users = create_each_user(1000)
    r, d = beta2r(h,25)
    re = []
    re_total = []
    raan_range = int(360/cycle)
    for raan in range(360):
        print(raan)
        create_onenode(h, 1426, incli = incli, snap = 1, raan =raan, maan = 0, file = 60)
        sate_list = get_sat_loca('data/one_node%d/60.txt'%h)
        max_user = 0
        total_user = 0
        for sate in sate_list:
            user_total = 0
            for user in users:
                if abs(user[0] - sate[1]) < r/100 and abs(user[1] - sate[2]) < 2*r/100:
                    d2 = latlon2d(user[0],sate[1],user[1],sate[2],6371,6371+h)
                    if d2<d: user_total += 1
            if user_total > max_user: max_user=user_total
            total_user += user_total
        re.append(max_user)
        re_total.append(total_user/len(sate_list))
    re_max = max(re)
    re = [i/re_max*100 for i in re]
    # re_total = [i/re_max for i in re_total]
    print(re)
    plt.xlabel('RAAN')
    plt.ylabel('GSL maximum utilization/%')
    plt.plot(re,label='maximum utilization/%')
    # plt.plot(re_total,label='average utilization')
    # plt.legend()
    plt.show()
# draw_same_access()

def draw_base_conste_given(sate, h):
    left = 0
    right = int(len(sate)*400*0.03)
    users = create_each_user(right)
    sate_num, user_link = minlin_each(users, sate, h, mx=400)
    while len(sate_num) == 0:
        user_num = int(right / 2)
        right = user_num
        users = create_each_user(right)
        sate_num, user_link = minlin_each(users, sate, h, mx=400)
        # 当找到下界的时候离开循环

    if len(sate_num) == 0:
        left = right
        right *= 2
    # 二分查找寻找上界
    while len(sate_num) == 0:
        user_num = right
        users = create_each_user(right)
        sate_num, user_link = minlin_each(users, sate, h, mx=400)
        if len(sate_num) == 1:
            left = user_num
            right = left * 2
        
    # 极值就在left和right 之间
    while (right - left) / right > 0.05:
        mid = int((right + left) / 2)
        user_num = mid
        users = create_each_user(user_num)
        sate_num, user_link = minlin_each(users, sate, h, mx=400)
        # print(user_num, len(sate_num))
        if len(sate_num) != 0:
            left = mid
        else:
            right = mid
    return user_num

def draw_same_access_use():
    '''
    同shell形成多个r初始raan不同的星座，获取这些不同星座的最大利用率
    首先给出每个卫星的raan和maan生成初始时刻的卫星分布，然后计算最大利用率
    '''
    n = 66
    cycle = 13
    h = get_same_point_h(cycle)
    T = 24*60 / cycle
    incli = 53
    for rainit in range(int(360/cycle)):
        # 生成对应于初始raan的星座
        raan_list = []
        maan_list = []
        for i in range(n * cycle):
            maan_list.append(-int(i*360/n))
            raan_list.append(360/n*T*i/1440+rainit)
        create_shell_init(raan_list, maan_list, h, incli, 10)
        sate_list = get_sat_loca('data/fengwo/0.txt')

        # 对该星座寻找适宜的用户数
        sate_num = draw_base_conste_given(sate_list, h)
        print("result",rainit, sate_num)
# draw_same_access_use()

def draw_space():
    '''
    画出一颗卫星一个周期内对不同区域的服务差异度
    '''
    h=523
    draw_onenode(h,60)

def draw_time_change():
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure(figsize=(12,6))
    map = Basemap()
    map.drawcoastlines()
    map.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1])
    map.drawmeridians(np.arange(-180,180,40),labels=[1,1,0,1])
    lis = [304,305,339,373,407,441,475,509,543,577,611,645]
    plt2=[]
    c_list=['b','g','r','c','m','y','k','orange','purple','navy']
    def drawMap(sate_num, fold,start_nodes,end_nodes,capacity,users,cur_min,c):
        sats = {}
        sate_lat = [0] *len(lis)
        satPosFile = '%s/%d.txt'%(fold, cur_min)
        with open(satPosFile, 'r') as f:
            lines = f.readlines()
            for line in lines:
                nums = line.split(',')
                sat = {}
                sats[int(nums[0])] = (float(nums[2]), float(nums[3]))
        re = []
        for i in range(len(capacity)):
            if capacity[i] <= 20: c = 'r'
            elif capacity[i] <= 30: c = 'orange'
            elif capacity[i] <=40: c = 'y'
            if capacity[i] <50:
                re.append([start_nodes[i],end_nodes[i]])
                
                lat1 = sats[start_nodes[i]][0]
                lon1 = sats[start_nodes[i]][1]
                for l in range(len(lis)):
                    if start_nodes[i] == lis[l]:
                        sate_lat[l] = [lat1,lon1]
                # print(end_nodes[i])
                lat2 = sats[end_nodes[i]][0]
                lon2 = sats[end_nodes[i]][1]
                if (lon1 < -150 and lon2 > 150 ) or (lon2 < -150 and lon1 > 150 ):
                    if lon1 > lon2:
                        lon1,lon2=lon2,lon1
                        lat1,lat2=lat2,lat1
                    map.plot([lon1,-180],[lat1,(lat2+lat1)/2],linewidth=2,color=c,latlon='True',marker='.')
                    map.plot([180,lon2],[(lat1+lat2)/2,lat2],linewidth=2,color=c,latlon='True',marker='.')
                else:
                    map.plot([lon1,lon2],[lat1,lat2],linewidth=2,color=c,latlon='True',marker='.')
                    if t == 135:x = 2
                    elif t== 138:x=0
                    else:x=-6
                    plt.text(lon2, lat2+x, end_nodes[i], ha='center', va= 'bottom',fontsize=10,color=c)
        plt2.append(sate_lat)
        # print(re)

    def MFMC_topo_t_store(orbit, num, h, users, link_cap, trans_per, t):
        start_nodes, end_nodes, capacity, cost, sate = create_link(orbit, num, h, "data/kuiper", link_cap)
        
        # 记录有多少连接是星间链接
        sate_num = orbit * num
        link_dic = {}
        user_num = len(users)

        sate = get_sat_loca('data/kuiper/%d.txt'%t)
        for i in range(len(start_nodes)):
            start_nodes[i] -= 1
            end_nodes[i] -= 1
        
        # 获得每个用户连接到哪颗卫星上。
        sate_num, user_link = minlin_each(users, sate, h, mx=400)
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
                return -2
                # break
            supply[k[0]] = 0
            supply[k[1]] = 0
        drawMap(len(sate_num), "data/kuiper",start_nodes,end_nodes,capacity,users,cur_min=t,c=c_list[int((t-135)/3)])
    # users_t = [[-15.47,-47.56],[28,77]] # 巴西利亚-新德里
    # users_t = [[-15.47,-47.56],[9,7]] # 巴西利亚-尼日利亚
    # users_t = [[38,-77],[51,0]] # 华盛顿-伦敦
    # users_t = [[51,0],[39,116]] # 伦敦-北京
    # users_t = [[39,116],[38,-77]] # 北京-华盛顿
    # users_t = [[38,-77],[9,7]] # 华盛顿-尼日利亚
    # users_t = [[38,-77],[28,77]] # 华盛顿-新德里
    # users_t = [[38,-77],[8,102]] # 华盛顿-河内
    users_t = [[38,-77],[19,72]] # 华盛顿-孟买
    for t in range(135,142,3):
        u = len(users_t)      
        cap = 50
        per = 1
        re = MFMC_topo_t_store(34, 34, 630, users_t, cap, per, t)

    # print(plt2)
    for i in range(len(lis)):
        lon=[]
        lat=[]
        for j in range(1, len(plt2)):
            y0=plt2[j-1][i][0]
            x0=plt2[j-1][i][1]
            y1=plt2[j][i][0]
            x1=plt2[j][i][1]
            plt.arrow(x0,y0,x1-x0,y1-y0,length_includes_head=True,width=0.2,head_length=4,head_width=3,overhang=0.4,color='y',linestyle=':')
    plt.show()
# draw_time_change()

def draw_topo():
    MFMC_topo_t_store(34,34,630,[],50,1,0)
    plt.show()
# draw_topo()

def cal_mim_capacity():
    '''
    计算链路的最大使用看最优能减少多少卫星
    '''
    link = [50]*6336
    sate = [0]*1584
    for t in range(480,480*7,3):
        with open('data/fengwo/cap_link%s.txt'%str(t), 'r') as f:
            data = f.readlines()
            for i in range(len(data)):
                row = data[i]
                b, e, c = map(int, row.split(','))
                if c < link[i]: link[i] = c
        with open('data/fengwo/cap_sate%s.txt'%str(t), 'r') as f:
            data = f.readlines()
            for i in range(len(data)):
                row = data[i]
                c =int(row)
                if c > sate[i]: sate[i] = c
    link = [50-i for i in link]
    
    # print(link)
    sate.sort()
    link.sort()
    plt.scatter([i for i in range(len(sate))],[i/4 for i in sate],label='GSL')
    plt.scatter([i/4 for i in range(len(link))],[i*2 for i in link],label='ISL')
    plt.xlabel('satellite index')
    plt.ylabel('maximum utilization/%')
    plt.legend()
    plt.show()
    print('link',sum(link)/6336/50)
    print('sate',sum(sate)/1584/400)
# cal_mim_capacity()

def time_space_diff():
    '''
    两颗卫星的轨迹
    '''
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    ra1=29.3
    ra2=0
    plt.tick_params(labelsize=15)
    map = Basemap()
    map.drawcoastlines()
    map.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1])
    map.drawmeridians(np.arange(-180,180,30),labels=[1,1,0,1])
    # ax2 = ax1.twinx()
    # plt.ylabel('GSL utilization/%',font_xy)
    # plt.xlabel('satellite index',font_xy)
    lat1, lon1 = lonca(0, ra1, 0, 53, 550)
    lat2, lon2 = lonca(0, ra2, 0, 53, 550)
    lat1_lst=[lat1*180/pi]
    lon1_lst=[lon1*180/pi]
    lat2_lst=[lat2*180/pi]
    lon2_lst=[lon2*180/pi]
    plt.text(lon1_lst[-1], lat1_lst[-1]+5, 16, ha='center', va= 'bottom',fontsize=15,color='orange')
    plt.text(lon2_lst[-1], lat1_lst[-1]+5, 0, ha='center', va= 'bottom',fontsize=15,color='blue')
    for i in range(1,96):
        lat1, lon1 = lonca(i, ra1, 0, 53, 550)
        lat2, lon2 = lonca(i, ra2, 0, 53, 550)
        if i == 24:
            plt.text(lon1_lst[-1], lat1_lst[-1]+5, 22, ha='center', va= 'bottom',fontsize=15,color='orange')
            plt.text(lon2_lst[-1], lat1_lst[-1]+5, 6, ha='center', va= 'bottom',fontsize=15,color='blue')
        elif i == 48:
            plt.text(lon1_lst[-1], lat1_lst[-1]+5, 4, ha='center', va= 'bottom',fontsize=15,color='orange')
            plt.text(lon2_lst[-1], lat1_lst[-1]+5, 12, ha='center', va= 'bottom',fontsize=15,color='blue')
        elif i == 72:
            plt.text(lon1_lst[-1], lat1_lst[-1]+5, 10, ha='center', va= 'bottom',fontsize=15,color='orange')
            plt.text(lon2_lst[-1], lat1_lst[-1]+5, 18, ha='center', va= 'bottom',fontsize=15,color='blue')
        elif i == 95:
            plt.text(lon1_lst[-1], lat1_lst[-1]+5, 16, ha='center', va= 'bottom',fontsize=15,color='orange')
            plt.text(lon2_lst[-1], lat1_lst[-1]+5, 0, ha='center', va= 'bottom',fontsize=15,color='blue')
        if lon1*180/pi < lon1_lst[-1]:
            plt.plot(lon1_lst, lat1_lst, color='orange',lw=3)
            lon1_lst=[]
            lat1_lst=[]
        if lon2*180/pi < lon2_lst[-1]:
            plt.plot(lon2_lst, lat2_lst, color='blue',lw=3)
            lon2_lst=[]
            lat2_lst=[]
        lat1_lst.append(lat1*180/pi)
        lon1_lst.append(lon1*180/pi)
        lat2_lst.append(lat2*180/pi)
        lon2_lst.append(lon2*180/pi)
    plt.plot(lon2_lst, lat2_lst, label='s1', color='b',lw=3)
    plt.plot(lon1_lst, lat1_lst, label='s2',color='orange',lw=3)
    
    plt.text(-200, 90, 'Initial\ntime', ha='center', va= 'bottom',fontsize=10)
    for i in range(-180, 181,30):
        t = int(((i+180)/15+12)%24)
        plt.text(i, 90, '%d:00'%t, ha='center', va= 'bottom',fontsize=10)
    plt.legend(loc='upper left',prop=font_legend)
    plt.show()
# time_space_diff()

def cal_use_per():
    '''
    计算超过百分比或者低于百分比的利用率的时间的比例
    '''
    c_link=0
    c_total_link=0
    c_sate=0
    c_total_sate=0
    for t in range (480, 480*7, 3):
        with open('data/fengwo/cap_link%d.txt'%t, 'r') as f:
            data = f.readlines()
            for row in data:
                b, e, c = map(int, row.split(','))
                if c < 20: c_link += 1
                c_total_link+=1
        
        with open('data/fengwo/cap_sate%d.txt'%t, 'r') as f:
            data = f.readlines()
            for row in data:
                c = int(row)
                if c > 240: c_sate += 1
                c_total_sate+=1
    print(c_link/c_total_link, c_sate/c_total_sate)
# cal_use_per()


def cal_use_max():
    '''
    计算卫星每个链路的最大使用
    '''
    link = [50] * 1584 * 4
    link_total = [0] * 1584 *4
    for t in range (480, 480*6, 3):
        with open('data/fengwo/cap_link%d.txt'%t, 'r') as f:
            data = f.readlines()
            for i in range(len(data)):
                b, e, c = map(int, data[i].split(','))
                if (50-c) > link[i]:
                    link[i]= 50 - c
                link_total[i] += 50 - c
    link_total=[i/800 for i in link_total]
    print(sum(link_total)/sum(link))
# cal_use_max()