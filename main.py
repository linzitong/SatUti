from constellation import *
from population import *
from link import *
from use_rate import *

def stat_user_each_time_each_sate():
    sate_loca = create_same_point_t(12,n=10,t=1440)
    h=get_same_point_h(10)
    users = create_each_user(3000,folder='data/population180.360.2.csv')
    num = []

    for minu in range(0, 1440):
        t=minu*60
        user_t = change_day_night_each(users, t)
        # print(minu)
        conste = [
            [0,sate_loca[0][minu][0], sate_loca[0][minu][1]], 
            [0,sate_loca[int(len(sate_loca)/4)][minu][0],sate_loca[int(len(sate_loca)/4)][minu][1]],
            [0,sate_loca[int(len(sate_loca)/4*2)][minu][0],sate_loca[int(len(sate_loca)/4*2)][minu][1]],
            [0,sate_loca[int(len(sate_loca)/4*3)][minu][0],sate_loca[int(len(sate_loca)/4*3)][minu][1]],
        ]
        sate_num = []
        for c in conste:
            sate_nu, block_user = minlin_block_user(user_t, [c], h,mx=1000)
            sate_num.append(sate_nu[0])
        num.append(sate_num)

    with open('data/sate_num/same_point.txt', 'w') as f:
        for i in num:
            f.writelines('%d,%d,%d,%d\n'%(i[0],i[1],i[2],i[3]))


def max_user_per_sate():
    '''
    地面每个经纬度设置为一个需求，统计每个时间片单星下用户量
    寻找最大的用户密度。
    '''
    m=0
    user_total=7969436034
    create_onenode(550, 1440, incli = 53)
    users = create_user(user_total, 0, fold='data/population180.360.2.csv')
    fold = 'data/one_node550/60.txt'
    sate_loca = get_sat_loca(fold)
    for sate in sate_loca:
        sate_num=minlen(users, [sate], 550)
        print(sate_num[0])
        if sate_num[0]>m:m=sate_num[0]
    print(m)
# max_user_per_sate()



def find_max_cap():
    left = 0
    right_begin = int(72*22*400*0.03)
    # link_cap = [10, 25, 50, 100, 200, 400, 800]
    # trans_per = [0.05, 0.2, 0.5, 0.75, 1]
    link_cap = [800]
    trans_per = [0.05]
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
find_max_cap()