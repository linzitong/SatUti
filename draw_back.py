import matplotlib.pyplot as plt
def draw_A():
    shell = 4
    elevation = 40
    popu_stat=[39917,39835,39739,39660,39698,39765,39801,39845,39891,39963,39957,40020,40027,39992,40014,39993,39972,39932,39937,39945,39912,39920,39932]
    popu_b = [18360,18361,18387,18410,18402,18402,18400,18375,18344,18296,18287,18239,18206,18229,18220,18226,18234,18261,18257,18283,18348,18379,18388]
    for i in range(len(popu_stat)):
        popu_stat[i] /=popu_b[i]
    # popu_stat.sort()
    print(popu_stat)
    x=[i for i in range(len(popu_stat))]
    plt.plot(x,popu_stat[0:len(popu_stat)])
    plt.xlabel('angle of deflection')
    plt.ylabel('ocean/land')
    plt.show()
# draw()
def temp():
	floder = 'tongji/ele=40/'
	diff = []
	for i in [2, 3, 4, 6, 8, 10, 12, 15, 16]:
		f = open('%sshell=%dele=40.txt'%(floder,i),'r')
		data = f.read()
		data = data[1:len(data)-6].split(', ')
		# print(data)
		for j in range(len(data)):
			data[j] = float(data[j])
		diff.append((max(data) - min(data))/max(data))
		f.close()
	x = [2, 3, 4, 6, 8, 10, 12, 15, 16]
	plt.plot(x, diff)
	plt.xlabel('cycle')
	plt.ylabel('diff:(max-min)/min')
	# plt.title('(max-min)/min')
	plt.show()

# main()

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
# basemap 教程 https://blog.csdn.net/maoye/article/details/90157850
def drawMap(sate_num, fold,start_nodes,end_nodes,capacity):
    fig = plt.figure(figsize=(12,6))
    map = Basemap()
    map.drawcoastlines()
    # map.fillcontinents()
    # 画经纬度标定 labels = [left,right,top,bottom]
    
    sats = {}
    cur_min = 3
    # cur_min = 
    satPosFile = '%s/%d.txt'%(fold, cur_min)
    print(satPosFile)
    with open(satPosFile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            nums = line.split(',')
            sat = {}
            sats[int(nums[0])] = (float(nums[2]), float(nums[3]))
    ground_latlon=[
        [22, 114],
        [51, -10],
        [35, -100],
        [31, 34],
        [-15, -47]
    ]
    for i in range(len(ground_latlon)):
        sats[sate_num + i] = (float(ground_latlon[i][0]), float(ground_latlon[i][1]))
    # print(len(start_nodes),len(end_nodes),len(capacity))
    for i in range(len(capacity)):
        if capacity[i] == 0:
            lat1 = sats[start_nodes[i]][0]
            lon1 = sats[start_nodes[i]][1]
            # print(end_nodes[i])
            lat2 = sats[end_nodes[i]][0]
            lon2 = sats[end_nodes[i]][1]
            if (lon1 < -150 and lon2 > 150 ) or (lon2 < -150 and lon1 > 150 ):
                pass
            else:
                map.plot([lon1,lon2],[lat1,lat2],linewidth=1,color='r',latlon='True')
        elif capacity[i] > 0 and capacity[i] < 100:
            lat1 = sats[start_nodes[i]][0]
            lon1 = sats[start_nodes[i]][1]
            lat2 = sats[end_nodes[i]][0]
            lon2 = sats[end_nodes[i]][1]
            if (lon1 < -150 and lon2 > 150 ) or (lon2 < -150 and lon1 > 150 ):
                pass
            else:
                map.plot([lon1,lon2],[lat1,lat2],linewidth=1,color='b',latlon='True')
            

    for sat in sats.values():
        xpt, ypt = sat[1], sat[0]
        map.scatter(xpt, ypt, s=5, color='b')
    map.drawparallels(np.arange(-90,90,30),labels=[1,1,0,1])
    map.drawmeridians(np.arange(-180,180,60),labels=[1,1,0,1])
    plt.show()
    # drawMap()