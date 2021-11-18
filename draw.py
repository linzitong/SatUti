from matplotlib import colors
import matplotlib.pyplot as plt
def draw_A():
    # shell = 4
    elevation = 53
    popu_stat=[66668643.759179994, 75201837.69338001, 91906409.83947997, 75831350.09588, 77428774.90108003, 83667134.85398002, 84091783.99208005, 83542492.36538002, 88933949.94609998, 86351736.91999999, 83851558.01017998, 63811822.72077997, 61830162.65467997, 70286788.81767999, 65995782.72667993, 66608466.32057994, 66067359.70057994, 62668216.46467993, 61444334.48637994, 60563408.03095995, 53861586.11145998, 53563171.607459985, 42134927.46275999]
    # popu_b = [18360,18361,18387,18410,18402,18402,18400,18375,18344,18296,18287,18239,18206,18229,18220,18226,18234,18261,18257,18283,18348,18379,18388]
    # for i in range(len(popu_stat)):
    #     popu_stat[i] /=popu_b[i]
    # popu_stat.sort()
    print(popu_stat)
    x=[i for i in range(len(popu_stat))]
    plt.plot(x,popu_stat[0:len(popu_stat)])
    plt.xlabel('angle of deflection')
    plt.ylabel('ocean/land')
    plt.show()
# draw_A()
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
def drawMap(sate_num, fold,start_nodes,end_nodes,capacity,users,cur_min):
    fig = plt.figure(figsize=(12,6))
    map = Basemap()
    map.drawcoastlines()
    # map.fillcontinents()
    # 画经纬度标定 labels = [left,right,top,bottom]
    
    sats = {}
    # cur_min = 
    satPosFile = '%s/%d.txt'%(fold, cur_min)
    with open(satPosFile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            nums = line.split(',')
            sat = {}
            sats[int(nums[0])] = (float(nums[2]), float(nums[3]))

    # for i in range(len(ground_latlon)):
    #     sats[sate_num + i] = (float(ground_latlon[i][0]), float(ground_latlon[i][1]))
    # print(len(start_nodes),len(end_nodes),len(capacity))
    for i in range(len(capacity)):
        c = 'b'
        if capacity[i] <= 20: c = 'r'
        elif capacity[i] <= 30: c = 'orange'
        elif capacity[i] <=40: c = 'y'
        if capacity[i] <=50:
            lat1 = sats[start_nodes[i]][0]
            lon1 = sats[start_nodes[i]][1]
            # print(end_nodes[i])
            lat2 = sats[end_nodes[i]][0]
            lon2 = sats[end_nodes[i]][1]
            if (lon1 < -150 and lon2 > 150 ) or (lon2 < -150 and lon1 > 150 ):
                if lon1 > lon2:
                    lon1,lon2=lon2,lon1
                    lat1,lat2=lat2,lat1
                map.plot([lon1,-180],[lat1,(lat2+lat1)/2],linewidth=1,color=c,latlon='True')
                map.plot([180,lon2],[(lat1+lat2)/2,lat2],linewidth=1,color=c,latlon='True')
            else:
                map.plot([lon1,lon2],[lat1,lat2],linewidth=1,color=c,latlon='True')
       
    for sat in sats.values():
        xpt, ypt = sat[1], sat[0]
        map.scatter(xpt, ypt, s=5, color='b')
    for u in users:
        x, y = u[1], u[0]
        map.scatter(x, y, s = 3, color='g')
    map.drawparallels(np.arange(-90,90,30),labels=[1,1,0,1])
    map.drawmeridians(np.arange(-180,180,60),labels=[1,1,0,1])
    # plt.savefig('img/kuiper2/link%d.jpg'%cur_min)
    # plt.close(fig)
    # plt.savefig('img/link%d.eps'%cur_min)
    # drawMap()


def load_geo():
    '''
    载入amazon的地站数据并绘制在地图上
    '''
    import json
    import matplotlib.pyplot as plt
    
    import numpy as np
    x = []
    y = []
    with open('aws_geojson.json') as f:
        load_dict = json.load(f)
        for dc in load_dict:
            x.append(dc["geometry"]["coordinates"][0])
            y.append(dc["geometry"]["coordinates"][1])
    from mpl_toolkits.basemap import Basemap
    map = Basemap()
    map.drawcoastlines()
    map.drawparallels(np.arange(-90,90,30),labels=[1,1,0,1])
    map.drawmeridians(np.arange(-180,180,60),labels=[1,1,0,1])
    plt.scatter(x, y, marker='o', edgecolors='b')
    plt.show()
# load_geo()