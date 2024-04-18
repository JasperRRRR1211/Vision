import numpy as np
import cv2

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)   #计算比例因子
		dim = (int(w * r), height)  #调整之后的尺寸
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation=inter)
	return resized

def makesure(pts):
    pent = np.zeros((5, 2), dtype = "float32")
    column_0 = pts[:, 0]    #第一列的各点的x拿出来
    sorted_indices = np.argsort(column_0)   #按x从小到大排序一下
    sorted_pts = pts[sorted_indices]    #按排序后的顺序对五个坐标重新排列
    vertex = sorted_pts[2]  #拿出来x第二大的那个坐标值
    column_1 = pts[:1]  #把各点的y拿出来
    ymin = np.min(column_1)     #计算最小的y
    if vertex[1]>ymin:     #判断
        is_Rotate = True
    else:
        is_Rotate = False
    return is_Rotate

def distance(zer, sec, thir):   #计算两点间的距离, zer->zero sec->second thir->third
    right = np.sqrt((sec[1]-thir[1])**2+(sec[0]-thir[0])**2)
    left = np.sqrt((zer[1]-thir[1])**2+(thir[0]-zer[0])**2)
    return right, left
def order_points(pts):

	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	# 获取输入坐标点
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 变换后对应坐标位置
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# 计算变换矩阵
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回变换后结果
	return warped

def cv_show(image):
    cv2.imshow('image', image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

def run(image):
    #输入并展示图片
    #image = cv2.imread(source)
    copy = image.copy()
    image = resize(copy, height = 200)

    #把图片变成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #中值滤波滤一遍
    gray = cv2.medianBlur(gray, 5)

    #二值化
    edged = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
    edged = cv2.medianBlur(edged, 5)

    #检测轮廓
    img_cpy = image.copy()
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cnts:  # 如果conts为空
        #print("No contours found.")
        return  # 结束run()函数，返回到调用点
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    TarCnt = cnts[0]
    cv2.drawContours(img_cpy, [TarCnt], -1, (255, 0, 0), 3)

    #得到外接矩形，和各个点的坐标
    rect = cv2.minAreaRect(TarCnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    #过滤掉错误图片
    width, height = rect[1]
    Area = width*height
    if Area>2000:       #小于这个面积的就是一些非目标噪音，过滤掉
        cv2.drawContours(img_cpy, [box], -1, (255, 0, 0), 3)

        #计算旋转角度
        zero = box[0]
        first = box[1]
        second = box[2]
        third = box[3]

        angle = rect[-1]
        if first[1]>third[1]:
            right_dis, left_dis = distance(zero, second, third)
        elif first[1]<third[1]:
            left_dis, right_dis = distance(zero, second, third)

        if right_dis>=left_dis:
            angle = angle-90
        else:
            angle = angle
    else:
        #print('wrong pic')
        return

    #旋转图片
    rotation_matrix = cv2.getRotationMatrix2D(rect[0], angle, 1)
    warped = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    orig = warped
    warped_cpy = warped.copy()
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = cv2.medianBlur(warped, 5)
    ref = cv2.threshold(warped, 180, 255, cv2.THRESH_BINARY)[1]

    #检测轮廓
    ref = cv2.medianBlur(ref, 5)
    conts = cv2.findContours(ref.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not conts:
        #print("No contours found.")
        return
    conts = sorted(conts, key = cv2.contourArea, reverse = True)[:5]
    pentagon = conts[0]
    epl = cv2.arcLength(pentagon, True)
    pent_outline = cv2.approxPolyDP(pentagon, 0.03 * epl, True)

    #确保五边形尖端向上
    if pent_outline.shape != (5, 1, 2):
        return
    else:
        is_Rotate = makesure(pent_outline.reshape(5, 2))
        if is_Rotate:
            height, width = warped.shape[:2]
            cenpoint = (width / 2, height / 2)
            angle = 180
            rotation_matrix = cv2.getRotationMatrix2D(cenpoint, angle, 1.0)
            ref = cv2.warpAffine(ref, rotation_matrix, (width, height))
            orig = cv2.warpAffine(orig, rotation_matrix, (width, height))
        CNTS = cv2.findContours(ref.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not CNTS:
            return
        CNTS = sorted(CNTS, key=cv2.contourArea, reverse=True)[:5]  #找到五边形标靶
        TarRegion = CNTS[0]

        #把五边形截出来
        ptg_rect = cv2.minAreaRect(TarRegion)
        ptg_box = cv2.boxPoints(ptg_rect)
        ptg_box = np.int0(ptg_box)
        width, height = ptg_rect[1]
        #print(width, height)
        width = np.int0(width)
        height = np.int0(height)
	    
        left_vertex = ptg_box[0]    #最左侧的坐标值拿出来
	right_vertex = ptg_box[2]   #最右侧的坐标值
        center = [(left_vertex[0]+right_vertex[0])/2, (left_vertex[1]+right_vertex[1])/2]		#计算中心点坐标!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
        x = left_vertex[0]
        y = left_vertex[1]
    croped_pentagon = orig[y:y+width, x:x+height]   #截取五边形标靶
    orig_PTG = croped_pentagon
    PTG_cpy = croped_pentagon.copy()
    if croped_pentagon.size == 0:
        #print('wrong picture')
        return
    croped_pentagon = cv2.cvtColor(croped_pentagon, cv2.COLOR_BGR2GRAY)
    croped_pentagon = cv2.medianBlur(croped_pentagon, 5)
    PTG_ref = cv2.threshold(croped_pentagon, 180, 255, cv2.THRESH_BINARY)[1]
    PTG_ref = cv2.medianBlur(PTG_ref, 5)
    Crop_conts = cv2.findContours(PTG_ref, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    #找到四边形
    square = None
    for c in Crop_conts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            square = approx
            break
    if square is not None:
        number = four_point_transform(orig_PTG, square.reshape(4, 2))
        number = cv2.resize(number, (60, 60))
	print(center)			#打印中心点坐标
        #cv_show(PTG_cpy)
        return number
    else:
        #print("No suitable target found.")
        return None

if __name__ == '__main__':
    run(source = 'runs/NGYFRUN/exp10/crops/Red/NGYF2373.jpg')




