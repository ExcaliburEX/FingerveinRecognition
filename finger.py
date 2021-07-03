import warnings
warnings.filterwarnings("ignore")

# VGG16
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from tensorflow.compat.v1 import logging as log
log.set_verbosity(log.ERROR)
import time
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from PIL import Image
import numpy as np
import random
import ssl


def get_feature(path,model):
    img = image.load_img(path, target_size=(224, 224))
    predict_img = preprocess_input(np.expand_dims(image.img_to_array(img),0))
    return model.predict(predict_img).flatten()

def cos_sim(a, b):
    a = np.mat(a)
    b = np.mat(b)
    return float(a * b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
# end



# 图像处理模块
import cv2 as cv
from math import pi, sqrt, sin,cos
from cmath import exp
import scipy.signal
import threading
import PySimpleGUI as sg
from io import BytesIO
import base64

def gaborfilter(I, S, F, W, P):
    size = int(1.5 / S)
    F = (S ** 2) / sqrt(2 * pi)
    k = 1
    G = [[0] * (2*size) for _ in range(2*size)]
    for x in range(-size, size):
        for y in range(-size, size):
            tmp = 2*pi*F*(x*cos(W)+y*sin(W))+P
            comp1 = complex(0,tmp)
            comp2 = complex(0,P)
            G[size + x ][size + y ] = k * exp(-pi*(S**2)*(x*x+y*y)) * (exp(comp1) - exp(-pi*((F/S) ** 2) + comp2))
    GABOUT = scipy.signal.convolve2d(I, G, 'same')
    GABOUT[np.imag(GABOUT) != 0] = np.real(GABOUT[np.imag(GABOUT) != 0])
    GABOUT[GABOUT < 0] = 0
    GABOUT = GABOUT.astype(np.double)
    return GABOUT


def ImgShow(pic,window,picnum):
    try:
        image = Image.fromarray(cv.cvtColor(pic,cv.COLOR_BGR2RGB))
    except:
        image = Image.open(pic)
    image = image.resize((300, 150))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    window['-IMG%s-'%(str(picnum))].update(data=img_str)


def Train(file,flag,window,model):
    if len(file.split('.')) == 0:
        return None,0
    elif file.split('.')[-1] not in {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF','PNG'}:
        return None,0
    name = file.split('/')[-1]
    recognized = []
    feature_data_list = []
    feature_name = []
    try:
        feature_data = np.load('feature_data.npz')
        for i in feature_data['arr_0']:
            feature_data_list.append(i)
        for j in feature_data['arr_1']:
            feature_name.append(j)
    except:
        pass
    if os.path.exists('history.txt'):
            with open('history.txt', 'r+') as f:
                lines = f.readlines()
                for line in lines:
                    recognized.append(line.replace("\n", ""))
    para_clipLimit = 10
    para_adaptiveThreshold = 31
    if name not in recognized:
        img = cv.imread(file, 0)
        clahe = cv.createCLAHE(clipLimit=para_clipLimit, tileGridSize=(8, 8))
        cl = clahe.apply(img)
        threading.Thread(target=ImgShow, args=(cl,window,1)).start()
        equalizeHist = cv.equalizeHist(cl)
        threading.Thread(target=ImgShow, args=(equalizeHist,window,2)).start()
        bw = cv.adaptiveThreshold(equalizeHist, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, para_adaptiveThreshold, 2)
        out = np.zeros(bw.shape, np.double)
        normalized = np.uint8(cv.normalize(bw, out, 1.0, 0.0,cv.NORM_MINMAX, dtype=cv.CV_64F))
        normalized[normalized > 0] = 255
        threading.Thread(target=ImgShow, args=(normalized,window,3)).start()
        normalized = normalized.astype(np.double)
        ga = gaborfilter(normalized, 0.035, 0.8, 0, 0)
        filedir = os.path.abspath(file)
        newdirList = filedir.split('\\')
        if flag == 1:
            newdirList[-1] = 'trained'
        else:
            newdirList[-1] = 'tested'
        newdir = ''
        for i in range(len(newdirList)):
            newdir = newdir + newdirList[i] + '\\'
        if ~os.path.exists(newdir):
            try:
                os.mkdir(newdir)
            except:
                pass
        cv.imwrite(newdir + name, ga*255)
        threading.Thread(target=ImgShow, args=(newdir + name,window,4)).start()
        feature = get_feature(newdir + name,model)
        feature_data_list.append(feature)
        feature_name.append(name)
        np.savez('feature_data.npz',feature_data_list,feature_name)
        with open('history.txt', 'a+') as f:
            f.writelines(name)
            f.writelines('\n')
            f.close()
        return feature,1
    else:
        print('\n',time.strftime("[%Y-%m-%d %H:%M:%S]: ",time.localtime()), "图片 %s 已识别过..."%(name))
    return None,2

def Test(trainPic, readPic, finger_num,num,window,model):
    name = readPic.split('/')[-1]
    feature_data_list = []
    feature_name = []
    feature_data = np.load('feature_data.npz')
    for i in feature_data['arr_0']:
        feature_data_list.append(i)
    for j in feature_data['arr_1']:
        feature_name.append(j)
    p1,flag = Train(readPic,2,window,model)

    if flag == 0:
        # 其他文件，不用识别
        return
    elif flag == 2:
        p1 = feature_data_list[feature_name.index(name)]
    MatchArray = []
    AveMatchArray = []
    m = len(os.listdir(trainPic))
    people = m // finger_num
    q = 0
    cnt = 1
    for i in range(people):
        random_finger = random.sample(range(1,finger_num+1),num)
        for k in random_finger:
            img = str(i+1) + '-' + str(k) + '.' + readPic.split('.')[-1]
            p2 = feature_data_list[feature_name.index(img)]
            similarity = cos_sim(p1,p2)
            MatchArray.append(similarity)
            q = q + 1
            if cnt % num == 0:
                AveMatchArray.append(np.mean(MatchArray[q-num:q-1]))
            cnt = cnt + 1
    M,I = max(AveMatchArray),AveMatchArray.index(max(AveMatchArray))
    print(time.strftime("[%Y-%m-%d %H:%M:%S]: ",time.localtime()),"%s 最佳匹配是手指 %d ，匹配平均概率为：%2.4f%%\n"%(name,I+1,M*100))
    return 
# end




# GUI
def GUI():
    # t1 = ' ' * 10
    # t2 = ' ' * 28
    t3 = ' ' * 24
    t4 = ' ' * 16
    col = []
    sg.theme('LightGrey1')
    col.append(
        [
            sg.Text('自适应直方图：', font=("KaiTi", 12),justification='left',relief=sg.RELIEF_RIDGE),
            sg.Text(t3, font=("KaiTi", 12),justification='left',background_color='papayawhip'),
            sg.Text('直方图均衡化：', font=("KaiTi", 12),justification='left',relief=sg.RELIEF_RIDGE),
            sg.Text(t4, font=("KaiTi", 12),justification='left',background_color='papayawhip')
        ])
    col.append(
        [
            sg.Image(data=None, background_color='plum', enable_events=True,
                        key='-IMG1-', size=(300, 150)),
            sg.Image(data=None, background_color='plum', enable_events=True,
                        key='-IMG2-', size=(300, 150))
        ]
    )

    col.append(
        [
            sg.Text('二值化：', font=("KaiTi", 12),justification='left',relief=sg.RELIEF_RIDGE),
            sg.Text(t3, font=("KaiTi", 12),justification='left',background_color='papayawhip'),
            sg.Text('Gaborfilter滤波：', font=("KaiTi", 12),justification='left',relief=sg.RELIEF_RIDGE),
            sg.Text(t4, font=("KaiTi", 12),justification='left',background_color='papayawhip')
        ])
    col.append(
        [
            sg.Image(data=None, background_color='plum', enable_events=True,
                        key='-IMG3-' , size=(300, 150)),
            sg.Image(data=None, background_color='plum', enable_events=True,
                        key='-IMG4-' , size=(300, 150))
        ]
    )

    layout = [
        [
            sg.Text('单人手指图片个数：', font=("KaiTi", 12),justification='left',relief=sg.RELIEF_RIDGE),
            sg.InputText('', font=("KaiTi", 12), size=(17, 2), key='-RANGE1-'),
            sg.Text('随机选取的图片测试个数：', font=("KaiTi", 12),justification='left',relief=sg.RELIEF_RIDGE),
            sg.InputText('', font=("KaiTi", 12), size=(17, 2), key='-RANGE2-')
            ],
        [
            sg.Text('训练集图片文件夹：', font=("KaiTi", 12),justification='left',relief=sg.RELIEF_RIDGE),
            sg.In(size=(60, 1), enable_events=True, key="-FOLDER1-"),
            sg.FolderBrowse('浏览', button_color=('Lavender', 'BlueViolet'), font=("KaiTi", 10),size=(8, 1))
        ],
        [
            sg.Text('待识别图片文件夹：', font=("KaiTi", 12),justification='left',relief=sg.RELIEF_RIDGE),
            sg.In(size=(60, 1), enable_events=True, key="-FOLDER2-"),
            sg.FolderBrowse('浏览', button_color=('Lavender', 'BlueViolet'), font=("KaiTi", 10),size=(8, 1))
        ],
        [sg.Column(col, background_color='papayawhip', size=(
            640, 380), scrollable=True, justification="center", element_justification="center",key='COLUMN')],
        [sg.Output(size=(90, 8), key='OUTPUT', echo_stdout_stderr=True)],
        [sg.Button('开始训练', button_color=('white', 'green'),key='Train_1', size=(42, 1), font=("KaiTi", 12)),
            sg.Exit('开始识别', button_color=('white', 'orange'),key='Rec_2', size=(37, 1), font=("KaiTi", 12))
        ],
        [sg.Button('清空输出', button_color=('white', 'blue'),key='CLEAR', size=(42, 1), font=("KaiTi", 12)),
            sg.Exit('退出', button_color=('white', 'firebrick4'),key='Exit', size=(37, 1), font=("KaiTi", 12))
        ]
    ]
    window = sg.Window('HighGuardFingerRecV1.0', layout,
            default_element_size=(80, 1), resizable=True, element_justification='center', text_justification='center',finalize=True)


    while True:
        event, values = window.read()
        
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Train_1':
            if str(values['-FOLDER1-']) == '':
                print(time.strftime("[%Y-%m-%d %H:%M:%S]: ",time.localtime()), "请先选择训练目录!" )
            else:
                threading.Thread(target=BatchTrain, args=(str(values['-FOLDER1-']),window)).start()
        elif event == 'Rec_2':
            if str(values['-FOLDER2-']) == '' or str(values['-RANGE1-']) == '' or str(values['-RANGE2-']) == '':
                print(time.strftime("[%Y-%m-%d %H:%M:%S]: ",time.localtime()), "请先选择识别目录以及手指个数和选取个数!" )
            else:
                threading.Thread(target=BatchTest, args=(str(values['-FOLDER1-']) + '/trained',str(values['-FOLDER2-']),str(values['-RANGE1-']),str(values['-RANGE2-']),window)).start()
        elif event == 'CLEAR':
            window['OUTPUT'].update(' ')
# end



# 批量训练与识别
def BatchTrain(directory,window):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]: ",time.localtime()), "正在加载识别模型..." )
    ssl._create_default_https_context = ssl._create_unverified_context # 下载模型的时候不想进行ssl证书校验
    starttime = time.time()
    model = VGG16(weights='imagenet', include_top=False)
    elapse = time.time() - starttime
    print(time.strftime("[%Y-%m-%d %H:%M:%S]: ",time.localtime()), "加载完毕，耗时：%.3fs" % (elapse))
    filelist = os.listdir(directory)
    cnt = 1
    starttime2 = time.time()
    for f in filelist:
        window['OUTPUT'].update(value=time.strftime("[%Y-%m-%d %H:%M:%S]: ",time.localtime()) +  "正在训练第 %d / %d 个图片..."%(cnt,len(filelist)))
        Train(directory + '/' + f,1,window,model)
        cnt += 1
    elapse2 = time.time() - starttime2
    print('\n',time.strftime("[%Y-%m-%d %H:%M:%S]: ",time.localtime()), "训练完毕，耗时：%.3fs" % (elapse2))


def BatchTest(trainPic, readPicDir, finger_num,num,window):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]: ",time.localtime()), "单人手指数：%d张，随机抽取其中 %d 张进行比对！"%(int(finger_num),int(num)) )
    print(time.strftime("[%Y-%m-%d %H:%M:%S]: ",time.localtime()), "正在加载识别模型..." )
    starttime = time.time()
    model = VGG16(weights='imagenet', include_top=False)
    elapse = time.time() - starttime
    print(time.strftime("[%Y-%m-%d %H:%M:%S]: ",time.localtime()), "加载完毕，耗时：%.3fs" % (elapse))
    filelist = os.listdir(readPicDir)
    starttime = time.time()
    cnt = 1
    for p in filelist:
        print(time.strftime("[%Y-%m-%d %H:%M:%S]: ",time.localtime()),"正在识别第 %d / %d 个图片..."%(cnt,len(filelist)))
        Test(trainPic, readPicDir + '/' + p, int(finger_num),int(num),window,model)
        cnt += 1
    elapse = time.time() - starttime
    print(time.strftime("[%Y-%m-%d %H:%M:%S]: ",time.localtime()), "识别完毕，总耗时：%.3fs" % (elapse))
    print(time.strftime("[%Y-%m-%d %H:%M:%S]: ",time.localtime()), "单张图片识别平均耗时：%.3fs" % (elapse / len(filelist)))
# end


if __name__ == "__main__":
    GUI()



