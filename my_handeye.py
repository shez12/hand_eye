from record_aruco import record_in_hand
from hand_eye_calib_arucos import *

'''
(1) 不管采集多少组用于标定的运动数据，每组运动使运动角度最大。
(2) 使两组运动的旋转轴角度最大。
(3) 每组运动中机械臂末端运动距离尽量小，可通路径规划实现该条件。
(4) 尽量减小相机中心到标定板的距离，可使用适当小的标定板。
(5) 尽量采集多组用于求解的数据。

                       
原文链接：https://blog.csdn.net/Thinkin9/article/details/123743924
'''



def main():
    print("=====================starting record data=====================")
    # path = record_in_hand()
    path = "data/images-20241126-155752"
    print("=====================calculate hand eye=====================")
    myHandEye= compute_model(path,method='eye_in_hand')
    print("=====================validate hand eye=====================")
    T_t2b = myHandEye.compute_t2b()
    for i in range(len(T_t2b)):
        T_b2t = SE3(T_t2b[i]).inv() # select one as T_t2b
        T_b2t.printline()


    


if __name__ == "__main__":
    main()
