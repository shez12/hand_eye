import rospy
import sys
import cv2
import json
import os
from spatialmath import SE3

from _utils.myImageSaver import MyImageSaver
from _utils.aruco_util import *

sys.path.append("/home/hanglok/work/ur_slam")
from ik_step import init_robot, lookup_action, key_map
from myIK import MyIK


def load_intrinsics(json_file):
    with open(json_file, "r") as file:
        intrinsic_params = json.load(file)
    return intrinsic_params

def get_aruco_poses(corners, ids):
    # make sure the aruco's orientation in the camera view! 
    intrinsics_d435 = load_intrinsics("camera_parameters/intrinsics_d435.json")
    poses = estimate_markers_poses(corners, marker_size=0.10, intrinsics=intrinsics_d435)  
    poses_dict = {}
    # detected
    if ids is not None:
        for k, iden in enumerate(ids):
            poses_dict[iden]=poses[k] 
    return poses_dict

def get_marker_pose(frame, id=0, draw=True):
    corners, ids = detect_aruco(frame, draw_flag=draw)# 
    if ids is not None and len(ids)>0:
        poses_dict = get_aruco_poses(corners=corners, ids=ids)
        for i, c in zip(ids, corners):
            if i == id:
                pose = poses_dict[id] 
                return pose, c
    
    return None, None


def record_in_hand():
    # for eye in hand
    rospy.init_node('follow_aruco')
    image_saver = MyImageSaver(cameraNS="camera1")
    rospy.sleep(1)
    framedelay = 1000//20
    robot = init_robot("robot1")
    goal_pose = None
    goal_corner = None
    robot_poses = []
    marker_poses = []# in frame of aruco marker
    home = image_saver.folder_path
    robot_fk = MyIK()

    while not rospy.is_shutdown():
        frame = image_saver.rgb_image
        
        marker_pose, corner = get_marker_pose(frame, 0)
        
        if goal_corner is not None and corner is not None:
            for (x1, y1), (x2, y2) in zip(corner, goal_corner):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,0), 2)

        cv2.imshow('Camera', frame)
        # Exit on 'q' key press
        key = cv2.waitKey(framedelay) & 0xFF 
        if key == ord('q'):

            break
        if key == ord('s'):
            if len(marker_pose) > 1:
                joints = robot.get_joints()
                robot_poses.append(robot_fk.fk_se3(joints))
                marker_poses.append(marker_pose)
                image_saver.record()
            else:
                print("cannot detect id 0")


        elif key in key_map:
            code  = key_map[key]
            print(f"action {code}")
            action = lookup_action(code)
            pose = robot.step(action=action, wait=False)
            print('robot pose', np.round(pose[:3], 3))

        if key == ord('m') and goal_pose is  not None and marker_pose is not None:
            move = goal_pose * marker_pose.inv()
            move = SE3(goal_pose.t) * SE3(marker_pose.t).inv()
            print('movement')
            move.printline()
            robot.step(action=move, wait=False)

        if key == ord('g'):
            # setup goal
            if marker_pose is not None:
                goal_pose = marker_pose.copy()
                goal_corner = corner.copy()
                print('set goal as')
                goal_pose.printline()
            else:
                print('no valid marker pose')
    np.save(os.path.join(home,  'robot_poses'), robot_poses)
    np.save(os.path.join(home,  'marker_poses'), marker_poses)
    cv2.destroyAllWindows()
    return home