import math
import time

import pybullet as p
import pybullet_data
import numpy as np

import DAF_config
from utils import *
from DAF_config import *
import random


class RULE:
    def __init__(self):
        super(RULE, self).__init__()

        self.defendId2index = {}
        self.id2Index = {}
        self.attackId2Index = {}
        self.physicsClientId = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.81)
        self.defendAgentIds = []
        self.attackAgentIds = []
        self.t_points = []
        self.x_1 = np.zeros((defend_num, 2))  # previous position
        self.v_1 = np.zeros((defend_num, 2))  # previous velocity
        self.cur_to_index = [1 for i in range(defend_num)]

        planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(planeId, 0, lateralFriction=0, spinningFriction=0, rollingFriction=0)

        # 禁止圈
        drawCircle(r1, [249 / 255, 205 / 255, 173 / 255], theta_delta=30 / 180 * math.pi, p=p)

        # 结束圈
        drawCircle(r2, [130 / 255, 32 / 255, 43 / 255], theta_delta=40 / 180 * math.pi, p=p)

        # 设置agent的初始朝向
        agentStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        # 加载守方智能体初始位置
        cur_angle = 0
        theta = math.pi * 2 / defend_num
        index = 0
        index_ = 0
        self.generate_points()

        width = int(map_width / map_res)
        # initializing the grid centers
        self.map_pos = np.zeros((width, width, 2))
        for g_i in range(width):
            for g_j in range(width):
                self.map_pos[g_i, g_j, 0] = map_res * g_i - map_width / 2
                self.map_pos[g_i, g_j, 1] = map_res * g_j - map_width / 2
        self.obs_map, self.total_count = mask_map(r1, r2, map_res, map_width)
        self.records = np.zeros((width, width))

        for i in range(defend_num):
            x = defend_radius * math.sin(cur_angle)
            y = defend_radius * math.cos(cur_angle)
            defendAgentStartPos = [x, y, 0]
            agentId = p.loadURDF("defendAgent.urdf", defendAgentStartPos, agentStartOrientation)
            self.defendAgentIds.append(agentId)
            self.defendId2index[agentId] = index
            self.id2Index[agentId] = index_
            index_ += 1
            index += 1
            cur_angle += theta
            self.x_1[i] = np.array([x, y])
        # 加载攻方智能体初始位置
        cur_angle = random.randint(0, 360)
        if attack_num > 0:
            theta = math.pi * 2 / attack_num
            index = 0
            for i in range(attack_num):
                x = attack_radius * math.sin(cur_angle)
                y = attack_radius * math.cos(cur_angle)
                attackAgentStartPos = [x, y, 0]
                agentId = p.loadURDF("attackAgent.urdf", attackAgentStartPos, agentStartOrientation)
                self.attackAgentIds.append(agentId)
                self.attackId2Index[agentId] = index
                self.id2Index[agentId] = index_
                index += 1
                index_ += 1
                cur_angle += theta

    def generate_points(self):
        t_points = []
        for i in range(defend_num):
            o_r = r2 - 5
            i_r = r1 + 5
            theta = 0 + (math.pi*2/defend_num) * i
            theta_delta = (math.pi*2/defend_num)/4
            points = []
            for j in range(4+1):
                point = [o_r * math.sin(theta), o_r * math.cos(theta)]
                points.append(point)
                theta += theta_delta
            for j in range(4+1):
                point = [i_r * math.sin(theta), i_r * math.cos(theta)]
                points.append(point)
                theta -= theta_delta
            t_points.append(points)
        self.t_points = np.array(t_points)

    def run(self):
        counter = 1
        # main loop
        run_flag = True
        while run_flag:
            T = counter * t_gap
            for i in range(defend_num):
                cur_v = self.v_1[i]
                cur_x = self.x_1[i]
                self.x_1[i] = cur_x + cur_v * t_gap
                cur_to_point = self.t_points[i][self.cur_to_index[i]]
                temp = cur_to_point - cur_x
                a = max_a * (temp / math.sqrt(temp[0] ** 2 + temp[1] ** 2))
                cur_v = cur_v + (a * t_gap)
                cur_v_scaler = np.sqrt(cur_v[0] ** 2 + cur_v[1] ** 2)
                if cur_v_scaler > max_v:
                    cur_v = max_v * (cur_v / cur_v_scaler)
                self.v_1[i] = cur_v

                speed = list(self.v_1[i])
                position = list(self.x_1[i])
                speed.append(0)
                position.append(0)
                new_orientation = p.getQuaternionFromEuler([0, 0, -0 * math.pi / 180])
                # print(agentId, position)
                agentId = i + 1
                p.resetBasePositionAndOrientation(agentId, position, new_orientation)

                dis = getDis(self.x_1[i], cur_to_point)
                arr_dis = 3
                if dis < arr_dis:
                    self.cur_to_index[i] = (self.cur_to_index[i] + 1) % 10
            # print(self.cur_to_index)
            update_records(self.records, self.map_pos, self.x_1, T)


            # 发生故障
            # 规则（考虑通信？  不考虑通信？）
            # DAF （                     ）

           # 统计剩余面积（=0    <t）
            rem_map = sum(sum(np.logical_and(self.records[:, :] == 0, self.obs_map[:, :] == 1)))
            cumu_covg = (self.total_count - rem_map) / (self.total_count * 0.01)
            # for i in range(20):
            p.stepSimulation()
            time.sleep(DAF_config.time_to_render)
            # termination condition
            # if cumu_covg == 100:
            #     print("100%!")
            #     break
            # else:
            #     counter += 1
            counter += 1
            if counter % 10 == 0:
                print(cumu_covg)


a = RULE()
a.run()