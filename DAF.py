import time

import pybullet as p
import pybullet_data

import DAF_config
from utils import *
from DAF_config import *
import random


class DAF:
    def __init__(self):
        super(DAF, self).__init__()

        self.defendId2index = {}
        self.id2Index = {}
        self.attackId2Index = {}
        self.physicsClientId = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.81)
        self.defendAgentIds = []
        self.attackAgentIds = []

        planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(planeId, 0, lateralFriction=0, spinningFriction=0, rollingFriction=0)

        # 禁止圈
        drawCircle(r1, [249 / 255, 205 / 255, 173 / 255], theta_delta=30 / 180 * math.pi, p=p)

        # 结束圈
        drawCircle(r2, [130 / 255, 32 / 255, 43 / 255], theta_delta=40 / 180 * math.pi, p=p)

        # defining vectors
        x = np.zeros((defend_num, 2))  # current position
        self.x_1 = np.zeros((defend_num, 2))  # previous position
        self.v_1 = np.zeros((defend_num, 2))  # previous velocity
        self.tra_dist = np.zeros((defend_num, 1))  # travel distance
        self.x_r_t = np.zeros((defend_num, 3))  # search location last visited time

        self.cell_map_size = (int(np.floor(map_width / map_res)), int(np.floor(map_width / map_res)), defend_num * 2)
        self.cell_map = np.zeros(self.cell_map_size)  # cell_map structure

        # initializing the grid centers
        self.map_pos = np.zeros((self.cell_map_size[0], self.cell_map_size[1], 2))
        for g_i in range(self.cell_map_size[0]):
            for g_j in range(self.cell_map_size[1]):
                self.map_pos[g_i, g_j, 0] = map_res * g_i - map_width / 2
                self.map_pos[g_i, g_j, 1] = map_res * g_j - map_width / 2

        # 初始化

        self.obs_map, self.total_count = mask_map(r1, r2, map_res, map_width)
        # 设置agent的初始朝向
        agentStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        # 加载守方智能体初始位置
        cur_angle = 0
        theta = math.pi * 2 / defend_num
        index = 0
        index_ = 0
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
            self.x_r_t[i, 0] = int((x + map_width / 2) // map_res)
            self.x_r_t[i, 1] = int((y + map_width / 2) // map_res)
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

        self.x_r = self.x_1  # search location
        self.x_r_b = [[] for i in range(defend_num)]
        self.r_s_m = r_s * 2
        self.x_r_b_degree = 360 / defend_num

        ini_v_t = 2 * np.pi * np.random.uniform(low=0.0, high=1.0, size=defend_num)
        ini_v_s = v_0 * np.random.uniform(low=0.0, high=1.0, size=defend_num)
        self.v_1[:, 0] = np.multiply(ini_v_s, np.cos(ini_v_t))
        self.v_1[:, 1] = np.multiply(ini_v_s, np.sin(ini_v_t))

        self.fused_scan_record = np.zeros((self.cell_map_size[0], self.cell_map_size[1], 2))

    def run(self):
        counter = 1
        # main loop
        run_flag = True
        while run_flag:
            u_d = np.zeros((defend_num, 2))  # control input for decentralizing
            u_o = np.zeros((defend_num, 2))  # control input for obstacle avoidance

            # calculating adjacency matrix
            dist_gap = get_gap(self.x_1)
            dist_2 = squd_norm(dist_gap)
            dist = np.sqrt(dist_2)
            adj = action_func(dist, d_a, c1)
            nbr = np.zeros((defend_num, defend_num))
            nbr[dist <= r_c] = 1
            nbr = nbr - np.diag(np.diag(nbr))  # set diagonal to 0
            adj = np.multiply(nbr, adj)

            T = counter * t_gap

            # 记录当前时间和看到的标记
            self.cell_map = update_individual_record(self.cell_map, self.map_pos, self.x_1, T, r_s)

            if np.sum(nbr) > 0:
                self.cell_map = fuse_record(self.cell_map, nbr)

            # fusing cell_map for calculations
            self.fused_scan_record = fuse_all_records(self.cell_map, defend_num, self.fused_scan_record)

            rem_map = sum(sum(np.logical_and(self.fused_scan_record[:, :, 0] == 0, self.obs_map[:, :] == 1)))
            # print((self.fused_scan_record[:, :, 0] == 0).shape)
            # print((self.obs_map[:, :] == 1).shape)
            # print('ab:', sum(sum(self.obs_map[:, :] == 1)), 'tt:', self.total_count)
            # print(rem_map)
            cur_covg = sum(sum(self.fused_scan_record[:, :, 0] == T))
            # print(cur_covg.shape)
            map_w, map_h = self.cell_map_size[0], self.cell_map_size[1]

            cumu_covg = (self.total_count - rem_map) / (self.total_count * 0.01)
            # print(cumu_covg)
            inst_covg = cur_covg / ((map_w * map_h) * 0.01)

            # selfishness
            self.x_r_m = self.x_r
            for i in range(defend_num):
                if self.x_r_b[i]:
                    self.x_r_m[i] = self.x_r_b[i][0]

            u_e = c2 * (self.x_r_m - self.x_1) - c3 * self.v_1

            # decentering
            for a in range(defend_num):
                for b in range(defend_num):
                    u_d[a, :] = u_d[a, :] + (self.x_1[b, :] - self.x_1[a, :]) * adj[a, b] / np.sqrt(efs + dist_2[a, b])

            u = u_d + u_o + u_e

            # calculating the new position and velocity
            x = self.x_1 + self.v_1 * t_gap
            v = self.v_1 + u * t_gap

            self.v_1 = v
            x_pre = self.x_1
            self.x_1 = x
            self.tra_dist = self.tra_dist + np.sqrt(
                np.square(x[:, 0] - x_pre[:, 0]) + np.square(x[:, 1] - x_pre[:, 1])).reshape(
                (defend_num, 1))

            # calculate goal distances

            # x_r是x的目标点
            goal_dist = np.zeros((defend_num, defend_num))
            goal_m_dist = np.zeros(defend_num)
            for s in range(defend_num):
                for t in range(defend_num):
                    goal_dist[s, t] = np.sqrt((x[s, 0] - self.x_r[t, 0]) ** 2 + (x[s, 1] - self.x_r[t, 1]) ** 2)
                goal_m_dist[s] = np.sqrt((x[s, 0] - self.x_r_m[s, 0]) ** 2 + (x[s, 1] - self.x_r_m[s, 1]) ** 2)

            for s in range(defend_num):
                obs_map_temp = np.empty_like(self.obs_map)
                # todo
                obs_map_temp[:] = self.obs_map
                temp_cell_map = self.cell_map[:, :, 2 * s]
                recalculate = 0

                if goal_m_dist[s] < self.r_s_m:
                    if self.x_r_b[s]:
                        self.x_r_b[s].pop(0)

                # 到达目标点
                if goal_dist[s, s] < r_s:
                    recalculate = 1
                # todo
                elif temp_cell_map[int(self.x_r_t[s, 0]), int(self.x_r_t[s, 1])] != self.x_r_t[s, 2]:
                    recalculate = 1
                else:
                    for i in range(defend_num):
                        if nbr[s, i] == 1:
                            # s 和 i 的目标距离靠近
                            if np.sqrt(np.square(self.x_r[i, 0] - self.x_r[s, 0]) + np.square(
                                    self.x_r[i, 1] - self.x_r[s, 1])) < r_s and \
                                    goal_dist[
                                        i, i] < goal_dist[s, s]:
                                # todo?
                                obs_map_temp[np.sqrt(np.square(self.map_pos[:, :, 0] - self.x_r[i, 0]) + np.square(
                                    self.map_pos[:, :, 1] - self.x_r[i, 1])) < r_s] = neg_inf
                                recalculate = 1
                            #  source bug todo
                            elif goal_dist[i, s] < goal_dist[s, s] and goal_dist[s, i] < goal_dist[
                                i, i] and recalculate == 0:
                                temp_x_r = self.x_r[i, :]
                                self.x_r[i, :] = self.x_r[s, :]
                                self.x_r[s, :] = temp_x_r

                if recalculate == 1:
                    # calculate distance to each grid
                    tar_dist = np.sqrt(
                        np.square(x[s, 0] - self.map_pos[:, :, 0]) + np.square(x[s, 1] - self.map_pos[:, :, 1]))
                    # calculate minimum distance to each grid from neighbors
                    nbr_tar_dist = np.inf * np.ones((self.cell_map_size[0], self.cell_map_size[1]))
                    for i in range(defend_num):
                        if nbr[s, i] == 1:
                            nbr_tar_dist = np.minimum(nbr_tar_dist, np.sqrt(
                                np.square(x[i, 0] - self.map_pos[:, :, 0]) + np.square(
                                    x[i, 1] - self.map_pos[:, :, 1])))

                    # 将己方的可运动范围用一个障碍地图表示
                    obs_map_temp[tar_dist != np.minimum(nbr_tar_dist, tar_dist)] = neg_inf
                    pre_tar_dist = np.sqrt(
                        np.square(self.x_r[s, 0] - self.map_pos[:, :, 0]) + np.square(
                            self.x_r[s, 1] - self.map_pos[:, :, 1]))
                    map_d = np.multiply((T - temp_cell_map),
                                        (rho + (1 - rho) * np.exp(-sigma1 * tar_dist - sigma2 * pre_tar_dist)))

                    # 在此时进行mask操作
                    map_d1 = np.multiply(map_d, (obs_map_temp + 1))

                    # 选取 值最大的进行 运动
                    coor = np.argwhere(map_d1 == np.amax(map_d1))
                    ind = random.sample(range(coor.shape[0]), 1)[0]
                    I, J = coor[ind, :]
                    self.x_r[s, :] = self.map_pos[I, J, :]
                    self.x_r_t[s, :] = [I, J, temp_cell_map[I, J]]
                    self.x_r_b[s] = get_before(x[s], self.x_r[s, :], center_point, self.x_r_b_degree)

            # plotting
            vdx = []
            vdy = []
            for a in range(defend_num):
                sssp = np.sqrt(np.sum(np.square(v[a, :])))
                if sssp == 0:
                    vdx.append(arrow_head_length)
                    vdy.append(arrow_head_length)
                else:
                    vdx.append(arrow_head_length * v[a, 0] / sssp)
                    vdy.append(arrow_head_length * v[a, 1] / sssp)

            for agentId in self.defendAgentIds:
                speed = list(self.v_1[self.defendId2index[agentId]])
                position = list(self.x_1[self.defendId2index[agentId]])
                speed.append(0)
                position.append(0)
                new_orientation = p.getQuaternionFromEuler([0, 0, -0 * math.pi / 180])
                # print(agentId, position)
                p.resetBasePositionAndOrientation(agentId, position, new_orientation)
                # p.resetBaseVelocity(agentId, speed, [0, 0, 0])

            # for i in range(20):
            p.stepSimulation()
            time.sleep(DAF_config.time_to_render)
            # termination condition
            if cumu_covg == 100:
                print("100%!")
                break
            else:
                counter += 1
            if counter % 10 == 0:
                print(cumu_covg)
