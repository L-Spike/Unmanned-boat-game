# coding=utf-8
import math
import random
from random import choice
import sys
from abc import ABC
import numpy as np
import pybullet as p
import time
import pybullet_data
import gym
from gym import spaces

from config import *


def velocityConversion(velocity, angle):
    velocityX = velocity * np.sin(angle * math.pi / 180)
    velocityY = velocity * np.cos(angle * math.pi / 180)
    velocityZ = 0
    return [velocityX, velocityY, velocityZ]


def velocityConversionVerse(speed):
    vx, vy, vz = speed
    velocity = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
    angle = azimuthAngleWPBase(vx, vy)
    return velocity, angle


# 计算方位角函数
def azimuthAngleWP(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    dx = x2 - x1
    dy = y2 - y1
    return azimuthAngleWPBase(dx, dy)


def azimuthAngleWPBase(dx, dy):
    angle = 0
    if dx == 0:
        # angle = math.pi / 2.0
        # if y2 == y1:
        #    angle = 0.0
        # elif y2 < y1:
        #    angle = 3.0 * math.pi / 2.0
        angle = 0
        if dy < 0:
            angle = 3.0 * math.pi / 2.0
    elif dx > 0 and dy > 0:
        angle = math.atan(dx / dy)
    elif dx > 0 > dy:
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif dx < 0 and dy < 0:
        angle = math.pi + math.atan(dx / dy)
    elif dx < 0 < dy:
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return angle * 180 / math.pi


def getDis(pos1, pos2):
    dis = math.sqrt(math.pow(pos1[0] - pos2[0], 2) + math.pow(pos1[1] - pos2[1], 2))
    return dis


def transferRelativePosition(pos1, pos2, pos3):
    s = getDis(pos1, pos2)
    angle1 = azimuthAngleWP(pos1, pos2)
    angle2 = azimuthAngleWP(pos1, pos3)
    phi = azimuthAngleWA(angle2, angle1)
    return s, phi


def transferRelativeAngle(angle1, pos1, pos2):
    angle = azimuthAngleWP(pos1, pos2)
    return azimuthAngleWA(angle, angle1)


def azimuthAngleWA(baseAngle, anotherAngle):
    angle = (anotherAngle - baseAngle) % 360
    return angle


def relativeAngle(baseAngle, anotherAngle):
    angle = (anotherAngle - baseAngle) % 360
    angle = angle if angle <= 180 else 360 - angle
    return angle


def relativeAngleWithSymbol(baseAngle, anotherAngle):
    angle = (anotherAngle - baseAngle) % 360
    angle = angle if angle <= 180 else -(360 - angle)
    return angle


# 进攻策略，可以采用规则化策略或者强化学习策略
class AttackStrategy:

    def __init__(self):
        self.local_state = 0

    def generate_actions(self, state):
        return 0


# 防守策略，可以采用规则化策略或者强化学习策略
class DefendStrategy:

    def __init__(self):
        self.local_state = 0

    def generate_actions(self, state):
        pass


# 简单的规则化进攻策略，采取部分可观测的环境设置
# 每个agent有一个局部观测的state，嵌入到环境中
class SimpleAttackStrategy(AttackStrategy):
    def __init__(self):
        super().__init__()

        self.max_oil = 5
        self.threat_angle = threat_angle
        self.threat_dis = attack_threat_dis
        self.max_threat_threshold = attack_threat_dis * threat_angle
        self.threat_angle_delta = threat_angle_delta
        self.small_angle = small_angle

    # 依据局部观测到的状态产生简单的进攻行为
    # 具体而言是朝者目标直线运动，直到出现碰撞行为，然后选择朝左或者右进行规避，当不出现碰撞行为时再朝着目标进行运动
    def generate_actions(self, states):
        actions = []
        # （友军设置碰撞范围小）（敌军设置碰撞范围大）
        #  简单将友军设置成为均匀分散的初始位置
        #  检测碰撞 & 选取合适的角度（注意规则和强化学习的差别和统一）
        #  不再有碰撞 & 朝着目标前进

        # 检测所有友军， 检测所有敌军， 找到一个最近的可能碰撞的，还得看方向
        # 状态的给定
        # 环境自身的状态给定依据转换，通信距离的体现，历史的位置(清除)

        # 遍历所有攻方agent
        # [[cur_velocity, cur_angle], [[other_agent_id, s, phi, velocity, angle]...],
        # [[other_agent_id, s, phi, velocity, angle],...], dis]
        for agent_index, agent_info in enumerate(states):
            min_threat_degree = self.max_threat_threshold
            min_d = -1
            min_delta = -1
            min_threat_agent_info = None

            velocity1 = agent_info[0][0]
            angle1 = agent_info[0][1]
            for attack_agent_info in agent_info[2]:
                s = attack_agent_info[1]
                phi = attack_agent_info[2]
                velocity2 = attack_agent_info[3]
                angle2 = attack_agent_info[4]
                threat_degree, d, delta = self.calThreatDegree(s, phi, angle1, angle2, velocity1, velocity2)
                if min_threat_degree > threat_degree:
                    # 处理  标记
                    min_threat_degree = threat_degree
                    min_d = d
                    min_delta = delta
                    min_threat_agent_info = attack_agent_info

            if min_threat_degree == self.max_threat_threshold:
                oil, rudder = self.getActionsSimple(angle1)
            else:
                # 计算
                s = min_threat_agent_info[1]
                phi = min_threat_agent_info[2]
                velocity2 = min_threat_agent_info[3]
                angle2 = min_threat_agent_info[4]
                oil, rudder = self.getActionWithObstacles(s, phi, angle1, angle2, velocity1, velocity2, min_d,
                                                          min_delta)
            action = (oil, rudder)
            actions.append(action)

        # 遍历所有攻方（友方）agent todo
        return actions

    def getActionsSimple(self, angle1):
        angle1 = angle1 if angle1 < 180 else angle1 - 360
        rudder = -5 if angle1 > 0 else +5
        oil = self.max_oil
        return oil, rudder

    def getActionWithObstacles(self, s, phi, angle1, angle2, velocity1, velocity2, d, delta):
        # 计算目标角度
        # delta_s = self.delta_t * velocity2
        # dx = s * math.cos(phi) + delta_s * math.cos(angle2)
        # dy = s * math.sin(phi) + delta_s * math.sin(angle2)
        # s_new = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        # target_angle = azimuthAngleWPBase(dx, dy)
        if d < self.threat_dis / 4:
            oil = 1
        else:
            oil = 5

        # relative_angle = relativeAngle(target_angle, angle1)
        # relative_angle_symbol = 1 if relative_angle > 0 else -1
        relative_angle_symbol = 1 if delta > 0 else -1
        if abs(delta) < self.small_angle:
            rudder = 1 * relative_angle_symbol
        else:
            rudder = 5 * relative_angle_symbol
        return oil, rudder

    def calThreatDegree(self, s, phi, angle1, angle2, velocity1, velocity2):
        # 计算相对角度，小于2倍的危险角度就认为可能发生碰撞
        re_angle = relativeAngle(phi, angle1)
        re_angle2 = relativeAngle(phi, angle2)

        # if s < self.threat_dis:
        #     if re_angle < self.threat_angle and re_angle2 > 180 - self.threat_angle:
        #         angle_ = azimuthAngleWA(phi, angle1)
        #         if angle_ > 180:
        #             angle__ = 270 * 2 - angle_
        #         else:
        #             angle__ = 90 * 2 - angle_
        #         angle___ = azimuthAngleWA(phi, angle2)
        #         delta_angle = relativeAngle(angle___, angle__)
        #         if delta_angle < self.threat_angle_delta:
        #             # todo 更加详细的威胁度评定
        #             return s * delta_angle

        theta = re_angle2 / 180 * math.sin(math.pi)
        sin_value = math.sin(theta) * velocity2 / (velocity1 + 0.01)
        if sin_value > 1:
            return
        alpha = math.atan(sin_value)
        # if alpha < theta:(只有一个解)
        delta = math.pi - alpha - theta
        # delta = alpha - theta  (数形结合)
        d = s / math.sin(delta) * math.sin(theta)
        symbol = 1 if relativeAngleWithSymbol(phi, angle2) > 0 else -1
        target_angle = phi + alpha * symbol
        target_angle_delta = relativeAngle(target_angle, angle1)
        target_angle_delta_with_symbol = target_angle_delta if relativeAngleWithSymbol(target_angle,
                                                                                       angle1) > 0 else -target_angle_delta

        threat_item = d * target_angle_delta

        if d > self.threat_dis or target_angle_delta > self.threat_angle:
            return self.max_threat_threshold, -1, -1

        return threat_item, d, target_angle_delta_with_symbol


class RandomDefendStrategy(DefendStrategy):
    def __init__(self):
        super().__init__()

    # 防守方随机选取动作
    def generate_actions(self, states):
        oil_range = [-5, -1, 0, 1, 5]
        rudder_range = [-5, 1, 0, 1, 5]
        actions = []
        for i in range(len(states[0])):
            rudder = choice(rudder_range)
            oil = choice(oil_range)
            actions.append([rudder, oil])
        return actions


def drawCircle(radius, color, theta_delta):
    from_angle = 0
    to_angle = 20 / 180 * math.pi
    froms = []
    tos = []
    while to_angle < 2 * math.pi:
        from_point = [radius * math.sin(from_angle),
                      radius * math.cos(from_angle), 0]
        to_point = [radius * math.sin(to_angle), radius * math.cos(to_angle), 0]
        from_angle = to_angle
        to_angle += theta_delta
        froms.append(from_point)
        tos.append(to_point)
    for f, t in zip(froms, tos):
        p.addUserDebugLine(
            lineFromXYZ=f,
            lineToXYZ=t,
            lineColorRGB=color,
            lineWidth=3
        )


class GlobalAgentsEnv:
    def __init__(self, defend_stratedy, attack_strategy,
                 render: bool = False):

        self.attack_reward_factor = attack_reward_factor
        self.turn_radius = turn_radius
        self.factor = factor
        self.capture_reward = capture_reward
        self.forbidden_reward = forbidden_reward
        self.not_find_reward = not_find_reward
        self.done_dis = done_dis
        self.attack_num = attack_num
        self.defend_num = defend_num
        self.attack_radius = attack_radius
        self.defend_radius = defend_radius
        self.forbidden_radius = forbidden_radius
        self.not_to_catch_reward = not_to_catch_reward
        self.threat_angle_delta = threat_angle_delta
        self.threat_angle = threat_angle
        self.threat_dis = defend_threat_dis
        self.capture_dis = capture_dis
        self.reward_agent_num = reward_agent_num
        self.target_position = target_position
        self.max_velocity = max_velocity
        self.max_turn_angle = max_turn_angle
        self.communicate_radius = communicate_radius
        self.observe_radius = observe_radius
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(2)

        self.defend_stratedy: DefendStrategy = defend_stratedy
        self.attack_strategy: AttackStrategy = attack_strategy

        # 连接到渲染环境
        self.physicsClientId = p.connect(p.GUI if render else p.DIRECT)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.81)
        self.defendAgentIds = []
        self.attackAgentIds = []
        self.id2Index = {}
        self.agentCurPositions = [0] * (self.attack_num + self.defend_num)
        self.agentCurVelocities = [0] * (self.attack_num + self.defend_num)
        self.defendId2index = {}
        self.attackId2Index = {}
        self.init_agent()

        self.defend_total_reward = 0
        self.state = None
        self.reward = None
        self.attack_adj = np.zeros((self.attack_num, self.attack_num))
        self.defend_adj = np.zeros((self.defend_num, self.defend_num))

        #         pKey = ord('p')
        #         key = p.getKeyboardEvents()
        #         if pKey in key and key[pKey] & p.KEY_WAS_TRIGGERED:

    def init_agent(self):
        # 将地面设置为光滑平面
        planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(planeId, 0, lateralFriction=0, spinningFriction=0, rollingFriction=0)

        # # 相机设置
        # p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=110, cameraPitch=-30,
        #                              cameraTargetPosition=[-5, 5, 0.3])

        # 禁止圈
        drawCircle(self.forbidden_radius, [1, 1, 1], theta_delta=30 / 180 * math.pi)

        # 结束圈
        drawCircle(self.done_dis, [1, 1, 1], theta_delta=40 / 180 * math.pi)

        # 设置agent的初始朝向
        agentStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        # 加载守方智能体初始位置
        cur_angle = 0
        theta = math.pi * 2 / self.defend_num
        index = 0
        index_ = 0
        for i in range(self.defend_num):
            x = self.defend_radius * math.sin(cur_angle)
            y = self.defend_radius * math.cos(cur_angle)
            defendAgentStartPos = [x, y, 0]
            agentId = p.loadURDF("defendAgent.urdf", defendAgentStartPos, agentStartOrientation)
            self.defendAgentIds.append(agentId)
            self.defendId2index[agentId] = index
            self.id2Index[agentId] = index_
            index_ += 1
            index += 1
            cur_angle += theta

        # 加载攻方智能体初始位置
        cur_angle = random.randint(0, 360)
        theta = math.pi * 2 / self.attack_num
        index = 0
        for i in range(self.attack_num):
            x = self.attack_radius * math.sin(cur_angle)
            y = self.attack_radius * math.cos(cur_angle)
            attackAgentStartPos = [x, y, 0]
            agentId = p.loadURDF("attackAgent.urdf", attackAgentStartPos, agentStartOrientation)
            self.attackAgentIds.append(agentId)
            self.attackId2Index[agentId] = index
            self.id2Index[agentId] = index_
            index += 1
            index_ += 1
            cur_angle += theta

    def reset(self):

        # 设置agent的初始朝向
        agentStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        # 加载守方智能体初始位置
        cur_angle = 0
        theta = math.pi * 2 / self.defend_num
        for agentId in self.defendAgentIds:
            x = self.defend_radius * math.sin(cur_angle)
            y = self.defend_radius * math.cos(cur_angle)
            defendAgentStartPos = [x, y, 0]
            p.resetBasePositionAndOrientation(agentId, defendAgentStartPos, agentStartOrientation)
            cur_angle += theta

        # 加载攻方智能体初始位置
        cur_angle = random.randint(0, 360)
        theta = math.pi * 2 / self.attack_num
        for agentId in self.attackAgentIds:
            x = self.attack_radius * math.sin(cur_angle)
            y = self.attack_radius * math.cos(cur_angle)
            attackAgentStartPos = [x, y, 0]
            p.resetBasePositionAndOrientation(agentId, attackAgentStartPos, agentStartOrientation)
            cur_angle += theta

    def attackReset(self):
        self.reset()
        self.updateStateReward()
        a_state, _ = self.getAttackStateReward()
        return a_state, self.attack_adj

    def defendReset(self):
        self.reset()
        self.updateStateReward()
        d_state, _ = self.getDefendStateReward()
        return d_state, self.defend_adj

    def transformDefendState(self, state):
        g_state = []
        for i in range(self.defend_num):
            cur_state = state[i]
            g_cur_state = []
            g_cur_state.extend(cur_state[0])
            for j in range(self.reward_agent_num):
                g_cur_state.extend(cur_state[1][j])
            for k in range(self.reward_agent_num):
                g_cur_state.extend(cur_state[2][k])
            g_cur_state.append(cur_state[-1])
            g_state.append(g_cur_state)
        return g_state

    # 以对象的方式设置防守和进攻策略
    def set_defend_stratedy(self, defend_stratedy: DefendStrategy):
        self.defend_stratedy: DefendStrategy = defend_stratedy

    # 以对象的方式设置防守和进攻策略
    def set_attack_strategy(self, attack_strategy: AttackStrategy):
        self.attack_strategy: AttackStrategy = attack_strategy

    # 守方reward
    def defendReward(self, s, phi, angle1, angle2, velocity1, velocity2):

        # # 没有危险，不用追捕
        # if s > self.threat_dis and 130 < phi < 230:
        #     return 0

        # 计算相对角度，小于2倍的危险角度就认为可能发生碰撞
        re_angle = relativeAngle(phi, angle1)
        re_angle2 = relativeAngle(phi, angle2)
        theta = re_angle2 / 180 * math.sin(math.pi)
        sin_value = math.sin(theta) * velocity2 / (velocity1 + 0.01)

        if sin_value > 1:
            return self.not_to_catch_reward
        alpha = math.atan(sin_value)
        # if alpha < theta:(只有一个解)
        delta = math.pi - alpha - theta
        # delta = alpha - theta  (数形结合)
        d = s / math.sin(delta) * math.sin(theta)
        symbol = 1 if relativeAngleWithSymbol(phi, angle2) > 0 else -1
        target_angle = phi + alpha * symbol
        target_angle_delta = relativeAngle(target_angle, angle1)

        reward_item = d * target_angle_delta * self.factor
        if s > self.capture_dis:
            return -reward_item
        else:
            return self.capture_reward

    # 攻方reward
    def attackReward(self, s, angle):
        if s < self.done_dis:
            return 10
        if angle < self.done_dis:
            return 5

    def updateCurPositionsVelocities(self):
        for agent_id in self.attackAgentIds:
            position, _ = p.getBasePositionAndOrientation(agent_id)
            position = position[:2]
            self.agentCurPositions[self.id2Index[agent_id]] = position
            velocity, _ = p.getBaseVelocity(agent_id)
            self.agentCurVelocities[self.id2Index[agent_id]] = velocity

        for agent_id in self.defendAgentIds:
            position, _ = p.getBasePositionAndOrientation(agent_id)
            position = position[:2]
            self.agentCurPositions[self.id2Index[agent_id]] = position
            velocity, _ = p.getBaseVelocity(agent_id)
            self.agentCurVelocities[self.id2Index[agent_id]] = velocity

    def attackRewardDisAngle(self, dis, angle):
        # math.pi*d
        # phi - sin(phi)
        angle = 360 - angle if angle > 180 else angle
        angle = angle * math.pi / 180
        change_item = self.turn_radius * abs(angle - math.sin(angle))
        return -(dis + change_item) * self.attack_reward_factor
        
    def attackRewardDis(self, dis):
        return -dis * self.attack_reward_factor

    def updateStateReward(self):
        state = [[], []]
        reward = [[], []]

        # 更新当前智能体位置和速度
        self.updateCurPositionsVelocities()
        # 计算邻接矩阵
        self.updateDefendAdj()
        self.updateAttackAdj()

        # 攻方的观测， 再此过程中得到奖励
        for cur_agent_id in self.attackAgentIds:
            cur_position = self.agentCurPositions[self.id2Index[cur_agent_id]]
            cur_speed = self.agentCurVelocities[self.id2Index[cur_agent_id]]
            cur_velocity, cur_angle = velocityConversionVerse(cur_speed)
            base_angle = azimuthAngleWP(cur_position, self.target_position)
            cur_angle = azimuthAngleWA(base_angle, cur_angle)

            # [[cur_velocity, cur_angle], [[ther_agent_id, s, phi,
            # velocity, angle]...], [[ther_agent_id, s, phi, velocity, angle],...], dis]

            cur_observe = [[cur_velocity, cur_angle], [], [], -1]
            dis = getDis(self.target_position, cur_position)
            cur_observe[3] = dis

            # calculate attack reward
            attack_reward = self.attackRewardDisAngle(dis, cur_angle)
            reward[0].append(attack_reward)

            for other_agent_id in self.attackAgentIds:
                if cur_agent_id == other_agent_id:
                    continue
                other_position = self.agentCurPositions[self.id2Index[cur_agent_id]]
                dis = getDis(cur_position, other_position)

                if dis < self.communicate_radius:
                    s, phi = transferRelativePosition(cur_position, other_position, self.target_position)
                    speed = self.agentCurVelocities[self.id2Index[other_agent_id]]
                    velocity, angle = velocityConversionVerse(speed)
                    angle = azimuthAngleWA(base_angle, angle)
                    cur_observe[1].append([other_agent_id, s, phi, velocity, angle])

            for other_agent_id in self.defendAgentIds:
                other_position = self.agentCurPositions[self.id2Index[other_agent_id]]
                dis = getDis(cur_position, other_position)
                if dis < self.observe_radius:
                    s, phi = transferRelativePosition(cur_position, other_position, self.target_position)
                    speed = self.agentCurVelocities[self.id2Index[other_agent_id]]
                    velocity, angle = velocityConversionVerse(speed)
                    angle = azimuthAngleWA(base_angle, angle)
                    cur_observe[2].append([other_agent_id, s, phi, velocity, angle])

            state[0].append(cur_observe)

        # observe for defend
        for cur_agent_id in self.defendAgentIds:
            cur_position = self.agentCurPositions[self.id2Index[cur_agent_id]]
            cur_speed = self.agentCurVelocities[self.id2Index[cur_agent_id]]
            cur_velocity, cur_angle = velocityConversionVerse(cur_speed)
            base_angle = azimuthAngleWP(cur_position, self.target_position)
            cur_angle = azimuthAngleWA(base_angle, cur_angle)
            cur_observe = [[cur_velocity, cur_angle], [], [], -1]
            cur_observe[3] = getDis(self.target_position, cur_position)

            for other_agent_id in self.defendAgentIds:
                if cur_agent_id == other_agent_id:
                    continue
                other_position = self.agentCurPositions[self.id2Index[other_agent_id]]
                dis = getDis(cur_position, other_position)
                if dis < self.communicate_radius:
                    s, phi = transferRelativePosition(cur_position, other_position, self.target_position)
                    speed = self.agentCurVelocities[self.id2Index[other_agent_id]]
                    velocity, angle = velocityConversionVerse(speed)
                    angle = azimuthAngleWA(base_angle, angle)
                    cur_observe[1].append([other_agent_id, s, phi, velocity, angle])

            cur_observe_sort = sorted(cur_observe[1], key=lambda x: x[1])[:self.reward_agent_num]
            cur_observe_sort_ = sorted(cur_observe_sort, key=lambda x: x[2])
            if len(cur_observe_sort) < self.reward_agent_num:
                for i in range(len(cur_observe_sort), self.reward_agent_num):
                    cur_observe_sort.append([0, 0, 0, 0, 0])
            cur_observe[1] = cur_observe_sort_

            for other_agent_id in self.attackAgentIds:
                other_position = self.agentCurPositions[self.id2Index[other_agent_id]]
                dis = getDis(cur_position, other_position)
                if dis < self.observe_radius:
                    s, phi = transferRelativePosition(cur_position, other_position, self.target_position)
                    speed = self.agentCurVelocities[self.id2Index[other_agent_id]]
                    velocity, angle = velocityConversionVerse(speed)
                    angle = azimuthAngleWA(base_angle, angle)
                    cur_observe[2].append([other_agent_id, s, phi, velocity, angle])

            # todo 排序依据
            cur_observe_sort = sorted(cur_observe[2], key=lambda x: (x[1], x[2]))[:self.reward_agent_num]
            if len(cur_observe_sort) < self.reward_agent_num:
                for i in range(len(cur_observe_sort), self.reward_agent_num):
                    cur_observe_sort.append([0, 0, 0, 0, 0])
            cur_observe[2] = cur_observe_sort

            # todo
            # 选择最近的攻击方智能体进行防守
            cur_observe_ = cur_observe[2][0]
            if cur_observe_[0] != 0:
                defend_reward = self.defendReward(cur_observe_[1], cur_observe_[2], cur_angle, cur_observe_[4],
                                                  cur_velocity, cur_observe_[3])
            else:
                # 处理lazy？  未发现敌方
                defend_reward = self.not_find_reward
            reward[1].append(defend_reward)
            state[1].append(cur_observe)

        # 进入极端距离的负奖励
        ex_threat_r = self.getForbiddenReward()
        for i in range(self.defend_num):
            reward[1][i] += ex_threat_r

        # 修改reward
        self.state = state
        self.reward = reward

    def updateDefendAdj(self):
        for agent_id in self.defendAgentIds:
            pos1 = self.agentCurPositions[self.id2Index[agent_id]]
            for other_agent_id in self.defendAgentIds:
                pos2 = self.agentCurPositions[self.id2Index[other_agent_id]]
                if getDis(pos1, pos2) < self.communicate_radius:
                    self.defend_adj[self.defendId2index[agent_id]][self.defendId2index[other_agent_id]] = 1
                    self.defend_adj[self.defendId2index[other_agent_id]][self.defendId2index[agent_id]] = 1
        return self.defend_adj

    def updateAttackAdj(self):
        for agent_id in self.attackAgentIds:
            pos1 = self.agentCurPositions[self.id2Index[agent_id]]
            for other_agent_id in self.attackAgentIds:
                pos2 = self.agentCurPositions[self.id2Index[other_agent_id]]
                if getDis(pos1, pos2) < self.communicate_radius:
                    self.attack_adj[self.defendId2index[agent_id]][self.defendId2index[other_agent_id]] = 1
                    self.attack_adj[self.defendId2index[other_agent_id]][self.defendId2index[agent_id]] = 1
        return self.attack_adj

    def getAttackStateReward(self):
        return self.state[0], self.reward[0]

    def getDefendStateReward(self):
        state = self.transformDefendState(self.state[1])
        return state, self.reward[1]

    def getDone(self):
        for agent_id in self.attackAgentIds:
            a_position = self.agentCurPositions[self.id2Index[agent_id]]
            if getDis(a_position, self.target_position) < self.done_dis:
                return True
        return False

    def getForbiddenReward(self):
        for agent_id in self.attackAgentIds:
            a_position = self.agentCurPositions[self.id2Index[agent_id]]
            if getDis(a_position, self.target_position) < self.forbidden_radius:
                return self.forbidden_reward
        return 0

    def getInfo(self):
        pass

    # 训练好之后的运行方式
    def run_one_step(self):
        self.updateStateReward()
        a_state = self.state[0]
        a_reward = self.reward[0]
        d_state = self.state[1]
        d_reward = self.reward[1]
        self.defend_total_reward += sum(d_reward)
        # print(self.defend_total_reward)
        d_actions = self.defend_stratedy.generate_actions(d_state)
        a_actions = self.attack_strategy.generate_actions(a_state)
        self.apply_defend_action2(d_actions)
        self.apply_attack_action2(a_actions)
        p.stepSimulation()
        time.sleep(1 / 10)

    def run(self):
        for i in range(1000):
            self.run_one_step()

    # 防守方强化学习接口
    def apply_defend_action(self, defend_actions):
        # defend_action n个防守方的油门舵角
        # [[油门，舵角],[]]
        # 对每个智能体的速度进行改变
        a_state, a_reward = self.getAttackStateReward()
        attack_actions = self.attack_strategy.generate_actions(a_state)
        self.apply_attack_action2(attack_actions)
        self.apply_defend_action2(defend_actions)
        p.stepSimulation()

        self.updateStateReward()

        state, reward = self.getDefendStateReward()
        done = self.getDone()
        # state, adj, reward, done
        return state, self.defend_adj, reward, done,

    # 进攻方强化学习接口
    def apply_attack_action(self, attack_actions):
        d_state, d_reward = self.getDefendStateReward()
        defend_actions = self.defend_stratedy.generate_actions(d_state)
        self.apply_defend_action2(defend_actions)
        self.apply_attack_action2(attack_actions)
        p.stepSimulation()

        self.updateStateReward()

        state, reward = self.getAttackStateReward()
        done = self.getDone()

        return state, reward, done, self.attack_adj

    # 防守方行为接口
    def apply_defend_action2(self, defend_actions):
        # defend_actions n个防守方的速度
        # [[大小，方向],[]]
        # 对每个智能体的速度进行改变
        for agentId, defend_action in zip(self.defendAgentIds, defend_actions):
            oil = defend_action[0]
            rudder = defend_action[1]
            # 速度转换
            # 得到速度
            old_speed, old_w = p.getBaseVelocity(agentId)
            # todo
            old_position, old_orientation = p.getBasePositionAndOrientation(agentId)
            # old_position = list(self.agentCurPositions[self.id2Index[agentId]])
            # old_position.append(0)
            velocity, angle = velocityConversionVerse(old_speed)
            velocity, angle = self.changeSpeed(velocity, angle, oil, rudder)
            # print('defend:')
            # print(f'angle:{angle}  oil:{oil} rudder:{rudder} velocity: {velocity} old_position{old_position}')
            speed = velocityConversion(velocity, angle)
            # 设置加速度
            new_orientation = p.getQuaternionFromEuler([0, 0, -angle * math.pi / 180])
            p.resetBasePositionAndOrientation(agentId, old_position, new_orientation)
            p.resetBaseVelocity(agentId, speed, [0, 0, 0])

    # 进攻方行为接口
    def apply_attack_action2(self, attack_actions):
        for agentId, attack_action in zip(self.attackAgentIds, attack_actions):
            oil = attack_action[0]
            rudder = attack_action[1]
            old_speed, old_w = p.getBaseVelocity(agentId)
            # old_position = list(self.agentCurPositions[self.id2Index[agentId]])
            # old_position.append(0)
            old_position, old_orientation = p.getBasePositionAndOrientation(agentId)
            velocity, angle = velocityConversionVerse(old_speed)
            # print('attack:')
            # print(f'angle:{angle}  oil:{oil} rudder:{rudder} velocity: {velocity} old_position{old_position}')
            velocity, angle = self.changeSpeed(velocity, angle, oil, rudder)
            speed = velocityConversion(velocity, angle)

            # 设置加速度
            new_orientation = p.getQuaternionFromEuler([0, 0, -angle * math.pi / 180])
            p.resetBasePositionAndOrientation(agentId, old_position, new_orientation)
            p.resetBaseVelocity(agentId, speed, [0, 0, 0])

    # 速度改变机制
    def changeSpeed(self, velocity, angle, oil, rudder):
        velocity = velocity + oil
        velocity = min(self.max_velocity, velocity)
        velocity = max(0, velocity)
        angle = (angle + rudder) % 360
        return velocity, angle


class DefendAgentsEnv(gym.Env, ABC):
    def __init__(self, global_agents_env: GlobalAgentsEnv):
        super().__init__()
        self.global_agents_env = global_agents_env
        self.n_agent = self.global_agents_env.defend_num
        m_v = global_agents_env.max_velocity
        m_a = global_agents_env.max_turn_angle
        self.n_observation = 3 + 10 * self.global_agents_env.reward_agent_num
        self.n_action = 25
        self.actionIndex2OilRudder = [[0, 0], [0, 0.2 * m_a], [0, -0.2 * m_a], [0, m_a], [0, -m_a],
                                      [0.2 * m_v, 0], [0.2 * m_v, 0.2 * m_a], [0.2 * m_v, -0.2 * m_a], [0.2 * m_v, m_a],
                                      [0.2 * m_v, -m_a],
                                      [-1 * 0.2 * m_v, 0], [-1 * 0.2 * m_v, 0.2 * m_a], [-1 * 0.2 * m_v, -0.2 * m_a],
                                      [-1 * 0.2 * m_v, m_a], [-1 * 0.2 * m_v, -m_a],
                                      [m_v, 0], [m_v, 0.2 * m_a], [m_v, -0.2 * m_a], [m_v, m_a], [m_v, -m_a],
                                      [-m_v, 0], [-m_v, 0.2 * m_a], [-m_v, -0.2 * m_a], [-m_v, m_a], [-m_v, -m_a]]

    def step(self, actions):
        actions_ = []
        for action in actions:
            actions_.append(self.actionIndex2OilRudder[action])
        state, adj, reward, done = self.global_agents_env.apply_defend_action(actions_)
        return state, adj, reward, done

    def reset(self):
        state, adj = self.global_agents_env.defendReset()
        return state, adj

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass


class AttackAgentsEnv(gym.Env, ABC):
    def __init__(self, global_agents_env: GlobalAgentsEnv):
        super().__init__()
        self.global_agents_env = global_agents_env
        self.n_agent = self.global_agents_env.defend_num
        m_v = global_agents_env.max_velocity
        m_a = global_agents_env.max_turn_angle
        self.n_observation = 3 + 10 * self.global_agents_env.reward_agent_num
        self.n_action = 25
        self.actionIndex2OilRudder = [[0, 0], [0, 0.2 * m_a], [0, -0.2 * m_a], [0, m_a], [0, -m_a],
                                      [0.2 * m_v, 0], [0.2 * m_v, 0.2 * m_a], [0.2 * m_v, -0.2 * m_a], [0.2 * m_v, m_a],
                                      [0.2 * m_v, -m_a],
                                      [-1 * 0.2 * m_v, 0], [-1 * 0.2 * m_v, 0.2 * m_a], [-1 * 0.2 * m_v, -0.2 * m_a],
                                      [-1 * 0.2 * m_v, m_a], [-1 * 0.2 * m_v, -m_a],
                                      [m_v, 0], [m_v, 0.2 * m_a], [m_v, -0.2 * m_a], [m_v, m_a], [m_v, -m_a],
                                      [-m_v, 0], [-m_v, 0.2 * m_a], [-m_v, -0.2 * m_a], [-m_v, m_a], [-m_v, -m_a]]

    def step(self, actions):
        actions_ = []
        for action in actions:
            actions_.append(self.actionIndex2OilRudder[action])
        state, adj, reward, done = self.global_agents_env.apply_attack_action(actions_)
        return state, adj, reward, done

    def reset(self):
        state, adj = self.global_agents_env.attackReset()
        return state, adj

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass
