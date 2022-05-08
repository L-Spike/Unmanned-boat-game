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
import torch
from gym import spaces

import logging
from config import *

from model import DGN

level = logging.DEBUG  # INFO 、WARNING、ERROR、CRITICAL
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s() - %(message)s',
    level=logging.INFO)


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


def transferRelativePositionReverse(pos1, s, phi, pos3):
    angle2 = azimuthAngleWP(pos1, pos3)
    angle = (angle2 + phi) % 360
    dx = s * math.sin(angle * math.pi / 180)
    dy = s * math.cos(angle * math.pi / 180)
    return [pos1[0] + dx, pos1[1] + dy]


# s, phi = transferRelativePosition(cur_position, other_position, target_position)


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


actionIndex2OilRudder = [[0, 0], [0, 0.2 * max_turn_angle], [0, -0.2 * max_turn_angle], [0, max_turn_angle],
                         [0, -max_turn_angle],
                         [0.2 * max_velocity, 0], [0.2 * max_velocity, 0.2 * max_turn_angle],
                         [0.2 * max_velocity, -0.2 * max_turn_angle], [0.2 * max_velocity, max_turn_angle],
                         [0.2 * max_velocity, -max_turn_angle],
                         [-1 * 0.2 * max_velocity, 0], [-1 * 0.2 * max_velocity, 0.2 * max_turn_angle],
                         [-1 * 0.2 * max_velocity, -0.2 * max_turn_angle],
                         [-1 * 0.2 * max_velocity, max_turn_angle], [-1 * 0.2 * max_velocity, -max_turn_angle],
                         [max_velocity, 0], [max_velocity, 0.2 * max_turn_angle], [max_velocity, -0.2 * max_turn_angle],
                         [max_velocity, max_turn_angle], [max_velocity, -max_turn_angle],
                         [-max_velocity, 0], [-max_velocity, 0.2 * max_turn_angle],
                         [-max_velocity, -0.2 * max_turn_angle], [-max_velocity, max_turn_angle],
                         [-max_velocity, -max_turn_angle]]

actinIndex2SpeedAll = []  # -1 stay   0 - 7   i * 45


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
def changeAngleToConcrete(angle):
    b = angle % 45
    a = angle // 45
    if b == 0:
        return angle
    else:
        if b > 22.5:
            return (a + 1) * 45
        else:
            return a * 45


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
        states = states[0]
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
            logging.debug(f'agent_info: {agent_info}')
            velocity1 = agent_info[0][0]
            angle1 = agent_info[0][1]
            base_angle = agent_info[-1]
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
                if action_setting == "speed" and actinIndex == "all":
                    # todo 更新绝对角度和速度
                    action = [max_velocity, changeAngleToConcrete(base_angle)]
                else:
                    action = self.getActionsSimple(angle1)
            else:
                # 计算
                s = min_threat_agent_info[1]
                phi = min_threat_agent_info[2]
                velocity2 = min_threat_agent_info[3]
                angle2 = min_threat_agent_info[4]
                if action_setting == "speed" and actinIndex == "all":
                    action = self.getActionWithObstaclesBySpeed(s, phi, angle1, angle2, velocity1, velocity2, min_d,
                                                                min_delta)
                    action[1] = changeAngleToConcrete((action[1] + base_angle) % 360)
                else:
                    action = self.getActionWithObstacles(s, phi, angle1, angle2, velocity1, velocity2, min_d,
                                                         min_delta)
            actions.append(action)

        # 遍历所有攻方（友方）agent todo
        return actions

    def getActionsSimple(self, angle):
        angle = angle if angle < 180 else angle - 360
        rudder = -5 if angle > 0 else +5
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
            rudder = 5 * relative_angle_symbol
        else:
            rudder = 1 * relative_angle_symbol
        return oil, rudder

    def getActionWithObstaclesBySpeed(self, s, phi, angle1, angle2, velocity1, velocity2, d, delta):
        if phi > 0:
            target = (phi - 90) % 360
        else:
            target = (phi + 90) % 360
        return [max_velocity, target]

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

        theta = re_angle2 / 180 * math.pi
        sin_value = math.sin(theta) * velocity2 / (velocity1 + 0.01)
        # todo
        if sin_value > 1:
            return self.max_threat_threshold, -1, -1
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


class DRLAttackStrategy(AttackStrategy):
    def __init__(self, model_path):
        # super(DRLAttackStrategy, self).__init__()
        super().__init__()
        self.n_ant = attack_num
        self.observation_space = 3 + 10 * reward_agent_num
        self.n_action = 25
        self.model = DGN(attack_num, self.observation_space, hidden_dim, self.n_action)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.n_agent = attack_num

    def generate_actions(self, states):
        obs, adj = states
        action = []
        n_adj = adj + np.eye(self.n_ant)
        logging.debug(f'obs:{obs} \n n_adj:{n_adj}')
        obs = transformState(obs)
        q, a_w = self.model(torch.Tensor(np.array([obs])), torch.Tensor(np.array([n_adj])))
        q = q[0]
        for i in range(self.n_ant):
            a = q[i].argmax().item()
            action.append(actionIndex2OilRudder[a])
        return action


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


def transformState(state):
    g_state = []
    for cur_state in state:
        g_cur_state = []
        g_cur_state.extend(cur_state[0])
        for j in range(reward_agent_num):
            g_cur_state.extend(cur_state[1][j])
        for k in range(reward_agent_num):
            g_cur_state.extend(cur_state[2][k])
        g_cur_state.append(cur_state[-2])
        g_cur_state.append(cur_state[-1])
        g_state.append(g_cur_state)
    return g_state


def attackRewardDisAngle(dis, angle):
    # math.pi*d
    # phi - sin(phi)
    angle = 360 - angle if angle > 180 else angle
    angle = angle * math.pi / 180
    change_item = turn_radius * abs(angle - math.sin(angle))
    return -(dis + change_item) * attack_reward_factor


def attackRewardDis(dis):
    return -dis * attack_reward_factor


def defendRewardDisAngle(dis, target_angle_delta):
    change_item = turn_radius * target_angle_delta * math.pi / 180
    reward_item = - (dis + change_item) * defend_reward_factor
    return reward_item


def defendRewardDisAngleDirect(dis, target_angle_delta):
    change_item = turn_radius * target_angle_delta * math.pi / 180
    reward_item = - (dis + change_item) * defend_reward_factor
    return reward_item


def defendReward(s, phi, angle1, angle2, velocity1, velocity2):
    # # 没有危险，不用追捕
    # if s > self.threat_dis and 130 < phi < 230:
    #     return 0

    # 计算相对角度，小于2倍的危险角度就认为可能发生碰撞
    re_angle = relativeAngle(phi, angle1)
    re_angle2 = relativeAngle(phi, angle2)
    # theta = re_angle2 / 180 * math.sin(math.pi)
    theta = re_angle2 / 180 * math.pi
    sin_value = math.sin(theta) * velocity2 / (velocity1 + 0.01)

    if sin_value > 1:
        return not_to_catch_reward
    alpha = math.atan(sin_value)
    # if alpha < theta:(只有一个解)
    delta = math.pi - alpha - theta
    # delta = alpha - theta  (数形结合)
    d = s / math.sin(delta) * math.sin(theta)
    symbol = 1 if relativeAngleWithSymbol(phi, angle2) > 0 else -1
    target_angle = phi + alpha * symbol
    target_angle_delta = relativeAngle(target_angle, angle1)

    if s > capture_dis:
        return defendRewardDisAngle(d, target_angle_delta)
    else:
        return capture_reward


def defendRewardSimple(s, phi, angle1, angle2, velocity1, velocity2):
    if s > capture_dis:
        return -s
    else:
        return capture_reward


def defendRewardSimpleV2(s, s_t):
    if s > capture_dis:
        return -s
    else:
        return capture_reward + (s_t - forbidden_radius) * 2


class GlobalAgentsEnv:
    def __init__(self, defend_stratedy, attack_strategy,
                 render: bool = False):
        self.defendWords = {}
        self.defendLineTime = -1
        self.defendLines = {}
        self.cur_step = 0
        # self.action_space = spaces.Discrete(5)
        # self.observation_space = spaces.Discrete(2)

        self.defend_stratedy: DefendStrategy = defend_stratedy
        self.attack_strategy: AttackStrategy = attack_strategy

        # 连接到渲染环境
        self.physicsClientId = p.connect(p.GUI if render else p.DIRECT)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.81)
        self.defendAgentIds = []
        self.attackAgentIds = []
        self.id2Index = {}
        self.agentCurPositions = [0] * (attack_num + defend_num)
        self.agentCurVelocities = [0] * (attack_num + defend_num)
        self.defendId2index = {}
        self.attackId2Index = {}
        self.init_agent()

        self.defend_total_reward = 0
        self.state = None
        self.reward = None
        self.attack_adj = np.zeros((attack_num, attack_num))
        self.defend_adj = np.zeros((defend_num, defend_num))

        #         pKey = ord('p')
        #         key = p.getKeyboardEvents()
        #         if pKey in key and key[pKey] & p.KEY_WAS_TRIGGERED:

    def init_agent(self):
        # 将地面设置为光滑平面
        planeId = p.loadURDF("plane.urdf")
        logging.debug(f'planeID:{planeId}')
        p.changeDynamics(planeId, 0, lateralFriction=0, spinningFriction=0, rollingFriction=0)

        # # 相机设置
        # p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=110, cameraPitch=-30,
        #                              cameraTargetPosition=[-5, 5, 0.3])

        # 禁止圈
        drawCircle(forbidden_radius, [249 / 255, 205 / 255, 173 / 255], theta_delta=30 / 180 * math.pi)

        # 结束圈
        drawCircle(done_dis, [130 / 255, 32 / 255, 43 / 255], theta_delta=40 / 180 * math.pi)

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

        # 加载攻方智能体初始位置
        cur_angle = random.randint(0, 360)
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

        logging.debug(f'attack ids:{self.attackAgentIds}')
        logging.debug(f'defend ids: {self.defendAgentIds}')

    def reset(self):

        self.cur_step = 0
        # 设置agent的初始朝向
        agentStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        # 加载守方智能体初始位置
        cur_angle = 0
        theta = math.pi * 2 / defend_num
        for agentId in self.defendAgentIds:
            x = defend_radius * math.sin(cur_angle)
            y = defend_radius * math.cos(cur_angle)
            defendAgentStartPos = [x, y, 0]
            p.resetBasePositionAndOrientation(agentId, defendAgentStartPos, agentStartOrientation)
            cur_angle += theta

        # 加载攻方智能体初始位置
        cur_angle = random.randint(0, 360)
        theta = math.pi * 2 / attack_num
        for agentId in self.attackAgentIds:
            x = attack_radius * math.sin(cur_angle)
            y = attack_radius * math.cos(cur_angle)
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
        d_state = transformState(d_state)
        return d_state, self.defend_adj

    # 以对象的方式设置防守和进攻策略
    def set_defend_stratedy(self, defend_stratedy: DefendStrategy):
        self.defend_stratedy: DefendStrategy = defend_stratedy

    # 以对象的方式设置防守和进攻策略
    def set_attack_strategy(self, attack_strategy: AttackStrategy):
        self.attack_strategy: AttackStrategy = attack_strategy

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

            # [[cur_velocity, cur_angle], [[ther_agent_id, s, phi,
            # velocity, angle]...], [[ther_agent_id, s, phi, velocity, angle],...], dis]

            cur_observe = [[cur_agent_id, cur_velocity, cur_angle], [], [], []]
            s = getDis(target_position, cur_position)
            phi = azimuthAngleWP(cur_position, target_position)
            cur_observe[3] = [s, phi]

            # calculate attack reward
            if use_angle:
                attack_reward = attackRewardDisAngle(s, cur_angle)
            else:
                attack_reward = attackRewardDis(s)
            reward[0].append(attack_reward)

            for other_agent_id in self.attackAgentIds:
                if cur_agent_id == other_agent_id:
                    continue
                other_position = self.agentCurPositions[self.id2Index[cur_agent_id]]
                dis = getDis(cur_position, other_position)
                if dis < communicate_radius:
                    phi = azimuthAngleWP(cur_position, other_position)
                    speed = self.agentCurVelocities[self.id2Index[other_agent_id]]
                    velocity, angle = velocityConversionVerse(speed)
                    cur_observe[1].append([other_agent_id, dis, phi, velocity, angle])

            cur_observe_sort = sorted(cur_observe[1], key=lambda x: (x[1], x[2]))[:reward_agent_num]
            if len(cur_observe_sort) < reward_agent_num:
                for i in range(len(cur_observe_sort), reward_agent_num):
                    cur_observe_sort.append([0, 0, 0, 0, 0])
            cur_observe[1] = cur_observe_sort

            for other_agent_id in self.defendAgentIds:
                other_position = self.agentCurPositions[self.id2Index[other_agent_id]]
                dis = getDis(cur_position, other_position)
                if dis < observe_radius:
                    phi = azimuthAngleWP(cur_position, other_position)
                    speed = self.agentCurVelocities[self.id2Index[other_agent_id]]
                    velocity, angle = velocityConversionVerse(speed)
                    cur_observe[2].append([other_agent_id, dis, phi, velocity, angle])

            cur_observe_sort = sorted(cur_observe[2], key=lambda x: (x[1], x[2]))[:reward_agent_num]
            if len(cur_observe_sort) < reward_agent_num:
                for i in range(len(cur_observe_sort), reward_agent_num):
                    cur_observe_sort.append([0, 0, 0, 0, 0])
            cur_observe[2] = cur_observe_sort
            state[0].append(cur_observe)

        # observe for defend
        self.defendLineTime = (self.defendLineTime + 1) % 100
        for cur_agent_id in self.defendAgentIds:
            cur_position = self.agentCurPositions[self.id2Index[cur_agent_id]]
            cur_speed = self.agentCurVelocities[self.id2Index[cur_agent_id]]
            cur_velocity, cur_angle = velocityConversionVerse(cur_speed)
            cur_observe = [[cur_agent_id, cur_velocity, cur_angle], [], [], []]
            s = getDis(target_position, cur_position)
            phi = azimuthAngleWP(cur_position, target_position)
            cur_observe[3] = [s, phi]

            for other_agent_id in self.defendAgentIds:
                if cur_agent_id == other_agent_id:
                    continue
                other_position = self.agentCurPositions[self.id2Index[other_agent_id]]
                dis = getDis(cur_position, other_position)
                if dis < communicate_radius:
                    phi = azimuthAngleWP(cur_position, other_position)
                    speed = self.agentCurVelocities[self.id2Index[other_agent_id]]
                    velocity, angle = velocityConversionVerse(speed)
                    cur_observe[1].append([other_agent_id, dis, phi, velocity, angle])

            cur_observe_sort = sorted(cur_observe[1], key=lambda x: x[1])[:reward_agent_num]
            cur_observe_sort_ = sorted(cur_observe_sort, key=lambda x: x[2])
            if len(cur_observe_sort) < reward_agent_num:
                for i in range(len(cur_observe_sort), reward_agent_num):
                    cur_observe_sort.append([0, 0, 0, 0, 0])
            cur_observe[1] = cur_observe_sort_

            for other_agent_id in self.attackAgentIds:
                other_position = self.agentCurPositions[self.id2Index[other_agent_id]]
                dis = getDis(cur_position, other_position)
                if dis < observe_radius:
                    phi = azimuthAngleWP(cur_position, other_position)
                    speed = self.agentCurVelocities[self.id2Index[other_agent_id]]
                    velocity, angle = velocityConversionVerse(speed)
                    cur_observe[2].append([other_agent_id, s, phi, velocity, angle])

            # todo 排序依据
            cur_observe_sort = sorted(cur_observe[2], key=lambda x: (x[1], x[2]))[:reward_agent_num]
            if len(cur_observe_sort) < reward_agent_num:
                for i in range(len(cur_observe_sort), reward_agent_num):
                    cur_observe_sort.append([0, 0, 0, 0, 0])
            cur_observe[2] = cur_observe_sort

            # todo
            # 选择最近的攻击方智能体进行防守
            cur_observe_ = cur_observe[2][0]

            from_ = list(cur_position[:])
            from_.append(0)
            if cur_observe_[0] != 0:
                defend_reward = defendRewardSimpleV2(cur_observe_[1], cur_observe[3][0])

                if self.defendLineTime == 0 and DEBUG:
                    if cur_agent_id in self.defendLines:
                        p.removeUserDebugItem(self.defendLines[cur_agent_id])
                    to_ = transferRelativePositionReverse(pos1=cur_position, s=cur_observe_[1], phi=cur_observe_[2],
                                                          pos3=target_position)
                    to_.append(0)
                    self.defendLines[cur_agent_id] = p.addUserDebugLine(
                        lineFromXYZ=from_,
                        lineToXYZ=to_,
                        lineColorRGB=[34 / 255, 34 / 255, 56 / 255],
                        lineWidth=3
                    )
            else:
                # 处理lazy？  未发现敌方
                defend_reward = not_find_reward
                if self.defendLineTime == 0 and DEBUG:
                    if cur_agent_id in self.defendWords:
                        p.removeUserDebugItem(self.defendWords[cur_agent_id])
                    self.defendWords[cur_agent_id] = p.addUserDebugText(
                        text="LAZY!",
                        textPosition=from_,
                        textColorRGB=[5 / 255, 39 / 255, 175 / 255],  # 5；G:39；B:175
                        textSize=3
                    )
            reward[1].append(defend_reward)
            state[1].append(cur_observe)

        # 进入极端距离的负奖励
        if use_forbidden_reward:
            ex_threat_r = self.getForbiddenReward()
            for i in range(defend_num):
                reward[1][i] += ex_threat_r

        # 修改reward
        self.state = state
        self.reward = reward

    def updateDefendAdj(self):
        for agent_id in self.defendAgentIds:
            pos1 = self.agentCurPositions[self.id2Index[agent_id]]
            for other_agent_id in self.defendAgentIds:
                pos2 = self.agentCurPositions[self.id2Index[other_agent_id]]
                if getDis(pos1, pos2) < communicate_radius:
                    self.defend_adj[self.defendId2index[agent_id]][self.defendId2index[other_agent_id]] = 1
                    self.defend_adj[self.defendId2index[other_agent_id]][self.defendId2index[agent_id]] = 1
        return self.defend_adj

    def updateAttackAdj(self):
        for agent_id in self.attackAgentIds:
            pos1 = self.agentCurPositions[self.id2Index[agent_id]]
            for other_agent_id in self.attackAgentIds:
                pos2 = self.agentCurPositions[self.id2Index[other_agent_id]]
                if getDis(pos1, pos2) < communicate_radius:
                    self.attack_adj[self.attackId2Index[agent_id]][self.attackId2Index[other_agent_id]] = 1
                    self.attack_adj[self.attackId2Index[other_agent_id]][self.attackId2Index[agent_id]] = 1
        return self.attack_adj

    def getAttackStateReward(self):
        # state = self.transformAttackState(self.state[0])
        state = self.state[0]
        return state, self.reward[0]

    def getDefendStateReward(self):
        # state = self.transformDefendState(self.state[1])
        state = self.state[1]
        return state, self.reward[1]

    def getDone(self):
        for agent_id in self.attackAgentIds:
            a_position = self.agentCurPositions[self.id2Index[agent_id]]
            if getDis(a_position, target_position) < done_dis:
                return True
        return False

    def getForbiddenReward(self):
        for agent_id in self.attackAgentIds:
            a_position = self.agentCurPositions[self.id2Index[agent_id]]
            if getDis(a_position, target_position) < forbidden_radius:
                return forbidden_reward
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
        self.apply_defend_action_by_oil(d_actions)
        self.apply_attack_action_by_oil(a_actions)
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
        self.cur_step += 1
        a_state, a_reward = self.getAttackStateReward()
        attack_actions = self.attack_strategy.generate_actions([a_state, self.attack_adj])
        logging.debug(f'a_state:{a_state}', f'attack_actions:{attack_actions}')
        if action_setting == "speed":
            self.apply_attack_action_by_speed(attack_actions)
            self.apply_defend_action_by_speed(defend_actions)
        else:
            self.apply_attack_action_by_oil(attack_actions)
            self.apply_defend_action_by_oil(defend_actions)
        p.stepSimulation()

        self.updateStateReward()

        state, reward = self.getDefendStateReward()
        state = transformState(state)
        done = self.getDone()

        if use_done:
            # todo
            if done:
                reward = [-defend_succeed_reward] * defend_num
            elif self.cur_step == max_step:
                reward = [defend_succeed_reward] * defend_num

            return state, self.defend_adj, reward, done

    # 进攻方强化学习接口
    def apply_attack_action(self, attack_actions):

        self.cur_step += 1
        d_state, d_reward = self.getDefendStateReward()
        defend_actions = self.defend_stratedy.generate_actions(d_state)
        self.apply_defend_action_by_oil(defend_actions)
        self.apply_attack_action_by_oil(attack_actions)
        p.stepSimulation()

        self.updateStateReward()

        state, reward = self.getAttackStateReward()
        state = transformState(state)
        done = self.getDone()

        if use_done:
            # todo
            if done:
                reward = [attack_succeed_reward] * attack_num
            elif self.cur_step == max_step:
                reward = [- attack_succeed_reward] * attack_num

        return state, self.attack_adj, reward, done

    # 防守方行为接口
    def apply_defend_action_by_oil(self, defend_actions):
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

    def apply_defend_action_by_speed(self, defend_actions):
        for agentId, defend_action in zip(self.defendAgentIds, defend_actions):
            velocity, angle = defend_action
            old_position, old_orientation = p.getBasePositionAndOrientation(agentId)
            speed = velocityConversion(velocity, angle)
            # 设置加速度
            new_orientation = p.getQuaternionFromEuler([0, 0, -angle * math.pi / 180])
            p.resetBasePositionAndOrientation(agentId, old_position, new_orientation)
            p.resetBaseVelocity(agentId, speed, [0, 0, 0])

    # 进攻方行为接口
    def apply_attack_action_by_oil(self, attack_actions):
        logging.debug(f'action:{attack_actions}')
        for agentId, attack_action in zip(self.attackAgentIds, attack_actions):
            logging.debug(f'df:{attack_action}')
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

    def apply_attack_action_by_speed(self, attack_actions):
        for agentId, attack_action in zip(self.attackAgentIds, attack_actions):
            velocity, angle = attack_action
            old_position, old_orientation = p.getBasePositionAndOrientation(agentId)
            speed = velocityConversion(velocity, angle)
            # 设置加速度
            new_orientation = p.getQuaternionFromEuler([0, 0, -angle * math.pi / 180])
            p.resetBasePositionAndOrientation(agentId, old_position, new_orientation)
            p.resetBaseVelocity(agentId, speed, [0, 0, 0])

    # 速度改变机制
    def changeSpeed(self, velocity, angle, oil, rudder):
        velocity = velocity + oil
        velocity = min(max_velocity, velocity)
        velocity = max(0, velocity)
        angle = (angle + rudder) % 360
        return velocity, angle


class DefendAgentsEnv(gym.Env, ABC):
    def __init__(self, global_agents_env: GlobalAgentsEnv):
        super().__init__()
        self.global_agents_env = global_agents_env
        self.n_agent = defend_num
        self.n_observation = 5 + 10 * reward_agent_num
        if action_setting == "speed" and actinIndex == "all":
            self.n_action = 9
        else:
            self.n_action = 25

    def step(self, actions):
        actions_ = []
        for action in actions:
            if action_setting == "speed" and actinIndex == "all":
                if action == -1:
                    actions_.append([0, 0])
                else:
                    actions_.append([max_velocity, action * 45])
            else:
                actions_.append(actionIndex2OilRudder[action])
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
        self.n_agent = attack_num
        self.n_observation = 3 + 10 * reward_agent_num
        self.n_action = 25

    def step(self, actions):
        actions_ = []
        for action in actions:
            actions_.append(actionIndex2OilRudder[action])
        state, adj, reward, done = self.global_agents_env.apply_attack_action(actions_)
        return state, adj, reward, done

    def reset(self):
        state, adj = self.global_agents_env.attackReset()
        return state, adj

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass
