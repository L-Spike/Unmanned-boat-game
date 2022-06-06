import numpy as np
import math
import colorsys
from config import *
from DAF_config import r_s

neg_inf = -9e+100


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
        if dy < 0:
            angle = math.pi
        else:
            angle = 0
    elif dy == 0:
        if dx > 0:
            angle = math.pi / 2.0
        else:
            angle = 3 * math.pi / 2.0
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


def drawCircle(radius, color, theta_delta, p):
    from_angle = 0
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
        g_cur_state.extend(cur_state[-1])
        g_state.append(g_cur_state)
    return g_state


def attackAction(s, phi, symbol):
    attack_epsilon = 0.9
    if s > 2 * attack_threat_dis / 3:
        if np.random.rand() < attack_epsilon:
            target_angle = (phi + (symbol * 45)) % 360
        else:
            target_angle = (phi + (-1 * symbol * 45)) % 360
    elif s > attack_threat_dis / 3:
        if np.random.rand() < attack_epsilon:
            target_angle = (phi + (symbol * 90)) % 360
        else:
            target_angle = (phi + (-1 * symbol * 90)) % 360
    else:
        if np.random.rand() < attack_epsilon:
            target_angle = (phi + (symbol * 135)) % 360
        else:
            target_angle = (phi + (-1 * symbol * 135)) % 360
    return target_angle


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
        return capture_reward + (s_t - done_dis) * 2


def defendRewardSimpleV3(s, s_t2, s_t):
    if s_t2 > ignore_radius:
        if s_t > ignore_radius + 4:
            return too_far_reward
        else:
            return defend_ok_reward
    else:
        if s > capture_dis:
            return -s + - (ignore_radius - s_t2)
        else:
            return - (ignore_radius - s_t2)


def defendRewardTotal(s_t2):
    if s_t2 > ignore_radius:
        return 0
    else:
        return -(ignore_radius - s_t2)


def defendRewardTotalV2(s_t2):
    if s_t2 > ignore_radius:
        return defend_ok_reward
    else:
        return -(ignore_radius - s_t2)


def defendRewardTotalInd(s_t):
    if s_t > ignore_radius + 4:
        return too_far_reward
    else:
        return 0


def get_before(x, x_r, x_center, degree):
    angle1 = azimuthAngleWP(x_center, x)
    angle2 = azimuthAngleWP(x_center, x_r)
    delta_angle = (angle2 - angle1) % 360
    a = 1
    if delta_angle > 180:
        a = -1
        delta_angle = 360 - delta_angle
    add_num = int(delta_angle // degree)
    add_points = []
    for i in range(add_num):
        angle = (angle1 + a * degree * (i + 1)) % 360
        dis = getDis(x_center, x_r)
        dx = dis * math.sin(angle * math.pi / 180)
        dy = dis * math.cos(angle * math.pi / 180)
        point = [x_center[0] + dx, x_center[1] + dy]
        add_points.append(point)
    return add_points


def mask_map(r1, r2, map_res, map_width):
    length = int(map_width / map_res)
    map_mask = neg_inf * np.ones((length, length))
    t_coord = [0, 0]
    counter = 0
    for i in range(length):
        for j in range(length):
            coord = [i * map_res - map_width / 2, j * map_res - map_width / 2]
            dis = math.sqrt((coord[0] - t_coord[0]) ** 2 + (coord[1] - t_coord[1]) ** 2)
            if r2 > dis > r1:
                map_mask[i][j] = 1
                counter += 1
    return map_mask, counter


def update_records(records, map_pos, x, T):
    # other_dist = np.sqrt(np.square(x[r, 0] - map_pos[:, :, 0]) + np.square(x[r, 1] - map_pos[:, :, 1]))
    n = x.shape[0]
    for i in range(n):
        other_dist = np.sqrt(np.square(x[i, 0] - map_pos[:, :, 0]) + np.square(x[i, 1] - map_pos[:, :, 1]))
        records[other_dist < r_s] = T


def squd_norm(z):
    return np.add(np.square(z[:, :, 0]), np.square(z[:, :, 1]))


def sigma_norm(z, efs):
    return (1.0 / efs) * (np.sqrt(1.0 + efs * z * z) - 1.0)


def get_gap(r):
    num_rows = r.shape[0]
    gap = np.zeros((num_rows, num_rows, 2), dtype=np.float32)

    for a in range(1, num_rows):
        for b in range(a):
            gap[a, b, :] = r[b, :] - r[a, :]

    gap[:, :, 0] = gap[:, :, 0] - gap[:, :, 0].T
    gap[:, :, 1] = gap[:, :, 1] - gap[:, :, 1].T

    return gap


def action_func(z, d, c1):
    z = z / d
    p_h = np.zeros(z.shape)
    i = z <= 1
    p_h[i] = -(c1 * 0.5 * np.pi / d) * np.sin(0.5 * np.pi * (z[i] + 1))

    return p_h


def update_individual_record(cell_map, map_pos, x, T, r_s, fail_list):
    for r in range(x.shape[0]):
        if r in fail_list:
            continue
        other_dist = np.sqrt(np.square(x[r, 0] - map_pos[:, :, 0]) + np.square(x[r, 1] - map_pos[:, :, 1]))
        temp_map = cell_map[:, :, 2 * r]
        temp_ind = cell_map[:, :, 2 * r + 1]
        temp_map[other_dist <= r_s] = T
        temp_ind[other_dist <= r_s] = r
        cell_map[:, :, 2 * r] = temp_map
        cell_map[:, :, 2 * r + 1] = temp_ind

    return cell_map


def fuse_record(cell_map, nbr, fail_list):
    # cell map 是全局的融合观测  事后分析存在（分析累计覆盖率）
    #
    for i in range(1, nbr.shape[0]):
        if i in fail_list:
            continue
        for j in range(i):
            if j in fail_list:
                continue
            if nbr[i, j] == 1:
                max_time = np.maximum(cell_map[:, :, 2 * i], cell_map[:, :, 2 * j])
                temp_cell_i = cell_map[:, :, 2 * i + 1]
                temp_cell_i[cell_map[:, :, 2 * i] != max_time] = j
                cell_map[:, :, 2 * i] = max_time
                cell_map[:, :, 2 * j] = max_time
                cell_map[:, :, 2 * i + 1] = temp_cell_i
                cell_map[:, :, 2 * j + 1] = temp_cell_i

    return cell_map


def fuse_all_records(cell_map, fused_scan_record, fail_list):
    num_agents = int(cell_map.shape[2] / 2)
    for i in range(0, num_agents * 2, 2):
        if i / 2 in fail_list:
            continue
        max_time = np.maximum(fused_scan_record[:, :, 0], cell_map[:, :, i])
        fused_scan_record[:, :, 1][max_time != fused_scan_record[:, :, 0]] = (i / 2) + 1
        fused_scan_record[:, :, 0] = max_time

    return fused_scan_record


def get_colors(num_colors):
    colors = []
    for c in np.arange(0., 360., 360. / num_colors):
        (r, g, b) = colorsys.hls_to_rgb(c / 360., (50 + np.random.rand() * 10) / 100.,
                                        (90 + np.random.rand() * 10) / 100.)
        colors.append((r, g, b))
    return colors
