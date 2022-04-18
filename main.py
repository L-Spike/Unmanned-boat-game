from attackDefendEnv import *

env = GlobalAgentsEnv(RandomDefenfStrategy(),
                      SimpleAttackStrategy(
                          threat_angle=45,
                          threat_dis=0.5,  # 攻击策略
                          threat_angle_delta=10,
                          small_angle=10,
                          delta_t=0
                      ),
                      attack_num=4,
                      attack_radius=7,
                      defend_num=4,
                      defend_radius=2.5,
                      forbidden_radius=1.5,
                      threat_angle=45,
                      threat_angle_delta=10,
                      threat_dis=0.5,  # 奖励
                      capture_dis=0.2,
                      render=False
                      )

env.run()
p.disconnect()

