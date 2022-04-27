from attackDefendEnv import *
env = GlobalAgentsEnv(RandomDefendStrategy(),
                      SimpleAttackStrategy(),
                      render=True
                      )

env.run()
p.disconnect()

