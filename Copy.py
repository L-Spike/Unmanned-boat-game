import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)


cylinderStartPos = [0,0,0.3]
cylinderStartPos1 = [1,1,0.3]
cylinderStartOrientation = p.getQuaternionFromEuler([0,0,0])

planeId = p.loadURDF("plane.urdf")
cylinderId = p.loadURDF("cylinder.urdf",cylinderStartPos,cylinderStartOrientation)
cylinderId2 = p.loadURDF("cylinder.urdf",cylinderStartPos1,cylinderStartOrientation)

p.changeDynamics(planeId,-1,lateralFriction = 0,spinningFriction = 0,rollingFriction = 0)


for k in range(3):
    for i in range (480):
        p.applyExternalForce(cylinderId,-1,[1,0,0],[0,0,0.3],p.LINK_FRAME)
        pKey = ord('p')
        key = p.getKeyboardEvents()
        #deccelKey = ord('e')
        if pKey in key and key[pKey] & p.KEY_WAS_TRIGGERED:
            baseinfo = p.getBasePositionAndOrientation(cylinderId)
            print(baseinfo)
        p.stepSimulation()
        time.sleep(1./240.)
    for j in range (480):
        p.applyExternalForce(cylinderId,-1,[-1.4,0,0],[0,0,0.3],p.LINK_FRAME)
        pKey = ord('p')
        key = p.getKeyboardEvents()
        #deccelKey = ord('e')
        if pKey in key and key[pKey] & p.KEY_WAS_TRIGGERED:
            baseinfo = p.getBasePositionAndOrientation(cylinderId)
            print(baseinfo)
        p.stepSimulation()
        time.sleep(1./240.)
p.disconnect()









