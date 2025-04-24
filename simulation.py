import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

from pathlib import Path
import argparse
import math
import time


# --------------------------------- Constants --------------------------------- #

L2 = 0.12
L3 = 0.12


def map_to_robot(joints):
    """
    Map the joint values from the kinematic model's joint space 
    to the actual robot's joint space.
    """
    map_joint_1 = joints[0]
    map_joint_2 = -joints[1] + math.pi / 2
    map_joint_3 = -joints[2] + math.pi
    map_joint_4 = joints[3]
    return (map_joint_1, map_joint_2, map_joint_3, map_joint_4)

def inverse_kinematics(point):
    """
    Compute the inverse kinematics for the SCARA arm.
    """

    x = point[0]
    y = point[1]
    z = point[2]
    yaw = point[3]

    distance = math.sqrt(x**2 + y**2)
    
    # Check if the point is reachable
    if distance > (L2 + L3):
        return None, None  # Point is out of reach
    
    q1 = z

    c3 = (x**2 + y**2 - L2**2 - L3**2) / (2 * L2 * L3)
    s3_a = math.sqrt(1 - c3**2)
    s3_b = -s3_a

    q3_a = math.atan2(s3_a, c3)
    q3_b = math.atan2(s3_b, c3)
    
    gamma = math.atan2(y, x)
    q2_a = gamma - math.atan2(L3 * s3_a, L2 + L3 * c3)
    q2_b = gamma - math.atan2(L3 * s3_b, L2 + L3 * c3)
    
    q4_a = yaw - (q2_a + q3_a)
    q4_b = yaw - (q2_b + q3_b)
    
    return (q1, q2_a, q3_a, q4_a), (q1, q2_b, q3_b, q4_b)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="OpenSCARA PyBullet simulation.")
    parser.add_argument('-x', '--x', type=float, default=0.2, help="X coordinate of the task point")
    parser.add_argument('-y', '--y', type=float, default=0, help="Y coordinate of the task point")
    parser.add_argument('-z', '--z', type=float, default=0.15, help="Z coordinate of the task point")
    parser.add_argument('-w', '--yaw', type=float, default=1.57, help="Yaw value of the task point")
    parser.add_argument("-u", "--URDF", type=str, default='OpenSCARA-Hardware/SCARA.urdf', help="Path to SCARA's URDF")
    parser.add_argument('-d', '--dt', type=float, default=0.02, help="Time delta for update (default=0.02=50Hz)")
    parser.add_argument('-v', '--video_path', type=str, default=None, help="If provided, the script will save an mp4 of the simulation on the path")

    args = parser.parse_args()

    # --------------------------------- PyBullet --------------------------------- #

    # Connect to physics server
    # physics = p.connect(p.GUI)

    physics = bc.BulletClient(connection_mode=p.GUI)
    physics.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    physics.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    physics.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    physics.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)  # Disable shadows

    # Set initial camera view
    default_camera_distance = 0.5
    default_camera_roll = 0
    default_camera_yaw = 0
    default_camera_pitch = -45
    default_camera_target = [0, 0, 0]

    physics.resetDebugVisualizerCamera(
        cameraDistance=default_camera_distance,
        cameraYaw=default_camera_yaw,
        cameraPitch=default_camera_pitch,
        cameraTargetPosition=default_camera_target
    )

    # Load additional data and set gravity
    physics.setAdditionalSearchPath(pybullet_data.getDataPath())
    physics.setGravity(0, 0, -9.81)

    # Load plane and robot URDF files
    planeId = physics.loadURDF("plane.urdf")
    cubeStartPos = [0, 0, 0]
    cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

    urdf_file = Path(args.URDF)
    robotID = physics.loadURDF(str(urdf_file), cubeStartPos, cubeStartOrientation, flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=True)

    # List revolute and prismatic joints
    revolute_joints = []
    prismatic_joints = []
    num_joints = physics.getNumJoints(robotID)
    for joint_index in range(num_joints):
        joint_info = physics.getJointInfo(robotID, joint_index)
        joint_type = joint_info[2]  # Joint type is the third element in joint_info tuple
        if joint_type == physics.JOINT_REVOLUTE:
            revolute_joints.append(joint_index)
            print(f"Joint {joint_index} is revolute: {joint_info[1].decode('utf-8')}")
        elif joint_type == physics.JOINT_PRISMATIC:
            prismatic_joints.append(joint_index)
            print(f"Joint {joint_index} is prismatic: {joint_info[1].decode('utf-8')}")

    # We know the joints are well formatted (they follow the order of the kinematic model i.e. joint 1 is prismatic, joint 2 to 4 are revolute)
    joint_mapping = {
        'joint_1': prismatic_joints[0],
        'joint_2': revolute_joints[0],
        'joint_3': revolute_joints[1],
        'joint_4': revolute_joints[2]
    }

    # ------------------------------ Simulation loop ----------------------------- #

    if args.video_path:

        video_path = Path(args.video_path)
        folder_path = video_path.parent
        if not folder_path.exists():
            folder_path.mkdir(exist_ok=True, parents=True)

        physics.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, args.video_path)

    try:

        t = 0.0
        # dt = 1. / 240.
        dt = args.dt

        # Create the target point (task space)
        point = (args.x, args.y, args.z, args.yaw)
        point = (0.2, 0, 0.17, 0)
        print(f"Target ee pose (point in task space): {point}")

        # Compute inverse kinematics
        solution_1, solution_2 = inverse_kinematics(point)
        print(f"Configuration 1: {solution_1}")
        print(f"Configuration 2: {solution_2}")

        while p.isConnected():

            # Step the simulations
            physics.stepSimulation()

            # If any, apply the solution
            solution = solution_1 if solution_1 is not None else solution_2
            if solution is not None:

                solution_mapped = map_to_robot(solution)
                q1 = solution_mapped[0]
                q2 = solution_mapped[1]
                q3 = solution_mapped[2]
                q4 = solution_mapped[3]

                # Move the joints to the target position
                p.setJointMotorControl2(robotID, joint_mapping['joint_1'], p.POSITION_CONTROL, targetPosition=q1)
                p.setJointMotorControl2(robotID, joint_mapping['joint_2'], p.POSITION_CONTROL, targetPosition=q2)
                p.setJointMotorControl2(robotID, joint_mapping['joint_3'], p.POSITION_CONTROL, targetPosition=q3)
                p.setJointMotorControl2(robotID, joint_mapping['joint_4'], p.POSITION_CONTROL, targetPosition=q4)

            time.sleep(dt)
            t += dt

    finally:

        print('Disconnected.')

        # Disconnect from the simulations when done
        physics.disconnect()
