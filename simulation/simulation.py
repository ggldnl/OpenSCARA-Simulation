import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

from pathlib import Path
import numpy as np
import argparse
import math
import time


# --------------------------------- Constants --------------------------------- #

L2 = 0.12
L3 = 0.12

# The first joint angle can move within the [0, 200] mm range
JOINT_1_MIN = 0
JOINT_1_MAX = 0.2  # 200 mm, 0.2 m

# The second joint angle can move within the [-pi/2, +pi/2] range
JOINT_2_MIN = -math.pi / 2
JOINT_2_MAX = math.pi / 2

# The third joint angle can move by almost 360 degrees
JOINT_3_MIN = -math.pi
JOINT_3_MAX = math.pi - math.radians(10)  # Almost +180 deg

# The fourth joint angle can also move by almost 360 degrees
JOINT_4_MIN = -math.pi
JOINT_4_MAX = math.pi - math.radians(10)  # Almost +180 deg

# -------------------------------- Kinematics -------------------------------- #

class Configuration:

    def __init__(self, q1, q2, q3, q4):
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4

    def distance(self, other):
        """
        Compute the distance between two configurations. Only the second joint is considered.
        The third and fourth joints are a consequence of the second joint.
        """
        x1 = L2 * math.cos(self.q2)
        y1 = L2 * math.sin(self.q2)
        x2 = L2 * math.cos(other.q2)
        y2 = L2 * math.sin(other.q2)
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def __str__(self):
        return f"(q1={self.q1}, q2={self.q2}, q3={self.q3}, q4={self.q4})"

    def __iter__(self):
        return iter([self.q1, self.q2, self.q3, self.q4])

    def __getitem__(self, item):
        if item == 0:
            return self.q1
        elif item == 1:
            return self.q2
        elif item == 2:
            return self.q3
        elif item == 3:
            return self.q4
        else:
            raise ValueError(f"Index out of range: {item}")
    
class TaskPoint:
    
    def __init__(self, x, y, z, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw

    def __str__(self):
        return f"(x={self.x}, y={self.y}, z={self.z}, yaw={self.yaw})"
    
class Trajectory:

    def __init__(self, qi, qf, vi, vf, ai, af, tf):
        self.qi = qi
        self.qf = qf
        self.vi = vi
        self.vf = vf
        self.ai = ai
        self.af = af
        self.tf = tf
        self.coefficients = self.solve()

    def solve(self):
        tf = self.tf 
        M = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, tf, tf**2, tf**3, tf**4, tf**5],
            [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
            [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3],
        ])
        b = np.array([self.qi, self.vi, self.ai, self.qf, self.vf, self.af])
        x = np.linalg.solve(M, b)
        return x

    def __getitem__(self, t):

        c = self.coefficients
        if len(c) == 0:
            raise ValueError("Coefficients not set. Solve the trajectory first.")

        if t >= self.tf:
            t = self.tf  # Cap the value to the trajectory period

        return c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5

def map_range(value, from_min, from_max, to_min, to_max):
    """
    Map a value from a range to another.
    """
    return to_min + ((value - from_min) / (from_max - from_min)) * (to_max - to_min)

def map_to_robot(joint_values):
    """
    Map the joint values from the kinematic model's joint space
    to the actual robot's joint space.
    """
    map_joint_1 = joint_values[0]
    map_joint_2 = map_range(joint_values[1], math.pi / 2, -math.pi / 2, 0, math.pi)
    map_joint_3 = map_range(joint_values[2], -math.pi, math.pi, 0, -2 * math.pi)
    map_joint_4 = joint_values[3]  # TODO fix this joint
    return map_joint_1, map_joint_2, map_joint_3, map_joint_4

def wrap_angle(angle):
    """
    Wrap angle to (-pi, pi] range.
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi

def inverse_kinematics(point):
    """
    Compute the inverse kinematics for the SCARA arm.
    """
    x = point.x
    y = point.y
    z = point.z
    yaw = point.yaw

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
    
    q4_a = wrap_angle(yaw - (q2_a + q3_a))
    q4_b = wrap_angle(yaw - (q2_b + q3_b))

    config_a_valid = (
            JOINT_1_MIN <= q1 <= JOINT_1_MAX and
            JOINT_2_MIN <= q2_a <= JOINT_2_MAX and
            JOINT_3_MIN <= q3_a <= JOINT_3_MAX and
            JOINT_4_MIN <= q4_a <= JOINT_4_MAX
    )
    config_a = Configuration(q1, q2_a, q3_a, q4_a) if config_a_valid else None

    config_b_valid = (
            JOINT_1_MIN <= q1 <= JOINT_1_MAX and
            JOINT_2_MIN <= q2_b <= JOINT_2_MAX and
            JOINT_3_MIN <= q3_b <= JOINT_3_MAX and
            JOINT_4_MIN <= q4_b <= JOINT_4_MAX
    )
    config_b = Configuration(q1, q2_b, q3_b, q4_b) if config_b_valid else None

    return config_a, config_b

def interpolate(point_a, point_b, steps):
    """
    Generate a list of configurations interpolating from config_a to config_b.
    """
    points = []
    for i in range(steps):
        alpha = i / (steps - 1)
        x = (1 - alpha) * point_a.x + alpha * point_b.x
        y = (1 - alpha) * point_a.y + alpha * point_b.y
        z = (1 - alpha) * point_a.z + alpha * point_b.z
        yaw = (1 - alpha) * point_a.yaw + alpha * point_b.yaw
        points.append(TaskPoint(x, y, z, yaw))
    return points


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="OpenSCARA PyBullet simulation.")
    parser.add_argument("-u", "--URDF", type=str, default='../OpenSCARA-Hardware/SCARA.urdf', help="Path to SCARA's URDF")
    parser.add_argument('-t', '--dt', type=float, default=0.02, help="Time delta for update (default=0.02=50Hz)")
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
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Enable shadows

    # Set initial camera view
    default_camera_distance = 0.8
    default_camera_roll = 0
    default_camera_yaw = -45
    default_camera_pitch = -25
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

    urdf_file = Path(args.URDF)
    start_position = [0, 0, 0]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robotID = physics.loadURDF(str(urdf_file), start_position, start_orientation, flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=True)

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

    # ---------------------------- Trajectory planning --------------------------- #

    # Target points (task space)
    task_points = []
    durations = []

    # Move forward and backward
    for x in [0.05, 0.2, 0.05, 0.2]:
        task_points.append(TaskPoint(x, 0.0, 0.16, 0))
        durations.append(3)  # Same for all the trajectories

    # Move left and right
    interpolation_points = interpolate(
        TaskPoint(0.1, -0.2, 0.16, 0),
        TaskPoint(0.1, 0.2, 0.16, 0),
        10
    )
    for point in interpolation_points:
        task_points.append(point)
        durations.append(3)  # Same for all the trajectories

    for point in reversed(interpolation_points):
        task_points.append(point)
        durations.append(3)  # Same for all the trajectories

    """
    Draw a square
    
    (0.2, 0.2) o------------o (0.2, 0)
               |            |
               |            |
               |            |
               |            |
    (  0, 0.2) o------------o (  0, 0)
    """
    for x, y in [(0.02, 0.02), (0.02, 0.12), (0.12, 0.12), (0.12, 0.02), (0.02, 0.02)]:
        task_points.append(TaskPoint(x, y, 0.16, 0))
        durations.append(3)

    # Current configuration (joint space)
    current_config = Configuration(q1=0.16, q2=0, q3=0, q4=0)
    p.setJointMotorControl2(robotID, joint_mapping['joint_1'], p.POSITION_CONTROL, targetPosition=current_config[0])
    p.setJointMotorControl2(robotID, joint_mapping['joint_2'], p.POSITION_CONTROL, targetPosition=current_config[1])
    p.setJointMotorControl2(robotID, joint_mapping['joint_3'], p.POSITION_CONTROL, targetPosition=current_config[2])
    p.setJointMotorControl2(robotID, joint_mapping['joint_4'], p.POSITION_CONTROL, targetPosition=current_config[3])
    print(f"Current configuration: {current_config}")

    # Smooth start and stop
    vi = vf = ai = af = 0.0

    # ------------------------------ Simulation loop ----------------------------- #

    if args.video_path:

        video_path = Path(args.video_path)
        folder_path = video_path.parent
        if not folder_path.exists():
            folder_path.mkdir(exist_ok=True, parents=True)

        physics.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, args.video_path)

    try:

        dt = args.dt

        for target, duration in zip(task_points, durations):

            print(f"\nReaching target {target} in time {duration}s")

            sol1, sol2 = inverse_kinematics(target)
            if sol1 is None and sol2 is None:
                print("No solution for target, skipping.")
                continue
            elif sol1 is not None or sol2 is not None:
                # Select the only available solution
                print("Only one solution found for the given point.")
                solution = sol1 if sol1 is not None else sol2
            else:
                # Two solutions found, select the closest configuration
                distance_1 = current_config.distance(sol1)
                distance_2 = current_config.distance(sol2)
                solution = sol1 if distance_1 < distance_2 else sol2

            print(f"Selected solution: {solution}")

            # Plan trajectories
            trajectories = [
                Trajectory(qi=qi, qf=qf, vi=vi, vf=vf, ai=ai, af=af, tf=duration)
                for qi, qf in zip(current_config, solution)
            ]

            t = 0.0
            while t < duration:

                physics.stepSimulation()
                joint_values = [traj[t] for traj in trajectories]
                mapped = map_to_robot(joint_values)

                p.setJointMotorControl2(robotID, joint_mapping['joint_1'], p.POSITION_CONTROL, targetPosition=mapped[0])
                p.setJointMotorControl2(robotID, joint_mapping['joint_2'], p.POSITION_CONTROL, targetPosition=mapped[1])
                p.setJointMotorControl2(robotID, joint_mapping['joint_3'], p.POSITION_CONTROL, targetPosition=mapped[2])
                p.setJointMotorControl2(robotID, joint_mapping['joint_4'], p.POSITION_CONTROL, targetPosition=mapped[3])

                time.sleep(dt)
                t += dt

            # Update current configuration to the one we just reached
            current_config = solution

        # Keep the simulation alive
        t = 0.0
        while physics.isConnected():
            physics.stepSimulation()
            time.sleep(dt)
            t += dt

    finally:

        print('Disconnected.')

        # Disconnect from the simulations when done
        physics.disconnect()
