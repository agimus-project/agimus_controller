import rospy
import time
from dynamic_graph_bridge_msgs.msg import Vector
from collections import deque
from threading import Lock
from agimus_controller.trajectory_point import TrajectoryPoint


class HPPSubscriberParameters:
    def __init__(self) -> None:
        self.op_frames = rospy.get_param("~op_frames", [])
        self.name = rospy.get_param("~name", "robot")
        self.prefix = rospy.get_param("~prefix", "agimus")
        self.rate = rospy.get_param("~rate", 100)


class FIFO:
    def __init__(self):
        self.deque = deque()
        self.mutex = Lock()

    def push_back(self, msg):
        with self.mutex:
            self.deque.append(msg)

    def pop_front(self):
        with self.mutex:
            ret = self.deque.popleft()
        return ret

    def get_size(self):
        with self.mutex:
            return len(self.deque)


class HPPSubscriber:
    def __init__(self, use_ros_params=True) -> None:
        if use_ros_params:
            rospy.loginfo("Load parameters")
            self.params = HPPSubscriberParameters()

            rospy.loginfo("Create the rate object.")
            self.wait_subscribers = rospy.Rate(10000)  # 10kHz

            rospy.loginfo("Parse the prefix/name params.")
            if not self.params.prefix.endswith("/"):
                self.params.prefix = self.params.prefix + "/"
            if not self.params.prefix.startswith("/"):
                self.params.prefix = "/" + self.params.prefix
            self.params.name.replace("/", "")

            rospy.loginfo("Spawn the subscribers.")
            self.subscribers = []

            # q
            rospy.loginfo("\t- Robot configuration subscriber.")
            self.subscribers += [
                rospy.Subscriber(
                    "/hpp/target/position",  # self.params.prefix + "position",
                    Vector,
                    self.position_callback,
                )
            ]
            # v
            rospy.loginfo("\t- Robot velocity subscriber.")
            self.subscribers += [
                rospy.Subscriber(
                    "/hpp/target/velocity",  # self.params.prefix + "velocity",
                    Vector,
                    self.velocity_callback,
                )
            ]
            # a
            rospy.loginfo("\t- Robot acceleration subscriber.")
            self.subscribers += [
                rospy.Subscriber(
                    "/hpp/target/acceleration",  # self.params.prefix + "acceleration",
                    Vector,
                    self.acceleration_callback,
                )
            ]

        self.fifo_q = FIFO()  # q
        self.fifo_v = FIFO()  # v
        self.fifo_a = FIFO()  # a
        self.fifo_com_pose = FIFO()  # com_pos
        self.fifo_com_velocity = FIFO()  # com_vel
        self.fifo_op_frame_pose = FIFO()  # op_pos
        self.fifo_op_frame_velocity = FIFO()  # op_vel

        self.index = 0
        # # com_q
        # rospy.loginfo("\t- CoM pose subscriber.")
        # self.subscribers += [
        #     rospy.Subscriber(
        #         self.params.prefix + "com/" + self.params.name,
        #         Vector3,
        #         self.com_pose_callback,
        #     )
        # ]
        # # com_v
        # rospy.loginfo("\t- CoM velocity subscriber.")
        # self.subscribers += [
        #     rospy.Subscriber(
        #         self.params.prefix + "velocity/com/" + self.params.name,
        #         Vector3,
        #         self.com_velocity_callback,
        #     )
        # ]
        # for op_frame in self.params.op_frames:
        #     # op_frame_q
        #     rospy.loginfo("\t- Operationnal Point (OP) frame pose subscriber.")
        #     self.subscribers += [
        #         rospy.Subscriber(
        #             self.params.prefix + op_frame + self.params.name,
        #             Transform,
        #             partial(self.params.op_frames_callback, op_frame),
        #         )
        #     ]
        #     # op_frame_v
        #     rospy.loginfo("\t- OP frame pos velocity subscriber.")
        #     self.subscribers += [
        #         rospy.Subscriber(
        #             self.params.prefix + op_frame + "/velocity",
        #             Transform,
        #             partial(self.params.op_frames_callback, op_frame),
        #         )
        #     ]

    # get q, v, a avec pubQ, pubV, pubA qui sont publish dans le fichier discretization.cc

    def print_fifo(self):
        print(self.fifo_q.get_size())

    def position_callback(self, msg):
        self.last_time_got_traj_point = time.time()
        self.fifo_q.push_back(msg)

    def velocity_callback(self, msg):
        self.fifo_v.push_back(msg)

    def acceleration_callback(self, msg):
        self.fifo_a.push_back(msg)

    def com_pose_callback(self, msg):
        rospy.logdebug("CoM pos msg = ", msg)
        self.fifo_com_pose.push_back(msg)

    def com_velocity_callback(self, msg):
        rospy.logdebug("CoM vel msg = ", msg)
        self.fifo_com_velocity.push_back(msg)

    def op_frames_callback(self, op_frame, msg):
        rospy.logdebug("Op frame = ", msg)
        rospy.logdebug(op_frame)
        self.fifo_op_frame_pose.push_back(msg)

    def min_all_deque(self):
        return min(
            self.fifo_q.get_size(),
            self.fifo_v.get_size(),
            self.fifo_a.get_size(),
        )

    def get_trajectory_point(self):
        if self.min_all_deque() == 0:
            return None
        else:
            q = self.fifo_q.pop_front().data
            v = self.fifo_v.pop_front().data
            tp = TrajectoryPoint(time=self.index, nq=len(q), nv=len(v))
            tp.q[:] = q[:]
            tp.v[:] = v[:]
            tp.a[:] = self.fifo_a.pop_front().data[:]

            self.index += 1
            return tp

    def fill_buffer_manually(self, poses, vels, accs):
        for idx in range(poses.shape[0]):
            self.fifo_q.push_back(Vector(poses[idx, 0]))
            self.fifo_v.push_back(Vector(vels[idx, 0]))
            self.fifo_a.push_back(Vector(accs[idx, 0]))
