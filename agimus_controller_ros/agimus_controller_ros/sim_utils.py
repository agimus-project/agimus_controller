from builtin_interfaces.msg import Duration
import numpy as np


def convert_float_to_ros_duration_msg(time: float):
    ros_duration = Duration()
    ros_duration.sec = int(time)
    ros_duration.nanosec = int((time % 1) * 1e9)
    return ros_duration


def save_plannif(whole_x_plan, whole_a_plan, file_name):
    planning = {}
    planning["whole_x_plan"] = whole_x_plan
    planning["whole_a_plan"] = whole_a_plan
    np.save(file_name, planning, allow_pickle=True)


def load_plannif(file_name):
    planning = np.load(file_name, allow_pickle=True).item()
    whole_x_plan = planning["whole_x_plan"]
    whole_a_plan = planning["whole_a_plan"]
    return whole_x_plan, whole_a_plan
