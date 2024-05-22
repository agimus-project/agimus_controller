from pathlib import Path
from .croco_hpp import *


with open(Path(__file__).parent / "resources" / "datas.npy", "rb") as f:
    x_plan_0 = np.load(f)
    x_plan_1 = np.load(f)

ball_init_pose = [0.2, 0, 0.02, 0, 0, 0, 1]
chc = CrocoHppConnection(None, "ur5", None, ball_init_pose)
chc.prob.hpp_paths.append(SubPath(None))
chc.prob.hpp_paths[0].x_plan = list(x_plan_0)
chc.prob.hpp_paths[0].T = len(x_plan_0) - 1

chc.prob.hpp_paths.append(SubPath(None))
chc.prob.hpp_paths[1].x_plan = list(x_plan_1)
chc.prob.hpp_paths[1].T = len(x_plan_1) - 1
chc.prob.nb_paths = 2
chc.search_best_costs(1, use_mim=False)
