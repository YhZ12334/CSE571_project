import argparse
import ast
import time
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.model.kinematics import RobotKinematics


ARM_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
ALL_MOTOR_NAMES = ARM_MOTOR_NAMES + ["gripper"]

# 你刚刚确认的默认抓取姿态
DEFAULT_ROTVEC = (-2.0, 2.0, -0.36)

GRIPPER_MAP = {
    "open": 50.0,
    "close": 0.0,
}


def precise_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def parse_points(points_str: str) -> list[np.ndarray]:
    """
    例如:
    --points "[(0.24, 0.10, 0.24), (0.26, 0.10, 0.20)]"
    """
    value = ast.literal_eval(points_str)
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError("points must be a non-empty list/tuple")
    pts = []
    for p in value:
        if not isinstance(p, (list, tuple)) or len(p) != 3:
            raise ValueError(f"bad point: {p}")
        pts.append(np.array([float(p[0]), float(p[1]), float(p[2])], dtype=float))
    return pts


def parse_gripper_actions(actions_str: str, n_points: int) -> list[float]:
    """
    例如:
    --gripper-actions "open,open,close"
    如果只给一个，就对所有点复用。
    """
    items = [x.strip().lower() for x in actions_str.split(",") if x.strip()]
    if len(items) == 0:
        raise ValueError("empty gripper-actions")

    if len(items) == 1:
        items = items * n_points
    elif len(items) != n_points:
        raise ValueError(
            f"gripper-actions length {len(items)} must be 1 or equal to number of points {n_points}"
        )

    out = []
    for item in items:
        if item not in GRIPPER_MAP:
            raise ValueError(f"unknown gripper action: {item}")
        out.append(GRIPPER_MAP[item])
    return out


def load_kinematics(urdf_path: str) -> RobotKinematics:
    return RobotKinematics(
        urdf_path=urdf_path,
        joint_names=ARM_MOTOR_NAMES,
        target_frame_name="gripper_frame_link",
    )


def get_obs_q(obs: dict) -> np.ndarray:
    return np.array([float(obs[f"{name}.pos"]) for name in ARM_MOTOR_NAMES], dtype=float)


def get_obs_gripper(obs: dict) -> float:
    return float(obs["gripper.pos"])


def fk_pose(kinematics: RobotKinematics, q_deg: np.ndarray) -> np.ndarray:
    return kinematics.forward_kinematics(q_deg)


def fk_xyz(kinematics: RobotKinematics, q_deg: np.ndarray) -> np.ndarray:
    T = fk_pose(kinematics, q_deg)
    return np.asarray(T[:3, 3], dtype=float)


def build_target_T(xyz: Sequence[float], rotvec: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def max_abs_delta_deg(q_curr: np.ndarray, q_goal: np.ndarray) -> float:
    return float(np.max(np.abs(q_goal - q_curr)))


def near_soft_limits(q_goal: np.ndarray, margin_deg: float = 8.0) -> bool:
    """
    这里用保守的软限位启发式，不是官方硬限位。
    你后面可以按真机关节范围再细调。
    """
    lower = np.array([-95.0, -95.0, -95.0, -95.0, -170.0], dtype=float)
    upper = np.array([95.0, 95.0, 95.0, 95.0, 170.0], dtype=float)
    return bool(np.any(q_goal < lower + margin_deg) or np.any(q_goal > upper - margin_deg))


def folded_pose_heuristic(q_goal: np.ndarray) -> bool:
    """
    粗略拒绝一些很容易导致折叠/不利发力的姿态。
    这不是严格碰撞检测，只是启发式。
    """
    shoulder_lift = q_goal[1]
    elbow_flex = q_goal[2]
    wrist_flex = q_goal[3]

    # shoulder 太低 + wrist 很高，常见不利构型
    if shoulder_lift < -80.0 and wrist_flex > 75.0:
        return True

    # elbow / wrist 同时很极端，也判坏
    if elbow_flex < -70.0 and wrist_flex > 70.0:
        return True

    return False


def ik_solution_ok(
    kinematics: RobotKinematics,
    q_curr: np.ndarray,
    q_goal: np.ndarray,
    target_xyz: np.ndarray,
    target_rotvec: np.ndarray,
    final_pos_tol_m: float,
    progress_margin_m: float,
    require_upward_progress: bool,
) -> tuple[bool, str, float]:
    """
    返回:
    - 是否接受
    - 原因
    - FK 预测位置误差
    """

    # 1) 关节别太贴近软限位
    if near_soft_limits(q_goal):
        return False, "near soft limits", np.inf

    # 2) 粗略折叠/坏构型检查
    if folded_pose_heuristic(q_goal):
        return False, "folded pose heuristic triggered", np.inf

    # 3) FK 回代
    curr_xyz = fk_xyz(kinematics, q_curr)
    goal_xyz = fk_xyz(kinematics, q_goal)

    err_curr = float(np.linalg.norm(curr_xyz - target_xyz))
    err_goal = float(np.linalg.norm(goal_xyz - target_xyz))

    # 4) 如果目标 z 比当前高，要求预测 z 也不能往下
    # if require_upward_progress:
    #     if target_xyz[2] > curr_xyz[2] + 1e-4 and goal_xyz[2] < curr_xyz[2] - 1e-3:
    #         return False, "upward task but predicted z goes downward", err_goal

    # 5) 三档判断
    if err_goal <= final_pos_tol_m:
        return True, "good", err_goal

    if err_goal < err_curr - progress_margin_m:
        return True, "progress", err_goal

    return False, f"FK improvement too small: curr={err_curr:.4f}, goal={err_goal:.4f}", err_goal


def solve_ik_with_retries(
    kinematics: RobotKinematics,
    q_curr: np.ndarray,
    target_xyz: np.ndarray,
    target_rotvec: np.ndarray,
    position_weight: float,
    orientation_weight: float,
    final_pos_tol_m: float = 0.02,
    progress_margin_m: float = 0.002,
    n_retries: int = 100,
) -> tuple[np.ndarray, float, str]:
    """
    对同一个目标多试几次 IK，挑一个通过 FK 和启发式筛选的最好解。
    这里只在“求某个点的 IK”这一层重试，不进入执行插值阶段重算。
    """
    T_target = build_target_T(target_xyz, target_rotvec)

    seeds = [q_curr.copy()]
    rng = np.random.default_rng()

    # 在当前关节附近加一点扰动，试不同初值
    for _ in range(n_retries - 1):
        seeds.append(q_curr + rng.normal(0.0, 16.0, size=q_curr.shape))

    best_q = None
    best_err = np.inf
    best_reason = "no candidate"

    require_upward_progress = target_xyz[2] > fk_xyz(kinematics, q_curr)[2] + 1e-4

    for seed in seeds:
        try:
            q_goal = kinematics.inverse_kinematics(
                current_joint_pos=seed,
                desired_ee_pose=T_target,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
            )
        except Exception as e:
            best_reason = f"IK exception: {e}"
            continue

        ok, reason, pos_err = ik_solution_ok(
            kinematics=kinematics,
            q_curr=q_curr,
            q_goal=q_goal,
            target_xyz=target_xyz,
            target_rotvec=target_rotvec,
            final_pos_tol_m=final_pos_tol_m,
            progress_margin_m=progress_margin_m,
            require_upward_progress=require_upward_progress,
        )

        if ok and pos_err < best_err:
            best_q = q_goal
            best_err = pos_err
        elif best_q is None and pos_err < best_err:
            # 没有通过的解时，也记住最接近的，方便报错
            best_q = q_goal
            best_err = pos_err
            best_reason = reason

    if best_q is None:
        raise RuntimeError(f"IK failed: {best_reason}")

    # 最终还要保证通过筛选
    ok, reason, pos_err = ik_solution_ok(
        kinematics=kinematics,
        q_curr=q_curr,
        q_goal=best_q,
        target_xyz=target_xyz,
        target_rotvec=target_rotvec,
        final_pos_tol_m=final_pos_tol_m,
        progress_margin_m=progress_margin_m,
        require_upward_progress=require_upward_progress,
    )
    if not ok:
        raise RuntimeError(f"IK candidate rejected: {reason}")

    return best_q, pos_err, reason


def build_joint_action(
    q_goal: np.ndarray,
    gripper_goal: float,
    wrist_roll_override: float | None = None,
) -> dict:
    action = {
        f"{name}.pos": float(q_goal[i])
        for i, name in enumerate(ARM_MOTOR_NAMES)
    }
    if wrist_roll_override is not None:
        action["wrist_roll.pos"] = float(wrist_roll_override)
    action["gripper.pos"] = float(gripper_goal)
    return action


def interpolate_joint_path(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    g_start: float,
    g_goal: float,
    max_step_deg: float = 1.0,
    min_steps: int = 1,
) -> list[tuple[np.ndarray, float]]:
    """
    只在关节空间插值，不重算 IK。
    """
    max_delta = max_abs_delta_deg(q_start, q_goal)
    n_steps = max(min_steps, int(np.ceil(max_delta / max_step_deg)))

    path = []
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        q = q_start + alpha * (q_goal - q_start)
        g = g_start + alpha * (g_goal - g_start)
        path.append((q, float(g)))
    return path


def execute_joint_goal_no_replan(
    robot: SO101Follower,
    kinematics: RobotKinematics,
    q_goal: np.ndarray,
    gripper_goal: float,
    target_xyz: np.ndarray,
    dt: float = 0.05,
    max_step_deg: float = 1.0,
    ee_tol_m: float = 0.02,
    joint_tol_deg: float = 2.0,
    log_every: int = 10,
) -> tuple[bool, float]:
    """
    严格按“固定 q_goal”执行，不重解 IK。
    """
    obs0 = robot.get_observation()
    q_curr = get_obs_q(obs0)
    g_curr = get_obs_gripper(obs0)

    path = interpolate_joint_path(
        q_start=q_curr,
        q_goal=q_goal,
        g_start=g_curr,
        g_goal=gripper_goal,
        max_step_deg=max_step_deg,
        min_steps=1,
    )

    for i, (q_cmd, g_cmd) in enumerate(path):
        loop_start = time.perf_counter()

        action = build_joint_action(q_cmd, g_cmd)
        sent = robot.send_action(action)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(dt - dt_s, 0.0))

        obs = robot.get_observation()
        q_now = get_obs_q(obs)
        xyz_now = fk_xyz(kinematics, q_now)

        ee_err = float(np.linalg.norm(xyz_now - target_xyz))
        joint_err = float(np.max(np.abs(q_goal - q_now)))

        if i % log_every == 0 or i == len(path) - 1:
            print(
                f"[STEP {i+1:03d}/{len(path):03d}] "
                f"ee={xyz_now.tolist()} target={target_xyz.tolist()} "
                f"ee_err={ee_err:.4f} joint_err={joint_err:.2f} "
                f"sent={sent}"
            )

        if ee_err < ee_tol_m:
            return True, ee_err

    obs_end = robot.get_observation()
    q_end = get_obs_q(obs_end)
    xyz_end = fk_xyz(kinematics, q_end)
    final_err = float(np.linalg.norm(xyz_end - target_xyz))
    return final_err < ee_tol_m, final_err


# def run_points(
#     robot: SO101Follower,
#     kinematics: RobotKinematics,
#     points_xyz: list[np.ndarray],
#     gripper_goals: list[float],
#     rotvec: Sequence[float],
#     position_weight: float,
#     orientation_weight: float,
#     dt: float,
#     ik_pos_tol_m: float,
#     exec_pos_tol_m: float,
#     interp_max_step_deg: float,
# ) -> None:
#     target_rotvec = np.asarray(rotvec, dtype=float)

#     for idx, target_xyz in enumerate(points_xyz):
#         obs = robot.get_observation()
#         q_curr = get_obs_q(obs)
#         g_goal = gripper_goals[idx]

#         print(f"\n[POINT {idx+1}/{len(points_xyz)}] target_xyz={target_xyz.tolist()} gripper={g_goal}")

#         q_goal, fk_err = solve_ik_with_retries(
#             kinematics=kinematics,
#             q_curr=q_curr,
#             target_xyz=target_xyz,
#             target_rotvec=target_rotvec,
#             position_weight=position_weight,
#             orientation_weight=orientation_weight,
#             pos_tol_m=ik_pos_tol_m,
#             n_retries=12,
#         )
#         print(f"[IK] accepted q_goal={q_goal.tolist()} fk_err={fk_err:.4f} m")

#         ok, final_err = execute_joint_goal_no_replan(
#             robot=robot,
#             kinematics=kinematics,
#             q_goal=q_goal,
#             gripper_goal=g_goal,
#             target_xyz=target_xyz,
#             dt=dt,
#             max_step_deg=interp_max_step_deg,
#             ee_tol_m=exec_pos_tol_m,
#             joint_tol_deg=2.0,
#             log_every=10,
#         )

#         if not ok:
#             raise RuntimeError(
#                 f"Failed at point {idx+1}: final EE error still {final_err:.4f} m"
#             )

#         print(f"[POINT {idx+1}] reached with final_err={final_err:.4f} m")

def run_points(
    robot: SO101Follower,
    kinematics: RobotKinematics,
    points_xyz: list[np.ndarray],
    gripper_goals: list[float],
    rotvec: Sequence[float],
    position_weight: float,
    orientation_weight: float,
    dt: float,
    ik_pos_tol_m: float,
    exec_pos_tol_m: float,
    interp_max_step_deg: float,
) -> None:
    target_rotvec = np.asarray(rotvec, dtype=float)

    # 每个输入点允许做多轮“局部IK + 执行推进”
    max_outer_iters_per_point = 8
    progress_margin_m = 0.002   # 至少改善 5 mm 才算推进

    for idx, target_xyz in enumerate(points_xyz):
        g_goal = gripper_goals[idx]
        print(f"\n[POINT {idx+1}/{len(points_xyz)}] target_xyz={target_xyz.tolist()} gripper={g_goal}")

        reached = False

        for outer in range(max_outer_iters_per_point):
            obs = robot.get_observation()
            q_curr = get_obs_q(obs)
            curr_xyz = fk_xyz(kinematics, q_curr)
            curr_err = float(np.linalg.norm(curr_xyz - target_xyz))

            print(
                f"[POINT {idx+1}] outer_iter={outer+1}/{max_outer_iters_per_point} "
                f"curr_xyz={curr_xyz.tolist()} curr_err={curr_err:.4f} m"
            )

            # 已经足够接近，就结束这个输入点
            if curr_err < exec_pos_tol_m:
                reached = True
                print(f"[POINT {idx+1}] already within tolerance: {curr_err:.4f} m")
                break

            q_goal, fk_err, quality = solve_ik_with_retries(
                kinematics=kinematics,
                q_curr=q_curr,
                target_xyz=target_xyz,
                target_rotvec=target_rotvec,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
                final_pos_tol_m=ik_pos_tol_m,
                progress_margin_m=progress_margin_m,
                n_retries=100,
            )

            print(
                f"[IK] accepted q_goal={q_goal.tolist()} "
                f"fk_err={fk_err:.4f} m quality={quality}"
            )

            ok, final_err = execute_joint_goal_no_replan(
                robot=robot,
                kinematics=kinematics,
                q_goal=q_goal,
                gripper_goal=g_goal,
                target_xyz=target_xyz,
                dt=dt,
                max_step_deg=interp_max_step_deg,
                ee_tol_m=exec_pos_tol_m,
                joint_tol_deg=2.0,
                log_every=10,
            )

            # 执行完以后再看当前位置，如果已经够近就算到达
            obs_after = robot.get_observation()
            q_after = get_obs_q(obs_after)
            xyz_after = fk_xyz(kinematics, q_after)
            err_after = float(np.linalg.norm(xyz_after - target_xyz))

            print(
                f"[POINT {idx+1}] after execute: xyz={xyz_after.tolist()} "
                f"err={err_after:.4f} m"
            )

            if err_after < exec_pos_tol_m:
                reached = True
                print(f"[POINT {idx+1}] reached with final_err={err_after:.4f} m")
                break

            # 如果这轮推进几乎没改善，就直接报失败
            if err_after > curr_err - progress_margin_m:
                raise RuntimeError(
                    f"Point {idx+1}: IK produced only weak progress "
                    f"(before={curr_err:.4f} m, after={err_after:.4f} m)"
                )

        if not reached:
            raise RuntimeError(
                f"Failed at point {idx+1}: not within tolerance after "
                f"{max_outer_iters_per_point} local IK iterations"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=str, required=True)
    parser.add_argument(
        "--urdf",
        type=str,
        default="/home/iann_zhang/project/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
    )

    # 输入点：可以单点，也可以多点
    parser.add_argument("--x", type=float, default=None)
    parser.add_argument("--y", type=float, default=None)
    parser.add_argument("--z", type=float, default=None)
    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help='例如: "[(0.24, 0.10, 0.24), (0.26, 0.10, 0.20)]"',
    )

    parser.add_argument(
        "--rotvec",
        type=str,
        default=str(DEFAULT_ROTVEC),
        help='例如: "(-2.0, 2.0, -0.36)"',
    )

    parser.add_argument(
        "--gripper-actions",
        type=str,
        default="open",
        help='例如: "open,open,close" 或单个 "open"',
    )

    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.005)

    # IK 质量门槛
    parser.add_argument("--ik-pos-tol", type=float, default=0.02)        # 1~2 cm
    parser.add_argument("--exec-pos-tol", type=float, default=0.025)     # 执行到点容差

    # 关节空间插值
    parser.add_argument("--interp-max-step-deg", type=float, default=1.0)

    # send_action 安全截断：建议先保留一点安全
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--no-calibrate", action="store_true")

    args = parser.parse_args()

    # 点输入
    if args.points is not None:
        points_xyz = parse_points(args.points)
    else:
        if args.x is None or args.y is None or args.z is None:
            raise ValueError("Either --points or all of --x --y --z must be provided")
        points_xyz = [np.array([args.x, args.y, args.z], dtype=float)]

    gripper_goals = parse_gripper_actions(args.gripper_actions, len(points_xyz))
    rotvec = ast.literal_eval(args.rotvec)
    rotvec = tuple(float(v) for v in rotvec)

    kinematics = load_kinematics(args.urdf)

    config = SO101FollowerConfig(
        port=args.port,
        id="my_follower",
        use_degrees=True,
        max_relative_target=args.max_relative_target,  # 可以先保留 5.0 这种温和保险
        cameras={},
    )

    robot = SO101Follower(config)
    robot.connect(calibrate=not args.no_calibrate)

    try:
        run_points(
            robot=robot,
            kinematics=kinematics,
            points_xyz=points_xyz,
            gripper_goals=gripper_goals,
            rotvec=rotvec,
            position_weight=args.position_weight,
            orientation_weight=args.orientation_weight,
            dt=args.dt,
            ik_pos_tol_m=args.ik_pos_tol,
            exec_pos_tol_m=args.exec_pos_tol,
            interp_max_step_deg=args.interp_max_step_deg,
        )
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()