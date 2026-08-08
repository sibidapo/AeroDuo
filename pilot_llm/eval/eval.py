import json
import math
import os
import re
import time
import traceback
import warnings

import sys

import cv2
import msgpackrpc
import numpy as np
import torch
import tqdm

# Keep this dir ahead of pilot_llm/high_uav (which inserts itself at sys.path[0])
# so top-level `config` resolves to eval2/config, not high_uav/config.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.evalconfig import args, data_args, model_args, EvalConfig
from dualuavpilot import DualUAVPilot
from vlnce_src.env_uav import AirVLNENV
from utils.logger import logger
from utils.utils import is_dist_avail_and_initialized

warnings.filterwarnings("ignore")

class BatchIterator:

    def __init__(self, env: AirVLNENV):
        self.env = env

    def __len__(self):
        return len(self.env.data)

    def __next__(self):
        # import ipdb;ipdb.set_trace()
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch


class Metrics:

    def __init__(self, log_path: str = None):
        self.log_path = log_path
        # ultimate metrics
        self.OSR = 0
        self.SR = 0
        self.CR = 0
        self.airsim_CR = 0
        self.SPL = 0
        self.SST = 0
        self.TL = 0  # average trajectory length
        self.NE = 0

        # intermediate statistics
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                data = json.load(f)
            self.total_cnt = data["total_cnt"]
            self.success_cnt = data["success_cnt"]
            self.oracle_success_cnt = data["oracle_success_cnt"]
            self.collision_cnt = data["collision_cnt"]
            self.airsim_collision_cnt = data["airsim_collision_cnt"]
            self.spl_cnt = data["spl_cnt"]
            self.sst_cnt = data["sst_cnt"]
            self.total_length = data["total_length"]
            self.total_error = data["total_error"]
            # Navigator extra statistics
            self.nav_time_stat = data["nav_time_stat"]
            self.nav_velocity_stat = data["nav_velocity_stat"]
            self.nav_stat_cnt = data["nav_stat_cnt"]
        else:
            self.total_cnt = 0
            self.success_cnt = 0
            self.oracle_success_cnt = 0
            self.collision_cnt = 0
            self.airsim_collision_cnt = 0
            self.spl_cnt = 0
            self.sst_cnt = 0
            self.total_length = 0
            self.total_error = 0
            # Navigator extra statistics
            self.nav_time_stat = 0
            self.nav_velocity_stat = 0
            self.nav_stat_cnt = 0

    def update_ultimate_metrics(self):
        self.OSR = self.oracle_success_cnt / self.total_cnt
        self.SR = self.success_cnt / self.total_cnt
        self.CR = self.collision_cnt / self.total_cnt
        self.airsim_CR = self.airsim_collision_cnt / self.total_cnt
        self.SPL = self.spl_cnt / self.total_cnt
        self.SST = self.sst_cnt / self.total_cnt
        self.TL = self.total_length / self.total_cnt
        self.NE = self.total_error / self.total_cnt
        print("OSR: ", self.OSR, " SR: ", self.SR, " CR: ", self.CR,
              "airsim_CR: ", self.airsim_CR, " SPL: ", self.SPL, " SST: ",
              self.SST, "TL:", self.TL, " NE: ", self.NE)

        # save intermeidate statistics
        with open(self.log_path, 'w') as f:
            json.dump(
                {
                    "total_cnt": self.total_cnt,
                    "success_cnt": self.success_cnt,
                    "oracle_success_cnt": self.oracle_success_cnt,
                    "collision_cnt": self.collision_cnt,
                    "airsim_collision_cnt": self.airsim_collision_cnt,
                    "spl_cnt": self.spl_cnt,
                    "sst_cnt": self.sst_cnt,
                    "total_length": self.total_length,
                    "total_error": self.total_error,
                    "nav_time_stat": self.nav_time_stat,
                    "nav_velocity_stat": self.nav_velocity_stat,
                    "nav_stat_cnt": self.nav_stat_cnt
                }, f)

class EvalRollout:

    def __init__(self, env_batches, train_env, object_desc_dict, args):
        self.ori_data_dirs = [b['seq_name'] for b in env_batches]
        self.map_names = [b['map_name'] for b in env_batches]
        self.target_positions = [b['object_position'] for b in env_batches]
        self.object_infos = [
            object_desc_dict.get(
                b['object']['asset_name'].replace("AA", ""),
                re.sub(r'(SM_|AASM_)?\d*([a-zA-Z]+)\d*', r'\2',
                       b['object']['asset_name'])) for b in env_batches
        ]
        self.gt_drone1_trajs = [b['drone1_traj'] for b in env_batches]
        self.gt_drone2_trajs = [b['drone2_traj'] for b in env_batches]
        self.drone1_trajs = [[] for _ in range(train_env.batch_size)] ## Low UAV
        self.drone2_trajs = [[] for _ in range(train_env.batch_size)] ## High UAV
        self.raw_instructions = [b['instruction'] for b in env_batches]

        self.train_env = train_env
        self.episodes = [[] for _ in range(train_env.batch_size)]
        self.skips = [False for _ in range(train_env.batch_size)]
        self.dones = [False for _ in range(train_env.batch_size)]
        self.collisions = [False for _ in range(train_env.batch_size)]
        self.distance_to_ends = [[] for _ in range(train_env.batch_size)]
        self.success = [False for _ in range(train_env.batch_size)]
        self.oracle_success = [False for _ in range(train_env.batch_size)]
        self.early_end = [False for _ in range(train_env.batch_size)]
        self.envs_to_pause = []

        self.maxWaypoints = args.maxWaypoints  ##TODO
        self.is_end = False
        self.eval_save_dir = args.eval_save_path

        self.bevs = [[] for _ in range(train_env.batch_size)]

        self.steps = 0
        self.distance = 0
        self.time = 0
        self.airsim_collision = False

    def calculate_traj_stats(self, metrics: Metrics):
        # calculate optimal distance
        tot_distance = 0
        traj_waypoint = self.gt_drone1_trajs[0]
        for i in range(len(traj_waypoint) - 1):
            p1 = np.array(traj_waypoint[i])
            p2 = np.array(traj_waypoint[i + 1])
            tot_distance += np.linalg.norm(p2 - p1)
        tot_distance = max(tot_distance - 20, 1e-6)

        # retrieve optimal time
        root_path = "data/HaL-13k"
        time_json_path = os.path.join(root_path, self.map_names[0],
                                      self.ori_data_dirs[0], "time.json")
        with open(time_json_path, 'r') as f:
            tot_time = json.load(f)["gt_time"]

        return tot_distance, tot_time

    def fetch_from_observations(self, key):
        result = []
        for i in range(self.train_env.batch_size):
            result.append(self.observations[i][-1][key])
        return result
    
    def update_bevs(self):
        bev = self.fetch_from_observations("bev")
        for i in range(self.train_env.batch_size):
            self.bevs[i].append(bev[i])
            print(f"Updating bev")

    def update_observation(self, outputs, pos_list, airsim_collision):
        observations, dones, collisions, oracle_success = [
            list(x) for x in zip(*outputs)
        ]
        self.observations = observations
        self.update_bevs()

        self.airsim_collision = airsim_collision

        for i in range(self.train_env.batch_size):
            if i in self.envs_to_pause:
                continue
            self.episodes[i].append(observations[i][-1])
            self.dones[i] = dones[i]
            self.collisions[i] = collisions[i]
            self.oracle_success[i] = oracle_success[i] or self.oracle_success[i]
            self.drone1_trajs[i].extend(pos_list)
            if len(pos_list) > 0:
                new_distance_to_ends = [
                    np.linalg.norm(
                        np.array(pos) - np.array(self.target_positions[i]))
                    for pos in pos_list
                ]
                for distance in new_distance_to_ends:
                    if distance <= 20:
                        self.oracle_success[i] = True
                        break
                self.distance_to_ends[i].extend(new_distance_to_ends)
            if len(self.drone1_trajs[i]) == 0: #Initial position of low UAV upon spin-up
                self.drone1_trajs[i].append(
                    observations[i][-1]['sensors']['state']['position'])
            if len(self.drone2_trajs[i]) > 0:
                last_point = self.drone2_trajs[i][-1]
                now_point = self.train_env.sim_states[i].drone2_traj[-1][
                    'position']
                direction = np.array(now_point) - np.array(last_point)
                distance = np.linalg.norm(direction)
                unit_direction = direction / distance if distance > 0 else np.zeros_like(
                    direction)
                # new point every 5 meters
                new_point = last_point
                while distance > 5:
                    new_point = np.array(new_point) + unit_direction * 5
                    self.drone2_trajs[i].append(new_point.tolist())
                    distance -= 5
            else:
                self.drone2_trajs[i].append(
                    self.train_env.sim_states[i].drone2_traj[-1]['position'])
                
            if self.oracle_success[i]:
                self.check_deviation(i)

    def save_to_dataset(self, root_path, i):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        folder_names = ['bevcamera', 'frontcamera', 'log', 'log2']
        for folder_name in folder_names:
            os.makedirs(os.path.join(root_path, folder_name), exist_ok=True)
        #self.save_images(root_path, i)
        self.save_logs(root_path, i)

    def save_images(self, trajectory_dir, i):
        episodes = self.episodes[i]
        for idx, episode in enumerate(episodes):
            # if 'bev' in episode:
            #     image = episode['bev']
            #     cv2.imwrite(
            #         os.path.join(trajectory_dir, 'bevcamera',
            #                      str(idx).zfill(6) + '.png'), image)
            if 'rgb' in episode and len(episode['rgb']) > 0:
                image = episode['rgb'][0]  # RGB_FOLDER[0] == 'frontcamera'
                cv2.imwrite(
                    os.path.join(trajectory_dir, 'frontcamera',
                                 str(idx).zfill(6) + '.png'), image)
                
    def save_logs(self, trajectory_dir, i):
        drone1_traj = self.drone1_trajs[i]
        save_dir = os.path.join(trajectory_dir, 'log')
        for idx, point in enumerate(drone1_traj):
            with open(os.path.join(save_dir,
                                   str(idx).zfill(6) + '.json'), 'w') as f:
                json.dump(point, f)

        # TODO: save drone2 traj
        drone2_traj = self.drone2_trajs[i]
        drone2_save_dir = os.path.join(trajectory_dir, 'log2')
        for idx, point in enumerate(drone2_traj):
            with open(
                    os.path.join(drone2_save_dir,
                                 str(idx).zfill(6) + '.json'), 'w') as f:
                json.dump(point, f)

            
    def check_deviation(self, i):
        def target_distance_increasing_for_10frames(lst):
            if len(lst) < 10:
                return False
            sublist = lst[-10:]
            for i in range(1, len(sublist)):
                if sublist[i] < sublist[i - 1]:
                    return False
            return True
        if target_distance_increasing_for_10frames(self.distance_to_ends[i]):
            self.dones[i] = True

    def check_traj_status(self, metrics):
        for i in range(self.train_env.batch_size):
            if len(self.drone1_trajs[i]) > 1:
                delta_distance = np.linalg.norm(
                    np.array(self.drone1_trajs[i][-1]) -
                    np.array(self.drone1_trajs[i][-2]))
                if delta_distance < 0.1:
                    self.collisions[i] = True
            if self.collisions[i]:
                self.dones[i] = True
            if self.early_end[i] and self.oracle_success[i]:
                self.dones[i] = True
            if self.dones[i] and not self.skips[i]:
                prex = ""
                self.envs_to_pause.append(i)
                if len(self.distance_to_ends[i]) > 0 and self.distance_to_ends[i][-1] <= 20:
                    if not self.collisions[i]:
                        self.success[i] = True
                    self.oracle_success[i] = True
                if self.success[i]:
                    prex = 'success_'
                    print(i, " has succeed!")
                elif self.oracle_success[i]:
                    prex = "oracle_"
                    print(i, " has oracle succeed!")
                new_traj_name = prex + self.ori_data_dirs[
                    i] 
                new_traj_dir = os.path.join(self.eval_save_dir, new_traj_name)
                self.save_to_dataset(new_traj_dir, i)
                self.skips[i] = True
                print(i, " has finished!")
            if np.array(self.skips).all():
                self.is_end = True

        if self.dones[0]:
            self.update_metrics(metrics, new_traj_dir)
            return True
        return False

    def calculate_spl(self, gt_distance):
        spl = gt_distance / max(gt_distance, self.distance)
        assert spl > 0, "SPL should be greater than 0"

        return spl

    def calculate_sst(self, gt_time):
        sst = gt_time / max(gt_time, self.time)
        assert sst > 0, "SST should be greater than 0"

        return sst

    def update_metrics(self, metrics: Metrics, new_traj_dir):
        # we temporarily take bs=1
        metrics.total_cnt += 1
        spl_cnt = 0
        sst_cnt = 0

        metrics.total_length += self.distance
        metrics.total_error += self.distance_to_ends[0][-1]

        if self.success[0]:
            metrics.success_cnt += 1
            metrics.oracle_success_cnt += 1
            gt_distance, gt_time = self.calculate_traj_stats(metrics)
            spl_cnt = self.calculate_spl(gt_distance)
            sst_cnt = self.calculate_sst(gt_time)
            metrics.spl_cnt += spl_cnt
            metrics.sst_cnt += sst_cnt
        elif self.oracle_success[0]:
            metrics.oracle_success_cnt += 1
        if self.collisions[0] and not self.success[0]:
            metrics.collision_cnt += 1
        if self.airsim_collision:
            metrics.airsim_collision_cnt += 1

        state_log = {
            "success": self.success[0],
            "oracle_success": self.oracle_success[0],
            "collision": self.collisions[0],
            "airsim_collision": self.airsim_collision,
            "spl": spl_cnt,
            "sst": sst_cnt,
            "steps": len(self.drone1_trajs[0]) - 1,
            "path_length": self.distance,
            "time": self.time,
            "distance_to_end": self.distance_to_ends[0][-1],
        }
        with open(os.path.join(new_traj_dir, "state_log.json"), 'w') as f:
            json.dump(state_log, f)

        metrics.update_ultimate_metrics()

    def update_navigator_extra_statistics(self, metrics: Metrics, nav_time, nav_velocity):
        metrics.nav_time_stat += nav_time
        metrics.nav_velocity_stat += nav_velocity
        metrics.nav_stat_cnt += 1
        print("-----------------------------")
        print(
            f"mean velocity: {nav_velocity:.2f} m/s, navigation time: {nav_time:.2f} s"
        )
        print("-----------------------------")

def main():

    config = EvalConfig(stage1_ckpt=args.stage1_ckpt, stage2_ckpt=args.stage2_ckpt)
    config.low_uav.use_zgraph = not args.no_zgraph

    save_path = args.eval_save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_path = os.path.join(save_path, "metrics.json")

    train_env = AirVLNENV(batch_size=args.batchSize,
                        dataset_path=args.dataset_path,
                        save_path=save_path)
    pilot = DualUAVPilot(config, airsim_client=None)
    metrics = Metrics(log_path=log_path)

    object_desc_dict = dict()
    with open("data/config/object_new_name.json") as f:  ##TODO REPLACE WITH DATASET PATH ARG 
        file = json.load(f)
        for item in file:
            object_desc_dict[item['object_name']] = item['new_name']
    drone2_collision_cnt = 0

    with torch.no_grad():
        dataset = BatchIterator(train_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)
        

        while True:
            env_batches = train_env.next_minibatch()
            if env_batches is None:
                break
            
            rollout = EvalRollout(env_batches, train_env, object_desc_dict, args)
            pbar.update(n=train_env.batch_size)
            seq_name = rollout.ori_data_dirs[0]

            outputs = train_env.reset()
            pilot.reset()
            pilot.update_airsim_client(train_env.simulator_tool.airsim_clients[0][0])
            rollout.update_observation(outputs, [], airsim_collision=False)

            pre_time = None
            t = -1
            while True:
                t += 1
                if t >= args.maxWaypoints:  ##TODO
                    rollout.dones[0] = True

                logger.info('Step: {} \t Completed: {} / {}'.format(
                    t,
                    int(train_env.index_data) - int(train_env.batch_size),
                    end_iter))
                
                cur_time = time.time()
                if pre_time is None:
                    pre_time = cur_time
                else:
                    logger.info('Time Cost : {} s'.format(
                        round(cur_time - pre_time, 2)))
                    pre_time = cur_time

                if rollout.check_traj_status(metrics):
                    break

                instruction = rollout.raw_instructions[0]
                init_pos: list | None = None
                all_predicted_wps: list = []

                obs = rollout.observations[0][-1]
                bev_np = obs["bev"]
                front_np = obs["rgb"][0]
                high_pose, low_pose = pilot.get_current_poses()

                if init_pos is None:
                    init_pos = list(low_pose[:3])
                
                pilot.push(bev_np, high_pose, low_pose)

                t1 = time.time()
                waypoints = pilot.predict(
                    instruction, front_np, low_pose,
                    goal_position=rollout.target_positions[0],
                )
                t2 = time.time()
                print(f"pilot prediction time: {t2-t1}")


                pos_list = []
                airsim_collision = False
                accumulated_distance = 0
                accumulated_time = 0
                start_time = time.time()
                for i, wp in enumerate(waypoints[:args.steps_per_plan]):
                    # Advance drone2 toward the goal in bounded legs paced by
                    # drone1, instead of commanding the full remaining distance.
                    drone1_pos = np.array(
                        pos_list[-1] if pos_list else rollout.drone1_trajs[0][-1])
                    leg_len = np.linalg.norm(np.array(wp[:3]) - drone1_pos)
                    drone2_pos = np.array(
                        train_env.sim_states[0].drone2_traj[-1]['position'])
                    to_goal = np.array(
                        rollout.target_positions[0])[:2] - drone2_pos[:2]
                    dist_to_goal = np.linalg.norm(to_goal)
                    advance = min(dist_to_goal, max(leg_len, 5.0))
                    drone2_xy = drone2_pos[:2]
                    if dist_to_goal > 1e-6:
                        drone2_xy = drone2_xy + to_goal / dist_to_goal * advance
                    drone2_waypoint = np.array(
                        [[drone2_xy[0], drone2_xy[1], drone2_pos[2]]])
                    drone1_waypoint = np.array([wp])  # keep full [x, y, z, heading]
                    drone1_collision, drone2_collision, delta_time, delta_distance, end_pos, airsim_collision = train_env.makeActions(
                        drone1_waypoint, drone2_waypoint, rollout)
                    
                    rollout.steps += 1
                    rollout.distance += delta_distance
                    rollout.time += delta_time

                    accumulated_distance += delta_distance
                    accumulated_time += delta_time
                
                    if len(pos_list) > 0:
                        previous_pos = pos_list[-1]
                    else:
                        previous_pos = rollout.drone1_trajs[0][-1]
                    delta = np.array(end_pos) - np.array(previous_pos)
                    length = np.linalg.norm(delta)
                    delta_unit = delta/length
                    cur_pos = np.array(previous_pos)
                    while length >= 10:
                        length -= 5
                        cur_pos = cur_pos + delta_unit * 5
                        pos_list.append(cur_pos.tolist())
                    pos_list.append(end_pos.tolist())
                    if drone2_collision:
                        drone2_collision_cnt += 1
                        print(f"Drone2 collides! {drone2_collision_cnt} times")
                        break
                    if drone1_collision:
                        break
                    if rollout.dones[0]:
                        break
                if accumulated_time > 1:
                    end_time = time.time()
                    total_nav_time = end_time - start_time
                    mean_velocity = accumulated_distance / accumulated_time
                    rollout.update_navigator_extra_statistics(
                                        metrics, total_nav_time, mean_velocity)
                    
                outputs = train_env.get_obs()
                rollout.update_observation(outputs, pos_list,
                                            airsim_collision)

if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except (msgpackrpc.error.TimeoutError, ConnectionError,
                TimeoutError) as e:
            print(f"AirSim connection error, retrying: {e}")
            traceback.print_exc()
            time.sleep(3)
            continue