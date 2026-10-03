import sys
import os
import subprocess
import json

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_eval_video_size(args):
    head_camera_cfg = get_camera_config(args["camera"]["head_camera_type"])
    video_w = int(head_camera_cfg["w"])
    video_h = int(head_camera_cfg["h"])

    if args["camera"].get("collect_wrist_camera", False):
        wrist_camera_cfg = get_camera_config(args["camera"]["wrist_camera_type"])
        wrist_w = int(wrist_camera_cfg["w"])
        wrist_h = int(wrist_camera_cfg["h"])
        video_w = max(video_w, wrist_w * 2)
        video_h = video_h + wrist_h

    return f"{video_w}x{video_h}"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _result_suffix_from_task_config(task_config):
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "random"
    raise ValueError(
        f"Unsupported `task_config` for fixed result naming: {task_config}. "
        "Expected one of: ['demo_clean', 'demo_randomized']."
    )


def _atomic_write_json(path, payload, *, durable=False):
    """Replace JSON atomically; optionally fsync benchmark-semantic state."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class EvaluationProgress:
    """Own the benchmark-semantic state needed to resume the exact next seed."""

    SCHEMA_VERSION = 1

    def __init__(self, progress_path, status_path, metadata):
        self.progress_path = None if not progress_path else Path(str(progress_path))
        self.status_path = None if not status_path else Path(str(status_path))
        self.metadata = metadata

        if self.progress_path is not None and self.progress_path.is_file():
            with self.progress_path.open("r", encoding="utf-8") as handle:
                self.state = json.load(handle)
            self._validate_loaded_state()
        else:
            now = datetime.now().astimezone().isoformat()
            self.state = {
                "schema_version": self.SCHEMA_VERSION,
                "status": "in_progress",
                **metadata,
                "completed_episodes": 0,
                "successes": 0,
                "next_seed": int(metadata["start_seed"]),
                "episodes": [],
                "created_at": now,
                "updated_at": now,
            }
            self._save()

    @property
    def completed_episodes(self):
        return int(self.state["completed_episodes"])

    @property
    def successes(self):
        return int(self.state["successes"])

    @property
    def next_seed(self):
        return int(self.state["next_seed"])

    def _validate_loaded_state(self):
        if self.state.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported evaluation progress schema in {self.progress_path}: "
                f"{self.state.get('schema_version')!r}."
            )

        mismatches = [
            f"{key}: stored={self.state.get(key)!r}, expected={value!r}"
            for key, value in self.metadata.items()
            if self.state.get(key, False if key in {"sample_adaptive_gate", "match_eval_in_train_instruction"} else None) != value
        ]
        if mismatches:
            raise ValueError(
                f"Incompatible RoboTwin episode progress: {self.progress_path}\n"
                + "\n".join(mismatches)
            )

        completed = self.completed_episodes
        successes = self.successes
        episodes = self.state.get("episodes")
        target = int(self.metadata["target_episodes"])
        if (
            not isinstance(episodes, list)
            or any(not isinstance(episode, dict) for episode in episodes)
        ):
            raise ValueError(f"Malformed episode history in {self.progress_path}.")
        if not 0 <= successes <= completed <= target:
            raise ValueError(f"Invalid episode counts in {self.progress_path}.")
        if len(episodes) != completed:
            raise ValueError(f"Inconsistent episode history lengths in {self.progress_path}.")
        if [episode.get("episode_index") for episode in episodes] != list(range(completed)):
            raise ValueError(f"Episode indices are not contiguous in {self.progress_path}.")
        if any(not isinstance(episode.get("success"), bool) for episode in episodes):
            raise ValueError(f"Episode success values must be booleans in {self.progress_path}.")
        if sum(episode["success"] for episode in episodes) != successes:
            raise ValueError(f"Success count does not match episode history in {self.progress_path}.")
        start_seed = int(self.metadata["start_seed"])
        next_seed = int(self.state.get("next_seed", -1))
        if next_seed < start_seed:
            raise ValueError(f"Invalid next_seed in {self.progress_path}.")
        episode_seeds = [int(episode.get("seed")) for episode in episodes]
        if any(seed < start_seed or seed >= next_seed for seed in episode_seeds):
            raise ValueError(f"Episode seed is outside the processed range in {self.progress_path}.")
        if any(left >= right for left, right in zip(episode_seeds, episode_seeds[1:])):
            raise ValueError(f"Episode seeds are not strictly increasing in {self.progress_path}.")
        expected_status = "complete" if completed == target else "in_progress"
        if self.state.get("status") != expected_status:
            raise ValueError(
                f"Progress status/count mismatch in {self.progress_path}: "
                f"status={self.state.get('status')!r}, completed={completed}/{target}."
            )

    def _save(self):
        self.state["updated_at"] = datetime.now().astimezone().isoformat()
        if self.progress_path is not None:
            _atomic_write_json(self.progress_path, self.state, durable=True)

    def stage(self, stage, *, seed=None, episode=None, **details):
        """Emit a human-readable marker and an atomic machine-readable status."""

        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": datetime.now().astimezone().isoformat(),
            "pid": os.getpid(),
            "stage": str(stage),
            "task_name": self.metadata["task_name"],
            "task_config": self.metadata["task_config"],
            "episode": self.completed_episodes if episode is None else int(episode),
            "seed": self.next_seed if seed is None else int(seed),
            "completed_episodes": self.completed_episodes,
            "successes": self.successes,
            "details": details,
        }
        if self.status_path is not None:
            _atomic_write_json(self.status_path, payload)
        fields = [
            f"stage={payload['stage']}",
            f"episode={payload['episode']}",
            f"seed={payload['seed']}",
        ]
        fields.extend(f"{key}={value}" for key, value in details.items())
        print("[ROBOTWIN_STAGE] " + " ".join(fields), flush=True)

    def reject_candidate(self, seed):
        seed = int(seed)
        if seed != self.next_seed:
            raise RuntimeError(
                f"Cannot reject seed {seed}; progress expects candidate seed {self.next_seed}."
            )
        self.state["next_seed"] = seed + 1
        self._save()

    def commit_episode(self, seed, success, *, instruction=None):
        seed = int(seed)
        success = bool(success)
        if seed != self.next_seed:
            raise RuntimeError(
                f"Cannot commit seed {seed}; progress expects candidate seed {self.next_seed}."
            )
        episode_index = self.completed_episodes
        self.state["episodes"].append(
            {
                "episode_index": episode_index,
                "seed": seed,
                "success": success,
                "timestamp": datetime.now().astimezone().isoformat(),
            }
        )
        if instruction is not None:
            self.state["episodes"][-1]["instruction"] = instruction
        self.state["completed_episodes"] = episode_index + 1
        self.state["successes"] = self.successes + int(success)
        self.state["next_seed"] = seed + 1
        if self.completed_episodes == int(self.metadata["target_episodes"]):
            self.state["status"] = "complete"
        self._save()


def main(usr_args):
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    match_eval_in_train_instruction = usr_args.get("match_eval_in_train_instruction", False)
    if not isinstance(match_eval_in_train_instruction, bool):
        raise ValueError("match_eval_in_train_instruction must be true or false.")
    episode_instruction = usr_args.get("episode_instruction") if match_eval_in_train_instruction else None
    skip_get_obs_within_replan = parse_bool(usr_args.get("skip_get_obs_within_replan", False))
    eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
    if eval_num_episodes <= 0:
        raise ValueError(f"`eval_num_episodes` must be > 0, got: {eval_num_episodes}")
    eval_output_dir = usr_args.get("eval_output_dir")
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if eval_output_dir is not None and str(eval_output_dir).strip() != "":
        save_dir = Path(str(eval_output_dir))
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{eval_ts}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        video_size = get_eval_video_size(args)
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    test_num = eval_num_episodes

    progress_metadata = {
        "task_name": str(task_name),
        "task_config": str(task_config),
        "checkpoint_dir": str(ckpt_setting),
        "dataset_stats_path": str(usr_args.get("dataset_stats_path")),
        "instruction_type": str(instruction_type),
        "match_eval_in_train_instruction": match_eval_in_train_instruction,
        "visual_gate": (
            None
            if usr_args.get("visual_gate") is None
            else float(usr_args.get("visual_gate"))
        ),
        "target_episodes": int(test_num),
        "start_seed": int(st_seed),
    }
    if match_eval_in_train_instruction:
        progress_metadata["episode_instruction"] = episode_instruction
    if str(usr_args.get("visual_gate_mode", "fixed")) == "policy":
        # Gate-policy identity is part of atomic resume semantics for GRPO evaluation.
        progress_metadata.update(
            visual_gate_mode="policy",
            policy_checkpoint=str(usr_args.get("policy_checkpoint")),
            sample_adaptive_gate=parse_bool(usr_args.get("sample_adaptive_gate", False)),
        )
    progress = EvaluationProgress(
        usr_args.get("episode_progress_path"),
        usr_args.get("worker_status_path"),
        progress_metadata,
    )
    usr_args["episode_start_index"] = progress.completed_episodes
    progress.stage(
        "progress_loaded",
        seed=progress.next_seed,
        episode=progress.completed_episodes,
        resumed=progress.completed_episodes > 0,
    )

    if progress.completed_episodes < test_num:
        progress.stage("model_load_begin")
        model = get_model(usr_args)
        progress.stage("model_load_done")
        _, suc_num = eval_policy(task_name,
                                 TASK_ENV,
                                 args,
                                 model,
                                 progress,
                                 test_num=test_num,
                                 video_size=video_size,
                                 instruction_type=instruction_type,
                                 match_eval_in_train_instruction=match_eval_in_train_instruction,
                                 episode_instruction=episode_instruction,
                                 skip_get_obs_within_replan=skip_get_obs_within_replan)
    else:
        progress.stage("evaluation_already_complete")
        suc_num = progress.successes

    result_suffix = _result_suffix_from_task_config(task_config)
    file_path = os.path.join(save_dir, f"_result_{result_suffix}.txt")
    result_text = (
        f"Timestamp: {eval_ts}\n\n"
        f"Instruction Type: {instruction_type}\n\n"
        f"{suc_num / test_num}"
    )
    Path(file_path).write_text(result_text, encoding="utf-8")
    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                progress,
                test_num=100,
                video_size=None,
                instruction_type=None,
                skip_get_obs_within_replan=False,
                match_eval_in_train_instruction=False,
                episode_instruction=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = progress.successes
    TASK_ENV.test_num = progress.completed_episodes

    now_id = progress.completed_episodes
    succ_seed = progress.completed_episodes

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = progress.next_seed
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    # Only the opt-in branch depends on the shared GRPO instruction selector.
    if match_eval_in_train_instruction:
        from dev.model.rl.robotwin_env import select_episode_instruction

    def close_task_env(stage, seed, episode, **kwargs):
        """Bracket native cleanup so the watchdog can identify cleanup stalls."""

        progress.stage(f"{stage}_begin", seed=seed, episode=episode)
        TASK_ENV.close_env(**kwargs)
        progress.stage(f"{stage}_done", seed=seed, episode=episode)

    while succ_seed < test_num:
        episode_index = progress.completed_episodes
        candidate_seed = now_seed
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                progress.stage(
                    "expert_setup_begin", seed=candidate_seed, episode=episode_index
                )
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                progress.stage(
                    "expert_setup_done", seed=candidate_seed, episode=episode_index
                )
                progress.stage(
                    "expert_play_begin", seed=candidate_seed, episode=episode_index
                )
                episode_info = TASK_ENV.play_once()
                progress.stage(
                    "expert_play_done",
                    seed=candidate_seed,
                    episode=episode_index,
                    plan_success=bool(TASK_ENV.plan_success),
                )
                close_task_env("expert_close", candidate_seed, episode_index)
            except UnStableError as e:
                progress.reject_candidate(candidate_seed)
                progress.stage(
                    "candidate_rejected",
                    seed=candidate_seed,
                    episode=episode_index,
                    reason="expert_unstable",
                    error=str(e),
                )
                close_task_env("expert_close", candidate_seed, episode_index)
                now_seed = progress.next_seed
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                print(" -------------")
                print("Error: ", e)
                print("Stack Trace: ", traceback.format_exc())
                print(" -------------")
                progress.reject_candidate(candidate_seed)
                progress.stage(
                    "candidate_rejected",
                    seed=candidate_seed,
                    episode=episode_index,
                    reason=f"expert_exception:{type(e).__name__}",
                    error=str(e),
                )
                close_task_env("expert_close", candidate_seed, episode_index)
                now_seed = progress.next_seed
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        expert_succeeded = (not expert_check) or (
            TASK_ENV.plan_success and TASK_ENV.check_success()
        )
        if expert_succeeded:
            succ_seed += 1
            progress.stage(
                "candidate_accepted",
                seed=candidate_seed,
                episode=episode_index,
            )
        else:
            progress.reject_candidate(candidate_seed)
            progress.stage(
                "candidate_rejected",
                seed=candidate_seed,
                episode=episode_index,
                reason="expert_check_failed",
            )
            now_seed = progress.next_seed
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        try:
            progress.stage(
                "rollout_setup_begin", seed=candidate_seed, episode=episode_index
            )
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            progress.stage(
                "rollout_setup_done", seed=candidate_seed, episode=episode_index
            )
        except UnStableError as e:
            # This seed passed expert_check but failed during rollout env init.
            # Roll back the accepted-seed counter and skip to next seed.
            succ_seed -= 1
            progress.reject_candidate(candidate_seed)
            progress.stage(
                "candidate_rejected",
                seed=candidate_seed,
                episode=episode_index,
                reason="rollout_setup_unstable",
                error=str(e),
            )
            close_task_env("rollout_setup_cleanup", candidate_seed, episode_index)
            now_seed = progress.next_seed
            continue
        except Exception as e:
            succ_seed -= 1
            print(" -------------")
            print("Error: ", e)
            print("Stack Trace: ", traceback.format_exc())
            print(" -------------")
            progress.reject_candidate(candidate_seed)
            progress.stage(
                "candidate_rejected",
                seed=candidate_seed,
                episode=episode_index,
                reason=f"rollout_setup_exception:{type(e).__name__}",
                error=str(e),
            )
            close_task_env("rollout_setup_cleanup", candidate_seed, episode_index)
            now_seed = progress.next_seed
            print("error occurs !")
            continue
        # Resolve once per episode; replans reuse TASK_ENV's instruction.
        if match_eval_in_train_instruction:
            if episode_instruction is not None:
                instruction = episode_instruction
            else:
                instruction = select_episode_instruction(
                    args["task_name"],
                    episode_info["info"],
                    instruction_type,
                    pin_seed=candidate_seed,
                )
        else:
            # Native evaluation: preserve its candidate count and RNG behavior.
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            instruction = np.random.choice(results[0][instruction_type])

        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        current_video_path = None
        if TASK_ENV.eval_video_path is not None:
            episode_idx = TASK_ENV.test_num
            current_video_path = Path(TASK_ENV.eval_video_path) / f"episode{episode_idx}.mp4"
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    str(current_video_path),
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        # Optional policy context only: do not alter expert filtering or environment RNG.
        if hasattr(model, "on_episode_start"):
            model.on_episode_start(scene_seed=candidate_seed)
        progress.stage("policy_rollout_begin", seed=candidate_seed, episode=episode_index)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            need_obs = True
            if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
                need_obs = bool(model.should_request_observation())

            observation = None
            if need_obs:
                observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        progress.stage(
            "policy_rollout_done",
            seed=candidate_seed,
            episode=episode_index,
            success=bool(succ),
            simulation_steps=int(TASK_ENV.take_action_cnt),
        )

        # Success/failure is final when the rollout ends. Commit it before any
        # optional artifact or simulator cleanup so a cleanup stall cannot
        # cause this episode to be sampled and scored a second time.
        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")
        now_id += 1
        TASK_ENV.test_num += 1
        progress.commit_episode(
            candidate_seed,
            succ,
            instruction=instruction if match_eval_in_train_instruction else None,
        )
        if (
            TASK_ENV.test_num != progress.completed_episodes
            or TASK_ENV.suc != progress.successes
        ):
            raise RuntimeError(
                "RoboTwin in-memory counters diverged from the durable episode progress: "
                f"env={TASK_ENV.suc}/{TASK_ENV.test_num}, "
                f"progress={progress.successes}/{progress.completed_episodes}."
            )
        progress.stage(
            "episode_committed",
            seed=candidate_seed,
            episode=episode_index,
            success=bool(succ),
            next_seed=progress.next_seed,
        )
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()
            if current_video_path is None or not current_video_path.exists():
                raise FileNotFoundError(f"Expected eval video file not found: {current_video_path}")
            is_randomized = "randomized" in str(args["task_config"]).lower()
            renamed_video_path = (
                Path(TASK_ENV.eval_video_path)
                / f"episode{episode_idx}_randomized-{str(is_randomized).lower()}_success-{str(succ).lower()}.mp4"
            )
            current_video_path.rename(renamed_video_path)
            stale_video_path = (
                Path(TASK_ENV.eval_video_path)
                / f"episode{episode_idx}_randomized-{str(is_randomized).lower()}_success-{str(not succ).lower()}.mp4"
            )
            stale_video_path.unlink(missing_ok=True)

        # Optional policy hook for episode-level diagnostics.  It runs after
        # the rollout MP4 is closed but before the simulator itself is closed.
        if hasattr(model, "on_episode_end"):
            model.on_episode_end(TASK_ENV, succ)
        close_task_env(
            "rollout_close",
            candidate_seed,
            episode_index,
            clear_cache=((succ_seed + 1) % clear_cache_freq == 0),
        )

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()
        # TASK_ENV._take_picture()
        now_seed = progress.next_seed

    progress.stage(
        "evaluation_complete",
        seed=progress.next_seed,
        episode=progress.completed_episodes,
        success_count=progress.successes,
    )
    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
