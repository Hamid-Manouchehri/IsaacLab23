import argparse
import importlib
import torch

from isaaclab.app import AppLauncher


def load_obj_from_entrypoint(entrypoint: str):
    """Load a python object from 'module.submodule:ObjName'."""
    module_path, obj_name = entrypoint.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--task", type=str, default="PegInsert", choices=["PegInsert", "GearMesh", "NutThread"])
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # IMPORTANT: import AFTER SimulationApp is created
    from isaaclab_tasks.direct.ur5e_factory.factory_env import FactoryEnv

    # Pick task cfg class
    task_cfg_entry = {
        "PegInsert": "isaaclab_tasks.direct.ur5e_factory.factory_env_cfg:FactoryTaskPegInsertCfg",
        "GearMesh": "isaaclab_tasks.direct.ur5e_factory.factory_env_cfg:FactoryTaskGearMeshCfg",
        "NutThread": "isaaclab_tasks.direct.ur5e_factory.factory_env_cfg:FactoryTaskNutThreadCfg",
    }[args.task]

    CfgCls = load_obj_from_entrypoint(task_cfg_entry)
    cfg = CfgCls()

    # Set number of envs (FactoryEnvCfg stores it inside scene cfg)
    cfg.scene.num_envs = args.num_envs

    # Create env directly (bypasses gym.make() issues)
    env = FactoryEnv(cfg=cfg, render_mode="rgb_array")

    obs, _ = env.reset()

    # Zero action, just advance physics so you can visually verify assets
    # action dimension (Factory uses cfg.action_space as an int = 6)
    act_dim = int(getattr(env.cfg, "action_space", 0))  # should be 6

    # fallback if you ever wrap it as a gym.Box later
    if act_dim == 0 and hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
        act_dim = int(env.action_space.shape[0])

    zero_action = torch.zeros((args.num_envs, act_dim), device=env.device)


    while simulation_app.is_running():
        obs, rew, terminated, truncated, info = env.step(zero_action)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
