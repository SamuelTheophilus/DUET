"""
Drop-in replacement for `python3 -m verl.trainer.main_ppo` that injects
torch.profiler into ActorRolloutRefWorker before Ray creates the workers.

Usage (hydra args pass through via sys.argv):
    python3 -m duet.training.profile_entry algorithm.adv_estimator=grpo ...
"""
#
# from duet.training.profile_patch import patch_verl_for_profiling
# patch_verl_for_profiling()
# from verl.trainer.main_ppo import main
#
# main()
