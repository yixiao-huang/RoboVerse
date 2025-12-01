task_name=stack_cube # ['stack_cube', 'close_box']
level=0
config_name=dp_distill_runner

port=50010
seed=42
gpu=0
obs_space=joint_pos
act_space=joint_pos
delta_ee=0
eval_num_envs=1
env_spacing=1.0
max_eval_instances=100
if [ "${task_name}" = "close_box" ]; then
  num_epochs=100            # Number of training epochs
  max_train_steps=250
  eval_max_step=500
# stack cube
elif [ "${task_name}" = "stack_cube" ]; then
  num_epochs=100            # Number of training epochs
  max_train_steps=250
  eval_max_step=500
else
  num_epochs=100            # Number of training epochs
  max_train_steps=250
  eval_max_step=100
fi
expert_data_num=100
sim_set=isaacsim
dp_camera=True
run_unfinished=True
run_all=True
run_failed=False

## Choose training or inference algorithm
# Supported models:
#   "ddpm_unet_model", "ddpm_dit_model", "ddim_unet_model", "fm_unet_model", "fm_dit_model", "score_model", "vita_model"
algo_choose=0

algo_model=""
eval_path=""

home_path=/media/volume/MOL/MOL-Robotics/RoboVerse
eval_ckpt_name=100
# debug_extra="-dt:0.001-decimation:15"
debug_extra="_camera"
# debug_extra=""
task_extra="${task_name}_obs:${obs_space}_act:${act_space}${debug_extra}"

eval_path="${home_path}/info/outputs/DP/${task_extra}/checkpoints/${eval_ckpt_name}.ckpt"
cust_name="${debug_extra}"

case $algo_choose in
    0)
        # DDPM settings
        export algo_model="ddpm_unet_model"
        if [ "${task_name}" = "stack_cube" ]; then
          run_id="2025.11.12/01.52.51_stack_cube_obs:joint_pos_act:joint_pos/checkpoints/200.ckpt"
          # eval_path="${home_path}/info/outputs/DP/${run_id}"
        elif [ "${task_name}" = "close_box" ]; then
          run_id="2025.10.28/08.05.06_close_box_obs:joint_pos_act:joint_pos/checkpoints/100.ckpt"
        fi

        ;;
    1)
        # DDIM settings
        export algo_model="ddim_unet_model"
        ;;
    2)
        # FM settings
        export algo_model="dp_FM_UNet"
        ;;
    3)
        # FM DiT Settings
        export algo_model="dp_FM_DiT"
        ;;
    4)
        # Score-based settings
        export algo_model="dp_Score"
        ;;
    5)
        # VITA Settings
        export algo_model="dp_VITA"
        ;;
    *)
        echo "Invalid algorithm choice: $algo_choose"
        echo "Available options: 0 (DDPM), 1 (DDIM), 2 (FM UNet), 3 (FM DiT), 4 (Score-based), 5 (VITA)"
        exit 1
        ;;
esac
echo "Selected model: $algo_model"
echo "Checkpoint path: $eval_path"

extra="obs:${obs_space}_act:${act_space}${debug_extra}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi


python roboverse_learn/il/dp/distill.py --config-name=${config_name}.yaml \
task_name="${task_name}_${extra}" \
dataset_config.zarr_path="data_policy/${task_name}FrankaL${level}_${extra}_${expert_data_num}.zarr" \
distill_config.policy_runner.obs.obs_type=${obs_space} \
distill_config.policy_runner.action.action_type=${act_space} \
distill_config.policy_runner.action.delta=${delta_ee} \
distill_config.distill_args.task=${task_name} \
distill_config.distill_args.max_step=${eval_max_step} \
distill_config.distill_args.num_envs=${eval_num_envs} \
distill_config.distill_args.env_spacing=${env_spacing} \
distill_config.distill_args.max_demo=${max_eval_instances} \
distill_config.distill_args.dp_camera=${dp_camera} \
distill_config.distill_args.sim=${sim_set} \
distill_config.seed=${seed} \
distill_config.ckpt_path=${eval_path} \
distill_config.distill_args.run_unfinished=${run_unfinished} \
distill_config.distill_args.run_all=${run_all} \
distill_config.distill_args.run_failed=${run_failed} \
distill_config.distill_args.cust_name=${cust_name} \
# distill_config.distill_args.random.level=${level} \
