
## run nvidia-smi to check available GPUs
export CUDA_VISIBLE_DEVICES=0

## Parameters
task_name_set=close_box # ['stack_cube', 'close_box']
task_name_sets=("stack_cube" "close_box")
random_level=0          # 0: No randomization 1: Randomize visual material 2: Randomize camera pose 3: Randomize object reflection and lighting
num_envs=1              # Number of parallel environments
demo_start_idx=0      # Index of the first demo to collect
num_demo_success=100
max_demo_idx=100    # Maximum index of demos to collect
expert_data_num=100  # Number of expert demonstration data to convert
sim_set=isaacsim # ['mujoco', 'isaacsim']
dp_camera=True
cust_name=test
retry_num=0   # Number of retries for each demo
obs_space=joint_pos
act_space=joint_pos
delta_ee=0              # Delta control
# debug_extra="-dt:0.001-decimation:10"  # Extra debug arguments
debug_extra="_camera"  # Extra debug arguments
# debug_extra=""  # Extra debug arguments
extra="obs:${obs_space}_act:${act_space}${debug_extra}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

cust_name="${cust_name}"

# python collect_demo.py \
#   --task CloseBox \
#   --robot franka \
#   --sim isaaclab \
#   --num_envs 1 \
#   --headless True \
#   --random.level 2 \
#   --run_unfinished

if [ "${dp_camera}" = "True" ]; then
  dp_camera_flag="--dp_camera"
else
  dp_camera_flag=""
fi
## Collecting demonstration data
mode=collect # collect / convert

# if [ "${mode}" = "collect" ]; then
echo "Collect demonstration data for task: ${task_name_set}"
# python scripts/advanced/collect_demo.py \
#   --sim=${sim_set} \
#   --task=${task_name_set} \
#   --num_envs=${num_envs} \
#   --run_unfinished \
#   --headless \
#   --demo_start_idx=${demo_start_idx} \
#   --num_demo_success=${num_demo_success} \
#   --no-run_all \
#   --retry_num=${retry_num} \
#   --cust_name=${cust_name} \
#   ${dp_camera_flag} \

# --enable_randomization
# elif [ "${mode}" = "convert" ]; then
echo "Convert demonstration data for task: ${task_name_set}"
## Convert demonstration data

# python roboverse_learn/il/multi-task/append_task_config.py \
#   --config_path roboverse_learn/il/multi-task/configs/task_configs${debug_extra}.json \
#   --task_name ${task_name_set}FrankaL${random_level}_${extra} \
#   --max_demo_idx ${max_demo_idx} \
#   --expert_data_num ${expert_data_num} \
#   --metadata_dir roboverse_demo/demo_${sim_set}/${task_name_set}-${cust_name}/robot-franka/success \
#   --downsample_ratio 1 \
#   --show


# python roboverse_learn/il/data2zarr_dp.py \
#   --task_name ${task_name_set}FrankaL${random_level}_${extra} \
#   --expert_data_num ${expert_data_num} \
#   --max_demo_idx ${max_demo_idx} \
#   --metadata_dir roboverse_demo/demo_${sim_set}/${task_name_set}-${cust_name}/robot-franka/success \
#   --action_space ${act_space} \
#   --observation_space ${obs_space}
# else
#   echo "Invalid mode"
#   exit 1
# fi

echo "Convert multitask demonstration data"
# python roboverse_learn/il/data2zarr_dp_multitask_gpt.py \
python roboverse_learn/il/data2zarr_dp_multitask.py \
  --task_config roboverse_learn/il/multi-task/configs/task_configs${debug_extra}.json \
  --output_name multitask_franka${debug_extra} \
  --observation_space joint_pos \
  --action_space joint_pos

