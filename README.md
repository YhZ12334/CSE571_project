# so101 on Lerobot setup & application cammand
## Env Setup
https://huggingface.co/docs/lerobot/installation

## Hardware Setup
```bash
lerobot-find-port
```
```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```
```bash
lerobot-find-cameras opencv
```

## Calibration
```bash
lerobot-calibrate \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM0 \
--robot.id=my_follower
```
```bash
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_leader
```

## Teleop
```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_leader \
  --display_data=false
```

## Record
Note: rename repo_id every time.
```bash
lerobot-record \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM0 \
--robot.id=my_follower \
--robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
--teleop.type=so101_leader \
--teleop.port=/dev/ttyACM1 \
--teleop.id=my_leader \
--dataset.repo_id=CLM0215/pick_cube_test2 \
--dataset.num_episodes=30 \
--dataset.single_task="pick a cube and put it on a yellow pad" \
--dataset.push_to_hub=false \
--play_sounds=false \
--dataset.reset_time_s=20 \
--dataset.episode_time_s=30
```

## Upload to Huggingface
First time login:
```bash
huggingface-cli login
```
Upload:
```bash
hf upload {repo_id}/{dataset_name_on_repo} {dataset_location} --repo-type=dataset
```
E.g.:
```bash
hf upload CLM0215/pick_cube_test2 \
$HOME/.cache/huggingface/lerobot/CLM0215/pick_cube_test2 \
--repo-type=dataset
```

## Low_level point to point
```bach
python low_level.py   --port /dev/ttyACM0   --x 0.24   --y 0.10   --z 0.24   --dt 0.05   --max-step 0.003   --max-relative-target 1.5   --position-weight 1.0   --orientation-weight 0.00   --no-calibrate
```

## Low_level_new trajectory_following
```bash
python low_level_new.py --port /dev/ttyACM0 --points "[(0.22,0.08,0.22),(0.24,0.10,0.22),(0.24,0.10,0.18)]"  --gripper-actions "open,open,close" --dt 0.05 --position-weight 1.0 --orientation-weight 0.0 --ik-pos-tol 0.02 --exec-pos-tol 0.025 --interp-max-step-deg 1.0 --max-relative-target 5.0 --no-calibrate
```
