# so101 on Lerobot setup & application cammand
## Setup
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
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --display_data=true
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
```bash
hf upload CLM0215/pick_cube pick_cube_/ --repo-type=dataset 
```
In case:
```bash
huggingface-cli login
```