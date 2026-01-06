## Tips:
* For training and playing the env with an RL workflow (e.g. rl_games) in `Direct` paradigm for any task (e.g. Isaac_Factory_PegInsert_Direct_v0), first change the directory to the IsaacLab dir, then:
  
Trainig (windows):
```
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task Isaac_Factory_PegInsert_Direct_v0 --headless --video
``` 
Playing (windows):
```
isaaclab.bat -p scripts\reinforcement_learning\rl_games\train.py --task Isaac_Factory_PegInsert_Direct_v0 --num_envs 128 --use_last_checkpoint
```

* For making **plots** out of the data (IsaacLab\logs\<Path-to-Data>), execute the follwoing command in CMD:
```
tensorboard --logdir summaries
```
