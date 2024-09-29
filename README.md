# HRL_thesis



## What is this Project
This is a master thesis. We propose a Hierarchical Reinforcement Learning(HRL) method, using SAC with termination 
gradient under OCAD(Option-Critic Architecutre with deterministic intra-option policy, named TSAC under OCDA. \
We implement and validate the method in air-hockey games, with the LBR iiwa, the 7 DoF robotic arm.\
The low level agents include ATACOM and CNP-B. The ATACOM is a besser choice.\
We improve the agent's performance using Self Learning(SL) and Curriculum Learning(CL).

## Different Branches
The base experiment is in branch development. The other experiments are in branch cl_line and cl_reward. To set self_learn=True to 
turn on self learning.\
The tournament_eval includes the tournament configuration to evaluate two agents in tournament.\
In atacom we abandon OCDA, the agent direct learn the joint command instead of option.






