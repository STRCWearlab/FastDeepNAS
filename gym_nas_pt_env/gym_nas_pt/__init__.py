from gym.envs.registration import register

register(
	id='nas_pt-v0',
	entry_point='gym_nas_pt.envs:NASPtEnv',
)