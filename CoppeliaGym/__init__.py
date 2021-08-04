import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)
register(
    id='QuadratotBulletEnv-v0',
    entry_point=
    'CoppeliaGym.envs:QuadratotEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)

register(
    id='ArgoV2Env-v0',
    entry_point=
    'CoppeliaGym.envs:ArgoV2Env',
    max_episode_steps=1200,
    kwargs={'dt': 0.05,
            'headless': True},
)