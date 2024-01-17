import itertools

ENVS = [
    "Alien-v5",
    "Amidar-v5",
    "Assault-v5",
    "Asterix-v5",
    "Asteroids-v5",
    "Atlantis-v5",
    "BankHeist-v5",
    "BattleZone-v5",
    "BeamRider-v5",
    "Berzerk-v5",
    "Bowling-v5",
    "Boxing-v5",
    "Breakout-v5",
    "Centipede-v5",
    "ChopperCommand-v5",
    "CrazyClimber-v5",
    "Defender-v5",
    "DemonAttack-v5",
    "DoubleDunk-v5",
    "Enduro-v5",
    "FishingDerby-v5",
    "Freeway-v5",
    "Frostbite-v5",
    "Gopher-v5",
    "Gravitar-v5",
    "Hero-v5",
    "IceHockey-v5",
    "Jamesbond-v5",
    "Kangaroo-v5",
    "Krull-v5",
    "MsPacman-v5",
    "KungFuMaster-v5",
    "MontezumaRevenge-v5",
    "NameThisGame-v5",
    "Phoenix-v5",
    "Pitfall-v5",
    "Pong-v5",
    "PrivateEye-v5",
    "Qbert-v5",
    "Riverraid-v5",
    "RoadRunner-v5",
    "Robotank-v5",
    "Seaquest-v5",
    "Skiing-v5",
    "Solaris-v5",
    "SpaceInvaders-v5",
    "StarGunner-v5",
    "Surround-v5",
    "Tennis-v5",
    "TimePilot-v5",
    "Tutankham-v5",
    "UpNDown-v5",
    "Venture-v5",
    "VideoPinball-v5",
    "WizardOfWor-v5",
    "YarsRevenge-v5",
    "Zaxxon-v5",
]

LEARNING_RATES = [5e-4, 1e-3, 2e-3]

MAX_GRAD_NORMS = [0.5, 1.0, 5.0, 10.0]

SEEDS = [0, 1, 2]

GAE_LAMBDAS = [0.9, 0.95]

UPDATE_EPOCHS = [4, 6, 8]

RESET_TYPE = ["count"]


def generate_product_with_labels(hyperparams):
    keys, param_settings = [list(a) for a in zip(*hyperparams.items())]
    for param_setting in itertools.product(*param_settings):
        config = {
            keys[i]: param_setting for i, param_setting in enumerate(param_setting)
        }
        yield config


if __name__ == "__main__":
    hyperparams = {
        "learning_rate": LEARNING_RATES,
        "max_grad_norm": MAX_GRAD_NORMS,
        "gae_lambda": GAE_LAMBDAS,
        "update_epochs": UPDATE_EPOCHS,
        "seed": SEEDS,
        "reset_type": RESET_TYPE,
        "env_id": ENVS,
    }
    run_strings = []
    for config in generate_product_with_labels(hyperparams):
        run_string = "python ppo_atari_envpool_xla_jax_scan.py "
        for k, v in config.items():
            run_string += f"--{k}={v} "
        run_strings.append(run_string)

    with open("cleanrl/full_atari_sweep.txt", mode="wt", encoding="utf-8") as f:
        f.write('\n'.join(run_strings))