"""Rocket League-style soccar as a Gymnasium env, on RocketSim physics and a homemade arena."""

import gymnasium

gymnasium.register(id="RocketSimSoccar-v0", entry_point="rocketsim_gym.env:SoccarEnv")
