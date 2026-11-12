"""Matches over a local network: a match server plays the one match, and every player's computer
is a client that sends its controls and draws what the server sends from its own car.

- :mod:`.protocol`: what servers and clients say to each other, and how.
- :mod:`.room`: a server's room, its seats and its match, and the rules they follow.
- :mod:`.server`: the server, ``rocketsim-server``.
- :mod:`.connection`: a client's connection to a server, and finding servers on the network.
- :mod:`.client`: a player's side of a match: its seat, its controls, the frames it shows.
- :mod:`.match_log`: the server's log of a match, and replaying it.
"""
