Simulations (Python 3 + numpy). Walker approximation of ~10k Starlink satellites, receiver in Seattle, 11.7 GHz.
overlap.py      visible count, nearest-neighbour Doppler gaps, sensitivity to position/TLE errors
assoc.py        single-track ID success, windows 1/10/60 s (60 trials)
assoc_table.py  single-track ID success table in Q21 (100 trials, ~5 min)
joint.py        joint ID of 3 tracks with a 2D position grid search (25 trials, ~8 min)
known_position.py  ID success with known receiver position: TLE quality, LO handling, beam gating (150 trials, ~8 min)
beam_neighbours.py other satellites inside a beam of given half-angle
joint_assign.py    per-track vs joint (common LO offset) NORAD matching, with Starlink CFO corrections (~4 min)
