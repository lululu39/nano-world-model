"""Keep episode records limited to metrics defined by their environment."""


def episode_record(record, environment):
    result = dict(record)
    if environment == 'point_maze':
        # The shared planner uses NaN for absent angles. PointMaze has only
        # position and velocity; retain strict JSON checks for all real metrics.
        result.pop('angle_dist', None)
    return result
