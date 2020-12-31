class BaseScenario(object):
    # create element of game world
    def make_world(self):
        raise NotImplementedError()

    # create initial condition of the world
    def reset_world(self, world):
        raise NotImplementedError()
