from space import Box
import numpy as np
from gym import spaces
import seeding

class Env(object):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc...
    """
    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        """Support with-statement for the environment. """
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment. """
        self.close()
        # propagate exception
        return False

class PendulumEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=9.8):

        self.min_x = -1.0
        self.max_x = 3.0
        self.value = 0
        self.viewer = None

        self.action_space = spaces.Discrete(3)  # 0, 1, 2: 向左 不动 向右
        self.observation_space = spaces.Box(np.array([-1, -4]), np.array([3, 4]))

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        x = self.state
        if action == 0:
            step_size = -0.1
        if action == 1:
            step_size = 0
        if action == 2:
            step_size = 0.1

        costs = 0.99*x**5 - 5*x**4 + 4.98*x**3 + 5*x**2 - 6*x - 1
        self.value = costs

        new_x = x + step_size*(0.99*5*x**4 - 5*4*x**3 + 4.98*3*x**2 + 5*2*x - 6)
        self.state = new_x
        self.counts += 1
        return self.state, costs, False, {}

    def reset(self):
        #self.state = np.random.uniform(-1,3)
        self.state = 2
        self.counts = 0
        return self.state

    def render(self, mode='human'):
        import rendering
        if self.viewer is None:

            self.viewer = rendering.Viewer(400, 800)
            #self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            #point1 = rendering.Point
            #self.point1.add_attr(self.circletrans)
            #rod = rendering.make_capsule(1, .2)
            #point1.set_color(.8, .3, .3)
            #self.pole_transform = rendering.Transform()
            #rod.add_attr(self.pole_transform)

            #axle = rendering.make_circle(.05)
            #axle.set_color(0, 0, 0)
            #self.viewer.add_geom(axle)
            self.line1 = rendering.Line((-1, 0), (3, 0))
            self.line2 = rendering.Line((0, -4), (0, 4))
            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
        x = self.state
        costs = 0.99 * x ** 5 - 5 * x ** 4 + 4.98 * x ** 3 + 5 * x ** 2 - 6 * x - 1
        self.point1 = rendering.make_circle(100)
        self.point1.set_color(0, 1, 1)
        self.circletrans = rendering.Transform(translation=(x, costs))
        self.point1.add_attr(self.circletrans)
        self.viewer.add_geom(self.point1)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


