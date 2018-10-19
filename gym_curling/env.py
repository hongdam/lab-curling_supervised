import numpy as np
from gym_curling.Simulator import simulate
import matplotlib.pyplot as plt


class Curling:

    def __init__(self, uncertainty):
        self.state = None
        self.current_turn = None
        self.order = 1

        self.uncertainty = uncertainty

        self._fig, self._ax = plt.subplots()
        self._fig.set_size_inches(4.75 / 2, 11.28 / 2)

        self.x = 0
        self.y = 0
        self.curl = 0


    def step(self, action):
        self.x, self.y, self.curl = action
        # xy = simulate(xy, i, random.random()*4.75, random.random()*11.28, random.randint(0,1), 0.145)[0]
        self.state, actual_action = simulate(self.state, self.current_turn,
                                             action[0], action[1], action[2],
                                             self.uncertainty)
        # actual_action: [ -1.0299234 -29.67658     0.       ]

        done = True if self.current_turn == 15 else False
        reward = self.get_score(self.state, self.order) if done else 0
        self.current_turn += 1
        return self._get_obs(), reward, done

    def reset(self):
        self.state = np.zeros(32, np.float32)
        self.current_turn = 0

        return self._get_obs()

    def _get_obs(self):
        return [self.state, self.current_turn]

    def render(self, save=False, additional=None):
        if additional is not None:
            for i, coord in enumerate(additional, 1):
                cir = plt.Circle((coord[0], coord[1]), 0.09, color='blue', alpha=(1./i))
                if coord[2] == 1:
                    cir.set_edgecolor("black")
                    cir.set_linewidth(2)
                else:
                    cir.set_edgecolor("white")
                    cir.set_linewidth(2)
                self._ax.add_artist(cir)

            if save:
                plt.title('Recommended {:d} SHOT'.format(self.current_turn))
                plt.savefig(
                    './img/' + str(self.current_turn - 1) + '_topk.png')

        plt.cla()

        plt.title(str(self.current_turn) + ' SHOT')

        a = [[self.state[i], self.state[i + 1]] for i in range(0, 32, 2)]
        my = np.asarray([x for x in a[self.order::2] if x[0] != 0 and x[1] != 0 ])
        op = np.asarray([x for x in a[1 - self.order::2] if x[0] != 0 and x[1] != 0 ])

        cir = plt.Circle((2.375, 4.88), 1.83, color='g', alpha=0.2)
        self._ax.add_artist(cir)

        cir = plt.Circle((2.375, 4.88), 0.5, color='g', alpha=0.2)
        self._ax.add_artist(cir)

        if my.size != 0:
            self._ax.scatter(my[:, 0], my[:, 1], color='r')
        if op.size != 0:
            self._ax.scatter(op[:, 0], op[:, 1], color='y')
        self._ax.axis('equal')



        self._ax.set_ylim(0, 11.28)
        self._ax.set_xlim(0, 4.75)


        if save:
            plt.savefig('./img/' + str(self.current_turn) + '_{:.3f}_{:.3f}_{:.0f}'.format(self.x,self.y,self.curl) + '.png')

        plt.pause(0.1)
        # plt.cla()

    @staticmethod
    def get_score(state, turn):

        score = 0

        t_coor = np.array([2.375, 4.88])
        coors = [np.array([state[i], state[i + 1]]) for i in range(0, 32, 2)]
        dists = [np.linalg.norm(t_coor - coor) for coor in coors]

        my = sorted(dists[turn::2])
        op = sorted(dists[1 - turn::2])

        if my[0] < op[0]:
            for my_dist in my:
                if my_dist < op[0] and my_dist < 1.97:
                    score += 1
                else:
                    break
        else:
            for op_dist in op:
                if op_dist < my[0] and op_dist < 1.97:
                    score -= 1
                else:
                    break

        return score
