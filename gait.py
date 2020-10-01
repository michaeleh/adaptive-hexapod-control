import matplotlib.pyplot as plt
import numpy as np


class Gait:
    def __init__(self):
        self.angle = np.deg2rad(20)
        self.finished = False
        self.legup = True
        self.coxa_angle = 0
        self.femur_angle = 0
        self.tibia_angle = 0

    def generate_action(self):
        dt = 0.005
        if self.legup:
            self.finished = False
            if self.tibia_angle <= self.angle:
                self.tibia_angle += dt
            elif self.femur_angle <= self.angle:
                self.femur_angle += dt
            elif self.coxa_angle <= self.angle:
                self.coxa_angle += dt
            else:
                self.legup = False
        else:
            if self.tibia_angle >= 0:
                self.tibia_angle -= dt
            elif self.femur_angle >= 0:
                self.femur_angle -= dt
            elif self.coxa_angle >= 0:
                self.coxa_angle -= dt
            else:
                self.legup = True
                self.finished = True

        return dict(
            coxa=self.coxa_angle,
            femur=self.femur_angle,
            tibia=self.tibia_angle,
            finished=self.finished
        )


if __name__ == "__main__":
    gait = Gait()
    motions = []
    for i in range(1000):
        motions.append(gait.generate_action())
    coxa = [d['coxa'] for d in motions]
    femur = [d['femur'] for d in motions]
    tibia = [d['tibia'] for d in motions]
    plt.plot(coxa, label='coxa')
    plt.plot(femur, label='femur')
    plt.plot(tibia, label='tibia')
    plt.legend()
    plt.show()
