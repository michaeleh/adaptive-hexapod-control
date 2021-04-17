import numpy as np

if __name__ == '__main__':
    size = np.array([0.01, 0.25])
    step_size = 0.001
    target_height = .08
    pos = np.array([-1, 0, -0.15])
    name = 'box'
    material = 'wood'

    for i, step in enumerate(np.arange(step_size, target_height, step_size)):

        x_s = size[0]
        p = pos - (np.array([x_s, 0, 0]) * i)
        p = ' '.join(list(p.astype(str)))

        s = list(size) + [step]
        s = ' '.join(np.array(s).astype(str))

        if i == len(np.arange(step_size, target_height, step_size)) - 1:
            size[0] *= 50
            s = list(size) + [step]
            s = ' '.join(np.array(s).astype(str))
            p = pos - (np.array([x_s, 0, 0]) * i) - np.array([size[0], 0, 0])
            p = ' '.join(list(p.astype(str)))
            print(
                f'<geom name = "{name}{i}" pos = "{p}" type = "box" size = "{s}" material = "{material}" />')
        else:
            s = list(size) + [step]
            s = ' '.join(np.array(s).astype(str))
            print(
                f'<geom name = "{name}{i}" pos = "{p}" type = "box" size = "{s}" material = "{material}" />')
