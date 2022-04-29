import sys


class ValueWatcher:
    def __init__(self, mode='maximize', threshold=5):
        self.current = None
        self.count = 0
        self.max_score = sys.float_info.min
        self.min_score = sys.float_info.max
        self.is_max = False
        self.is_min = False
        self.is_new = False
        self.mode = mode
        self.threshold = threshold
        self.epoch = 0

    def update(self, val):
        if self.current is None:
            self.current = val
            self.max_score = val
            self.min_score = val
            self.is_new = True
        else:
            self.current = val
            self.is_new = False
            if self.mode == 'maximize':
                if val > self.max_score:
                    self.count = 0
                    self.max_score = val
                    self.is_new = True
                else:
                    self.count += 1
            elif self.mode == 'minimize':
                if val < self.min_score:
                    self.count = 0
                    self.min_score = val
                    self.is_new = True
                else:
                    self.count += 1

        if self.count >= self.threshold:
            if self.mode == 'maximize':
                self.is_max = True
            elif self.mode == 'minimize':
                self.is_min = True
        else:
            self.is_max = False
            self.is_min = False

        self.epoch += 1

    def is_over(self):
        if self.threshold == -1:
            return True

        if self.epoch >= 20:
            return True

        if self.mode == 'maximize':
            return self.is_max
        elif self.mode == 'minimize':
            return self.is_min
        else:
            return None

    def is_updated(self):
        return self.is_new


if __name__ == '__main__':
    vw = ValueWatcher()
    for i in [0., 1.1, 2.1, 3.1, 2.5, 1.9, 3.3, 3.3 ,0.1 ,1.0, 2.9, 2.9, 2.9, 1.9]:
        vw.update(i)
        if vw.is_updated():
            print(f'updated: {vw.max_score}, {vw.count}')
        else:
            print(f'not-updated: {i}, {vw.count}')

        if vw.is_over():
            print('break')
            break

