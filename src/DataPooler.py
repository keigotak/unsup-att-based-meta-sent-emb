
class DataPooler:
    def __init__(self):
        self.pools = {}

    def set(self, key, val):
        if type(val) != list:
            val = [val]

        if key not in self.pools.keys():
            self.pools[key] = val
        else:
            self.pools[key].extend(val)

    def get(self, key=None):
        if key is None:
            return self.pools
        else:
            return self.pools[key]

    def get_best(self, key=None):
        return self.pools[key][-1]