        path_ok = x[0] == x[-1] and len(set(x)) == len(self.cities) and all([(x[i],x[i+1]) in self.G for i in range(len(x)-1)]) 