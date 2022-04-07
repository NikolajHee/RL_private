                    self.returns_sum[s] += G
                    self.returns_count[s] += 1.0
                    self.v[s] = self.returns_sum[s] / self.returns_count[s] 