        e = self.target - x 
        self.I = self.I + e * self.dt
        u = self.Kp * e + self.Ki * self.I + self.Kd * (e - self.e_prior)/self.dt
        self.e_prior = e 