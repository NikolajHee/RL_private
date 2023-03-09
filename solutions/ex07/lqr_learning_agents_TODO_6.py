        xl = x0 
        for k in range(self.NH):
            x_bar.append(xl)
            u_bar.append(L[k] @ xl + l[k])
            xl = A[k] @ x_bar[k] + B[k] @ u_bar[k] + d[k] 