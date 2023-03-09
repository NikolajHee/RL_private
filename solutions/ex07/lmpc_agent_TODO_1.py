        
        dG1 = np.hstack([A11, A12, np.zeros( (n*(N+1), nL + n - 2) ) ])
        dG2 = np.hstack( [np.zeros( (n, n * N ) ), np.eye(n), np.zeros( (n, N*2) ), - SS_v, np.eye(n)]  )
        dG3 = np.hstack( [np.zeros( (1, n*(N+1)+d*N)), np.ones( (1,nL)), np.zeros( (1,n)) ])    
        A = np.vstack( [dG1, dG2, dG3])
        b = np.hstack( [x0.T, np.concatenate( Ctv)[:,0], np.zeros( (n,) ), 1 ])
        