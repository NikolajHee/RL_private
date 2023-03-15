        
        dF1 = linalg.block_diag(*([Fx] * N) )
        dF1 = np.hstack( [dF1, np.zeros( ( dF1.shape[0], n) ) ] )
        dF2 = np.hstack( [ -np.eye( nL ), np.zeros((nL,n) ) ] )
        G = linalg.block_diag(dF1, linalg.block_diag(*([Fu] * N)), dF2  )
        h = np.hstack( [hx]*N + [hu]*N + [np.zeros( (nL ,) )] )
        