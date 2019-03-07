import numpy as np 


def compute_lambda(J, S):
    if len(J) == len(S):
        lambda_m = np.array([J[i]/S[j] for i in range(len(J)) for j in range(len(S))])
        return (1./len(S))*lambda_m.reshape((len(J),len(S)))
    else:
        raise ValueError('J and S arrays must have the same length.')


def split_lambda(L, mtype = 'diagonal'):

    L_star = np.zeros(L.shape)
    diag = np.diag_indices_from(L)
    L_star[diag] = L[diag].copy()

    if mtype == 'diagonal':
        return L_star

    elif mtype == 'tridiagonal':
        lower_diag = (diag[0][:-1],diag[1][1:])
        upper_diag = (diag[0][1:],diag[1][:-1])
        L_star[lower_diag] = L[lower_diag].copy()
        L_star[upper_diag] = L[upper_diag].copy()
        return L_star 
    
    else:
        raise NotImplementedError('%s' % mtype)


def compute_S_step(J, S, mtype='diagonal'):

    '''
    Computes the next iteration source-function using Accelerated Lambda Iteration.

    The gray version of the equation (RT: S=J) is:
    $S^{n+1} = (1-\Lambda^*)^{-1} (\Lambda - \Lambda^*) S^{n}
    '''
    
    L = compute_lambda(J, S)
    L_star = split_lambda(L, mtype=mtype)
    L_star_I_inv = np.linalg.inv(np.identity(len(L_star)) - L_star)
    
    return np.matmul(np.matmul(L_star_I_inv, L-L_star), S)


