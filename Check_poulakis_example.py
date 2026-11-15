import random
import math
import time
import numpy as np
from math import isqrt
from math import sqrt

from fpylll import IntegerMatrix, LLL, GSO

import gmpy2
from gmpy2 import mpq

q = 9001


def lattice_matrix(A, q):
    """
    Construct an  (n+1) x (n+1)  lattice matrix.

    Parameters
    ----------
    A : list
        List of n integers.
    q : int
        Modulus.

    Returns
    -------
    
        The matrix of  lattice.
    """
    n = len(A)
    M = []

    for i in range(n):
        row = [0] * (n + 1)
        row[i] = 1
        row[-1] = A[i]
        M.append(row)

    # Last line of q
    last_row = [0] * n + [q]
    M.append(last_row)

    return np.array(M, dtype=int)




def target_vector(n, B):
    """
    Returns a NumPy vector of length (n+1), with all coordinates equal to 0 except for the last coordinate, 
    which is (-B).

    n: integer
    B: integer 
    """
    t = np.zeros(n + 1, dtype=int)    
    t[-1] = -B                     
    return t


def My_Gram_Schmidt_with_norms(basis):
    from fpylll import IntegerMatrix, GSO

    nrows = len(basis)
    ncols = len(basis[0])

    B = IntegerMatrix(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            B[i, j] = int(basis[i][j])

    M = GSO.Mat(B)
    M.update_gso()

    # Get mu and norms from fpylll
    mu_matrix = [[0]*nrows for _ in range(nrows)]
    gso_norms = [0]*nrows

    for i in range(nrows):
        gso_norms[i] = M.get_r(i, i)  #  ||b_i*||^2
        for j in range(i):
            mu_matrix[i][j] = M.get_mu(i, j)

    # construct b_i*
    basis_star = []

    for i in range(nrows):
        # first vector
        b_star = [B[i, k] for k in range(ncols)]

        # substract the projections
        for j in range(i):
            mu = mu_matrix[i][j]
            for k in range(ncols):
                b_star[k] -= mu * basis_star[j][k]

        basis_star.append(b_star)

    return basis_star, gso_norms, mu_matrix


def do_LLL(basis, delta=0.75, eta=0.501):
    """
    Performs LLL reduction on a NumPy basis matrix (of integers)
    and returns the LLL-reduced basis as a 2D NumPy array.

    delta, eta: LLL parameters (recommended default values).
    """
    #  NumPy array -> IntegerMatrix
    mat = IntegerMatrix.from_matrix(basis.tolist())
    
    # LLL reduction inplace
    LLL.reduction(mat, delta=delta, eta=eta)
    
    # Return as NumPy array
    reduced_basis = np.array([[mat[i,j] for j in range(mat.ncols)] for i in range(mat.nrows)], dtype=int)
    
    return reduced_basis


def express_in_gso_basis(vector, basis_star, gso_norms):

    """
    Expresses a vector as a linear combination of the orthogonal
    Gram-Schmidt vectors and returns the corresponding coefficients.

    The coefficients are computed using orthogonal projections:
        t[i] = <vector, basis_star[i]> / ||basis_star[i]||^2

    Input:
        vector: The input vector.
        basis_star: The orthogonal Gram-Schmidt vectors.
        gso_norms: The squared norms of the Gram-Schmidt vectors.

    Returns:
        A list of exact rational coefficients.
    """
    n = len(basis_star)
    t = [mpq(0) for _ in range(n)]

    for i in range(n):
        dot = mpq(0)

        for j in range(len(vector)):
            dot += mpq(int(vector[j])) * basis_star[i][j]

        t[i] = dot / gso_norms[i]

    return t



#New Kannan
def Kannan_algorithm(n, R, basis,  vector_t, looking_for_v = None):

    
    """
    Enumerates the lattice vectors lying inside the ball centered at
    vector_t with radius R, using a Gram-Schmidt-based enumeration
    algorithm.

    The algorithm first applies LLL reduction to the input basis and
    then uses the Gram-Schmidt orthogonalization to systematically
    enumerate the integer coefficients of lattice vectors satisfying

        ||v - vector_t|| <= R.

    If looking_for_v is provided, the enumeration terminates as soon
    as the specified lattice vector is found.

    Parameters
    ----------
    n : int
        Dimension of the lattice.
    R : float
        Radius of the ball.
    basis : list of numpy arrays
        Basis vectors generating the lattice.
    vector_t : numpy array
        Center of the ball.
    looking_for_v : numpy array or None, optional
        Specific lattice vector to search for. If None, the algorithm
        enumerates all lattice vectors within the ball.

    Returns
    -------
    lattice_points_in_Ball : list
        List of lattice vectors found inside the ball.
    """

    start = time.time()


    lattice_points_in_Ball =[]

    l = np.zeros(n)
    x = np.zeros(n, dtype=int)
    
    vector_x = np.zeros_like(basis[0], dtype=int)
    plithos = 0

    # --- LLL Reduction ---
    lll_basis =do_LLL(basis, delta=0.99, eta=0.501)

    

    basis_star, gso_norms, mu_matrix = My_Gram_Schmidt_with_norms(lll_basis)

    t = express_in_gso_basis(vector_t, basis_star, gso_norms)
    
    
    # Minimum and maximum value of x[n-1] (explicit float)

    x[n-1] = int(math.ceil(t[n-1] -R / math.sqrt(gso_norms[n-1])  ))
    print("The minimum value of x[n-1]", x[n-1])
    
    #print("The maximum value of x[n-1]", int(math.ceil(t[n-1] +R / math.sqrt(gso_norms[n-1])  )))
    
    
    
    


    i = n-1
    
    while i<=n-1:

        Sum =0
        for j in range(i+1, n):
                Sum += x[j] * mu_matrix[j][i] 
        
        l[i] = ((x[i] - t[i] + Sum )**2) * gso_norms[i]

        sum_elements_of_l2 = np.sum(l[i+1:])
        sum_elements_of_l1 = np.sum(l[i:])

        if  sum_elements_of_l1 <= R**2:


          if i ==0:
  
            vector_x = np.zeros_like(basis[0], dtype=int)

            for j in range(n):
                vector_x += x[j] * lll_basis[j]

            lattice_points_in_Ball.append(vector_x) 

            if looking_for_v is not None and np.all(vector_x == looking_for_v):
                print("We found the  lattice vector!")
                print("--- minutes ---", (time.time() - start)/60)
                print("x =", x)
                print("The maximum value of x[n-1]", int(math.ceil(t[n-1] +R / math.sqrt(gso_norms[n-1])  )))
                print("vector_x",vector_x)
                
                
                
                break

            x[i] += 1
            

            
            plithos += 1
            if np.linalg.norm(vector_x - vector_t) <= R:
                   print("Within radius R")
            print("Vectors found:", plithos)

            
            print("---------")
        

        

          else:
            i -= 1
            Sum =0
            for j in range(i+1, n):
              Sum += x[j] * mu_matrix[j][i]


            sum_elements_of_l2 = np.sum(l[i+1:])

            x[i] = int(math.ceil(t[i] - Sum - math.sqrt((R**2 - sum_elements_of_l2) / gso_norms[i])  )) 
          

        else:
           i += 1
           if i==n:
               print("The maximum value of x[n-1]", int(math.ceil(t[n-1] +R / math.sqrt(gso_norms[n-1])  )))
               print("x =", x)
               break
           
           x[i] += 1

    return lattice_points_in_Ball       



def Solutions_of_Congruence(m, vectors, B, q):

    """
    Transforms a list of integer vectors by replacing the last
    coordinate of each vector with (-z_(n+1) - B) mod q,
    using the symmetric system of residues.

    Each input vector must have length m. The remaining
    coordinates are left unchanged, and the transformed
    vectors are returned as a new list.

    Parameters:
        m       : Expected length of each vector.
        vectors : List of integer vectors.
        B       : Integer constant.
        q       : Positive modulus.

    Returns:
        A list of transformed vectors.
    """

    result = []

    for v in vectors:
        if v.shape[0] != m:
            raise ValueError(f"The vector {v} does not have length {m}.")

        new_v = v.copy()

        last = (-new_v[-1] - B) % q

        # Conversion to the symmetric system of residues
        if last >= q // 2 + (q % 2):
            last -= q

        new_v[-1] = last
        result.append(new_v)

    return result


def Check_Solutions_of_Congruence(A, B, q, Solutions):
    """
      Checks whether each vector z = (z1, ..., zn, z_(n+1))
      satisfies the congruence

      z_(n+1) + A[0] z1 + ... + A[n-1] zn + B ≡ 0 (mod q).

    Parameters
     ----------
    A : list or numpy.ndarray
    A list of n integers.

    B : int
    An integer.

    q : int
    The modulus.


    Solutions : list of numpy.ndarray
    A list of vectors of length n+1.

   Returns
   -------
    True if all vectors satisfy the congruence,
    False otherwise.
    """

    n = len(A)

    for z in Solutions:

        if len(z) != n + 1:
            raise ValueError("Incorrect vector dimension.")

        value = np.dot(A, z[:n]) + z[n] + B

        if value % q != 0:
            return False

    return True




# Example 1



q=9001
n=1
A=[4513]
B=1155     
R=200



M = lattice_matrix(A,q)
print(M)

vector_t = target_vector(n, B)
vector_t = np.array(vector_t, dtype=int)
print("vector_t =",vector_t)


lattice_points_in_Ball = Kannan_algorithm(n+1, R, M,  vector_t, looking_for_v = None)
print("lattice_points_in_Ball",lattice_points_in_Ball)
print("---------")
print("---------")


Solutions = Solutions_of_Congruence(n+1, lattice_points_in_Ball, B, q)
print("Solutions",Solutions)






check = Check_Solutions_of_Congruence(A, B, q, Solutions)

if check == True:
    print("All points are solutions")
else:
    print("There are points that are not solutions")


