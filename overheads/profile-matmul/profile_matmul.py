import cupynumeric as np
if __name__ == "__main__":

    N = 10_000

    A = np.random.uniform(size = (N,N), dtype = np.float32)
    B = np.random.uniform(size = (N,N), dtype = np.float32)
    C = np.zeros((N, N), dtype=np.float32)
   
    np.dot(A, B, out=C)

    print(C[0,0])
