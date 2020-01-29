def use_coupons(A):
    for i in range(len(A) - 1):
        if A[i] % 2 == 1:
            if A[i+1] == 0:
                return False
            else:
                A[i+1] -= 1
    if A[len(A)-1] % 2 == 1:
        return False
    return True
        
        

if __name__ == "__main__":
    n = input()
    n = int(n)
    A = input()
    A = A.split()
    for i in range(n):
        A[i] = int(A[i])

    if use_coupons(A):
        print("YES")
    else:
        print("NO")
    
