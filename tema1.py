import numpy as np

def fitness(p):
    n=len(p)
    val=0
    for i in range(n-1):
        for j in range(i+1,n):
            if p[i]==j and p[j]==i:
                val+=1
    return val

def cerinta_a(dim,n):
    populatie=np.zeros([dim,n+1],dtype="int")
    for i in range(dim):
        populatie[i,:n]=np.random.permutation(n)
        populatie[i,n]=fitness(populatie[i,:n])
    return populatie

def inserare(p):
    n=len(p)
    i=np.random.randint(0,n-1)
    j=np.random.randint(i+1,n)
    r=p.copy()
    r[i+1]=p[j]
    if i+1<j:
        r[i+2:j+1]=p[i+1:j]
    return r

def cerinta_b(populatie,pm):
    dim=populatie.shape[0]
    populatie_m=populatie.copy()
    for i in range(dim):
        r=np.random.uniform(0,1)
        if r<=pm:
            print('Mutatie in ',populatie[i,:-1], ' calitatea ',populatie[i,-1])
            populatie_m[i,:-1]=inserare(populatie[i,:-1])
            populatie_m[i,-1]=fitness(populatie_m[i,:-1])
            print('Rezulta    ',populatie_m[i,:-1], ' calitatea ',populatie_m[i,-1])

    return populatie_m

if __name__=="__main__":
    n=20
    dim=10
    pm=0.2
    populatie=cerinta_a(dim,n)
    print(populatie)

    populatie_m=cerinta_b(populatie,pm)
