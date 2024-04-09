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


////////////


def dec_to_bin(n,m):
    repr = bin(n)[2:]
    repr_f = repr.zfill(m)
    x=[int(repr_f[i]) for i in range(m)]
    return x

#transfer invers
def bin_to_dec(x,m):
    y=''
    for i in range(m):
        y+=str(x[i])
    n=int(y,2)
    return n

#functia fitness

def fitness(sir):
    x=bin_to_dec(sir[0:11],11)
    y=bin_to_dec(sir[11:23],12)
    return (y-1)*(np.sin(x-2)**2)

def cerinta_a(dim):
    populatie=[]
    print("POPULATIA INITIALA")
    for i in range(dim):
        x=np.random.randint(0,1501)
        y=np.random.randint(0,2502)
        print("Componentele in baza 10 (fenotipul):",x,y)
        individ=dec_to_bin(x,11)+dec_to_bin(y+1,12)
        print("Reprezentarea genotipului",individ)
        calitate=fitness(individ)
        print("Fitness:",calitate)
        individ=individ+[calitate]
        populatie=populatie+[individ]
    return populatie

///////

def fitness(x):
    return 1+2*np.sin(x[0]-x[2])+np.cos(x[1])

def cerinta_a(dim):
    populatie=np.zeros([dim,4])
    for i in range(dim):
        populatie[i, 0]=np.random.uniform(-1,1)
        populatie[i, 1] = np.random.uniform(0, 1)
        populatie[i, 2] = np.random.uniform(-2, 1)
        populatie[i,3]=fitness(populatie[i,:3])
    return populatie


////

def elitism(pop_curenta, pop_mutanta, dim, n):
    pop_curenta = numpy.asarray(pop_curenta)
    pop_mutanta = numpy.asarray(pop_mutanta)
    pop_urmatoare = numpy.copy(pop_mutanta)

    max_curent = numpy.max(pop_curenta[:, -1])
    max_mutant = numpy.max(pop_mutanta[:, -1])

    if max_curent > max_mutant:
        poz = numpy.where(pop_curenta[:, -1] == max_curent)
        imax = poz[0][0]
        ir = numpy.random.randint(dim)
        pop_urmatoare[ir, 0:n] = pop_curenta[imax, 0:n].copy()
        pop_urmatoare[ir, n] = max_curent
    return pop_urmatoare

/////

def ruleta(pop_initiala, dim, n):
    pop_initiala = numpy.asarray(pop_initiala)
    parinti = pop_initiala.copy()
    fitnessuri = numpy.zeros(dim, dtype="float")
    for i in range(dim):
        fitnessuri[i] = pop_initiala[i][n]
    qfps = fps(fitnessuri, dim)
    for i in range(dim):
        r = numpy.random.uniform(0, 1)
        pozitie = numpy.where(qfps >= r)
        index_buzunar_ruleta = pozitie[0][0]
        parinti[i][0:n] = pop_initiala[index_buzunar_ruleta][0:n]
        parinti[i][n] = fitnessuri[index_buzunar_ruleta]
    return parinti

////

def SUS(pop_initiala, dim, n):
    pop_initiala = numpy.asarray(pop_initiala)
    parinti = pop_initiala.copy()  # gene si fitness-uri
    fitnessuri = numpy.zeros(dim, dtype="float")
    for i in range(dim):
        fitnessuri[i] = pop_initiala[i][n]
    qfps = fps(fitnessuri, dim)
    r = numpy.random.uniform(0, 1 / dim)
    k, i = 0, 0
    while k < dim:
        while r <= qfps[i]:
            parinti[k][0:n] = pop_initiala[i][0:n]
            parinti[k][n] = fitnessuri[i]
            r = r + 1 / dim
            k = k + 1
        i = i + 1
    return parinti