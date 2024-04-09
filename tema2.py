import numpy as np

#reprezentare n pe m biti prin sir binar memorat ca vector de 0-1
def dec_to_bin(n,m):
    #reprezentare standard, dar prin sir de caractere
    repr = bin(n)[2:]
    #transformare in string de m caractere
    repr_f = repr.zfill(m)
    # transformare in sir binar
    x=[int(repr_f[i]) for i in range(m)]
    return x

#transfer invers
def bin_to_dec(x,m):
    #transfer din lista de int in lista de caractere
    y=''
    for i in range(m):
        y+=str(x[i])
    #reprezentarea in baza 10
    n=int(y,2)
    return n

#functia fitness

def fitness(sir):
    x=bin_to_dec(sir[0:11],11)
    y=bin_to_dec(sir[11:23],12)
    return (y-1)*(np.sin(x-2)**2)

#un individ are 23 biti - reprezentarea binara a numerelor din {1,...,1500} concatenata cu
# reprezentarea binara a numerelor din {0,...,2501}
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


def recombinare_3puncte(sir1,sir2):
    n=23
    i=np.random.randint(0,n-2)
    j=np.random.randint(i+1,n-1)
    k=np.random.randint(j+1,n)
    #print(i,j,k)
    copil1=sir1.copy()
    copil2=sir2.copy()
    copil1[j:k+1]=sir2[j:k+1]
    copil2[j:k+1]=sir1[j:k+1]
    return copil1, copil2


def cerinta_b(populatie,pc):
    print("\n\nPOPULATIA DE COPII")
    dim=len(populatie)
    copii=populatie.copy()
    for i in range(0,dim-1,2):
        r=np.random.uniform(0,1)
        if r<=pc:
            #selectarea indivizilor, fara calitatile lor
            p1=populatie[i][:23].copy()
            p2=populatie[i+1][:23].copy()
            c1,c2=recombinare_3puncte(p1,p2)
            copii[i][:23]=c1
            copii[i][23]=fitness(c1)
            copii[i+1][:23] = c2
            copii[i+1][23] = fitness(c2)
        print("Individ+calitate",copii[i])
        print("Individ+calitate", copii[i+1])
    return copii

if __name__=="__main__":
    p=cerinta_a(10)
    c=cerinta_b(p,0.8)


