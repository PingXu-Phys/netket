from Hubbard import Hubbard


H, hi = Hubbard(
    Lx=4,
    Ly=4,
    t1=1.0,
    t2=0.2,
    U=8.0,
    pbc_x=True,
    pbc_y=False,
    n_fermions_per_spin=(8, 8),
)

print("H type =", type(H).__name__)
print("hi type =", type(hi).__name__)
print("hi size =", hi.size)
