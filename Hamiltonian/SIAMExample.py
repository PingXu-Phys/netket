from SIAM import SIAM


H, hi = SIAM(
    L=7,
    U=4.0,
    V=0.2,
    ti=1.0,
    pbc=False,
    n_fermions_per_spin=(4, 4),
)

print("H type =", type(H).__name__)
print("hi type =", type(hi).__name__)
print("hi size =", hi.size)
