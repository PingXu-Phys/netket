from KondoHeisenberg import KondoHeisenberg, joint_sector_reference_state, joint_total_two_sz_of_states


H, hi = KondoHeisenberg(
    Lx=4,
    n_legs=2,
    t1=1.0,
    t2=0.2,
    t3=0.0,
    J_K=1.0,
    J1=1.0,
    J2=0.3,
    J3=0.0,
    pbc_x=True,
    pbc_y=False,
    n_fermions=8,
)

reference = joint_sector_reference_state(n_sites=8, n_fermions=8, two_sz=0)
sector_value = int(joint_total_two_sz_of_states(reference[None, :], 8)[0])

print("H type =", type(H).__name__)
print("hi type =", type(hi).__name__)
print("hi size =", hi.size)
print("reference 2Sz_total =", sector_value)
