from KondoHeisenbergChain import KondoHeisenbergChain, joint_sector_reference_state, joint_total_two_sz_of_states


H, hi = KondoHeisenbergChain(
    Lx=8,
    t=1.0,
    J_K=1.0,
    J1=1.0,
    J2=0.3,
    pbc=False,
    n_fermions=8,
)

reference = joint_sector_reference_state(n_sites=8, n_fermions=8, two_sz=0)
sector_value = int(joint_total_two_sz_of_states(reference[None, :], 8)[0])

print("H type =", type(H).__name__)
print("hi type =", type(hi).__name__)
print("hi size =", hi.size)
print("reference 2Sz_total =", sector_value)
