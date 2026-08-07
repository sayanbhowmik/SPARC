# Si2_nmag (spin-unpolarized)

Diamond Si2 primitive cell with `SPIN_TYP: 0`.

Paired with `../Si2_nmag_spin` (collinear spin, zero initial magnetization).
Both should converge to a non-magnetic solution with matching total DOS / PDOS.

## Run SPARC

```bash
# from this directory; PRINT_ORBITAL requires single MPI rank
export LD_LIBRARY_PATH="${HOME}/opt/plumed/lib:${LD_LIBRARY_PATH:-}"
mpirun -np 1 /path/to/sparc -name Si2_nmag
```

## Run PDOS

```bash
cd ../../
python calculate_pdos.py --config=config_Si2_nmag.yaml
```
