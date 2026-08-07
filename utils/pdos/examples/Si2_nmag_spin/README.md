# Si2_nmag_spin (collinear, non-magnetic)

Same Si2 cell as `../Si2_nmag`, with `SPIN_TYP: 1` and `SPIN: 0.0` on both atoms.
Closed-shell semiconductor -> magnetization stays ~0 and DOS should match the unpolarized case.

## Run SPARC

```bash
export LD_LIBRARY_PATH="${HOME}/opt/plumed/lib:${LD_LIBRARY_PATH:-}"
mpirun -np 1 /path/to/sparc -name Si2_nmag_spin
```

## Run PDOS

```bash
cd ../../
python calculate_pdos.py --config=config_Si2_nmag_spin.yaml
```
