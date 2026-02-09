# PARCO_D2

## Instructions for reproducibility

In this folder there are available several .pbs files, each corresponding to a particular test. For each test that you need to execute see the specification below:

# 1) STRONG SCALING AND WEAK SCALING TESTS

# test SPMV4 Real Matrices

- configuration 4 nodes thread on
  ```bash
  dos2unix run_spmv_base4_ton.pbs
  qsub run_spmv_base4_ton.pbs
 
- configuration 4 nodes thread off
  ```bash
  dos2unix run_spmv_base4_toff.pbs
  qsub run_spmv_base4_toff.pbs

- configuration 8 nodes thread on
  ```bash
  dos2unix run_spmv_base8_ton.pbs
  qsub run_spmv_base8_ton.pbs

- configuratuin 8 nodes threading off
  ```bash
  dos2unix run_spmv_base8_toff.pbs
  qsub run_spmv_base8_toff.pbs

# test SPMV4 synthetic Matrices 
  ```bash
  dos2unix weak_spmv.pbs
  qsub weak_spmv.pbs 
   ```
# test 1D-overlapp Real Matrices

- configuration 4 nodes thread on
  ```bash
  dos2unix overlapped_4_ton.pbs
  qsub overlapped_4_ton.pbs
 
- configuration 4 nodes thread off
  ```bash
  dos2unix overlapped_4_toff.pbs
  qsub overlapped_4_toff.pbs 

- configuration 8 nodes thread on
  ```bash
  dos2unix overlapped_8_ton.pbs
  qsub overlapped_8_ton.pbs 

- configuratuin 8 nodes threading off
  ```bash
  dos2unix overlapped_8_toff.pbs
  qsub overlapped_8_toff.pbs

# test 1D-overlapp synthetic Matrices 
  ```bash
  dos2unix weak_overlapped.pbs
  qsub weak_overlapped.pbs 
   ```

# test 2D-overlapp Real Matrices

- configuration 4 nodes thread on

   ```bash
  dos2unix runTest2d_4_ton.pbs
  qsub runTest2d_4_ton.pbs 

- configuration 4 nodes thread off
   ```bash
  dos2unix runTest2d_4_toff.pbs
  qsub runTest2d_4_toff.pbs 

- configuration 8 nodes thread on
   ```bash
  dos2unix runTest2d_8_ton.pbs
  qsub runTest2d_8_ton.pbs 

- configuration 8 nodes threading off
   ```bash
  dos2unix  runTest2d_8_toff.pbs
  qsub runTest2d_8_toff.pbs 

# test 2D-overlapp synthetic Matrices
```bash
  dos2unix weak_2d_overlapped.pbs
  qsub weak_2d_overlapped.pbs 
   ```


# 2) PLOTTING THE GRAPHS
This script has to be executed in the folder with all of the csv results  
```bash
  python plot_scaling.py
```
the output is a folder called plot with all the png files of the strong/weak scaling and the efficiency
