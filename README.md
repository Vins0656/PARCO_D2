# PARCO_D2

## Instructions

In this folder there are available several .pbs files, each corresponding to a particular test. For each test that you need to execute see the specification below:

# test SPMV4

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

# test 1D-overlapp

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
  dos2unix overlapped_8_toff.pbs
  qsub overlapped_8_toff.pbs 

- configuratuin 8 nodes threading off
  ```bash
  dos2unix  .pbs
  qsub   .pbs 

# test 2D-overlapp

- configuration 4 nodes thread on

   ```bash
  dos2unix  runTest2d_4_ton.pbs
  qsub   runTest2d_4_ton.pbs 

- configuration 4 nodes thread off
   ```bash
  dos2unix  runTest2d_4_toff.pbs
  qsub   runTest2d_4_toff.pbs 

- configuration 8 nodes thread on
   ```bash
  dos2unix  runTest2d_8_ton.pbs
  qsub   runTest2d_8_ton.pbs 

- configuratuin 8 nodes threading off
   ```bash
  dos2unix  runTest2d_8_toff.pbs
  qsub   runTest2d_8_toff.pbs 
