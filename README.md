# PARCO_D2
This is a brief overview about how to execute the code.

# Parallel Matrix Transposition and Symmetry Check

This program demonstrates parallel matrix transposition and symmetry checking using MPI. The code was tested on a cluster via interactive sessions, utilizing the `gcc91` and `mpich-3.2.1--gcc-9.1.0` compilers.

## Compilation Instructions

To compile the program, you will need to load the necessary modules and use the following command:

1. After loading the files in this repository on your personal account on the cluster log into it and start an interactive session using 
```bash
qsub -I -q short_cpuQ -l select=2:ncpus=32:mpiprocs=32:mem=1gb
```
2.  Load the required modules:
   ```bash
module load gcc91
module load mpich-3.2.1--gcc-9.1.0
```
3. Compile the program using the following command:

```bash
mpicxx -std=c++11  Parallel_computing_D2.c++ -o parco_h2
```
4. Run the program either using
 ```bash
 mpirun -np (number of processes) ./parco_h2
```
or
 ```bash
 mpirun -np (number of processes) ./parco_h2 Scalabity_Test_Transpose.txt Scalabity_Test_Symmetry.txt
```
Without additional files:
The program will call both parallel and sequential functions while recording the wall clock time using MPI_Wtime() inside the functions. It will execute the matrix transposition and symmetry check on matrices of various sizes. After showing in the terminal the results of the computation for both The sequential and Parallel execution of each function, the parallel results run with the previously specified number of processor will be updated inside the two files  Scalabity_Test_Transpose.txt Scalabity_Test_Symmetry.txt

With additional files ("Scalability_Test_Transpose.txt" and "Scalability_Test_Sym.txt"):
If you include these files in the toolchain, the program will use the data inside them to compute strong/weak scaling and efficiency and store them in the Results.txt file. Unfortunately, the program does not update the text files when executed on the cluster but it updates the files only when executed on a local host.The results are still reproducible executing the function but they are not immediately updated inside the two files. To work around this limitation and still use the results fo the cluster, the program was run multiple times in an interactive session, varying the number of processes and manually updating the files. Afterward, the program was executed on a local machine with the updated files to generate the results.

The instructions to run the program on a local pc are pretty much the same. All it is required is to download and configure the mpi microsoft library and compile the program with gcc specifying the patch to library

 ```bash
 g++ Parallel_computing_D2.c++ -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi -o send_rec
```

this is a typical output example:


Running test on matrix: 4096 × 4096
Sequential Execution:
• Transposition: 0.539472
• Symmetry check: 0.106679
Transposition MPI:
• --4
• 16 × 16: 0.000014
• 32 × 32: 0.000033
• 64 × 64: 0.000116
• 128 × 128: 0.000368
• 256 × 256: 0.001043
• 512 × 512: 0.001952
• 1024 × 1024: 0.012243
• 2048 × 2048: 0.044632
• 4096 × 4096: 0.218019
Check Sym MPI:
• --4
• 16 × 16: 0.000006
• 32 × 32: 0.000009
• 64 × 64: 0.000049
• 128 × 128: 0.000189
• 256 × 256: 0.000495
• 512 × 512: 0.001183
• 1024 × 1024: 0.008276
• 2048 × 2048: 0.030743
• 4096 × 4096: 0.105162Running test on matrix: 4096 × 4096
Sequential Execution:
• Transposition: 0.539472
• Symmetry check: 0.106679
Transposition MPI:
• --4
• 16 × 16: 0.000014
• 32 × 32: 0.000033
• 64 × 64: 0.000116
• 128 × 128: 0.000368
• 256 × 256: 0.001043
• 512 × 512: 0.001952
• 1024 × 1024: 0.012243
• 2048 × 2048: 0.044632
• 4096 × 4096: 0.218019
Check Sym MPI:
• --4
• 16 × 16: 0.000006
• 32 × 32: 0.000009
• 64 × 64: 0.000049
• 128 × 128: 0.000189
• 256 × 256: 0.000495
• 512 × 512: 0.001183
• 1024 × 1024: 0.008276
• 2048 × 2048: 0.030743
• 4096 × 4096: 0.105162

