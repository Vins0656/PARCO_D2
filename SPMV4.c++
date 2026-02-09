/*
 * SPMV4 HYBRID - Logica IO Originale preservata + OpenMP + Metriche
 */

#include <mpi.h>
#include <omp.h> // Aggiunto OpenMP
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

using namespace std;

int main(int argc, char* argv[]) {
    int provided;
    // Inizializzazione ibrida (necessaria per OpenMP)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) cerr << "Usage: " << argv[0] << " <matrix_file.mtx>" << endl;
        MPI_Finalize();
        return 1;
    }

    // Variabili per il timing
    double t_io = 0, t_setup = 0, t_comm = 0, t_calc = 0;
    long long global_nnz_cnt = 0; // Per GFLOPS

    // ==================================================================================
    // FASE 1: LETTURA E DISTRIBUZIONE (Logica originale SPMV4.c++)
    // ==================================================================================
    double t_start_io = MPI_Wtime();

    int M, N, nz;
    vector<double> all_val;
    vector<int> all_col;
    vector<int> all_row_ptr;
    
    // Strutture per Scatterv
    vector<int> send_counts(size);
    vector<int> displs(size);
    vector<int> rows_per_proc_vec(size);

    if (rank == 0) {
        ifstream file(argv[1]);
        while (file.peek() == '%') file.ignore(2048, '\n');
        file >> M >> N >> nz;
        global_nnz_cnt = nz;

        vector<vector<pair<int, double>>> rows(M);
        for (int i = 0; i < nz; i++) {
            int r, c;
            double v;
            file >> r >> c >> v;
            rows[r - 1].push_back({c - 1, v});
        }

        all_row_ptr.push_back(0);
        for (int i = 0; i < M; i++) {
            for (auto& p : rows[i]) {
                all_col.push_back(p.first);
                all_val.push_back(p.second);
            }
            all_row_ptr.push_back(all_col.size());
        }

        int rows_per_proc = M / size;
        int remainder = M % size;
        int current_disp = 0;
        int current_row_idx = 0;

        for (int i = 0; i < size; i++) {
            int r = rows_per_proc + (i < remainder ? 1 : 0);
            rows_per_proc_vec[i] = r;
            
            // Calcolo NNZ per questo rank
            int start_idx = all_row_ptr[current_row_idx];
            int end_idx = all_row_ptr[current_row_idx + r];
            
            send_counts[i] = end_idx - start_idx;
            displs[i] = current_disp;
            
            current_disp += send_counts[i];
            current_row_idx += r;
        }
    }

    // Broadcast dimensioni globali
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_nnz_cnt, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Distribuzione numero righe
    int rows_assigned;
    MPI_Scatter(rows_per_proc_vec.data(), 1, MPI_INT, &rows_assigned, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribuzione Valori e Colonne
    int my_nnz;
    MPI_Scatter(send_counts.data(), 1, MPI_INT, &my_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<double> val(my_nnz);
    vector<int> col(my_nnz);
    MPI_Scatterv(all_val.data(), send_counts.data(), displs.data(), MPI_DOUBLE, 
                 val.data(), my_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_col.data(), send_counts.data(), displs.data(), MPI_INT, 
                 col.data(), my_nnz, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribuzione Row Ptr (Logica originale Send/Recv)
    vector<int> row_ptr(rows_assigned + 1);
    if (rank == 0) {
        int current_row = 0;
        // Copia locale per rank 0
        for (int i = 0; i <= rows_per_proc_vec[0]; i++) {
            row_ptr[i] = all_row_ptr[i];
        }
        current_row += rows_per_proc_vec[0];

        // Invio agli altri rank
        for (int i = 1; i < size; i++) {
            int count = rows_per_proc_vec[i] + 1;
            // Invia il blocco di row_ptr che parte da current_row
            MPI_Send(&all_row_ptr[current_row], count, MPI_INT, i, 0, MPI_COMM_WORLD);
            current_row += rows_per_proc_vec[i];
        }
    } else {
        MPI_Recv(row_ptr.data(), rows_assigned + 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Normalizzazione row_ptr
    int offset = row_ptr[0];
    for (int i = 0; i <= rows_assigned; i++) {
        row_ptr[i] -= offset;
    }

    t_io = MPI_Wtime() - t_start_io;

    // ==================================================================================
    // FASE 2: SETUP (Allocazione Vettori & Comm Plan)
    // ==================================================================================
    double t_start_setup = MPI_Wtime();

    vector<double> x(N, 1.0);
    vector<double> y(rows_assigned, 0.0);

    // Preparazione Allgatherv per x (copiato dall'originale)
    vector<int> recv_counts(size);
    vector<int> recv_displs(size);
    int chunks = N / size;
    int rem = N % size;
    for (int i = 0; i < size; i++) {
        recv_counts[i] = chunks + (i < rem ? 1 : 0);
        recv_displs[i] = (i == 0) ? 0 : recv_displs[i - 1] + recv_counts[i - 1];
    }

    t_setup = MPI_Wtime() - t_start_setup;

    // ==================================================================================
    // FASE 3: COMPUTATION (Kernel SpMV - PARALLELIZZATO + METRICHE)
    // ==================================================================================
    int N_ITER = 100;
    
    // Warmup (Parallelizzato per First Touch)
    for(int w=0; w<5; ++w) {
        #pragma omp parallel for
        for(int i=0; i<rows_assigned; ++i) y[i] = 0.0;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double loop_start_time = MPI_Wtime();

    for (int iter = 0; iter < N_ITER; iter++) {
        double t1 = MPI_Wtime();

        // Comunicazione (Logica originale: Allgatherv per aggiornare X)
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                       x.data(), recv_counts.data(), recv_displs.data(), 
                       MPI_DOUBLE, MPI_COMM_WORLD);

        double t2 = MPI_Wtime();
        t_comm += (t2 - t1);

        // Calcolo SpMV (ADESSO PARALLELO CON OPENMP)
        #pragma omp parallel for schedule(dynamic, 128)
        for (int i = 0; i < rows_assigned; i++) {
            double sum = 0.0;
            for (int k = row_ptr[i]; k < row_ptr[i+1]; k++) {
                sum += val[k] * x[col[k]];
            }
            y[i] = sum;
        }
        
        // Evita ottimizzazioni compiler dead-code
        if(rows_assigned > 0) x[0] += y[0] * 1e-9;

        t_calc += (MPI_Wtime() - t2);
    }

    double total_loop = MPI_Wtime() - loop_start_time;

    // ==================================================================================
    // FASE 4: REPORTING E CSV
    // ==================================================================================
    double max_io, max_setup, max_comm, max_calc, max_loop;

    MPI_Reduce(&t_io, &max_io, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_setup, &max_setup, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_calc, &max_calc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_loop, &max_loop, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Calcolo GFLOPS
        double gflops = (2.0 * global_nnz_cnt * N_ITER) / (max_loop * 1e9);
        double imbalance = max_loop - (max_comm + max_calc);

        // CSV OUTPUT COMPATIBILE
        cout << "CSV_DATA:" << argv[1] << "," 
             << size << "," 
             << omp_get_max_threads() << ","
             << max_io << "," 
             << max_setup << "," 
             << max_comm << "," 
             << max_calc << "," 
             << imbalance << "," 
             << max_loop << "," 
             << gflops << endl;
             
        // Debug Output
        cout << "Report SPMV4 Hybrid:" << endl;
        cout << "NNZ: " << global_nnz_cnt << " | GFLOPS: " << gflops << endl;
    }

    MPI_Finalize();
    return 0;
}