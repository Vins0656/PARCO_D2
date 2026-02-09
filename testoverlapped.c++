/*
 * HPC Optimized Hybrid MPI+OpenMP SpMV with Communication Overlap & GFLOPS Metric
 * STRATEGY: 1D Row Partitioning + Split-Phase Computation
 *
 * 1. Parallel IO (from test2)
 * 2. Setup: Split matrix into Diagonal (local X) and Off-Diagonal (remote X) parts.
 * 3. Loop:
 * a. Start Non-blocking Comm (Irecv/Isend)
 * b. Compute Diagonal Block (Overlap!)
 * c. Wait Comm
 * d. Compute Off-Diagonal Block & Accumulate
 * 4. Metric: Calculate GFLOPS based on Global NNZ.
 */

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <map>
#include <unordered_map>
#include <cmath>
#include <cstring>

using idx_t = long long;
using nnz_t = long long;

// Struct semplice per CSR
struct CSRMatrix {
    std::vector<double> values;
    std::vector<idx_t> col_ind;
    std::vector<nnz_t> row_ptr;
    idx_t rows;
};

// Matrice "Splittata" per Overlap
struct SplitCSRMatrix {
    idx_t local_rows;
    idx_t global_row_offset;
    
    // Parte Diagonale (usa X locale)
    CSRMatrix D; 
    // Parte Off-Diagonale (usa X remoto/ghost)
    CSRMatrix O; 
};

struct CommPlan {
    std::vector<int> recv_neighbors, send_neighbors;
    std::vector<double> send_buffer, recv_buffer;
    std::vector<idx_t> send_indices; // Indici locali da inviare (linearizzati)
    std::vector<int> send_offsets;   // Offset nel buffer di invio per ogni vicino
    std::vector<int> recv_offsets;   // Offset nel buffer di ricezione per ogni vicino
    int total_recv_size;
};

struct CooElem { int row, col; double val; };

// --- HELPER IO ---
MPI_Offset find_next_line_start(MPI_File fh, MPI_Offset start, MPI_Offset end) {
    char c; MPI_Status status; MPI_Offset current = start;
    while (current < end) {
        MPI_File_read_at(fh, current, &c, 1, MPI_CHAR, &status);
        if (c == '\n') return current + 1;
        current++;
    }
    return end;
}

// --- PARALLEL IO & BALANCE ---
// Modificato per restituire global_nnz
void parallel_read_and_distribute(const std::string& filename, int rank, int size,
                                  std::vector<CooElem>& local_elems, 
                                  idx_t& global_M, nnz_t& global_nnz, // Aggiunto parametro output
                                  idx_t& row_start, idx_t& row_end,
                                  double& io_time) {
    
    double t_start = MPI_Wtime();
    MPI_File fh;
    idx_t g_M_int = 0, g_N_int = 0;
    nnz_t g_nnz_int = 0;
    MPI_Offset file_size, data_start = 0;

    if (MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        if(rank == 0) std::cerr << "Error opening file." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        std::ifstream fs(filename);
        std::string line;
        while (std::getline(fs, line) && (line.empty() || line[0] == '%'));
        std::stringstream ss(line);
        ss >> g_M_int >> g_N_int >> g_nnz_int;
        data_start = fs.tellg();
    }
    MPI_Bcast(&g_M_int, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g_N_int, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g_nnz_int, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&data_start, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);
    
    global_M = g_M_int;
    global_nnz = g_nnz_int; // Salva il valore globale per il calcolo GFLOPS

    MPI_File_get_size(fh, &file_size);
    MPI_Offset chunk_size = (file_size - data_start) / size;
    MPI_Offset my_start_ideal = data_start + rank * chunk_size;
    MPI_Offset next_start_ideal = (rank == size - 1) ? file_size : data_start + (rank + 1) * chunk_size;

    MPI_Offset my_start = my_start_ideal;
    if(rank != 0) my_start = find_next_line_start(fh, my_start_ideal, file_size);
    MPI_Offset my_end = next_start_ideal;
    if(rank != size - 1) my_end = find_next_line_start(fh, next_start_ideal, file_size);

    long long read_size = my_end - my_start;
    if(read_size < 0) read_size = 0;
    std::vector<char> buffer(read_size + 1);
    if(read_size > 0) {
        MPI_File_read_at_all(fh, my_start, buffer.data(), read_size, MPI_CHAR, MPI_STATUS_IGNORE);
        buffer[read_size] = '\0';
    } else {
        MPI_File_read_at_all(fh, my_start, buffer.data(), 0, MPI_CHAR, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&fh);

    std::vector<CooElem> temp_elems;
    if(read_size > 0) {
        std::stringstream ss(buffer.data());
        int r, c; double v;
        while(ss >> r >> c >> v) temp_elems.push_back({r-1, c-1, v});
    }

    // Bilanciamento NNZ
    std::vector<int> local_row_cnt(global_M, 0);
    for(const auto& e : temp_elems) if(e.row >= 0 && e.row < global_M) local_row_cnt[e.row]++;
    std::vector<int> global_row_cnt(global_M);
    MPI_Allreduce(local_row_cnt.data(), global_row_cnt.data(), global_M, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::vector<idx_t> limits(size + 1, 0);
    nnz_t target = g_nnz_int / size;
    nnz_t current_sum = 0;
    int cp = 0;
    for(idx_t r=0; r<global_M; ++r) {
        current_sum += global_row_cnt[r];
        if(current_sum >= target && cp < size - 1) {
            limits[++cp] = r + 1;
            current_sum = 0;
        }
    }
    limits[size] = global_M;
    row_start = limits[rank];
    row_end = limits[rank+1];

    // Redistribuzione elementi
    std::vector<std::vector<CooElem>> send_bufs(size);
    for(const auto& e : temp_elems) {
        auto it = std::upper_bound(limits.begin(), limits.end(), e.row);
        int owner = std::distance(limits.begin(), it) - 1;
        owner = std::max(0, std::min(owner, size-1));
        send_bufs[owner].push_back(e);
    }
    
    std::vector<int> scounts(size), rcounts(size);
    for(int i=0; i<size; ++i) scounts[i] = send_bufs[i].size();
    MPI_Alltoall(scounts.data(), 1, MPI_INT, rcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<CooElem> sflat;
    std::vector<int> sdispls(size), rdispls(size);
    int tot_s = 0, tot_r = 0;
    for(int i=0; i<size; ++i) {
        sdispls[i] = tot_s;
        sflat.insert(sflat.end(), send_bufs[i].begin(), send_bufs[i].end());
        tot_s += scounts[i];
        rdispls[i] = tot_r;
        tot_r += rcounts[i];
    }
    local_elems.resize(tot_r);
    
    MPI_Datatype mpi_coo;
    MPI_Type_contiguous(sizeof(CooElem), MPI_BYTE, &mpi_coo);
    MPI_Type_commit(&mpi_coo);
    MPI_Alltoallv(sflat.data(), scounts.data(), sdispls.data(), mpi_coo,
                  local_elems.data(), rcounts.data(), rdispls.data(), mpi_coo, MPI_COMM_WORLD);
    MPI_Type_free(&mpi_coo);

    io_time += (MPI_Wtime() - t_start);
}

// --- SETUP AGGRESSIVO (SPLIT DIAGONAL / OFF-DIAGONAL) ---
void setup_split_communication(const std::vector<CooElem>& elems, int rank, int size,
                               idx_t row_start, idx_t row_end, idx_t global_M,
                               SplitCSRMatrix& mat, CommPlan& comm, double& setup_time) {
    double t_start = MPI_Wtime();

    mat.local_rows = row_end - row_start;
    if(mat.local_rows < 0) mat.local_rows = 0;
    mat.global_row_offset = row_start;

    // 1. Separazione Elementi Locali vs Remoti
    std::vector<CooElem> diag_elems, off_elems;
    std::map<idx_t, int> ghost_cols; // col_idx -> owner

    std::vector<idx_t> all_starts(size);
    MPI_Allgather(&row_start, 1, MPI_LONG_LONG, all_starts.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
    std::vector<idx_t> limits = all_starts; 
    limits.push_back(global_M);

    for(const auto& e : elems) {
        idx_t local_r = e.row - row_start;
        if(local_r < 0 || local_r >= mat.local_rows) continue;

        if(e.col >= row_start && e.col < row_end) {
            diag_elems.push_back(e);
        } else {
            off_elems.push_back(e);
            if(ghost_cols.find(e.col) == ghost_cols.end()) {
                auto it = std::upper_bound(limits.begin(), limits.end(), e.col);
                int owner = std::distance(limits.begin(), it) - 1;
                if(owner < 0) owner = 0; if(owner >= size) owner = size-1;
                ghost_cols[e.col] = owner;
            }
        }
    }

    // 2. Setup Comunicazione
    std::map<int, std::vector<idx_t>> req_per_proc;
    for(auto const& [col, owner] : ghost_cols) req_per_proc[owner].push_back(col);

    std::vector<int> send_cnt(size, 0), recv_cnt(size, 0);
    for(auto const& [p, vec] : req_per_proc) recv_cnt[p] = vec.size();
    MPI_Alltoall(recv_cnt.data(), 1, MPI_INT, send_cnt.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<idx_t> sbuf_req, rbuf_req;
    std::vector<int> sdispls(size, 0), rdispls(size, 0);
    int off=0;
    for(int i=0; i<size; ++i) {
        sdispls[i] = off;
        if(req_per_proc.count(i)) sbuf_req.insert(sbuf_req.end(), req_per_proc[i].begin(), req_per_proc[i].end());
        off += recv_cnt[i];
    }
    off=0;
    for(int i=0; i<size; ++i) { rdispls[i] = off; off += send_cnt[i]; }
    rbuf_req.resize(off);

    MPI_Alltoallv(sbuf_req.data(), recv_cnt.data(), sdispls.data(), MPI_LONG_LONG,
                  rbuf_req.data(), send_cnt.data(), rdispls.data(), MPI_LONG_LONG, MPI_COMM_WORLD);

    // 3. Costruzione CommPlan (Send Side)
    comm.send_neighbors.clear(); comm.send_indices.clear(); comm.send_offsets.clear();
    comm.send_buffer.resize(rbuf_req.size());
    int current_send_off = 0;
    idx_t my_start = row_start; 

    for(int i=0; i<size; ++i) {
        if(send_cnt[i] > 0) {
            comm.send_neighbors.push_back(i);
            comm.send_offsets.push_back(current_send_off);
            for(int k=0; k<send_cnt[i]; ++k) {
                idx_t global_c = rbuf_req[rdispls[i] + k];
                comm.send_indices.push_back(global_c - my_start); 
            }
            current_send_off += send_cnt[i];
        }
    }
    comm.send_offsets.push_back(current_send_off);

    // 4. Costruzione CommPlan (Recv Side)
    comm.recv_neighbors.clear(); comm.recv_offsets.clear();
    comm.total_recv_size = sbuf_req.size();
    comm.recv_buffer.resize(comm.total_recv_size);
    
    std::unordered_map<idx_t, idx_t> global_to_ghost_idx;
    int current_recv_off = 0;

    for(int i=0; i<size; ++i) {
        if(recv_cnt[i] > 0) {
            comm.recv_neighbors.push_back(i);
            comm.recv_offsets.push_back(current_recv_off);
            for(idx_t col : req_per_proc[i]) {
                global_to_ghost_idx[col] = current_recv_off++;
            }
        }
    }
    comm.recv_offsets.push_back(current_recv_off);

    // 5. Costruzione Matrici CSR
    std::sort(diag_elems.begin(), diag_elems.end(), [](const CooElem& a, const CooElem& b){
        return a.row < b.row || (a.row == b.row && a.col < b.col);
    });
    std::sort(off_elems.begin(), off_elems.end(), [](const CooElem& a, const CooElem& b){
        return a.row < b.row || (a.row == b.row && a.col < b.col);
    });

    // BUILD D
    mat.D.rows = mat.local_rows;
    mat.D.row_ptr.assign(mat.local_rows + 1, 0);
    for(const auto& e : diag_elems) mat.D.row_ptr[e.row - row_start + 1]++;
    for(int i=0; i<mat.local_rows; ++i) mat.D.row_ptr[i+1] += mat.D.row_ptr[i];
    mat.D.values.resize(diag_elems.size());
    mat.D.col_ind.resize(diag_elems.size());
    std::vector<int> work(mat.local_rows, 0);
    for(const auto& e : diag_elems) {
        int r = e.row - row_start;
        int dest = mat.D.row_ptr[r] + work[r]++;
        mat.D.values[dest] = e.val;
        mat.D.col_ind[dest] = e.col - row_start; 
    }

    // BUILD O
    mat.O.rows = mat.local_rows;
    mat.O.row_ptr.assign(mat.local_rows + 1, 0);
    for(const auto& e : off_elems) mat.O.row_ptr[e.row - row_start + 1]++;
    for(int i=0; i<mat.local_rows; ++i) mat.O.row_ptr[i+1] += mat.O.row_ptr[i];
    mat.O.values.resize(off_elems.size());
    mat.O.col_ind.resize(off_elems.size());
    std::fill(work.begin(), work.end(), 0);
    for(const auto& e : off_elems) {
        int r = e.row - row_start;
        int dest = mat.O.row_ptr[r] + work[r]++;
        mat.O.values[dest] = e.val;
        mat.O.col_ind[dest] = global_to_ghost_idx[e.col];
    }
    setup_time += (MPI_Wtime() - t_start);
}

// --- MAIN ---
int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) { MPI_Finalize(); return 1; }
    std::string filename = argv[1];

    SplitCSRMatrix mat;
    CommPlan comm;
    idx_t global_M, r_start, r_end;
    nnz_t global_nnz = 0; // Per GFLOPS
    std::vector<CooElem> local_elems;
    double t_io = 0, t_setup = 0;

    // 1. Lettura e Distribuzione (Passiamo global_nnz)
    parallel_read_and_distribute(filename, rank, size, local_elems, global_M, global_nnz, r_start, r_end, t_io);

    // 2. Setup (Split Phase)
    setup_split_communication(local_elems, rank, size, r_start, r_end, global_M, mat, comm, t_setup);
    
    local_elems.clear(); local_elems.shrink_to_fit();

    std::vector<double> X(mat.local_rows, 1.0);
    std::vector<double> Y(mat.local_rows, 0.0);
    
    int n_iter = 100;
    std::vector<MPI_Request> reqs;
    reqs.resize(comm.recv_neighbors.size() + comm.send_neighbors.size());
    
    MPI_Barrier(MPI_COMM_WORLD);
    double loop_start_wall = MPI_Wtime();
    double t_calc = 0, t_comm = 0;

    for(int iter=0; iter<n_iter; ++iter) {
        double t1 = MPI_Wtime();
        int r_idx = 0;

        // STEP A: START COMMS
        for(size_t i=0; i<comm.recv_neighbors.size(); ++i) {
            int count = comm.recv_offsets[i+1] - comm.recv_offsets[i];
            MPI_Irecv(&comm.recv_buffer[comm.recv_offsets[i]], count, MPI_DOUBLE, 
                      comm.recv_neighbors[i], 0, MPI_COMM_WORLD, &reqs[r_idx++]);
        }
        
        #pragma omp parallel for schedule(static)
        for(size_t i=0; i<comm.send_neighbors.size(); ++i) {
            int start = comm.send_offsets[i];
            int count = comm.send_offsets[i+1] - start;
            for(int k=0; k<count; ++k) comm.send_buffer[start+k] = X[comm.send_indices[start+k]];
        }

        for(size_t i=0; i<comm.send_neighbors.size(); ++i) {
            int start = comm.send_offsets[i];
            int count = comm.send_offsets[i+1] - start;
            MPI_Isend(&comm.send_buffer[start], count, MPI_DOUBLE, 
                      comm.send_neighbors[i], 0, MPI_COMM_WORLD, &reqs[r_idx++]);
        }

        // STEP B: COMPUTE DIAGONAL (OVERLAP)
        double t2 = MPI_Wtime();
        #pragma omp parallel for schedule(dynamic, 128)
        for(idx_t i=0; i<mat.local_rows; ++i) {
            double sum = 0.0;
            for(nnz_t j=mat.D.row_ptr[i]; j<mat.D.row_ptr[i+1]; ++j) {
                sum += mat.D.values[j] * X[mat.D.col_ind[j]];
            }
            Y[i] = sum;
        }
        t_calc += (MPI_Wtime() - t2);

        // STEP C: WAIT COMMS
        MPI_Waitall(r_idx, reqs.data(), MPI_STATUSES_IGNORE);
        t_comm += (MPI_Wtime() - t1) - (MPI_Wtime() - t2);

        // STEP D: COMPUTE OFF-DIAGONAL
        double t3 = MPI_Wtime();
        #pragma omp parallel for schedule(dynamic, 128)
        for(idx_t i=0; i<mat.local_rows; ++i) {
            double sum = 0.0;
            for(nnz_t j=mat.O.row_ptr[i]; j<mat.O.row_ptr[i+1]; ++j) {
                sum += mat.O.values[j] * comm.recv_buffer[mat.O.col_ind[j]];
            }
            Y[i] += sum;
            X[i] = Y[i] * 1e-5 + 0.01;
        }
        t_calc += (MPI_Wtime() - t3);
    }
    
    double total_loop = MPI_Wtime() - loop_start_wall;
    
    double max_io, max_setup, max_comm, max_calc, max_loop;
    MPI_Reduce(&t_io, &max_io, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_setup, &max_setup, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_calc, &max_calc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_loop, &max_loop, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        // --- CALCOLO GFLOPS ---
        // Formula: (2 * NNZ * Iterations) / (Time_Sec * 1e9)
        double gflops = (2.0 * global_nnz * n_iter) / (max_loop * 1e9);

        double imbalance = max_loop - (max_comm + max_calc);
        // Aggiunto GFLOPS alla fine del CSV
        std::cout << "CSV_DATA:" << filename << "," << size << "," << omp_get_max_threads() << ","
                  << max_io << "," << max_setup << "," 
                  << max_comm << "," << max_calc << "," 
                  << imbalance << "," << max_loop << "," << gflops << std::endl;
                  
        std::cout << "\n=== Report (Overlap + GFLOPS) ===" << std::endl;
        std::cout << "Matrix:     " << filename << std::endl;
        std::cout << "Global NNZ: " << global_nnz << std::endl;
        std::cout << "Time Loop:  " << max_loop << " s" << std::endl;
        std::cout << "GFLOPS:     " << gflops << std::endl;
    }

    MPI_Finalize();
    return 0;
}