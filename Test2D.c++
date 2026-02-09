/*
 * HPC Optimized 2D SpMV con OVERLAPPING (Split-Phase Expand) e METRICHE GFLOPS
 * FIXED: Variabile t5 dichiarata correttamente
 */

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <limits>

using idx_t = long long;
using nnz_t = long long;

struct CooElem { idx_t row, col; double val; };

struct CSRMatrix {
    std::vector<double> values;
    std::vector<idx_t> col_ind;
    std::vector<nnz_t> row_ptr;
    idx_t rows;
};

struct SplitCSRMatrix2D {
    idx_t local_rows_count;
    CSRMatrix Inner;
    CSRMatrix Outer;
    std::vector<idx_t> local_to_global_row;
};

struct GridInfo {
    int rank, size;
    int dims[2], coords[2];
    MPI_Comm comm_cart, comm_row, comm_col;
};

struct Comm2D {
    std::vector<double> x_col_buffer; 
    std::vector<int> recv_counts, displs;
    idx_t my_x_count;
};

// --- LEGGEREZZA E DISTRIBUZIONE ---
void read_and_setup_split_2d(const std::string& filename, SplitCSRMatrix2D& mat, 
                            Comm2D& comm, idx_t& total_rows, idx_t& total_cols, nnz_t& total_nnz_out,
                            GridInfo& grid, double& t_io, double& t_setup) {
    
    double t_start_io = MPI_Wtime();
    std::ifstream file(filename);
    nnz_t total_nnz_global = 0;
    
    if (grid.rank == 0) {
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '%') continue;
            std::stringstream ss(line);
            ss >> total_rows >> total_cols >> total_nnz_global;
            break;
        }
    }
    MPI_Bcast(&total_rows, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_cols, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_nnz_global, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    
    total_nnz_out = total_nnz_global;

    if (grid.rank != 0) file.open(filename);
    file.clear(); file.seekg(0);
    std::string line; 
    while(std::getline(file, line)) if (!line.empty() && line[0] != '%') break;

    nnz_t chunk = total_nnz_global / grid.size;
    nnz_t start = grid.rank * chunk;
    nnz_t end = (grid.rank == grid.size - 1) ? total_nnz_global : (grid.rank + 1) * chunk;
    for(nnz_t k=0; k<start; ++k) file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::vector<CooElem> my_coo;
    idx_t r, c; double v;
    for(nnz_t k=start; k<end; ++k) {
        if(!(file >> r >> c >> v)) break;
        r--; c--; 
        if ((r % grid.dims[0] == grid.coords[0]) && (c % grid.dims[1] == grid.coords[1])) {
            my_coo.push_back({r, c, v});
        }
    }
    t_io = MPI_Wtime() - t_start_io;

    double t_start_setup = MPI_Wtime();
    int col_peers; MPI_Comm_size(grid.comm_col, &col_peers);
    comm.recv_counts.resize(col_peers); comm.displs.resize(col_peers);
    
    idx_t cols_in_grid_col = (total_cols + grid.dims[1] - 1 - grid.coords[1]) / grid.dims[1];
    idx_t base = cols_in_grid_col / grid.dims[0];
    idx_t rem = cols_in_grid_col % grid.dims[0];
    comm.my_x_count = base + (grid.coords[0] < rem ? 1 : 0);
    
    MPI_Allgather(&comm.my_x_count, 1, MPI_INT, comm.recv_counts.data(), 1, MPI_INT, grid.comm_col);
    
    comm.displs[0] = 0;
    for(int i=1; i<col_peers; i++) comm.displs[i] = comm.displs[i-1] + comm.recv_counts[i-1];
    comm.x_col_buffer.resize(comm.displs.back() + comm.recv_counts.back());

    std::sort(my_coo.begin(), my_coo.end(), [](const CooElem& a, const CooElem& b){
        return a.row < b.row || (a.row == b.row && a.col < b.col);
    });

    std::map<idx_t, idx_t> row_map; idx_t l_row_idx = 0;
    std::vector<CooElem> inner_elems, outer_elems;
    
    for(const auto& e : my_coo) {
        if(row_map.find(e.row) == row_map.end()) {
            row_map[e.row] = l_row_idx++;
            mat.local_to_global_row.push_back(e.row);
        }
        
        idx_t compressed_col = e.col / grid.dims[1];
        int target_rank = -1;
        idx_t local_offset = -1;
        idx_t cur = 0;
        for(int p=0; p<col_peers; ++p) {
            idx_t cnt = base + (p < rem ? 1 : 0);
            if(compressed_col < cur + cnt) {
                target_rank = p;
                local_offset = compressed_col - cur;
                break;
            }
            cur += cnt;
        }

        if(target_rank == grid.coords[0]) {
            inner_elems.push_back({row_map[e.row], local_offset, e.val});
        } else {
            idx_t buf_idx = comm.displs[target_rank] + local_offset;
            outer_elems.push_back({row_map[e.row], buf_idx, e.val});
        }
    }
    
    mat.local_rows_count = l_row_idx;
    
    // Build CSR Inner
    mat.Inner.rows = mat.local_rows_count;
    mat.Inner.row_ptr.assign(mat.local_rows_count+1, 0);
    for(const auto& e : inner_elems) mat.Inner.row_ptr[e.row+1]++;
    for(int i=0; i<mat.local_rows_count; ++i) mat.Inner.row_ptr[i+1] += mat.Inner.row_ptr[i];
    mat.Inner.values.resize(inner_elems.size()); mat.Inner.col_ind.resize(inner_elems.size());
    std::vector<int> work(mat.local_rows_count, 0);
    for(const auto& e : inner_elems) {
        idx_t dest = mat.Inner.row_ptr[e.row] + work[e.row]++;
        mat.Inner.values[dest] = e.val;
        mat.Inner.col_ind[dest] = e.col;
    }

    // Build CSR Outer
    mat.Outer.rows = mat.local_rows_count;
    mat.Outer.row_ptr.assign(mat.local_rows_count+1, 0);
    for(const auto& e : outer_elems) mat.Outer.row_ptr[e.row+1]++;
    for(int i=0; i<mat.local_rows_count; ++i) mat.Outer.row_ptr[i+1] += mat.Outer.row_ptr[i];
    mat.Outer.values.resize(outer_elems.size()); mat.Outer.col_ind.resize(outer_elems.size());
    std::fill(work.begin(), work.end(), 0);
    for(const auto& e : outer_elems) {
        idx_t dest = mat.Outer.row_ptr[e.row] + work[e.row]++;
        mat.Outer.values[dest] = e.val;
        mat.Outer.col_ind[dest] = e.col;
    }
    t_setup = MPI_Wtime() - t_start_setup;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    GridInfo grid;
    MPI_Comm_rank(MPI_COMM_WORLD, &grid.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &grid.size);
    
    grid.dims[0]=0; grid.dims[1]=0;
    MPI_Dims_create(grid.size, 2, grid.dims);
    int periods[2]={0,0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, grid.dims, periods, 1, &grid.comm_cart);
    MPI_Cart_coords(grid.comm_cart, grid.rank, 2, grid.coords);
    MPI_Comm_split(grid.comm_cart, grid.coords[0], grid.coords[1], &grid.comm_row);
    MPI_Comm_split(grid.comm_cart, grid.coords[1], grid.coords[0], &grid.comm_col);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Usiamo una barriera per evitare che l'output si mescoli troppo con la lettura del file
    MPI_Barrier(MPI_COMM_WORLD);

    if (grid.rank < 128) { // Puoi limitare la stampa se hai troppi rank
        // Ogni rank stampa la sua posizione logica vs fisica
        printf("[Rank %3d] Logico: (%2d,%2d) | Fisico: %s\n", 
            grid.rank, grid.coords[0], grid.coords[1], processor_name);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::string filename = (argc>1)? argv[1] : "matrix.mtx";
    SplitCSRMatrix2D mat;
    Comm2D comm;
    idx_t tr, tc;
    nnz_t global_nnz = 0;
    double t_io = 0, t_setup = 0;
    
    read_and_setup_split_2d(filename, mat, comm, tr, tc, global_nnz, grid, t_io, t_setup);
    
    std::vector<double> x_owned(comm.my_x_count, 1.0);
    std::vector<double> y_local(mat.local_rows_count, 0.0);
    idx_t max_rows_row_comm = (tr + grid.dims[0] - 1)/grid.dims[0];
    std::vector<double> y_dense_contrib(max_rows_row_comm, 0.0);
    std::vector<double> y_result(max_rows_row_comm, 0.0);

    int n_iter = 100;
    MPI_Request req;
    double t_comm = 0, t_calc = 0;
    
    // WARMUP (Opzionale ma consigliato per evitare First Touch overhead)
    for(int w=0; w<5; ++w) {
         #pragma omp parallel for
         for(idx_t i=0; i<mat.local_rows_count; ++i) y_local[i] = 0.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double loop_start = MPI_Wtime();
    
    for(int i=0; i<n_iter; ++i) {
        double t1 = MPI_Wtime();
        
        // 1. START EXPAND (Async)
        MPI_Iallgatherv(x_owned.data(), comm.my_x_count, MPI_DOUBLE,
                        comm.x_col_buffer.data(), comm.recv_counts.data(), comm.displs.data(),
                        MPI_DOUBLE, grid.comm_col, &req);
        
        double t2 = MPI_Wtime();
        
        // 2. COMPUTE INNER (Overlap)
        #pragma omp parallel for
        for(idx_t r=0; r<mat.Inner.rows; ++r) {
            double sum = 0.0;
            for(nnz_t j=mat.Inner.row_ptr[r]; j<mat.Inner.row_ptr[r+1]; ++j) {
                sum += mat.Inner.values[j] * x_owned[mat.Inner.col_ind[j]];
            }
            y_local[r] = sum;
        }
        
        double t3 = MPI_Wtime();
        
        // 3. WAIT EXPAND
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        
        t_comm += (MPI_Wtime() - t1) - (t3 - t2); 
        
        double t4 = MPI_Wtime();

        // 4. COMPUTE OUTER
        #pragma omp parallel for
        for(idx_t r=0; r<mat.Outer.rows; ++r) {
            double sum = 0.0;
            for(nnz_t j=mat.Outer.row_ptr[r]; j<mat.Outer.row_ptr[r+1]; ++j) {
                sum += mat.Outer.values[j] * comm.x_col_buffer[mat.Outer.col_ind[j]];
            }
            y_local[r] += sum;
        }
        
        t_calc += (t3 - t2) + (MPI_Wtime() - t4); 

        // 5. FOLD
        // === FIX HERE: Dichiarazione di t5 ===
        double t5 = MPI_Wtime();
        
        std::fill(y_dense_contrib.begin(), y_dense_contrib.end(), 0.0);
        for(idx_t r=0; r<mat.local_rows_count; ++r) {
            idx_t g_row = mat.local_to_global_row[r];
            idx_t idx = g_row / grid.dims[0];
            if(idx < max_rows_row_comm) y_dense_contrib[idx] = y_local[r];
        }
        MPI_Allreduce(y_dense_contrib.data(), y_result.data(), max_rows_row_comm, MPI_DOUBLE, MPI_SUM, grid.comm_row);
        t_comm += (MPI_Wtime() - t5);
        
        if(!x_owned.empty() && !y_result.empty()) x_owned[0] = y_result[0]*0.01;
    }
    
    double total_time = MPI_Wtime() - loop_start;
    
    // Aggregazione Metriche
    double max_io, max_setup, max_comm, max_calc, max_loop;
    MPI_Reduce(&t_io, &max_io, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_setup, &max_setup, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_calc, &max_calc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_loop, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(grid.rank == 0) {
        double gflops = (2.0 * global_nnz * n_iter) / (max_loop * 1e9);
        double imbalance = max_loop - (max_comm + max_calc);

        std::cout << "CSV_DATA:" << filename << "," << grid.size << "," << omp_get_max_threads() << ","
                  << max_io << "," << max_setup << "," 
                  << max_comm << "," << max_calc << "," 
                  << imbalance << "," << max_loop << "," << gflops << std::endl;
                  
        std::cout << "\n=== Report 2D Overlapped ===" << std::endl;
        std::cout << "Global NNZ: " << global_nnz << std::endl;
        std::cout << "Time Loop:  " << max_loop << " s" << std::endl;
        std::cout << "GFLOPS:     " << gflops << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}