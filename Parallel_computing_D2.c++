using namespace std;
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <vector>
#include <random>
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iomanip>



typedef std::vector<float> Matrix;

//_____________________________________________________UTILITY FUNCTIONS_____________________________________________________


void random_init(Matrix & a, int size){
   std::uniform_real_distribution<float> dist(0.0f, 100.0f);
   std::random_device rd;
   std::default_random_engine gen(rd());

    a.resize(size*size);
      for (int i=0; i<size; i++){
        for (int j=0; j<size; j++){
             a[i*size+j]=dist(gen);
        }
    }
}


double harmonicMean(double arr[], int n) 
{ 
    // Declare sum variables and initialize with zero. 
    double sum = 0.0; 
    for (int i = 0; i < n; i++) 
    {
        //cout<<arr[i]<<" ";    
        sum = sum + (double)1 / arr[i]; 
    }
    return (double)n / sum; 
}




//_____________________________________________________DELIVERABLE FUNCTIONS_____________________________________________________


//SYMMETRY

void checkSym(Matrix & a, Matrix & b, int mat_size){
    int dim = mat_size;
    bool symmetric=true;

    for (int i =0; i< dim; i++){
        for (int j =0; j<dim; j++){
            if((a[j+dim*i]-b[j+dim*i])!=0){
                symmetric=false;
            }
        }
    }

    // if (symmetric){
    //     cout<<"The matrix is symmetric!";
    // }
    // else{
    //     cout<<"The matrix is not symmetric! ";
    // }
}


void checkSymMpi(Matrix & a, Matrix & b,int size_mat ,int rank, int size){

    const int rows = size_mat, cols = size_mat;
   
    std::vector<float> recvbuf_a((rows * cols) / size);
    std::vector<float> recvbuf_b((rows * cols) / size);  

    int rw_per_proc=size_mat/size;
    
    bool local_zero=true;
    
    MPI_Scatter(a.data(), (rows * cols) / size, MPI_FLOAT,
                 recvbuf_a.data(), (rows * cols) / size, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    MPI_Scatter(b.data(), (rows * cols) / size, MPI_FLOAT,
                 recvbuf_b.data(), (rows * cols) / size, MPI_FLOAT,
                 0, MPI_COMM_WORLD);             

    for(int j=0; j<rw_per_proc; j++){
        for(int i=0; i<rows; i++){
            if((recvbuf_a[i+rows*j]-recvbuf_b[i+rows*j])!=0){
                local_zero=false;
            }
        }
    }

    bool global_all_zero=true;
     
    MPI_Allreduce(&local_zero, &global_all_zero, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    
    // if (rank == 0) {
    //     if (global_all_zero) {
    //         std::cout << "The matrix is symmetric. " << std::endl;
    //     } else {
    //         std::cout << "The matrix is not symmetric. " << std::endl;
    //     }
    // }
}

//------------------------------------------------------------------------------
void matTranspose(Matrix & a, Matrix & b ,int size_mat){
    for (int i=0; i<size_mat; i++){
        for(int j=0; j<size_mat; j++){
            b[i*size_mat+j]=a[j*size_mat+i];
        }
    }
}

void matTransposeMpi(Matrix & matrix, Matrix & transposed, int size_mat, int rank, int size){
    const int rows = size_mat, cols = size_mat;
   
    std::vector<float> recvbuf((rows * cols) / size); 
    
    MPI_Scatter(matrix.data(), (rows * cols) / size, MPI_FLOAT,
                 recvbuf.data(), (rows * cols) / size, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    int rw_per_proc=size_mat/size;

    std::vector<float> copy=recvbuf;

    if (rw_per_proc>1){
    int count=1;
        for(int i=0; i<rows; i++){
            for(int j=0; j<rw_per_proc; j++){
               recvbuf[i*rw_per_proc+j]=copy[j*rows+i];
            }   
        }
    }

    std::vector<float> rcv_all(rows*rw_per_proc);

    for (int i=0; i<rw_per_proc; i++){
        MPI_Alltoall(recvbuf.data()+(i*rows), rw_per_proc, MPI_FLOAT,
                 rcv_all.data()+(i*rows), rw_per_proc, MPI_FLOAT,
                 MPI_COMM_WORLD);

        MPI_Gather(rcv_all.data()+(i*rows), rows , MPI_FLOAT,
                    transposed.data()+(rows*(rows/rw_per_proc)*i) , rows, MPI_FLOAT,
                    0, MPI_COMM_WORLD);      

    }
}


//____________________________________________________________MAIN_____________________________________________________

int main (int argc, char *argv[]){
 
if (argc==1){
    int dim_vector[9]={16,32,64,128,256,512,1024,2048,4096}; 
    double tran_seq_count[16]={0.0};
    double tran_mpi_count[16]={0.0};
    double sym_seq_count[16]={0.0};
    double sym_mpi_count[16]={0.0};

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dim=8;

    Matrix M(1), T(1);
    random_init(M,dim);
    random_init(T,dim);

    std::string s1,s2;

    for (int entry: dim_vector){
            if(rank==0){
                cout<<"Running test on matrix: "<<entry<<"x"<<entry<<endl;
            }
            for(int i=0; i<16; i++){
                random_init(M,entry);
                random_init(T,entry);
                
                if (rank==0){
                    double start_time = MPI_Wtime();
                    matTranspose(M,T,entry);
                    double end_time = MPI_Wtime();
                    double parallel_duration = end_time - start_time;
                    tran_seq_count[i]=parallel_duration;

                    start_time = MPI_Wtime();
                    checkSym(M,T,entry);
                    end_time = MPI_Wtime();
                    parallel_duration = end_time - start_time;
                    sym_seq_count[i]=parallel_duration;
                }
                random_init(T,entry);

                MPI_Barrier(MPI_COMM_WORLD);  
                double start_time = MPI_Wtime();
                matTransposeMpi(M,T,entry,rank,size);
                double end_time = MPI_Wtime();
                double parallel_duration = end_time - start_time;
                tran_mpi_count[i]=parallel_duration;

                MPI_Barrier(MPI_COMM_WORLD); 
                start_time = MPI_Wtime();
                checkSymMpi(M,T,entry,rank,size);
                end_time = MPI_Wtime();
                parallel_duration = end_time - start_time;
                sym_mpi_count[i]=parallel_duration;
            }
            if (rank==0){
                    
                bool symm=true;
                for(int i=0; i<entry; i++){
                    for(int j=0; j<entry; j++){
                        if (M[i*entry+j]!=T[j*entry+i]){
                                symm=false;
                            }
                        }
                    
                    }
                double a, b, c, d;

                a=harmonicMean(tran_seq_count,16);
                b=harmonicMean(tran_mpi_count,16);
                c=harmonicMean(sym_seq_count, 16);
                d=harmonicMean(sym_mpi_count, 16);
                cout<<"\nSequential execution Transposition: "<<a<<endl;
                cout<<"\nSequential execution Symmetry check:"<<c<<endl<<endl;

                if (entry==16){
                    s1+="--"+std::to_string(size)+"\n";
                    s2+="--"+std::to_string(size)+"\n";
                }

                
                s1 += std::to_string(entry) + "x" + std::to_string(entry) + ": " + std::to_string(b) + "\n";  // For entry and b
                s2 += std::to_string(entry) + "x" + std::to_string(entry) + ": " + std::to_string(d) + "\n";  // For entry and b

            }
        }

    MPI_Finalize();
    if (rank==0){
        cout<<"Transposition MPI\n"<<s1<<endl<<"Check Sym MPI\n"<<s2;
    }
    
    if (rank==0){
        std::string filename_1 = "Scalabity_Test_Transpose.txt";
        std::string filename_2 = "Scalabity_Test_Symmetry.txt";
        
        std::string  target="--"+std::to_string(size);


        std::string line;
        bool skipping = false;
        int skippedCount = 0;


        std::ifstream infile(filename_1);  // File in lettura
        std::ofstream tempFile("temp.txt", std::ios::trunc);

        if (!infile || !tempFile) {
            std::cerr << "Error opening file 1." << std::endl;
            return 0;
        }


        while (getline(infile, line)) {
            if (!skipping && line == target) {
                skipping = true;  // Inizia a saltare le righe
                skippedCount = 0;  // Conta le righe saltate
                continue;  // Non copia la linea di stop
            }
            if (skipping) {
                skippedCount++;
                if (skippedCount == 9) {
                    tempFile << s1 << "\n";
                    tempFile.flush();   // Scrive il nuovo testo dopo aver saltato le righe
                    skipping = false;  // Termina il salto e torna a copiare le righe
                }
                continue;  // Salta le righe durante il conteggio
            }

            tempFile << line << "\n"; 
            tempFile.flush();  // Copia le righe nel file temporaneo
        }

        infile.close();
        tempFile.close();

        // Sostituisce il file originale con il file modificato
        std::remove(filename_1.c_str());
        std::rename("temp.txt", filename_1.c_str());  




        line="";
        skipping = false;
        skippedCount = 0;


        std::ifstream infile_2(filename_2);  // File in lettura
        std::ofstream tempFile_2("temp.txt",std::ios::trunc);

        if (!infile_2 || !tempFile_2) {
            std::cerr << "Error opening file 2." << std::endl;
            return 0;
        }


        while (getline(infile_2, line)) {
            if (!skipping && line == target) {
                skipping = true;  // Inizia a saltare le righe
                skippedCount = 0;  // Conta le righe saltate
                continue;  // Non copia la linea di stop
            }
            if (skipping) {
                skippedCount++;
                if (skippedCount == 9) {
                    tempFile_2 << s2 << "\n";
                    tempFile_2.flush();  // Scrive il nuovo testo dopo aver saltato le righe
                    skipping = false;  // Termina il salto e torna a copiare le righe
                }
                continue;  // Salta le righe durante il conteggio
            }

            tempFile_2 << line << "\n";
            tempFile_2.flush();   // Copia le righe nel file temporaneo
        }

        infile_2.close();
        tempFile_2.close();

        // Sostituisce il file originale con il file modificato
        std::remove(filename_2.c_str());
        std::rename("temp.txt", filename_2.c_str());     

    
        }
    }
    else if(argc==3){
        MPI_Init(&argc, &argv);
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (rank==0){
            const int rows = 6, cols = 9;  // Dimensions of the matrix
            double matrix_tran[rows][cols];
                 // Matrix to store the values
            std::ifstream infile("Scalabity_Test_Transpose.txt");
            // Input file containing the data

            if (!infile) {
                std::cerr << "Error opening file 1." << std::endl;
                return 1;
            }

            std::string line;
            int rowIndex = 0;

            while (std::getline(infile, line)) {
                // Skip lines starting with '--'
                if (line.empty() || line[0] == '-') {
                    continue;
                }

                std::istringstream lineStream(line);
                std::string size_s, value;
                
                std::getline(lineStream, size_s, ':');  // Ignore size label (e.g., "16x16")
                std::getline(lineStream, value);      // Extract the value after the colon

                if (!value.empty() && rowIndex < rows * cols) {
                    int colIndex = rowIndex % cols;
                    int currentRow = rowIndex / cols;
                    matrix_tran[currentRow][colIndex] = std::stod(value);  // Convert value to double and store
                    rowIndex++;
                }
            }

            infile.close();

            double matrix_sym[rows][cols];
            std::ifstream infile_2("Scalabity_Test_Symmetry.txt");


             if (!infile_2) {
                std::cerr << "Error opening file 2." << std::endl;
                return 1;
            }

            line="";
            rowIndex = 0;

            while (std::getline(infile_2, line)) {
                // Skip lines starting with '--'
                if (line.empty() || line[0] == '-') {
                    continue;
                }

                std::istringstream lineStream_2(line);
                std::string size_s, value;
                
                std::getline(lineStream_2, size_s, ':');  // Ignore size label (e.g., "16x16")
                std::getline(lineStream_2, value);      // Extract the value after the colon

                if (!value.empty() && rowIndex < rows * cols) {
                    int colIndex = rowIndex % cols;
                    int currentRow = rowIndex / cols;
                    matrix_sym[currentRow][colIndex] = std::stod(value);  // Convert value to double and store
                    rowIndex++;
                }
            }

            infile_2.close();

            // for(int i=0; i<6; i++){
            //     for (int j=0; j<9; j++){
            //         cout<<matrix_tran[i][j]<<" ";
            //     }
            //     cout<<endl;
            // }



            //computing data

            double str_scl_tran [5][9], efficiency_tran[5][9];
            double str_scl_sym [5][9], efficiency_sym[5][9];
            double weak_scl_tran[5], weak_scl_sym[5];

            for(int i=1; i<6; i++){
                for (int j=0; j<9; j++){
                    str_scl_tran[i-1][j]=matrix_tran[0][j]/matrix_tran[i][j];
                    str_scl_sym[i-1][j]=matrix_sym[0][j]/matrix_sym[i][j];
                }
            }

            int num=2;
            for(int i=0; i<5; i++){
                for (int j=0; j<9; j++){
                    efficiency_tran[i][j]=(str_scl_tran[i][j]/num)*100;
                    efficiency_sym[i][j]=(str_scl_sym[i][j]/num)*100;
                }
                num*=2;
            }

            for(int i=1; i<6; i++){
                weak_scl_tran[i]=matrix_tran[0][0]/matrix_tran[i][i];
                weak_scl_sym[i]=matrix_sym[0][0]/matrix_sym[i][i];
            }


            std::ofstream outFile("Results.txt");
        if (!outFile) {
            std::cerr << "Error opening file!" << std::endl;
            return 1;
        }

        outFile << "Strong Scaling Transpose:\n";
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 9; ++j) {
                outFile << std::setw(10) << str_scl_tran[i][j] << " ";
            }
            outFile << "\n";
        }

        outFile << "\nEfficiency Transpose:\n";
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 9; ++j) {
                outFile << std::setw(10) << efficiency_tran[i][j] << " ";
            }
            outFile << "\n";
        }

        outFile << "\nStrong Scaling Symmetry:\n";
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 9; ++j) {
                outFile << std::setw(10) << str_scl_sym[i][j] << " ";
            }
            outFile << "\n";
        }

        outFile << "\nEfficiency Symmetry:\n";
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 9; ++j) {
                outFile << std::setw(10) << efficiency_sym[i][j] << " ";
            }
            outFile << "\n";
        }

        outFile << "\nWeak Scaling Transpose:\n";
        for (int i = 0; i < 5; ++i) {
            outFile << std::setw(10) << weak_scl_tran[i] << "\n";
        }

        outFile << "\nWeak Scaling Symmetry:\n";
        for (int i = 0; i < 5; ++i) {
            outFile << std::setw(10) << weak_scl_sym[i] << "\n";
        }

        outFile.close();
        cout << "Results written to Results.txt" << std::endl;


        }
         MPI_Finalize();
        
    }
    else{
        cout<<"Please, either execute this using 'mpiexec -np <var>' or  using 'mpiexec -np <var> Scalabity_Test_Transpose.txt Scalabity_Test_Symmetry.txt'."<<endl
        <<"The first updates, in the two txt files, the time required to execute the functions with <var> processor. "<<endl
        <<"The latter computes strong/weak scaling and efficiency saving the results in a txt file.";
    }
}
