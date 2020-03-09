#include "utils.h"
#include <string.h>

void printCenteredTitle(const char *title, char fill, int dim) {

    int len = strlen(title);
    int first = dim/2 + len/2;
    
    if(len >0)
        std::cout<<"\n";
    std::cout.width(first); std::cout.fill(fill); std::cout<<std::right<<title;
    std::cout.width(dim - first); std::cout<<"\n"; 
    std::cout.fill(' ');
}

bool fileExist(const char *fname) {
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    if(!dataFile)
        return false;
    return true;
}


void readBinaryFile(std::string fname, int size, dnnType** data_h, dnnType** data_d, int seek)
{
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    std::stringstream error_s;
    if (!dataFile)
    {
        error_s << "Error opening file " << fname; 
        FatalError(error_s.str());
    }

    if(seek != 0) {
        dataFile.seekg(seek*sizeof(dnnType), dataFile.cur);
    }

    int size_b = size*sizeof(dnnType);
    *data_h = new dnnType[size];
    if (!dataFile.read ((char*) *data_h, size_b)) 
    {
        error_s << "Error reading file " << fname << " with n of float: "<<size;
        error_s << " seek: "<<seek << " size: "<<size_b<<"\n";
        FatalError(error_s.str());
    }
    
    checkCuda( cudaMalloc(data_d, size_b) );
    checkCuda( cudaMemcpy(*data_d, *data_h, size_b, cudaMemcpyHostToDevice) );
}

void printDeviceVector(int size, dnnType* vec_d, bool device)
{
    dnnType *vec;
    if(device) {
        vec = new dnnType[size];
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaMemcpy(vec, vec_d, size*sizeof(dnnType), cudaMemcpyDeviceToHost));
        checkCuda(cudaDeviceSynchronize());
    } else {
        vec = vec_d;
    }
    
    for (int i = 0; i < size; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;

    if(device)
        delete [] vec;
}

int checkResult(int size, dnnType *data_d, dnnType *correct_d, bool device, int limit) {

    dnnType *data_h, *correct_h;
    const float eps = 0.02f;

    if(device) {
        data_h = new dnnType[size];
        correct_h = new dnnType[size];
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaMemcpy(data_h, data_d, size*sizeof(dnnType), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(correct_h, correct_d, size*sizeof(dnnType), cudaMemcpyDeviceToHost));
        checkCuda(cudaDeviceSynchronize());

    } else {
        data_h = data_d;
        correct_h = correct_d;
    }

    int diffs = 0;
    for(int i=0; i<size; i++) {
        if(data_h[i] != data_h[i] || correct_h[i] != correct_h[i] || //nan control
           fabs(data_h[i] - correct_h[i]) > eps) {
            diffs += 1;
            if(diffs == 1)
                std::cout<<"\n";
            if(diffs < limit)
                std::cout<<" | [ "<<i<<" ]: "<<data_h[i]<<" "<<correct_h[i]<<"\n";
        }
    }

    if(device) {
        delete [] data_h;
        delete [] correct_h;
    }

    std::cout<<" | ";
    if(diffs == 0)
        std::cout<<COL_GREENB<<"OK";
    else
        std::cout<<COL_REDB<<"Wrongs: "<<diffs;

    std::cout<<COL_END<<" ~"<<eps<<"\n";
    return diffs;
}

void resize(int size, dnnType **data)
{
    if (*data != NULL)
        checkCuda( cudaFree(*data) );
    checkCuda( cudaMalloc(data, size*sizeof(dnnType)) );
}

void matrixTranspose(cublasHandle_t handle, dnnType* srcData, dnnType* dstData, int rows, int cols) {

    dnnType *A = srcData, *clone = dstData;
    int m = rows, n= cols;
    checkCuda( cudaMemcpy(clone, A, m*n*sizeof(dnnType), cudaMemcpyDeviceToDevice));
   
    float const alpha(1.0);
    float const beta(0.0);
    checkERROR( cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, A, n, &beta, A, m, clone, m ));
}

void matrixMulAdd(  cublasHandle_t handle, dnnType* srcData, dnnType* dstData, 
                    dnnType* add_vector, int dim, dnnType mul) {

    checkCuda( cudaMemcpy(dstData, add_vector, dim*sizeof(dnnType), cudaMemcpyDeviceToDevice));
        
    dnnType alpha = mul;
    checkERROR( cublasSaxpy(handle, dim, &alpha, srcData, 1, dstData, 1));

}
