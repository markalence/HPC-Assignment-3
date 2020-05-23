#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <tuple>
#include <helper_functions.h>
#include <helper_cuda.h>
//#include "cuda_func.h"PPPPPPPPP

using namespace std;

const uint MAX_FILTER_SIZE = 49;
const char *fname;
int* dwd;
int* dht;
int ht;
int wd;
__constant__ float ConstFilter[MAX_FILTER_SIZE];
texture<float, 2, cudaReadModeElementType> tex;


void getEdgeDet(float *filter) {
    for (int i = 0; i < ht; ++i) {
        for (int j = 0; j < wd; ++j) {
            float v = wd / 2 - j;
            filter[i + i*j] = (i == ht / 2) ? v : 2 * v;
        }
    }
}

void getAvg(float *filter) {
    int n = ht * wd;
    float val = 1.0 / n;
    for (int i = 0; i < n; i++) {
        filter[i] = val;
    }
}

void getSharpen(float *filter) {
    for (int i = 0; i < ht; ++i) {
        for (int j = 0; j < wd; ++j) {
            filter[i*wd + j] = -1;
        }
    }
    filter[ht / 2 * wd / 2] = (float) (ht * wd);
}

tuple<float *, char *, uint, uint> loadImage(const char *fname, const char *exe) {
    printf("\n\n");
    float *image = nullptr;
    unsigned int width, height;
    char *imagepath = sdkFindFilePath(fname, exe);
    sdkLoadPGM(imagepath, &image, &width, &height);
    printf("'%s', %d x %d pixels\n", fname, width, height);
    return make_tuple(image, imagepath, height, width);
}

void getFilter(float *filter, char choice) {
    switch (choice) {
        case 'b':
            getAvg(filter);
            break;
        case 's':
            getSharpen(filter);
            break;
    }

}


__global__ void convolutionTextureGPU(float *output, float *filter, int width, int height, int* dht, int* dwd) {
    uint idxX = threadIdx.x + blockIdx.x * blockDim.x;
    uint idxY = threadIdx.y + blockIdx.y * blockDim.y;
    float val, fval;
    float sum = 0.0;
    int imRow, imCol;

    for (int r = -dht[0] / 2; r <= dht[0] / 2; r++) {
        for (int c = -dwd[0] / 2; c <= dwd[0] / 2; c++) {
            imRow = idxX - r;
            imCol = idxY - c;
            if (imRow < 0 || imCol < 0 || imRow > height - 1 || imCol > width - 1) {
                val = 0.0;
            } else {
                val = tex2D(tex, imCol + 0.5f, imRow + 0.5f);
            }
            fval = filter[(c + dwd[0] / 2) + (r + dht[0] / 2) * dwd[0]];
            sum += val * fval;
        }
    }

    if (sum < 0) sum = 0.0;
    if (sum > 1) sum = 1.0;
    output[idxY + width * idxX] = sum;
}


void convolveCPU(const float *image, float *output, const float *filter, unsigned int width, unsigned int height) {
    float sum;
    float val, fval;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            sum = 0.0;
            for (int r = -ht / 2; r <= ht / 2; r++) {
                for (int c = -wd / 2; c <= wd / 2; c++) {
                    if ((i - r) < 0 || (i - r) > height - 1 || (j - c) < 0 || (j - c) > width - 1) {
                        val = 0.0;
                    } else {
                        val = image[(j - c) + (i - r) * width];
                    }
                    fval = filter[(c + wd / 2) + (r + ht / 2) * wd];
                    sum += val * fval;
                }

            }
            if (sum < 0.0) sum = 0.0;
            if (sum > 1.0) sum = 1.0;
            output[j + i * width] = sum;
        }
    }
}


__global__ void convolutionConstantGPU(const float *image, float *output, uint height, uint width, int* dht, int* dwd) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val, fval;
    float sum = 0.0;
    int imRow, imCol;
    if (idx < height * width * sizeof(float)) {
        for (int r = -dht[0] / 2; r <= dht[0] / 2; r++) {
            for (int c = -dwd[0] / 2; c <= dwd[0] / 2; c++) {
                imRow = blockIdx.x - r;
                imCol = threadIdx.x - c;
                if (imRow < 0 || imCol < 0 || imRow > height - 1 || imCol > width - 1) {
                    val = 0.0;
                } else {
                    val = image[imCol + imRow * width];
                }
                fval = ConstFilter[(c + dwd[0] / 2) + (r + dht[0] / 2) * dwd[0]];
                sum += val * fval;
            }
        }
        if (sum < 0) sum = 0.0;
        if (sum > 1) sum = 1.0;
        output[idx] = sum;
    }
}

__global__ void
convolutionNaiveGPU(float *image, float *output, float *filter, uint height, uint width, int* dht, int* dwd) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val, fval;
    float sum = 0.0;
    int imRow, imCol;
    if (idx < height * width * sizeof(float)) {
        for (int r = -dht[0] / 2; r <= dht[0] / 2; r++) {
            for (int c = -dwd[0] / 2; c <= dwd[0] / 2; c++) {
                imRow = blockIdx.x - r;
                imCol = threadIdx.x - c;
                if (imRow < 0 || imCol < 0 || imRow > height - 1 || imCol > width - 1) {
                    val = 0.0;
                } else {
                    val = image[imCol + imRow * width];
                }
                fval = filter[(c + dwd[0] / 2) + (r + dht[0] / 2) * dwd[0]];
                sum += val * fval;
            }
        }
        if (sum < 0) sum = 0.0;
        if (sum > 1) sum = 1.0;
        output[idx] = sum;
    }
}

__global__ void
convolutionSharedGPU(const float *image, float *output, const float *filter, uint height, uint width, int* dht, int* dwd) {

    __shared__ float sharedFilter[MAX_FILTER_SIZE];
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val, fval;
    float sum = 0.0;
    int imRow, imCol;
    uint tid = threadIdx.x;
//    printf("%d %d\n",dwd[0],dht[0]);
    if (tid < dwd[0]*dht[0]) {
        sharedFilter[threadIdx.x] = filter[threadIdx.x];
    }
    __syncthreads();
    if (idx < height * width * sizeof(float)) {
        for (int r = -dht[0] / 2; r <= dht[0] / 2; r++) {
            for (int c = -dwd[0] / 2; c <= dwd[0] / 2; c++) {
                imRow = blockIdx.x - r;
                imCol = threadIdx.x - c;
                if (imRow < 0 || imCol < 0 || imRow > height - 1 || imCol > width - 1) {
                    val = 0.0;
                } else {
                    val = image[imCol + imRow * width];
                }
                fval = sharedFilter[(c + dwd[0] / 2) + (r + dht[0] / 2) * dwd[0]];

                sum += val * fval;
            }
        }

        if (sum < 0) sum = 0.0;
        if (sum > 1) sum = 1.0;
        output[idx] = sum;
    }
}

void SharedGPU(const char *exe, float *filter) {

    cudaDeviceSynchronize();
    StopWatchInterface *Otimer = nullptr;
    sdkCreateTimer(&Otimer);
    sdkStartTimer(&Otimer);
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = loadImage(fname, exe);
    float output[width * height];
    unsigned int size = width * height * sizeof(float);
    unsigned int filtersize = wd * ht * sizeof(float);

    float *dFilter = nullptr;
    float *dImage = nullptr;
    float *dResult = nullptr;
    cudaMalloc((void **) &dImage, size);
    cudaMalloc((void **) &dResult, size);
    cudaMalloc((void **) &dFilter, filtersize);
    cudaMemcpy(dImage, image, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dFilter, filter, filtersize, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    convolutionSharedGPU<<<height, width>>>(dImage, dResult, dFilter, height, width, dht, dwd);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    float KernelTime = sdkGetTimerValue(&timer);
    printf("Processing time for Shared: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f GLOPS\n",
           (width * height * ht * wd * 2) / (sdkGetTimerValue(&timer) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timer);
    cudaDeviceSynchronize();
    cudaMemcpy(output, dResult, size, cudaMemcpyDeviceToHost);
    char outputFilename[1024];
    strcpy(outputFilename, imagepath);
    strcpy(outputFilename + strlen(imagepath) - 4, "_shared_out.pgm");
    sdkSavePGM(outputFilename, output, width, height);
    printf("Wrote '%s'\n", outputFilename);
    cudaFree(dImage);
    cudaFree(dFilter);
    cudaFree(dResult);
    cudaDeviceSynchronize();
    sdkStopTimer(&Otimer);
    printf("Overhead: %f (ms)\n", sdkGetTimerValue(&Otimer) - KernelTime);
    sdkDeleteTimer(&Otimer);
}

void ConstantGPU(const char *exe, float *filter) {
    cudaDeviceSynchronize();
    StopWatchInterface *Otimer = nullptr;
    sdkCreateTimer(&Otimer);
    sdkStartTimer(&Otimer);
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = loadImage(fname, exe);
    float output[width * height];
    unsigned int size = width * height * sizeof(float);
    unsigned int filtersize = (wd * ht) * sizeof(float);
    float *dFilter = nullptr;
    float *dImage = nullptr;
    float *dResult = nullptr;
    cudaMalloc((void **) &dImage, size);
    cudaMalloc((void **) &dResult, size);
    cudaMalloc((void **) &dFilter, filtersize);
    cudaMemcpy(dImage, image, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(ConstFilter, filter, filtersize);
    cudaDeviceSynchronize();
    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    convolutionConstantGPU<<<height, width>>>(dImage, dResult, height, width, dht,dwd);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    float KernelTime = sdkGetTimerValue(&timer);
    printf("Processing time for Constant: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f GLOPS\n",
           (width * height * ht * wd * 2) / (sdkGetTimerValue(&timer) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timer);
    cudaDeviceSynchronize();
    cudaMemcpy(output, dResult, size, cudaMemcpyDeviceToHost);
    char outputFilenameNaive[1024];
    strcpy(outputFilenameNaive, imagepath);
    strcpy(outputFilenameNaive + strlen(imagepath) - 4, "_constant_out.pgm");
    sdkSavePGM(outputFilenameNaive, output, width, height);
    printf("Wrote '%s'\n", outputFilenameNaive);
    cudaFree(dImage);
    cudaFree(dFilter);
    cudaFree(dResult);
    cudaDeviceSynchronize();
    sdkStopTimer(&Otimer);
    printf("Overhead: %f (ms)\n", sdkGetTimerValue(&Otimer) - KernelTime);
    sdkDeleteTimer(&Otimer);

}

void TextureGPU(const char *exe, float *filter) {
    cudaDeviceSynchronize();
    StopWatchInterface *Otimer = nullptr;
    sdkCreateTimer(&Otimer);
    sdkStartTimer(&Otimer);
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = loadImage(fname, exe);
    float output[width * height];
    unsigned int size = width * height * sizeof(float);
    unsigned int filtersize = wd * ht * sizeof(float);
    cudaArray *dImage = nullptr;
    float *dFilter = nullptr;
    float *dResult = nullptr;
    cudaMalloc((void **) &dResult, size);
    cudaMalloc((void **) &dFilter, filtersize);
    cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(8 * sizeof(float), 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&dImage, &channelDesc, width, height);
    cudaMemcpyToArray(dImage, 0, 0, image, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dFilter, filter, filtersize, cudaMemcpyHostToDevice);
    tex.addressMode[0] = cudaAddressModeBorder;
    tex.addressMode[1] = cudaAddressModeBorder;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = false;
    cudaBindTextureToArray(tex, dImage, channelDesc);
    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    cudaDeviceSynchronize();
    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    convolutionTextureGPU<<<dimGrid, dimBlock, 0>>>(dResult, dFilter, width, height, dht, dwd);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    float KernelTime = sdkGetTimerValue(&timer);
    printf("Processing time for Texture: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f GLOPS\n",
           (width * height * wd * ht * 2) / (sdkGetTimerValue(&timer) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timer);
    cudaMemcpy(output, dResult, size, cudaMemcpyDeviceToHost);
    char outputFilename[1024];
    strcpy(outputFilename, imagepath);
    strcpy(outputFilename + strlen(imagepath) - 4, "_texture_out.pgm");
    sdkSavePGM(outputFilename, output, width, height);
    printf("Wrote '%s'\n", outputFilename);
    cudaFree(dImage);
    cudaFree(dFilter);
    cudaFree(dResult);
    cudaDeviceSynchronize();
    sdkStopTimer(&Otimer);
    printf("Overhead: %f (ms)\n", sdkGetTimerValue(&Otimer) - KernelTime);
    sdkDeleteTimer(&Otimer);
    printf("Reached End of Texture\n\n");
}


void NaiveGPU(const char *exe, float *filter) {

    cudaDeviceSynchronize();
    StopWatchInterface *Otimer = nullptr;
    sdkCreateTimer(&Otimer);
    sdkStartTimer(&Otimer);
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = loadImage(fname, exe);
    float output[width * height];
    unsigned int size = width * height * sizeof(float);
    unsigned int filtersize = wd * ht * sizeof(float);
    float *dFilter = nullptr;
    float *dImage = nullptr;
    float *dResult = nullptr;
    cudaMalloc((void **) &dImage, size);
    cudaMalloc((void **) &dResult, size);
    cudaMalloc((void **) &dFilter, filtersize);
    cudaMemcpy(dImage, image, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dFilter, filter, filtersize, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    convolutionNaiveGPU<<<height, width>>>(dImage, dResult, dFilter, height, width, dht,dwd);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    float KernelTime = sdkGetTimerValue(&timer);
    printf("Processing time for Naive: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f GLOPS\n",
           (width * height * wd * ht * 2) / (sdkGetTimerValue(&timer) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timer);
    cudaDeviceSynchronize();
    cudaMemcpy(output, dResult, size, cudaMemcpyDeviceToHost);
    char outputFilenameNaive[1024];
    strcpy(outputFilenameNaive, imagepath);
    strcpy(outputFilenameNaive + strlen(imagepath) - 4, "_naive_out.pgm");
    sdkSavePGM(outputFilenameNaive, output, width, height);
    printf("Wrote '%s'\n", outputFilenameNaive);
    cudaFree(dImage);
    cudaFree(dFilter);
    cudaFree(dResult);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    printf("Overhead: %f (ms)\n", sdkGetTimerValue(&Otimer) - KernelTime);
    sdkDeleteTimer(&Otimer);
}
void CPU(const char *exe, float *filter) {
    cudaDeviceSynchronize();
    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = loadImage(fname, exe);
    float outputCPU[width * height];
    cudaDeviceSynchronize();
    StopWatchInterface *timerCPU = nullptr;
    sdkCreateTimer(&timerCPU);
    sdkStartTimer(&timerCPU);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    convolveCPU(image, outputCPU, filter, width, height);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    sdkStopTimer(&timerCPU);
    float KernelTime = sdkGetTimerValue(&timerCPU);
    printf("Processing time for CPU: %f (ms)\n", sdkGetTimerValue(&timerCPU));
    printf("%.2f GLOPS\n",
           (float) (width * height * ht * wd * 2) / (sdkGetTimerValue(&timerCPU) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timerCPU);
    char outputFilename[1024];
    strcpy(outputFilename, imagepath);
    strcpy(outputFilename + strlen(imagepath) - 4, "_out.pgm");
    sdkSavePGM(outputFilename, outputCPU, width, height);
    printf("Wrote '%s'\n", outputFilename);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    printf("Overhead: %f (ms)\n", sdkGetTimerValue(&timer) - KernelTime);
    sdkDeleteTimer(&timer);
}


int main(int argc, char *argv[]) {

    const char *exe = argv[0];
    fname = argv[1];
    char filterchoice = *argv[2];
    ht = atoi(argv[3]);
    wd = atoi(argv[4]);
    cout << ht << " " << wd << endl;
    cudaMalloc((void**)&dht, sizeof(int));
    cudaMalloc((void**)&dwd, sizeof(int));
    cudaMemcpy(dht,&ht,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dwd,&wd,sizeof(int),cudaMemcpyHostToDevice);
    float filter[ht * wd];

    cout << "Using " << ht << "x" << wd << " filter" << endl;

    getFilter(filter, filterchoice);
    CPU(exe, filter);
    NaiveGPU(exe, filter);
    ConstantGPU(exe, filter);
    TextureGPU(exe, filter);
    SharedGPU(exe, filter);
    return 0;
}