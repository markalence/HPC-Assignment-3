#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <tuple>
#include <helper_functions.h>
#include <helper_cuda.h>
//#include "cuda_func.h"PPPPPPPPP

using namespace std;

const uint MAX_SIZE = 49;
const char *fname;
int* dwd;
int* dht;
int ht;
int wd;
FILE * OUT = fopen("results.txt","w");
__constant__ float constfilter[MAX_SIZE];
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

tuple<float *, char *, uint, uint> getPicture(const char *fname, const char *exe) {
    float *image = nullptr;
    unsigned int width, height;
    char *imagepath = sdkFindFilePath(fname, exe);
    sdkLoadPGM(imagepath, &image, &width, &height);
    fprintf(OUT,"Image size is %d x %d pixels\n", width, height);
    return make_tuple(image, imagepath, height, width);
}

void getFilter(float *filter, char choice) {
    switch (choice) {
        case 'a':
            getAvg(filter);
            break;
        case 'e':
            getEdgeDet(filter);
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
    int imrow, imcol;

    for (int r = -dht[0] / 2; r <= dht[0] / 2; r++) {
        for (int c = -dwd[0] / 2; c <= dwd[0] / 2; c++) {
            imrow = idxX - r;
            imcol = idxY - c;
            if (imrow < 0 || imcol < 0 || imrow > height - 1 || imcol > width - 1) {
                val = 0.0;
            } else {
                val = tex2D(tex, imcol + 0.5f, imrow + 0.5f);
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
    int imrow, imcol;
    if (idx < height * width * sizeof(float)) {
        for (int r = -dht[0] / 2; r <= dht[0] / 2; r++) {
            for (int c = -dwd[0] / 2; c <= dwd[0] / 2; c++) {
                imrow = blockIdx.x - r;
                imcol = threadIdx.x - c;
                if (imrow < 0 || imcol < 0 || imrow > height - 1 || imcol > width - 1) {
                    val = 0.0;
                } else {
                    val = image[imcol + imrow * width];
                }
                fval = constfilter[(c + dwd[0] / 2) + (r + dht[0] / 2) * dwd[0]];
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
    int imrow, imcol;
    if (idx < height * width * sizeof(float)) {
        for (int r = -dht[0] / 2; r <= dht[0] / 2; r++) {
            for (int c = -dwd[0] / 2; c <= dwd[0] / 2; c++) {
                imrow = blockIdx.x - r;
                imcol = threadIdx.x - c;
                if (imrow < 0 || imcol < 0 || imrow > height - 1 || imcol > width - 1) {
                    val = 0.0;
                } else {
                    val = image[imcol + imrow * width];
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

    __shared__ float sharedFilter[MAX_SIZE];
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val, fval;
    float sum = 0.0;
    int imrow, imcol;
    uint tid = threadIdx.x;
//    fprintf(OUT,"%d %d\n",dwd[0],dht[0]);
    if (tid < dwd[0]*dht[0]) {
        sharedFilter[threadIdx.x] = filter[threadIdx.x];
    }
    __syncthreads();
    if (idx < height * width * sizeof(float)) {
        for (int r = -dht[0] / 2; r <= dht[0] / 2; r++) {
            for (int c = -dwd[0] / 2; c <= dwd[0] / 2; c++) {
                imrow = blockIdx.x - r;
                imcol = threadIdx.x - c;
                if (imrow < 0 || imcol < 0 || imrow > height - 1 || imcol > width - 1) {
                    val = 0.0;
                } else {
                    val = image[imcol + imrow * width];
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

void shared_gpu(const char *exe, float *filter) {

    cudaDeviceSynchronize();
    StopWatchInterface *Otimer = nullptr;
    sdkCreateTimer(&Otimer);
    sdkStartTimer(&Otimer);
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = getPicture(fname, exe);
    float output[width * height];
    unsigned int size = width * height * sizeof(float);
    unsigned int filtersize = wd * ht * sizeof(float);

    float *d_filter = nullptr;
    float *dImage = nullptr;
    float *d_result = nullptr;
    cudaMalloc((void **) &dImage, size);
    cudaMalloc((void **) &d_result, size);
    cudaMalloc((void **) &d_filter, filtersize);
    cudaMemcpy(dImage, image, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filtersize, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    convolutionSharedGPU<<<height, width>>>(dImage, d_result, d_filter, height, width, dht, dwd);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    float KernelTime = sdkGetTimerValue(&timer);
    fprintf(OUT,"Processing time for Shared: %f (ms)\n", sdkGetTimerValue(&timer));
    fprintf(OUT,"%.2f GLOPS\n",
           (width * height * ht * wd * 2) / (sdkGetTimerValue(&timer) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timer);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_result, size, cudaMemcpyDeviceToHost);
    char outputFilename[1024];
    strcpy(outputFilename, imagepath);
    strcpy(outputFilename + strlen(imagepath) - 4, "_shared.pgm");
    sdkSavePGM(outputFilename, output, width, height);
    fprintf(OUT,"Results for shared memory \n");
    cudaFree(dImage);
    cudaFree(d_filter);
    cudaFree(d_result);
    cudaDeviceSynchronize();
    sdkStopTimer(&Otimer);
    fprintf(OUT,"Overhead: %f (ms)\n", sdkGetTimerValue(&Otimer) - KernelTime);
    fprintf(OUT,"\n");
    sdkDeleteTimer(&Otimer);
}

void const_gpu(const char *exe, float *filter) {
    cudaDeviceSynchronize();
    StopWatchInterface *Otimer = nullptr;
    sdkCreateTimer(&Otimer);
    sdkStartTimer(&Otimer);
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = getPicture(fname, exe);
    float output[width * height];
    unsigned int size = width * height * sizeof(float);
    unsigned int filtersize = (wd * ht) * sizeof(float);
    float *d_filter = nullptr;
    float *dImage = nullptr;
    float *d_result = nullptr;
    cudaMalloc((void **) &dImage, size);
    cudaMalloc((void **) &d_result, size);
    cudaMalloc((void **) &d_filter, filtersize);
    cudaMemcpy(dImage, image, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constfilter, filter, filtersize);
    cudaDeviceSynchronize();
    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    convolutionConstantGPU<<<height, width>>>(dImage, d_result, height, width, dht,dwd);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    float KernelTime = sdkGetTimerValue(&timer);
    fprintf(OUT,"Processing time for Constant: %f (ms)\n", sdkGetTimerValue(&timer));
    fprintf(OUT,"%.2f GLOPS\n",
           (width * height * ht * wd * 2) / (sdkGetTimerValue(&timer) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timer);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_result, size, cudaMemcpyDeviceToHost);
    char outputFilenameNaive[1024];
    strcpy(outputFilenameNaive, imagepath);
    strcpy(outputFilenameNaive + strlen(imagepath) - 4, "_constant.pgm");
    sdkSavePGM(outputFilenameNaive, output, width, height);
    fprintf(OUT,"Results for constant memory \n");
    cudaFree(dImage);
    cudaFree(d_filter);
    cudaFree(d_result);
    cudaDeviceSynchronize();
    sdkStopTimer(&Otimer);
    fprintf(OUT,"Overhead: %f (ms)\n", sdkGetTimerValue(&Otimer) - KernelTime);
    sdkDeleteTimer(&Otimer);
    fprintf(OUT,"\n");


}

void tex_gpu(const char *exe, float *filter) {
    cudaDeviceSynchronize();
    StopWatchInterface *Otimer = nullptr;
    sdkCreateTimer(&Otimer);
    sdkStartTimer(&Otimer);
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = getPicture(fname, exe);
    float output[width * height];
    unsigned int size = width * height * sizeof(float);
    unsigned int filtersize = wd * ht * sizeof(float);
    cudaArray *dImage = nullptr;
    float *d_filter = nullptr;
    float *d_result = nullptr;
    cudaMalloc((void **) &d_result, size);
    cudaMalloc((void **) &d_filter, filtersize);
    cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(8 * sizeof(float), 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&dImage, &channelDesc, width, height);
    cudaMemcpyToArray(dImage, 0, 0, image, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filtersize, cudaMemcpyHostToDevice);
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
    convolutionTextureGPU<<<dimGrid, dimBlock, 0>>>(d_result, d_filter, width, height, dht, dwd);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    float KernelTime = sdkGetTimerValue(&timer);
    fprintf(OUT,"Processing time for Texture: %f (ms)\n", sdkGetTimerValue(&timer));
    fprintf(OUT,"%.2f GLOPS\n",
           (width * height * wd * ht * 2) / (sdkGetTimerValue(&timer) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timer);
    cudaMemcpy(output, d_result, size, cudaMemcpyDeviceToHost);
    char outputFilename[1024];
    strcpy(outputFilename, imagepath);
    strcpy(outputFilename + strlen(imagepath) - 4, "_texture.pgm");
    sdkSavePGM(outputFilename, output, width, height);
    fprintf(OUT,"Results for texture memory \n");
    cudaFree(dImage);
    cudaFree(d_filter);
    cudaFree(d_result);
    cudaDeviceSynchronize();
    sdkStopTimer(&Otimer);
    fprintf(OUT,"Overhead: %f (ms)\n", sdkGetTimerValue(&Otimer) - KernelTime);
    sdkDeleteTimer(&Otimer);
    fprintf(OUT,"Reached End of Texture\n\n");
}


void naive_gpu(const char *exe, float *filter) {

    cudaDeviceSynchronize();
    StopWatchInterface *swtime = nullptr;
    sdkCreateTimer(&swtime);
    sdkStartTimer(&swtime);
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = getPicture(fname, exe);
    float output[width * height];
    unsigned int imsize = width * height * sizeof(float);
    unsigned int filtersize = wd * ht * sizeof(float);
    float *d_filter = nullptr;
    float *d_image = nullptr;
    float *d_result = nullptr;
    cudaMalloc((void **) &d_image, imsize);
    cudaMalloc((void **) &d_result, imsize);
    cudaMalloc((void **) &d_filter, filtersize);
    cudaMemcpy(d_image, image, imsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filtersize, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    convolutionNaiveGPU<<<height, width>>>(d_image, d_result, d_filter, height, width, dht, dwd);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    float KernelTime = sdkGetTimerValue(&timer);
    fprintf(OUT,"Processing time for Naive: %f (ms)\n", sdkGetTimerValue(&timer));
    fprintf(OUT,"%.2f GLOPS\n",
           (width * height * wd * ht * 2) / (sdkGetTimerValue(&timer) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timer);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_result, imsize, cudaMemcpyDeviceToHost);
    char output_naive[1024];
    strcpy(output_naive, imagepath);
    strcpy(output_naive + strlen(imagepath) - 4, "_naive.pgm");
    sdkSavePGM(output_naive, output, width, height);
    fprintf(OUT, "Results for naive method\n");
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_result);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    fprintf(OUT,"Overhead: %f (ms)\n", sdkGetTimerValue(&swtime) - KernelTime);
    sdkDeleteTimer(&swtime);
    fprintf(OUT,"\n");

}
void cpu_conv(const char *exe, float *filter) {
    cudaDeviceSynchronize();
    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = getPicture(fname, exe);
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
    fprintf(OUT,"Processing time for CPU: %f (ms)\n", sdkGetTimerValue(&timerCPU));
    fprintf(OUT,"%.2f GLOPS\n",
           (float) (width * height * ht * wd * 2) / (sdkGetTimerValue(&timerCPU) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timerCPU);
    char outfile[1024];
    strcpy(outfile, imagepath);
    strcpy(outfile + strlen(imagepath) - 4, "_cpu.pgm");
    sdkSavePGM(outfile, outputCPU, width, height);
    fprintf(OUT,"Results for CPU");
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    fprintf(OUT,"Overhead: %f (ms)\n", sdkGetTimerValue(&timer) - KernelTime);
    sdkDeleteTimer(&timer);
    fprintf(OUT,"\n");

}


int main(int argc, char *argv[]) {


    const char *prog = argv[0];

    fname = "home-mscluster/malence/data/pipe.pgm";
    char filters[] = {'a','e','s'};
    for(int i = 3; i <= 9; i+=2) {
        for(int j = 3; j <= 9; j+=2) {
            char filterchoice = *argv[2];
            ht = i;
            wd = j;
            cout << ht << " " << wd << endl;
            cudaMalloc((void **) &dht, sizeof(int));
            cudaMalloc((void **) &dwd, sizeof(int));
            cudaMemcpy(dht, &ht, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dwd, &wd, sizeof(int), cudaMemcpyHostToDevice);
            float filter[ht * wd];

            fprintf(OUT, "\n\n-------------------- %d x %d filter --------------------\n",ht,wd);

            getFilter(filter, filterchoice);
            cpu_conv(prog, filter);
            naive_gpu(prog, filter);
            const_gpu(prog, filter);
            tex_gpu(prog, filter);
            shared_gpu(prog, filter);
            fprintf(OUT, "\n-------------------- %d x %d filter --------------------",ht,wd);

            cudaFree(dht);
            cudaFree(dwd);
        }
    }
    return 0;
}