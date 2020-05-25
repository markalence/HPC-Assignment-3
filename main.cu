#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <tuple>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <ldap.h>

using namespace std;

int g = 0;
const uint MAX_SIZE = 100;
const char *filename;
int *dwd;
int *dht;
int ht;
int wd;
string filterchoice;
bool shouldprint = true;
FILE *OUT = fopen("results.txt", "w");
__constant__ float constfilter[MAX_SIZE];
texture<float, 2, cudaReadModeElementType> tex;


void getEdgeDet(float *filter) {
    for (int i = 0; i < ht; ++i) {
        for (int j = 0; j < wd; ++j) {
            float v = wd / 2 - j;
            filter[i + i * j] = (i == ht / 2) ? v : 2 * v;
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
            filter[i * wd + j] = -1;
        }
    }
    filter[ht / 2 * wd / 2] = (float) (ht * wd);
}

tuple<float *, char *, uint, uint> getImg(const char *filename, const char *exe) {
    float *image = nullptr;
    unsigned int width, height;
    char *imagepath = sdkFindFilePath(filename, exe);
    sdkLoadPGM(imagepath, &image, &width, &height);
    if (shouldprint) {
        fprintf(OUT, "Image size is %d x %d pixels\n\n", width, height);
        shouldprint = false;
    }
    return make_tuple(image, imagepath, height, width);
}

void getFilter(float *filter, string choice) {
    switch (choice[0]) {
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


__global__ void convolutionTextureGPU(float *output, float *filter, int width, int height, int *dht, int *dwd) {
    uint idxX = threadIdx.x + blockIdx.x * blockDim.x;
    uint idxY = threadIdx.y + blockIdx.y * blockDim.y;
    float val, fval;
    float sum = 0.0;
    int imrow, imcol;

    for (int r = -dht[0] / 2; r <= dht[0] / 2; r++) {
        for (int c = -dwd[0] / 2; c <= dwd[0] / 2; c++) {
            imrow = idxX - r;
            imcol = idxY - c;
            if (!(imrow < 0 || imcol < 0 || imrow > height - 1 || imcol > width - 1)) {
                val = tex2D(tex, imcol + 0.5f, imrow + 0.5f);
                fval = filter[(c + dwd[0] / 2) + (r + dht[0] / 2) * dwd[0]];
                sum += val * fval;
            }

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
                    if (!((i - r) < 0 || (i - r) > height - 1 || (j - c) < 0 || (j - c) > width - 1)) {
                        val = image[(j - c) + (i - r) * width];
                        fval = filter[(c + wd / 2) + (r + ht / 2) * wd];
                        sum += val * fval;
                        g++;
                    }
                }

            }
            if (sum < 0.0) sum = 0.0;
            if (sum > 1.0) sum = 1.0;
            output[j + i * width] = sum;
        }
    }
}


__global__ void convolutionConstantGPU(const float *image, float *output, uint height, uint width, int *dht, int *dwd) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val, fval;
    float sum = 0.0;
    int imrow, imcol;
    if (idx < height * width * sizeof(float)) {
        for (int r = -dht[0] / 2; r <= dht[0] / 2; r++) {
            for (int c = -dwd[0] / 2; c <= dwd[0] / 2; c++) {
                imrow = blockIdx.x - r;
                imcol = threadIdx.x - c;
                if (!(imrow < 0 || imcol < 0 || imrow > height - 1 || imcol > width - 1)) {
                    val = image[imcol + imrow * width];
                    fval = constfilter[(c + dwd[0] / 2) + (r + dht[0] / 2) * dwd[0]];
                    sum += val * fval;
                }

            }
        }
        if (sum < 0) sum = 0.0;
        if (sum > 1) sum = 1.0;
        output[idx] = sum;
    }
}

__global__ void
convolutionNaiveGPU(float *image, float *output, float *filter, uint height, uint width, int *dht, int *dwd) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val, fval;
    float sum = 0.0;
    int imrow, imcol;
    if (idx < height * width * sizeof(float)) {
        for (int r = -dht[0] / 2; r <= dht[0] / 2; r++) {
            for (int c = -dwd[0] / 2; c <= dwd[0] / 2; c++) {
                imrow = blockIdx.x - r;
                imcol = threadIdx.x - c;
                if (!(imrow < 0 || imcol < 0 || imrow > height - 1 || imcol > width - 1)) {
                    val = image[imcol + imrow * width];
                    fval = constfilter[(c + dwd[0] / 2) + (r + dht[0] / 2) * dwd[0]];
                    sum += val * fval;
                }
            }
        }
        if (sum < 0) sum = 0.0;
        if (sum > 1) sum = 1.0;
        output[idx] = sum;
    }
}

__global__ void
convolutionSharedGPU(const float *image, float *output, const float *filter, uint height, uint width, int *dht,
                     int *dwd) {

    __shared__ float sharedFilter[MAX_SIZE];
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val, fval;
    float sum = 0.0;
    int imrow, imcol;
    uint tid = threadIdx.x;
//    fprintf(OUT,"%d %d\n",dwd[0],dht[0]);
    if (tid < dwd[0] * dht[0]) {
        sharedFilter[threadIdx.x] = filter[threadIdx.x];
    }
    __syncthreads();
    if (idx < height * width * sizeof(float)) {
        for (int r = -dht[0] / 2; r <= dht[0] / 2; r++) {
            for (int c = -dwd[0] / 2; c <= dwd[0] / 2; c++) {
                imrow = blockIdx.x - r;
                imcol = threadIdx.x - c;
                if (!(imrow < 0 || imcol < 0 || imrow > height - 1 || imcol > width - 1)) {
                    val = image[imcol + imrow * width];
                    fval = sharedFilter[(c + dwd[0] / 2) + (r + dht[0] / 2) * dwd[0]];
                    sum += val * fval;
                }
            }
        }

        if (sum < 0) sum = 0.0;
        if (sum > 1) sum = 1.0;
        output[idx] = sum;
    }
}

void shared_gpu(const char *exe, float *filter) {

    cudaDeviceSynchronize();
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = getImg(filename, exe);
    float output[width * height];
    unsigned int size = width * height * sizeof(float);
    unsigned int filtersize = wd * ht * sizeof(float);

    float *d_filter = nullptr;
    float *d_image = nullptr;
    float *d_result = nullptr;
    cudaMalloc((void **) &d_image, size);
    cudaMalloc((void **) &d_result, size);
    cudaMalloc((void **) &d_filter, filtersize);
    cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filtersize, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    convolutionSharedGPU<<<height, width>>>(d_image, d_result, d_filter, height, width, dht, dwd);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    fprintf(OUT, "Processing time for Shared: %f (ms)\n", sdkGetTimerValue(&timer));
    fprintf(OUT, "%.2f GFLOPS\n",
            ((2 * ht * wd - 1) * height * width) / (sdkGetTimerValue(&timer) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timer);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_result, size, cudaMemcpyDeviceToHost);
    char outputfile[1024];
    strcpy(outputfile, imagepath);
    string imgstr(outputfile);
    imgstr = imgstr + "_shared_" + to_string(ht) + "by" + to_string(wd) + "_" + filterchoice + ".pgm";
    strcpy(outputfile + strlen(imagepath) - 4, imgstr.c_str());
    sdkSavePGM(outputfile, output, width, height);
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_result);
    cudaDeviceSynchronize();
}

void const_gpu(const char *exe, float *filter) {
    cudaDeviceSynchronize();
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = getImg(filename, exe);
    float output[width * height];
    unsigned int size = width * height * sizeof(float);
    unsigned int filtersize = (wd * ht) * sizeof(float);
    float *d_filter = nullptr;
    float *d_image = nullptr;
    float *d_result = nullptr;
    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    cudaMalloc((void **) &d_image, size);
    cudaMalloc((void **) &d_result, size);
    cudaMalloc((void **) &d_filter, filtersize);
    cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constfilter, filter, filtersize);
    cudaDeviceSynchronize();
    convolutionConstantGPU<<<height, width>>>(d_image, d_result, height, width, dht, dwd);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    fprintf(OUT, "Processing time for Constant: %f (ms)\n", sdkGetTimerValue(&timer));
    fprintf(OUT, "%.2f GFLOPS\n",
            ((2 * ht * wd - 1) * height * width) / (sdkGetTimerValue(&timer) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timer);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_result, size, cudaMemcpyDeviceToHost);
    char outputfile[1024];
    strcpy(outputfile, imagepath);
    string imgstr = "_const_" + to_string(ht) + "by" + to_string(wd) + "_" + filterchoice + ".pgm";
    strcpy(outputfile + strlen(imagepath) - 4, imgstr.c_str());
    sdkSavePGM(outputfile, output, width, height);
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_result);
    cudaDeviceSynchronize();
    fprintf(OUT, "\n");


}

void tex_gpu(const char *exe, float *filter) {
    cudaDeviceSynchronize();
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = getImg(filename, exe);
    float output[width * height];
    unsigned int size = width * height * sizeof(float);
    unsigned int filtersize = wd * ht * sizeof(float);
    cudaArray *d_image = nullptr;
    float *d_filter = nullptr;
    float *d_result = nullptr;
    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    cudaMalloc((void **) &d_result, size);
    cudaMalloc((void **) &d_filter, filtersize);
    cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(8 * sizeof(float), 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&d_image, &channelDesc, width, height);
    cudaMemcpyToArray(d_image, 0, 0, image, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filtersize, cudaMemcpyHostToDevice);
    tex.addressMode[0] = cudaAddressModeBorder;
    tex.addressMode[1] = cudaAddressModeBorder;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = false;
    cudaBindTextureToArray(tex, d_image, channelDesc);
    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    cudaDeviceSynchronize();

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    convolutionTextureGPU<<<dimGrid, dimBlock, 0>>>(d_result, d_filter, width, height, dht, dwd);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    fprintf(OUT, "Processing time for Texture: %f (ms)\n", sdkGetTimerValue(&timer));
    fprintf(OUT, "%.2f GFLOPS\n\n",
            (width * height * wd * ht * 2) / (sdkGetTimerValue(&timer) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timer);
    cudaMemcpy(output, d_result, size, cudaMemcpyDeviceToHost);
    char outputfile[1024];
    strcpy(outputfile, imagepath);
    string imgstr = "_tex_" + to_string(ht) + "by" + to_string(wd) + "_" + filterchoice + ".pgm";
    strcpy(outputfile + strlen(imagepath) - 4, imgstr.c_str());
    sdkSavePGM(outputfile, output, width, height);
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_result);
    cudaDeviceSynchronize();
}


void naive_gpu(const char *exe, float *filter) {

    cudaDeviceSynchronize();
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = getImg(filename, exe);
    float output[width * height];
    unsigned int imsize = width * height * sizeof(float);
    unsigned int filtersize = wd * ht * sizeof(float);
    float *d_filter = nullptr;
    float *d_image = nullptr;
    float *d_result = nullptr;
    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    cudaMalloc((void **) &d_image, imsize);
    cudaMalloc((void **) &d_result, imsize);
    cudaMalloc((void **) &d_filter, filtersize);
    cudaMemcpy(d_image, image, imsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filtersize, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    convolutionNaiveGPU<<<height, width>>>(d_image, d_result, d_filter, height, width, dht, dwd);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    fprintf(OUT, "Processing time for Naive: %f (ms)\n", sdkGetTimerValue(&timer));
    fprintf(OUT, "%.2f GFLOPS\n",
            (width * height * wd * ht * 2) / (sdkGetTimerValue(&timer) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timer);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_result, imsize, cudaMemcpyDeviceToHost);
    char outputfile[1024];
    strcpy(outputfile, imagepath);
    string imgstr = "_naive_" + to_string(ht) + "by" + to_string(wd) + "_" + filterchoice + ".pgm";
    strcpy(outputfile + strlen(imagepath) - 4, imgstr.c_str());
    sdkSavePGM(outputfile, output, width, height);
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_result);
    cudaDeviceSynchronize();
    fprintf(OUT, "\n");

}

void cpu_conv(const char *exe, float *filter) {
    cudaDeviceSynchronize();
    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    float *image = nullptr;
    char *imagepath = nullptr;
    unsigned int width, height;
    tie(image, imagepath, height, width) = getImg(filename, exe);
    float outputCPU[width * height];
    cudaDeviceSynchronize();
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    convolveCPU(image, outputCPU, filter, width, height);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    fprintf(OUT, "Processing time for CPU: %f (ms)\n", sdkGetTimerValue(&timer));
    fprintf(OUT, "%.2f GFLOPS\n",
            (float) ((2 * ht * wd - 1) * height * width) / (sdkGetTimerValue(&timer) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timer);
    char outputfile[1024];
    strcpy(outputfile, imagepath);
    string imgstr = "_cpu_" + to_string(ht) + "by" + to_string(wd) + "_" + filterchoice + ".pgm";
    strcpy(outputfile + strlen(imagepath) - 4, imgstr.c_str());
    sdkSavePGM(outputfile, outputCPU, width, height);
    cudaDeviceSynchronize();
    fprintf(OUT, "\n");

}


int main(int argc, char *argv[]) {


    const char *prog = argv[0];

    string names[] = {"home-mscluster/malence/hpc3/data/image21.pgm",
                      "home-mscluster/malence/hpc3/data/lena_bw.pgm",
                      "home-mscluster/malence/hpc3/data/man.pgm",
                      "home-mscluster/malence/hpc3/data/mandrill.pgm",
                      "home-mscluster/malence/hpc3/data/ref_rotated.pgm"};


    string filters[] = {"avg", "edge", "sharp"};
    for (int g = 0; g < 3; ++g) {
        filterchoice = filters[g];
        for (int h = 0; h < 5; ++h) {
            filename = names[h].c_str();
            for (int i = 3; i <= 9; i += 2) {
                for (int j = 3; j <= 9; j += 2) {
                    ht = i;
                    wd = j;
                    cout << ht << " " << wd << endl;
                    cudaMalloc((void **) &dht, sizeof(int));
                    cudaMalloc((void **) &dwd, sizeof(int));
                    cudaMemcpy(dht, &ht, sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpy(dwd, &wd, sizeof(int), cudaMemcpyHostToDevice);
                    float filter[ht * wd];
                    shouldprint = true;
                    fprintf(OUT, "\n\n-------------------- %d x %d filter --------------------\n", ht, wd);

                    getFilter(filter, filterchoice);
                    cpu_conv(prog, filter);
                    naive_gpu(prog, filter);
                    const_gpu(prog, filter);
                    tex_gpu(prog, filter);
                    shared_gpu(prog, filter);
                    fprintf(OUT, "-------------------- %d x %d filter --------------------", ht, wd);

                    cudaFree(dht);
                    cudaFree(dwd);
                }
            }
        }
    }
//    filterchoice = "avg";
//    ht = 23;
//    wd = 23;
//    float filter[ht * wd];
//    filename = names[0].c_str();
//    getFilter(filter, filterchoice);
//    cpu_conv(prog,filter);
//    return 0;
}