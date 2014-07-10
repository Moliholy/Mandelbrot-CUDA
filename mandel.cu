#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <time.h>

int imageRows;
int imageColumns;
double imagePixelSize;
double centerPointReal;
double centerPointImaginary;
int maximumIterations;

void sequentialMandelbrot(int* matrix) {
        //initial coordinate (pixel (0, 0))
        double startReal = -((double) imageColumns * imagePixelSize) / 2.0
                        + centerPointReal;
        double startImaginary = -((double) imageRows * imagePixelSize) / 2.0
                        + centerPointImaginary;

        double transformation = 255.0 / (double) maximumIterations;

        //same as Wikipedia code
        int k;
        for (k = 0; k < imageRows * imageColumns; ++k) {
                int i = k / imageColumns;
                int j = k % imageColumns;

                double x0 = startReal + j * imagePixelSize;
                double y0 = startImaginary + i * imagePixelSize;
                double x = 0.0;
                double y = 0.0;
                int iteration = 0;

                while (x * x + y * y < 4.0 && iteration < maximumIterations) {
                        double xtemp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = xtemp;
                        iteration++;
                }
                //changing the color
                matrix[i * imageColumns + j] = iteration * transformation;
        }
}

__global__
void parallelMandelbrot(int* matrix, int imageRows, int imageColumns,
                double imagePixelSize, double centerPointReal,
                double centerPointImaginary, int maximumIterations) {

        //same than wikipedia code
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int pos = i * imageColumns + j;
        if (pos < imageRows * imageColumns) {

                //initial coordinate (pixel (0, 0))
                double startReal = -((double) imageColumns * imagePixelSize) / 2.0
                                + centerPointReal;
                double startImaginary = -((double) imageRows * imagePixelSize) / 2.0
                                + centerPointImaginary;

                double transformation = 255.0 / (double) maximumIterations;

                double x0 = startReal + j * imagePixelSize;
                double y0 = startImaginary + i * imagePixelSize;
                double x = 0.0;
                double y = 0.0;
                int iteration = 0;

                while (x * x + y * y < 4.0 && iteration < maximumIterations) {
                        double xtemp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = xtemp;
                        iteration++;
                }
                //changing the color
                matrix[pos] = iteration * transformation;
        }
}

/**
 * Usage: ./mandel-magic imageRows imageColumns pixelSize centerPointReal
 *  centerPointImaginary maximumIterations
 */
int main(int argc, char *argv[]) {
        if (argc < 7) {
                printf("No enough arguments.");
                return -1;
        }

        imageRows = atoi(argv[1]);
        imageColumns = atoi(argv[2]);
        imagePixelSize = atof(argv[3]);
        centerPointReal = atof(argv[4]);
        centerPointImaginary = atof(argv[5]);
        maximumIterations = atoi(argv[6]);

        int* matrix = (int*) malloc(imageRows * imageColumns * sizeof(int));

        //sequential execution
        clock_t t = clock();
        sequentialMandelbrot(matrix);
        float sequentialExecutionTime = ((double) (clock() - t))
                        / ((double) (CLOCKS_PER_SEC));

        //starting parallel execution
        t = clock();
        int* d_matrix;

        //allocating memory
        cudaMalloc(&d_matrix, imageRows * imageColumns * sizeof(int));

        //dimensions
        dim3 threadblock(16, 16);
        dim3 grid(1 + imageColumns / threadblock.x, 1 + imageRows / threadblock.y);

        //calling function
        parallelMandelbrot<<<grid, threadblock>>>(d_matrix, imageRows, imageColumns,
                        imagePixelSize, centerPointReal, centerPointImaginary,
                        maximumIterations);

        //synchronizing
        cudaDeviceSynchronize();

        //once the function has been called I copy the result in matrix
        cudaMemcpy(matrix, d_matrix, imageRows * imageColumns * sizeof(int),
                        cudaMemcpyDeviceToHost);

        double parallelExecutionTime = ((double) (clock() - t))
                        / ((double) (CLOCKS_PER_SEC));

        printf("%d;%f;%f\n", imageRows, sequentialExecutionTime,
                        parallelExecutionTime);

        cudaFree(d_matrix);
        free(matrix);

        return 0;
}
