#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP          
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <emmintrin.h>

int numOfThread, height, width, iters;
double left, right, upper, lower, heightInterval, widthInterval;
int* image;
png_bytep* rows;

void save_color(png_bytep color, int p){
    if (p != iters) {
        if (p & 16) {
            color[0] = 240;
            color[1] = color[2] = p % 16 * 16;
        } else {
            color[0] = p % 16 * 16;
        }
    }
}

void write_png(const char* filename) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    for (int y = 0; y < height; ++y) {
        png_write_row(png_ptr, rows[y]);
        free(rows[y]);
    }
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void* mandelbrot(void *id){
    int tid = *(int*) id;
    
    for (int curHeight = tid; curHeight < height; curHeight+=numOfThread) {
        __m128d cImag = _mm_set1_pd(curHeight * heightInterval + lower);
        __m128d cReal =  _mm_setzero_pd();
        __m128d zImag =  _mm_setzero_pd();
        __m128d zReal =  _mm_setzero_pd();
        __m128d zzImag =  _mm_setzero_pd();
        __m128d zzReal =  _mm_setzero_pd();
        __m128d zRealzImag = _mm_setzero_pd();
        __m128d length_squared = _mm_setzero_pd();

        int curPointer[2] = {-1, -1};
        int repeats[2] = {0, 0};
        int curWidth = 0;

        size_t row_size = 3 * width * sizeof(png_byte);
        png_bytep curRow = (png_bytep)malloc(row_size);
        memset(curRow, 0, row_size);
        rows[height - 1 - curHeight] = curRow;
        
        // iterate every pixel in one row
        while(curWidth < width){
            // 1. initialize 2 value
            if(curPointer[0] == -1){
                cReal[0] = curWidth * widthInterval + left;
                zReal[0] = zImag[0] = zzReal[0] = zzImag[0] = length_squared[0] = 0;
                repeats[0] = 0;
                curPointer[0] = curWidth;
                curWidth++;
            }
            if(curPointer[1] == -1){
                cReal[1] = curWidth * widthInterval + left;
                zReal[1] = zImag[1] = zzReal[1] = zzImag[1] = length_squared[1] = 0;
                repeats[1] = 0;
                curPointer[1] = curWidth;
                curWidth++;
            }

            // 2. mandelbrot set
            while(repeats[0] < iters && repeats[1] < iters && length_squared[0] < 4 && length_squared[1] < 4){
                zRealzImag = _mm_mul_pd(zReal, zImag); 
                zImag = _mm_add_pd(_mm_add_pd(zRealzImag, zRealzImag), cImag); 
                zReal = _mm_add_pd(_mm_sub_pd(zzReal, zzImag), cReal); 
                zzReal = _mm_mul_pd(zReal, zReal);
                zzImag = _mm_mul_pd(zImag, zImag);
                length_squared = _mm_add_pd(zzReal, zzImag);
                repeats[0]++;
                repeats[1]++;
            }

            // 3. TODO: set color for the done one
            if(repeats[0] >= iters || length_squared[0] >= 4){
                save_color(curRow+curPointer[0]*3, repeats[0]);
                curPointer[0] = -1;
            }
            if(repeats[1] >= iters || length_squared[1] >= 4){
                save_color(curRow+curPointer[1]*3, repeats[1]);
                curPointer[1] = -1;
            }
        }

        // finish the rest value in one row
        double zRealzImag_d;
        if(curPointer[0] != -1){
            while(repeats[0] < iters && length_squared[0] < 4){
                zRealzImag_d = zReal[0]*zImag[0]; 
                zImag[0] = zRealzImag_d + zRealzImag_d + cImag[0]; 
                zReal[0] = zzReal[0] - zzImag[0] + cReal[0]; 
                zzReal[0] = zReal[0] * zReal[0];
                zzImag[0] = zImag[0] * zImag[0];
                length_squared[0] = zzReal[0] + zzImag[0];
                repeats[0]++;
            }
            // TODO: set color for the done one
            save_color(curRow+curPointer[0]*3, repeats[0]);
        }
        if(curPointer[1] != -1){
            while(repeats[1] < iters && length_squared[1] < 4){
                zRealzImag_d = zReal[1]*zImag[1]; 
                zImag[1] = zRealzImag_d + zRealzImag_d + cImag[1]; 
                zReal[1] = zzReal[1] - zzImag[1] + cReal[1]; 
                zzReal[1] = zReal[1] * zReal[1];
                zzImag[1] = zImag[1] * zImag[1];
                length_squared[1] = zzReal[1] + zzImag[1];
                repeats[1]++;
            }
            // TODO: set color for the done one
            save_color(curRow+curPointer[1]*3, repeats[1]);
        }
    }

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    numOfThread = CPU_COUNT(&cpu_set);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    heightInterval = ((double)(upper - lower) / (double)height);
    widthInterval = ((double)(right - left) / (double)width);

    /* allocate memory for image */
    rows = (png_bytep*)malloc(height * sizeof(png_bytep));

    pthread_t threads[numOfThread];
	int* tid = new int[numOfThread];

    for(int i = 0; i < numOfThread; i++){
		tid[i] = i; 
		pthread_create(&threads[i], NULL, mandelbrot, &tid[i]);
	}

    /* mandelbrot set */

	for(int i = 0; i < numOfThread; i++){
		pthread_join(threads[i], NULL);
	}
    
    /* draw and cleanup */
    write_png(filename);
}
