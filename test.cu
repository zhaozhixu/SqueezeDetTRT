#include <time.h>
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "tensorUtil.h"
#include "trtUtil.h"

clock_t start, end;
Tensor *t, *tcuda;
float *tcuda_data;

int ndim = 3;
int dims[] = {3, 2, 3};
float data[] = {0.0, 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0, 7.0,
                8.0, 9.0, 10.0, 11.0,
                12.0, 13.0, 14.0, 15.0,
                16.0, 17.0};
/*  data[] = {0.0, 2.0, 1.0, */
/*                 5.0, 4.0, 3.0, */
/*                 6.0, 7.0, 6.0, */
/*                 9.0, 10.0, 11.0, */
/*                 12.0, 12.0, 14.0, */
/*                 15.0, 16.0, 16.0}; */


void init()
{
     /* int ndim = 4; */
     /* int dims[] = {20, 24, 78, 72}; */
     /* int dims[] = {20, 24, 1872, 3}; */
     /* size_t data_size = 2695680; */
     /* float *data = (float *)malloc(sizeof(float) * data_size); */
     /* int i; */
     /* for (i = 0; i < data_size; i++) */
     /*      data[i] = 1.0; */

     t = createTensor(data, ndim, dims);
     /* tcuda_data = (float *)cloneMem(t->data, sizeof(float) * t->len, H2D); */
     tcuda = cloneTensor(t, H2D);
     /* tcuda = createTensor(tcuda_data, t->ndim, t->dims); */
     printTensor(t, "%.2f");
}

void testSliceTensor()
{
     /* Tensor *st = createSlicedTensor(t, 2, 2, 1800); */
     Tensor *st = createSlicedTensor(t, 2, 1, 2);
     start = clock();
     /* sliceTensor(t, st, 2, 2, 1800); */
     sliceTensor(t, st, 2, 1, 2);
     end = clock();
     printf("sliceTensor in %ld\n", end - start);
     printTensor(st, "%.2f");

     /* Tensor *stcuda = creatSlicedTensorCuda(tcuda, 2, 2, 1800); */
     Tensor *stcuda = creatSlicedTensorCuda(tcuda, 2, 1, 2);
     start = clock();
     /* sliceTensorCuda(tcuda, stcuda, 2, 2, 1800); */
     sliceTensorCuda(tcuda, stcuda, 2, 1, 2);
     end = clock();
     printf("sliceTensorCuda in %ld\n", end - start);
     float *sthost_data = (float *)cloneMem(stcuda->data, stcuda->len * sizeof(float), D2H);
     Tensor *sthost = createTensor(sthost_data, stcuda->ndim, stcuda->dims);
     printTensor(sthost, "%.2f");
}

void testReshapeTensor()
{
     /* printTensor(t, "%.2f"); */

     /* int newNdim = 3; */
     /* int newDims[] = {3, 3, 2}; */
     int newNdim = 2;
     int newDims[] = {3, 6};
     start = clock();
     Tensor *rt = reshapeTensor(t, newNdim, newDims);
     end = clock();
     printf("reshapeTensor in %ld\n", end - start);
     printTensor(rt, "%.2f");
}

void testReduceArgMax()
{
     assert(isTensorValid(tcuda));
     /* printTensor(t, "%.2f"); */
     Tensor *dst = createReducedTensor(tcuda, tcuda->ndim-2);
     Tensor *arg = createReducedTensor(tcuda, tcuda->ndim-2);
     start = clock();
     reduceArgMax(tcuda, dst, arg, tcuda->ndim-2);
     end = clock();
     printf("reduceArgMax in %ld\n", end - start);

     float *dst_host_data = (float *)cloneMem(dst->data, sizeof(float) * dst->len, D2H);
     Tensor *dst_host = createTensor(dst_host_data, dst->ndim, dst->dims);
     printTensor(dst_host, "%.2f");
     float *arg_host_data = (float *)cloneMem(arg->data, sizeof(float) * arg->len, D2H);
     Tensor *arg_host = createTensor(arg_host_data, arg->ndim, arg->dims);
     printTensor(arg_host, "%.2f");
}

void testMultiplyElement()
{
     float *dst_cuda_data;
     cudaMalloc(&dst_cuda_data, sizeof(float) * tcuda->len);
     Tensor *dst = createTensor(dst_cuda_data, tcuda->ndim, tcuda->dims);
     Tensor * src1 = createTensor(tcuda->data, tcuda->ndim, tcuda->dims);
     Tensor * src2 = createTensor(tcuda->data, tcuda->ndim, tcuda->dims);

     start = clock();
     multiplyElement(src1, src2, dst);
     end = clock();
     printf("multiplyElement in %ld\n", end - start);

     float *dst_host_data = (float *)cloneMem(dst->data, sizeof(float) * dst->len, D2H);
     Tensor *dst_host = createTensor(dst_host_data, dst->ndim, dst->dims);
     printTensor(dst_host, "%.2f");
}

void testTransformBboxSQD()
{
     float *delta_host_data = (float *)malloc(sizeof(float) * 24);
     for (int i = 0; i < 24; i++)
          delta_host_data[i] = i * 0.1;
     float *anchor_host_data = (float *)cloneMem(delta_host_data, sizeof(float) * 24, H2H);
     float *delta_cuda_data = (float *)cloneMem(delta_host_data, sizeof(float) * 24, H2D);
     float *anchor_cuda_data = (float *)cloneMem(anchor_host_data, sizeof(float) * 24, H2D);
     float *res_cuda_data;
     cudaMalloc(&res_cuda_data, sizeof(float) * 24);

     int dims[] = {1, 4, 6};
     Tensor *delta_host = createTensor(delta_host_data, 3, dims);
     Tensor *anchor_host = createTensor(anchor_host_data, 3, dims);
     Tensor *delta_cuda = createTensor(delta_cuda_data, 3, dims);
     Tensor *anchor_cuda = createTensor(anchor_cuda_data, 3, dims);
     Tensor *res_cuda = createTensor(res_cuda_data, 3, dims);
     float x_scales[] = {1.}, y_scales[] = {1.};
     float *x_scales_device = (float *)cloneMem(x_scales, sizeof(float), H2D);
     float *y_scales_device = (float *)cloneMem(y_scales, sizeof(float), H2D);

     printf("delta_host:\n");
     printTensor(delta_host, "%.6f");
     printf("anchor_host:\n");
     printTensor(anchor_host, "%.6f");
     start =clock();
     transformBboxSQD(delta_cuda, anchor_cuda, res_cuda, 1248, 384, x_scales_device, y_scales_device);
     end = clock();
     printf("transformBboxSQD in %ld\n", end - start);
     float *res_host_data = (float *)cloneMem(res_cuda_data, sizeof(float) * 24, D2H);
     Tensor *res_host = createTensor(res_host_data, 3, dims);
     printTensor(res_host, "%.6f");
}

void testAnchor()
{
     /* int H = 24, W = 78, B = 9; */
     int H = 4, W = 2, B = 9;
     int width = 1248, height = 384;
     float anchor_shape[] = {36, 37, 366, 174, 115, 59, /* w x h, 2 elements one group*/
                             162, 87, 38, 90, 258, 173,
                             224, 108, 78, 170, 72, 43};
     float center_x[W], center_y[H];
     float anchors[H*W*B*4];
     /* int i, j, k; */
     int i;
     for (i = 1; i <= W; i++)
          center_x[i-1] = i * width / (W + 1.0);
     for (i = 1; i <= H; i++)
          center_y[i-1] = i * height / (H + 1.0);
     /* int anchors_dims[] = {W, H, B, 4}; */
     int anchors_dims[] = {4, B, H, W};
     Tensor *anchor_tensor = createTensor(anchors, 4, anchors_dims);
     int a_vol = B * H * W;
     int b_vol = H * W;
     for (i = 0; i < a_vol; i++) {
          anchors[i] = center_x[i % W];
          anchors[a_vol + i] = center_y[i / W % H];
          anchors[a_vol * 2 + i] = anchor_shape[i / b_vol * 2];
          anchors[a_vol * 3 + i] = anchor_shape[i / b_vol * 2 + 1];
     }
     /* int w_vol = H*B*4; */
     /* int h_vol = B*4; */
     /* int b_vol = 4; */
     /* for (i = 0; i < W; i++) { */
     /*      for (j = 0; j < H; j++) { */
     /*           for (k = 0; k < B; k++) { */
     /*                anchors[i*w_vol+j*h_vol+k*b_vol] = center_x[i]; */
     /*                anchors[i*w_vol+j*h_vol+k*b_vol+1] = center_y[j]; */
     /*                anchors[i*w_vol+j*h_vol+k*b_vol+2] = anchor_shape[k*2]; */
     /*                anchors[i*w_vol+j*h_vol+k*b_vol+3] = anchor_shape[k*2+1]; */
     /*           } */
     /*      } */
     /* } */
     printf("anchor_tensor:\n");
     printTensor(anchor_tensor, "%.2f");
}

void testThrustSort()
{
     int n = 9;
     float f[] = {3.1, 9.2, 7.3, 5.4, 4.5, 0.6, 2.7, 6.8, 1.9};
     int id[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
     float *f_device = (float *)cloneMem(f, n * sizeof(float), H2D);
     int *id_device = (int *)cloneMem(id, n * sizeof(int), H2D);
     thrust::sort_by_key(thrust::device, f_device, f_device + n, id_device);
     float *f_host = (float *)cloneMem(f_device, n * sizeof(float), D2H);
     int *id_host = (int *)cloneMem(id_device, n * sizeof(int), D2H);
     for (int i = 0; i < n; i++) {
          printf("%.2f ", f[i]);
     }
     printf("\n");
     for (int i = 0; i < n; i++) {
          printf("%d ", id[i]);
     }
     printf("\n");
     for (int i = 0; i < n; i++) {
          printf("%.2f ", f_host[i]);
     }
     printf("\n");
     for (int i = 0; i < n; i++) {
          printf("%d ", id_host[i]);
     }
     printf("\n");
}

void findThrustBug()
{
     const int SIZE = 16848;
     FILE *infile = fopen("sort_bug.txt", "r");
     FILE *outfile = fopen("sorted.txt", "w");
     float array[SIZE];
     float *array_device;
     int id[SIZE];
     int *id_device;
     int *id_host;

     for (int i = 0; i < SIZE; i++) {
          fscanf(infile, "%f", &array[i]);
          id[i] = i;
     }
     cudaMalloc(&array_device, SIZE * sizeof(float));
     cudaMemcpy(array_device, array, SIZE * sizeof(float), cudaMemcpyHostToDevice);
     cudaMalloc(&id_device, SIZE * sizeof(int));
     cudaMemcpy(id_device, id, SIZE * sizeof(int), cudaMemcpyHostToDevice);
     thrust::sort_by_key(thrust::device, array_device, array_device + SIZE, id_device);
     id_host = (int *)malloc(SIZE * sizeof(int));
     cudaMemcpy(id_host, id_device, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
     for (int i = 0; i < SIZE; i++)
          fprintf(outfile, "%d\n", id_host[i]);
     fclose(infile);
     fclose(outfile);
}

void testOpencv()
{
     std::vector<std::string> imglist = getImageList("data/example");
     /* cv::Mat img = readImage(imglist.begin(), 1248, 768); */
     std::string filename = imglist[0];
     printf("filename: %s\n", filename.c_str());

     cv::Mat img = cv::imread(filename);
     printf("img.total(): %ld\n", img.total());
     cv::resize(img, img, cv::Size(1248, 384));
}

void testIou()
{
     float bbox0[] = {0, 0, 0, 0};
     float bbox1[] = {0, 0, 0, 0};
     printf("%f\n", computeIou(bbox0, bbox1));
}

void testPickElements()
{
     float src_host[] = {0.0, 1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0, 7.0,
                       8.0, 9.0, 10.0, 11.0,
                       12.0, 13.0, 14.0, 15.0,
                       16.0, 17.0};
     int index[] = {9, 15, 2, 17, 11};
     int len = 5;
     int *index_device = (int *)cloneMem(index, len * sizeof(int), H2D); /* remember this! */
     float *src_device = (float *)cloneMem(src_host, 18 * sizeof(float), H2D);
     float *dst_device;
     cudaMalloc(&dst_device, len * sizeof(float));

     pickElements(src_device, dst_device, 1, index_device, len);

     float *dst_host = (float *)cloneMem(dst_device, len * sizeof(float), D2H);
     for (int i = 0; i < 18; i++)
          printf("%.2f ", src_host[i]);
     printf("\n");
     for (int i = 0; i < len; i++)
          printf("%.2f ", dst_host[i]);
     printf("\n");
}

int main(int argc, char *argv[])
{
     init();
     /* testSliceTensor(); */
     /* testReshapeTensor(); */
     /* testReduceArgMax(); */
     /* testMultiplyElement(); */
     testTransformBboxSQD();
     /* testAnchor(); */
     /* testThrustSort(); */
     /* findThrustBug(); */
     /* testOpencv(); */
     /* testIou(); */
     /* testPickElements(); */
}
