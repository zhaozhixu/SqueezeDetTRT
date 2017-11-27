#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <dirent.h>
#include <unistd.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "common.h"
#include "tensorUtil.h"
#include "trtUtil.h"

static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

static const int INPUT_N = 1;
static const int INPUT_C = 3;
static const int INPUT_H = 384;
static const int INPUT_W = 1248;

static const int CONVOUT_C = 72;
static const int CONVOUT_H = 24;
static const int CONVOUT_W = 78;

static const int CLASS_SLICE_C = 27;
static const int CONF_SLICE_C = 9;
static const int BBOX_SLICE_C = 36;

static const int OUTPUT_CLS_SIZE = 3;
static const int OUTPUT_BBOX_SIZE = 4;

static const int TOP_N_DECTION = 64;
static const float NMS_THRESH = 0.4;
// static const float PROB_THRESH = 0.005;
static const float PROB_THRESH = 0.9;
// static const float EPSILON = 1e-16;

static const char* INPUT_NAME = "data";
static const char* CONVOUT_NAME = "conv_out";
static const char* CLASS_INPUT_NAME = "class_slice";
static const char* CONF_INPUT_NAME = "confidence_slice";
static const char* BBOX_INPUT_NAME = "bbox_slice";
static const char* CLASS_OUTPUT_NAME = "pred_class_probs";
static const char* CONF_OUTPUT_NAME = "pred_confidence_score";
static const char* BBOX_OUTPUT_NAME = "bbox_delta";

static const int ANCHORS_PER_GRID = 9;
static const int ANCHOR_SIZE = 4;
static const float ANCHOR_SHAPE[] = {36, 37, 366, 174, 115, 59, /* w x h, 2 elements one group*/
                              162, 87, 38, 90, 258, 173,
                              224, 108, 78, 170, 72, 43};

static const char *CLASS_NAMES[] = {"car", "pedestrian", "cyclist"};

std::string locateFile(const std::string& input)
{
     std::vector<std::string> dirs{"data/"};
     return locateFile(input, dirs);
}

ILayer*
addFireLayer(INetworkDefinition* network, ITensor& input, int ns1x1, int ne1x1, int ne3x3,
             Weights wks1x1, Weights wke1x1, Weights wke3x3,
             Weights wbs1x1, Weights wbe1x1, Weights wbe3x3)
{
     auto sq1x1 = network->addConvolution(input, ns1x1, DimsHW{1, 1}, wks1x1, wbs1x1);
     assert(sq1x1 != nullptr);
     sq1x1->setStride(DimsHW{1, 1});
     // sq1x1->setPadding(); TODO: add padding
     auto relu1 = network->addActivation(*sq1x1->getOutput(0), ActivationType::kRELU);
     assert(relu1 != nullptr);

     auto ex1x1 = network->addConvolution(*relu1->getOutput(0) , ne1x1, DimsHW{1, 1}, wke1x1, wbe1x1);
     assert(ex1x1 != nullptr);
     ex1x1->setStride(DimsHW{1, 1});
     auto relu2 = network->addActivation(*ex1x1->getOutput(0), ActivationType::kRELU);
     assert(relu2 != nullptr);

     auto ex3x3 = network->addConvolution(*relu1->getOutput(0), ne3x3, DimsHW{3, 3}, wke3x3, wbe3x3);
     assert(ex3x3 != nullptr);
     ex3x3->setStride(DimsHW{1, 1});
     ex3x3->setPadding(DimsHW{1, 1});
     auto relu3 = network->addActivation(*ex3x3->getOutput(0), ActivationType::kRELU);
     assert(relu3 != nullptr);

     ITensor *concatTensors[] = {relu2->getOutput(0), relu3->getOutput(0)};
     auto concat = network->addConcatenation(concatTensors, 2);
     assert(concat != nullptr);

     return concat;
}

// Creat the Engine using only the API and not any parser.
ICudaEngine *
createConvEngine(unsigned int maxBatchSize, IBuilder *builder, DataType dt)
{
     INetworkDefinition* network = builder->createNetwork();

     auto data = network->addInput(INPUT_NAME, dt, DimsCHW{INPUT_C, INPUT_H, INPUT_W});
     assert(data != nullptr);

     std::map<std::string, Weights> weightMap = loadWeights(locateFile("sqdtrt.wts")); // ?
     auto conv1 = network->addConvolution(*data, 64, DimsHW{3, 3},
                                          weightMap["conv1_kernels"],
                                          weightMap["conv1_bias"]);
     assert(conv1 != nullptr);
     conv1->setStride(DimsHW{2, 2});
     conv1->setPadding(DimsHW{1, 1});
     auto relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
     assert(relu1 != nullptr);

     auto pool1 = network->addPooling(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
     assert(pool1 != nullptr);
     pool1->setStride(DimsHW{2, 2});
     pool1->setPadding(DimsHW{1, 1});

     auto fire2 = addFireLayer(network, *pool1->getOutput(0), 16, 64, 64,
                               weightMap["fire2_squeeze1x1_kernels"],
                               weightMap["fire2_expand1x1_kernels"],
                               weightMap["fire2_expand3x3_kernels"],
                               weightMap["fire2_squeeze1x1_biases"],
                               weightMap["fire2_expand1x1_biases"],
                               weightMap["fire2_expand3x3_biases"]);
     auto fire3 = addFireLayer(network, *fire2->getOutput(0), 16, 64, 64,
                               weightMap["fire3_squeeze1x1_kernels"],
                               weightMap["fire3_expand1x1_kernels"],
                               weightMap["fire3_expand3x3_kernels"],
                               weightMap["fire3_squeeze1x1_biases"],
                               weightMap["fire3_expand1x1_biases"],
                               weightMap["fire3_expand3x3_biases"]);

     auto pool3 = network->addPooling(*fire3->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
     assert(pool3 != nullptr);
     pool3->setStride(DimsHW{2, 2});
     pool3->setPadding(DimsHW{1, 1});

     auto fire4 = addFireLayer(network, *pool3->getOutput(0), 32, 128, 128,
                               weightMap["fire4_squeeze1x1_kernels"],
                               weightMap["fire4_expand1x1_kernels"],
                               weightMap["fire4_expand3x3_kernels"],
                               weightMap["fire4_squeeze1x1_biases"],
                               weightMap["fire4_expand1x1_biases"],
                               weightMap["fire4_expand3x3_biases"]);
     auto fire5 = addFireLayer(network, *fire4->getOutput(0), 32, 128, 128,
                               weightMap["fire5_squeeze1x1_kernels"],
                               weightMap["fire5_expand1x1_kernels"],
                               weightMap["fire5_expand3x3_kernels"],
                               weightMap["fire5_squeeze1x1_biases"],
                               weightMap["fire5_expand1x1_biases"],
                               weightMap["fire5_expand3x3_biases"]);

     auto pool5 = network->addPooling(*fire5->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
     assert(pool3 != nullptr);
     pool3->setStride(DimsHW{2, 2});
     pool5->setPadding(DimsHW{1, 1});

     auto fire6 = addFireLayer(network, *pool5->getOutput(0), 48, 192, 192,
                               weightMap["fire6_squeeze1x1_kernels"],
                               weightMap["fire6_expand1x1_kernels"],
                               weightMap["fire6_expand3x3_kernels"],
                               weightMap["fire6_squeeze1x1_biases"],
                               weightMap["fire6_expand1x1_biases"],
                               weightMap["fire6_expand3x3_biases"]);
     auto fire7 = addFireLayer(network, *fire6->getOutput(0), 48, 192, 192,
                               weightMap["fire7_squeeze1x1_kernels"],
                               weightMap["fire7_expand1x1_kernels"],
                               weightMap["fire7_expand3x3_kernels"],
                               weightMap["fire7_squeeze1x1_biases"],
                               weightMap["fire7_expand1x1_biases"],
                               weightMap["fire7_expand3x3_biases"]);
     auto fire8 = addFireLayer(network, *fire7->getOutput(0), 64, 256, 256,
                               weightMap["fire8_squeeze1x1_kernels"],
                               weightMap["fire8_expand1x1_kernels"],
                               weightMap["fire8_expand3x3_kernels"],
                               weightMap["fire8_squeeze1x1_biases"],
                               weightMap["fire8_expand1x1_biases"],
                               weightMap["fire8_expand3x3_biases"]);
     auto fire9 = addFireLayer(network, *fire8->getOutput(0), 64, 256, 256,
                               weightMap["fire9_squeeze1x1_kernels"],
                               weightMap["fire9_expand1x1_kernels"],
                               weightMap["fire9_expand3x3_kernels"],
                               weightMap["fire9_squeeze1x1_biases"],
                               weightMap["fire9_expand1x1_biases"],
                               weightMap["fire9_expand3x3_biases"]);

     auto fire10 = addFireLayer(network, *fire9->getOutput(0), 96, 384, 384,
                                weightMap["fire10_squeeze1x1_kernels"],
                                weightMap["fire10_expand1x1_kernels"],
                                weightMap["fire10_expand3x3_kernels"],
                                weightMap["fire10_squeeze1x1_biases"],
                                weightMap["fire10_expand1x1_biases"],
                                weightMap["fire10_expand3x3_biases"]);
     auto fire11 = addFireLayer(network, *fire10->getOutput(0), 96, 384, 384,
                                weightMap["fire11_squeeze1x1_kernels"],
                                weightMap["fire11_expand1x1_kernels"],
                                weightMap["fire11_expand3x3_kernels"],
                                weightMap["fire11_squeeze1x1_biases"],
                                weightMap["fire11_expand1x1_biases"],
                                weightMap["fire11_expand3x3_biases"]);

     // TODO: add dropout11
     ILayer *dropout11 = fire11;

     auto preds = network->addConvolution(*dropout11->getOutput(0), CONVOUT_C, DimsHW{3, 3},
                                          weightMap["conv12_kernels"],
                                          weightMap["conv12_biases"]);
     assert(preds != nullptr);
     preds->setStride(DimsHW{1, 1}); // what is xavier, stddev?
     preds->setPadding(DimsHW{1, 1});

     preds->getOutput(0)->setName(CONVOUT_NAME);
     network->markOutput(*preds->getOutput(0));

     // Build the engine
     builder->setMaxBatchSize(maxBatchSize);
     builder->setMaxWorkspaceSize(1 << 20);

     auto engine = builder->buildCudaEngine(*network);
     // we don't need the network any more
     // network->destroy();	// SIGSEGV, don't know why

     // Once we have built the cuda engine, we can release all of our held memory.
     for (auto &mem : weightMap)
     {
          free((void*)(mem.second.values));
     }
     return engine;
}

ICudaEngine *
createInterpretEngine(unsigned int maxBatchSize, IBuilder *builder, DataType dt)
{
     INetworkDefinition* network = builder->createNetwork();

     auto class_tensor = network->addInput(CLASS_INPUT_NAME, dt, DimsCHW{OUTPUT_CLS_SIZE, 1, INPUT_N * CONVOUT_H * CONVOUT_W * CLASS_SLICE_C / OUTPUT_CLS_SIZE});
     assert(class_tensor != nullptr);
     auto confidence_tensor = network->addInput(CONF_INPUT_NAME, dt, DimsNCHW{INPUT_N, 1, 1, CONVOUT_W * CONVOUT_H * ANCHORS_PER_GRID});
     assert(confidence_tensor != nullptr);

     auto class_softmax = network->addSoftMax(*class_tensor);
     assert(class_softmax != nullptr);
     auto pred_conf = network->addActivation(*confidence_tensor, ActivationType::kSIGMOID);
     assert(pred_conf != nullptr);

     class_softmax->getOutput(0)->setName(CLASS_OUTPUT_NAME);
     pred_conf->getOutput(0)->setName(CONF_OUTPUT_NAME);
     network->markOutput(*class_softmax->getOutput(0));
     network->markOutput(*pred_conf->getOutput(0));

     // Build the engine
     builder->setMaxBatchSize(maxBatchSize);
     builder->setMaxWorkspaceSize(1 << 20);

     auto engine = builder->buildCudaEngine(*network);
     // we don't need the network any more
     // network->destroy(); // SIGSEGV, don't know why

     return engine;
}

void doInference(IExecutionContext& convContext, IExecutionContext& interpretContext, float* input, float* anchors, float *outProbs, float *outClass, float *outBbox, int batchSize)
{
     const ICudaEngine& convEngine = convContext.getEngine();
     const ICudaEngine& interpretEngine = interpretContext.getEngine();

     assert(convEngine.getNbBindings() == 2);
     assert(interpretEngine.getNbBindings() == 4);
     void* convBuffers[2];
     void* interpretBuffers[4];

     // In order to bind the buffers, we need to know the names of the input and output tensors.
     // note that indices are guaranteed to be less than IEngine::getNbBindings()
     int inputIndex = convEngine.getBindingIndex(INPUT_NAME),
          convoutIndex = convEngine.getBindingIndex(CONVOUT_NAME),
          classInputIndex = interpretEngine.getBindingIndex(CLASS_INPUT_NAME),
          confInputIndex = interpretEngine.getBindingIndex(CONF_INPUT_NAME),
          classOutputIndex = interpretEngine.getBindingIndex(CLASS_OUTPUT_NAME),
          confOutputIndex = interpretEngine.getBindingIndex(CONF_OUTPUT_NAME);

     // create GPU buffers and a stream
     float *bboxInput; // don't need to go into interpret engine
     int anchorsNum = CONVOUT_W * CONVOUT_H * ANCHORS_PER_GRID;
     size_t inputSize = batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float);
     size_t convoutSize = batchSize * CONVOUT_H * CONVOUT_W * CONVOUT_C * sizeof(float);
     size_t classInputSize = batchSize * CONVOUT_H * CONVOUT_W * CLASS_SLICE_C * sizeof(float);
     size_t confInputSize = batchSize * CONVOUT_H * CONVOUT_W * CONF_SLICE_C * sizeof(float);
     size_t bboxInputSize = batchSize * CONVOUT_H * CONVOUT_W * BBOX_SLICE_C * sizeof(float);
     size_t classOutputSize = batchSize * OUTPUT_CLS_SIZE * anchorsNum * sizeof(float);
     size_t confOutputSize = batchSize * anchorsNum * sizeof(float);
     CHECK(cudaMalloc(&convBuffers[inputIndex], inputSize));
     CHECK(cudaMalloc(&convBuffers[convoutIndex], convoutSize));
     CHECK(cudaMalloc(&interpretBuffers[classInputIndex], classInputSize));
     CHECK(cudaMalloc(&interpretBuffers[confInputIndex], confInputSize));
     CHECK(cudaMalloc(&bboxInput, bboxInputSize));
     CHECK(cudaMalloc(&interpretBuffers[classOutputIndex], classOutputSize));
     CHECK(cudaMalloc(&interpretBuffers[confOutputIndex], confOutputSize));

     int convout_dims[] = {INPUT_N, CONVOUT_H, CONVOUT_W, CONVOUT_C};
     int classInputDims[] = {INPUT_N, CONVOUT_H, CONVOUT_W, CLASS_SLICE_C};
     int confInputDims[] = {INPUT_N, CONVOUT_H, CONVOUT_W, CONF_SLICE_C};
     int bboxInputDims[] = {INPUT_N, CONVOUT_H, CONVOUT_W, BBOX_SLICE_C};
     int classOutputDims[] = {INPUT_N, anchorsNum, OUTPUT_CLS_SIZE};
     int confOutputDims[] = {INPUT_N, anchorsNum, 1};
     int bboxOutputDims[] = {INPUT_N, anchorsNum, OUTPUT_BBOX_SIZE};
     Tensor *convoutTensor = createTensor((float *)convBuffers[convoutIndex], 4, convout_dims);
     Tensor *classInputTensor = createTensor((float *)interpretBuffers[classInputIndex], 4, classInputDims);
     Tensor *confInputTensor = createTensor((float *)interpretBuffers[confInputIndex], 4, confInputDims);
     Tensor *bboxInputTensor = createTensor(bboxInput, 4, bboxInputDims);
     Tensor *classOutputTensor = createTensor((float *)interpretBuffers[classOutputIndex], 3, classOutputDims);
     Tensor *confOutputTensor = createTensor((float *)interpretBuffers[confOutputIndex], 3, confOutputDims);
     Tensor *bboxOutputTensor = reshapeTensor(bboxInputTensor, 3, bboxOutputDims);

     float *reduceMaxRes, *reduceArgRes, *mulRes, *bboxRes, *anchorsCuda;
     size_t reduceMaxResSize = batchSize * anchorsNum * sizeof(float);
     size_t reduceArgResSize = batchSize * anchorsNum * sizeof(float);
     size_t mulResSize = batchSize * anchorsNum * sizeof(float);
     size_t bboxResSize = batchSize * anchorsNum * OUTPUT_BBOX_SIZE * sizeof(float);
     size_t anchorsCudaSize = batchSize * anchorsNum * ANCHOR_SIZE * sizeof(float);
     CHECK(cudaMalloc(&reduceMaxRes, reduceMaxResSize));
     CHECK(cudaMalloc(&reduceArgRes, reduceArgResSize));
     CHECK(cudaMalloc(&mulRes, mulResSize));
     CHECK(cudaMalloc(&bboxRes, bboxResSize));
     anchorsCuda = (float *)cloneMem(anchors, anchorsCudaSize, H2D);

     int reduceMaxResDims[] = {INPUT_N, anchorsNum, 1};
     int reduceArgResDims[] = {INPUT_N, anchorsNum, 1};
     int mulResDims[] = {INPUT_N, anchorsNum, 1};
     int bboxResDims[] = {INPUT_N, anchorsNum, OUTPUT_BBOX_SIZE};
     int anchorsCudaDims[] = {INPUT_N, anchorsNum, ANCHOR_SIZE};
     Tensor *reduceMaxResTensor = createTensor(reduceMaxRes, 3, reduceMaxResDims);
     Tensor *reduceArgResTensor = createTensor(reduceArgRes, 3, reduceArgResDims);
     Tensor *mulResTensor = createTensor(mulRes, 3, mulResDims);
     Tensor *bboxResTensor = createTensor(bboxRes, 3, bboxResDims);
     Tensor *anchorsCudaTensor = createTensor(anchorsCuda, 3, anchorsCudaDims);

     int *orderDevice, *orderHost; // for top-n-detecion
     assert(orderHost = (int *)malloc(batchSize * anchorsNum * sizeof(int)));
     for (int i = 0; i < batchSize * anchorsNum; i++)
          orderHost[i] = i;
     orderDevice = (int *)cloneMem(orderHost, batchSize * anchorsNum * sizeof(int), H2D);

     cudaStream_t stream;
     CHECK(cudaStreamCreate(&stream));

     // timer create, timer start
     cudaEvent_t start, stop;
     float timeDetect;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     cudaEventRecord(start, 0);

     // FILE * probs_file = fopen("probs.txt", "w");
     // FILE * conf0_file = fopen("conf0.txt", "w");
     // FILE * conf_file = fopen("conf.txt", "w");
     // FILE * class_file = fopen("class.txt", "w");
     // FILE * bbox_file = fopen("bbox.txt", "w");
     // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
     CHECK(cudaMemcpyAsync(convBuffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice, stream));

     convContext.enqueue(batchSize, convBuffers, stream, nullptr);
     sliceTensorCuda(convoutTensor, classInputTensor, 3, 0, CLASS_SLICE_C);
     sliceTensorCuda(convoutTensor, confInputTensor, 3, CLASS_SLICE_C, CONF_SLICE_C);
     sliceTensorCuda(convoutTensor, bboxInputTensor, 3, CLASS_SLICE_C + CONF_SLICE_C, BBOX_SLICE_C);
     interpretContext.enqueue(batchSize, interpretBuffers, stream, nullptr);
     reduceArgMax(classOutputTensor, reduceMaxResTensor, reduceArgResTensor, 2);
     multiplyElement(reduceMaxResTensor, confOutputTensor, mulResTensor);
     transformBboxSQD(bboxOutputTensor, anchorsCudaTensor, bboxResTensor, INPUT_W, INPUT_H);

     cudaEventRecord(stop, 0);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&timeDetect, start, stop);

     // Tensor *probs_host = cloneTensor(classOutputTensor, D2H);
     // Tensor *conf0_host = cloneTensor(confOutputTensor, D2H);
     // Tensor *conf_host = cloneTensor(mulResTensor, D2H);
     // Tensor *class_host = cloneTensor(reduceArgResTensor, D2H);
     // Tensor *bbox_host = cloneTensor(bboxResTensor, D2H);
     // fprintTensor(probs_file, probs_host, "%f");
     // fprintTensor(conf0_file, conf0_host, "%f");
     // fprintTensor(conf_file, conf_host, "%f");
     // fprintTensor(class_file, class_host, "%f");
     // fprintTensor(bbox_file, bbox_host, "%f");
     // fclose(probs_file);
     // fclose(conf0_file);
     // fclose(conf_file);
     // fclose(class_file);
     // fclose(bbox_file);
     // filter top-n-detection
     // TODO: only batchSize = 1 supported
     tensorIndexSort(mulResTensor, orderDevice);

     CHECK(cudaMemcpyAsync(outProbs, mulResTensor->data, mulResTensor->len * sizeof(float), cudaMemcpyDeviceToHost, stream));
     CHECK(cudaMemcpyAsync(outClass, reduceArgResTensor->data, reduceArgResTensor->len * sizeof(float), cudaMemcpyDeviceToHost, stream));
     CHECK(cudaMemcpyAsync(outBbox, bboxResTensor->data, bboxResTensor->len * sizeof(float), cudaMemcpyDeviceToHost, stream));
     cudaStreamSynchronize(stream);

     // timer destroy
     cudaEventDestroy(start);
     cudaEventDestroy(stop);

     printf("detect in %f ms\n", timeDetect);

     // release the stream and the buffers
     cudaStreamDestroy(stream);
     CHECK(cudaFree(convBuffers[inputIndex]));
     CHECK(cudaFree(convBuffers[convoutIndex]));
     CHECK(cudaFree(interpretBuffers[classInputIndex]));
     CHECK(cudaFree(interpretBuffers[confInputIndex]));
     CHECK(cudaFree(bboxInput));
     CHECK(cudaFree(interpretBuffers[classOutputIndex]));
     CHECK(cudaFree(interpretBuffers[confOutputIndex]));
     CHECK(cudaFree(reduceMaxRes));
     CHECK(cudaFree(reduceArgRes));
     CHECK(cudaFree(mulRes));
     CHECK(cudaFree(bboxRes));
     CHECK(cudaFree(anchorsCuda));
}

// maxBatch - NB must be at least as large as the batch we want to run with)
void APIToModel(unsigned int maxBatchSize, IHostMemory **convModelStream, IHostMemory **interpretModelStream)
{
     // create the builder
     IBuilder* builder = createInferBuilder(gLogger);

     // create the model to populate the network, then set the outputs and create an engine
     ICudaEngine* convEngine = createConvEngine(maxBatchSize, builder, DataType::kFLOAT);
     ICudaEngine* interpretEngine = createInterpretEngine(maxBatchSize, builder, DataType::kFLOAT);

     assert(convEngine != nullptr);
     assert(interpretEngine != nullptr);

     // serialize the engine, then close everything down
     (*convModelStream) = convEngine->serialize();
     (*interpretModelStream) = interpretEngine->serialize();
     convEngine->destroy();
     interpretEngine->destroy();
     builder->destroy();
}

float *prepareData(std::vector<std::string> &imageList)
{
     std::vector<cv::Mat> images; // available images
     float* data = new float[INPUT_N * INPUT_C * INPUT_H * INPUT_W];

     int N = 1;// TODO: make it dynamic
     // srand(unsigned(time(nullptr))); // read a random sample image
     // std::random_shuffle(imageList.begin(), imageList.end(), [](int i) {return rand() % i; });
     assert(images.size() <= imageList.size());
     for (int i = 0; i < N; ++i)
          images.push_back(readImage(imageList[i], INPUT_W, INPUT_H));

     // pixel mean used by the SqueezeDet's author
     float pixelMean[3]{ 103.939f, 116.779f, 123.68f }; // also in BGR order
     for (int i = 0, volImg = INPUT_C*INPUT_H*INPUT_W; i < N; ++i)
     {
          for (int c = 0; c < INPUT_C; ++c)
          {
               // the color image to input should be in BGR order
               for (unsigned j = 0, volChl = INPUT_H*INPUT_W; j < volChl; ++j)
                    data[i*volImg + c*volChl + j] = float(images[i].data[j*INPUT_C]) - pixelMean[c];
          }
     }

     return data;
}

float *prepareAnchors(const float *anchor_shape, int width, int height, int H, int W, int B)
{
     assert(anchor_shape);
     float center_x[W], center_y[H];
     float *anchors = new float[H * W * B * 4];
     int i, j, k;

     for (i = 1; i <= W; i++)
          center_x[i-1] = i * width / (W + 1.0);
     for (i = 1; i <= H; i++)
          center_y[i-1] = i * height / (H + 1.0);

     int w_vol = H * B * 4;
     int h_vol = B * 4;
     int b_vol = 4;
     for (i = 0; i < W; i++) {
          for (j = 0; j < H; j++) {
               for (k = 0; k < B; k++) {
                    anchors[i*w_vol+j*h_vol+k*b_vol] = center_x[i];
                    anchors[i*w_vol+j*h_vol+k*b_vol+1] = center_y[j];
                    anchors[i*w_vol+j*h_vol+k*b_vol+2] = anchor_shape[k*2];
                    anchors[i*w_vol+j*h_vol+k*b_vol+3] = anchor_shape[k*2+1];
               }
          }
     }
     return anchors;
}

void detectionFilter(float *bboxes, float *classes, float *probs, int *keep, int num_probs, float nms_thresh, float prob_thresh)
{
     assert(bboxes && classes && probs && keep);

     int i, j;
     for (i = 0; i < num_probs; i++) {
          keep[i] = 1;
          if (probs[i] < prob_thresh) {
               keep[i] = 0;
               continue;
          }
          for (j = i - 1; j >= 0 ; j--) {
               if (!keep[j] || classes[i] != classes[j])
                    continue;
               if (computeIou(&bboxes[i*OUTPUT_BBOX_SIZE],&bboxes[j*OUTPUT_BBOX_SIZE]) > nms_thresh)
                    keep[j] = 0;
          }

     }
}

void fprintResult(FILE *fp, float *bboxes, float *classes, float *probs, int *keep, int num_probs)
{
     assert(fp && bboxes && classes && probs && keep);

     int i;
     float *bbox;
     for (i = 0; i < num_probs; i++) {
          if (!keep[i])
               continue;
          bbox = &bboxes[i * OUTPUT_BBOX_SIZE];
          fprintf(fp, "%s -1 -1 0.0 %.2f %.2f %.2f %.2f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %.3f\n",
                  CLASS_NAMES[(int)classes[i]], bbox[0], bbox[1], bbox[2], bbox[3], probs[i]);
     }
}

int main(int argc, char** argv)
{
     if (argc < 2) {
          std::cout << "usage: sqdtrt IMAGE_DIR\n";
          exit(EXIT_SUCCESS);
     }

     std::vector<std::string> imageList = getImageList(argv[1]);
     printf("image num: %ld\n", imageList.size());
     float *data = prepareData(imageList);
     float *anchors = prepareAnchors(ANCHOR_SHAPE, INPUT_W, INPUT_H, CONVOUT_H, CONVOUT_W, ANCHORS_PER_GRID);
     // float *outProbs = new float[INPUT_N * TOP_N_DECTION];
     // float *outClass = new float[INPUT_N * TOP_N_DECTION];
     // float *outBbox = new float[INPUT_N * TOP_N_DECTION * OUTPUT_BBOX_SIZE];
     int probsNum = INPUT_N * CONVOUT_H * CONVOUT_W * ANCHORS_PER_GRID;
     float *outProbs = new float[probsNum];
     float *outClass = new float[probsNum];
     float *outBbox = new float[probsNum * OUTPUT_BBOX_SIZE];
     int *keep = new int[probsNum];

     // create engines
     IHostMemory *convModelStream{ nullptr };
     IHostMemory *interpretModelStream{ nullptr };
     APIToModel(INPUT_N, &convModelStream, &interpretModelStream);

     // deserialize engines
     IRuntime* runtime = createInferRuntime(gLogger);
     ICudaEngine* convEngine = runtime->deserializeCudaEngine(convModelStream->data(), convModelStream->size(), nullptr);
     ICudaEngine* interpretEngine = runtime->deserializeCudaEngine(interpretModelStream->data(), interpretModelStream->size(), nullptr);
     IExecutionContext *convContext = convEngine->createExecutionContext();
     IExecutionContext *interpretContext = interpretEngine->createExecutionContext();

     // run inference
     doInference(*convContext, *interpretContext, data, anchors, outProbs, outClass, outBbox, INPUT_N);
     detectionFilter(outBbox, outClass, outProbs, keep, probsNum, NMS_THRESH, PROB_THRESH);
     fprintResult(stdout, outBbox, outClass, outProbs, keep, probsNum);

     // destroy the engine
     convContext->destroy();
     interpretContext->destroy();
     convEngine->destroy();
     interpretEngine->destroy();
     runtime->destroy();

     delete[] data;
     delete[] anchors;
     delete[] outProbs;
     delete[] outClass;
     delete[] outBbox;
     delete[] keep;
     return 0;
}
