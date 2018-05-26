#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <time.h>
#include <unistd.h>
#include <err.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// #define _GNU_SOURCE
// #include <getopt.h>

#include "NvInfer.h"
#include "common.h"
#include "tensorUtil.h"
#include "trtUtil.h"
#include "sdt_alloc.h"
#include "sdt_infer.h"

static const int INPUT_N = 1;   // one image at a time
static const int INPUT_C = 3;
static const int INPUT_H = 368;
static const int INPUT_W = 640;
// static const int INPUT_H = 512;
// static const int INPUT_W = 896;

// static const int CONVOUT_C = 108;
// static const int CONVOUT_C = 144;
static const int CONVOUT_C = 340;
static const int CONVOUT_H = 23;
static const int CONVOUT_W = 40;
// static const int CONVOUT_H = 32;
// static const int CONVOUT_W = 56;

static const int SPLIT_C = 128;
static const int FIRE9_C = 512;
static const int FIRE6_C = 384;
static const int FIRE7_C = 384;
static const int CONSTS_C = 384; // for concatation
static const float CONSTS_VALUE = 0.1; // for concatation

// static const int CLASS_SLICE_C = 63;
// static const int CLASS_SLICE_C = 99;
// static const int CLASS_SLICE_C = 108;
// static const int CONF_SLICE_C = 9;
// static const int BBOX_SLICE_C = 36;
static const int CLASS_SLICE_C = 240;
static const int CONF_SLICE_C = 20;
static const int BBOX_SLICE_C = 80;


// static const int OUTPUT_CLS_SIZE = 7;
// static const int OUTPUT_CLS_SIZE = 11;
static const int OUTPUT_CLS_SIZE = 12;
static const int OUTPUT_BBOX_SIZE = 4;

static const int TOP_N_DETECTION = 64;
static const float NMS_THRESH = 0.4;
// static const float PROB_THRESH = 0.005;
static const float PROB_THRESH = 0.3;
static const float PLOT_PROB_THRESH = 0.4;
// static const float EPSILON = 1e-16;

static const char *CONV1_POOL3_NAME = "conv1_pool3";
static const char *CONV1_FIRE6_NAME = "conv1_fire6";
static const char *CONV1_FIRE7_NAME = "conv1_fire7";
static const char *CONV1_FIRE9_NAME = "conv1_fire9";
static const char *CONV2_SPLIT1_NAME = "conv2_split1";
static const char *CONV2_SPLIT2_NAME = "conv2_split2";
static const char *CONV2_SPLIT3_NAME = "conv2_split3";
static const char *CONV2_SPLIT4_NAME = "conv2_split4";
static const char *CONV2_FIRE6_NAME = "conv2_fire6";
static const char *CONV2_FIRE7_NAME = "conv2_fire7";
static const char *CONV2_FIRE9_NAME = "conv2_fire9";
static const char* CONSTS_NAME = "CONST";
static const char* INPUT_NAME = "data";
static const char* CONVOUT_NAME = "conv_out";
static const char* CLASS_INPUT_NAME = "class_slice";
static const char* CONF_INPUT_NAME = "confidence_slice";
// static const char* BBOX_INPUT_NAME = "bbox_slice";
static const char* CLASS_OUTPUT_NAME = "pred_class_probs";
static const char* CONF_OUTPUT_NAME = "pred_confidence_score";
// static const char* BBOX_OUTPUT_NAME = "bbox_delta";

// static const int ANCHORS_PER_GRID = 9;
static const int ANCHORS_PER_GRID = 20;
static const int ANCHOR_SIZE = 4;
// static const float ANCHOR_SHAPE[] = {36, 37, 366, 174, 115, 59,
//                               162, 87, 38, 90, 258, 173,
//                               224, 108, 78, 170, 72, 43};
// static const float ANCHOR_SHAPE[] = {229, 137, 48, 71, 289, 245,
//                                      185, 134, 85, 142, 31, 41,
//                                      197, 191, 237, 206, 63, 108};
// static const float ANCHOR_SHAPE[] = {1.4*229, 1.4*137, 1.4*48, 1.4*71, 1.4*289, 1.4*245,
//                                      1.4*185, 1.4*134, 1.4*85, 1.4*142, 1.4*31, 1.4*41,
//                                      1.4*197, 1.4*191, 1.4*237, 1.4*206, 1.4*63, 1.4*108};
static const float ANCHOR_SHAPE[] = {10, 10, 20, 20, 30, 30, 30, 60,
                                     30, 90, 60, 30, 60, 60, 60, 90,
                                     90, 30, 90, 60, 90, 90, 56, 121,
                                     77, 202, 38, 103, 69, 150, 98, 49,
                                     100, 71, 121, 81, 223, 84, 141, 210};

// static const char *CLASS_NAMES[] = {"car", "pedestrian", "cyclist"};
// static const char *CLASS_NAMES[] = {"car", "person", "riding", "bike_riding", "boat", "truck", "horse_riding"};
// static const char *CLASS_NAMES[] = {"person", "car", "riding", "boat", "drone", "truck", "parachute", "whale", "building", "bird", "horse_riding"};
static const char *CLASS_NAMES[] = {"person", "car", "riding", "boat", "group", "wakeboard", "drone", "truck", "paraglider", "whale", "building", "horseride"};

// pixel mean used by the SqueezeDet's author
static const float PIXEL_MEAN[3]{ 103.939f, 116.779f, 123.68f }; // in BGR order

static const double DEFAULT_FPS = 10;

static int anchorsNum;
static int inputIndex, pool3Index, fire6OutIndex, fire7OutIndex, fire9OutIndex, split1Index, split2Index, split3Index, split4Index, fire6InIndex, fire7InIndex, fire9InIndex, constsIndex, convoutIndex, classInputIndex, confInputIndex, classOutputIndex, confOutputIndex;

// device buffers
static void *conv1Buffers[5],*conv2Buffers[9],  *interpretBuffers[4];
static float *bboxInput; // don't need to go into interpret engine
static int *transAxesDevice;
static Tensor *pool3Tensor;
static Tensor *split1Tensor;
static Tensor *split2Tensor;
static Tensor *split3Tensor;
static Tensor *split4Tensor;
static Tensor *convoutTensor;
static Tensor *classInputTensor;
static Tensor *confInputTensor;
static Tensor *bboxInputTensor;
static Tensor *classOutputTensor;
static Tensor *confOutputTensor;
static Tensor *bboxOutputTensor;
static Tensor *classTransTensor;
static Tensor *confTransTensor;
static Tensor *bboxTransTensor;
static int *classTransWorkspace[2], *confTransWorkspace[2], *bboxTransWorkspace[2];
static float *anchorsDevice;
static Tensor *reduceMaxResTensor;
static Tensor *reduceArgResTensor;
static Tensor *mulResTensor;
static Tensor *bboxResTensor;
static Tensor *anchorsDeviceTensor;
static int *orderDevice, *orderDeviceTmp; // for top-n-detecion
static Tensor *finalClassTensor;
static Tensor *finalProbsTensor;
static Tensor *finalBboxTensor;
static cudaStream_t stream;
static cudaEvent_t start_detect, stop_detect, start_misc, stop_misc;
static float timeDetect, timeMisc;

using namespace nvinfer1;

// not used
cv::Mat readImage(const std::string& filename, int width, int height, float *img_width, float *img_height)
{
     cv::Mat img = cv::imread(filename);
     if (img.empty())
          return img;

     if (img_width && img_height) {
          *img_width = img.size().width;
          *img_height = img.size().height;
     }
     cv::resize(img, img, cv::Size(width, height));
     return img;
}

void preprocessFrame(cv::Mat &frame, cv::Mat &frame_origin, int width, int height, float *img_width, float *img_height)
{
     assert(!frame_origin.empty());

     if (img_width && img_height) {
          *img_width = frame_origin.size().width;
          *img_height = frame_origin.size().height;
     }
     cv::resize(frame_origin, frame, cv::Size(width, height));
}

// Our weight files are in a very simple space delimited format.
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::map<std::string, Weights> weightMap;
	std::ifstream input(file);
	assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while(count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t type, size;
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);
        if (wt.type == DataType::kFLOAT) {
            uint32_t *val = reinterpret_cast<uint32_t*>(sdt_alloc(sizeof(*val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];

            }
            wt.values = val;
        } else if (wt.type == DataType::kHALF) {
            uint16_t *val = reinterpret_cast<uint16_t*>(sdt_alloc(sizeof(*val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}

// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
     void log(nvinfer1::ILogger::Severity severity, const char* msg) override
     {
          // suppress info-level messages
          if (severity == Severity::kINFO) return;

          switch (severity)
               {
               case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
               case Severity::kERROR: std::cerr << "ERROR: "; break;
               case Severity::kWARNING: std::cerr << "WARNING: "; break;
               case Severity::kINFO: std::cerr << "INFO: "; break;
               default: std::cerr << "UNKNOWN: "; break;
               }
          std::cerr << msg << std::endl;
     }
};

static Logger gLogger;

static ILayer*
addFireLayer(INetworkDefinition* network, ITensor &input, int ns1x1, int ne1x1, int ne3x3,
             Weights &wks1x1, Weights &wke1x1, Weights &wke3x3,
             Weights &wbs1x1, Weights &wbe1x1, Weights &wbe3x3)
{
     auto sq1x1 = network->addConvolution(input, ns1x1, DimsHW{1, 1}, wks1x1, wbs1x1);
     assert(sq1x1 != nullptr);
     sq1x1->setStride(DimsHW{1, 1});
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
static ICudaEngine *
createConv1Engine(unsigned int maxBatchSize, IBuilder *builder, DataType dt, const char *wts)
{
     INetworkDefinition* network = builder->createNetwork();

     auto data = network->addInput(INPUT_NAME, dt, DimsCHW{INPUT_C, INPUT_H, INPUT_W});
     assert(data != nullptr);

     std::map<std::string, Weights> weightMap = loadWeights(std::string(wts));
     auto conv1 = network->addConvolution(*data, 64, DimsHW{3, 3},
                                          weightMap["conv1_kernels"],
                                          weightMap["conv1_bias"]);
     assert(conv1 != nullptr);
     // conv1->setStride(DimsHW{2, 2});
     conv1->setStride(DimsHW{1, 1});
     conv1->setPadding(DimsHW{1, 1}); // all kernels of size 3x3 need to set padding 1x1
     auto relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
     assert(relu1 != nullptr);

     // auto pool1 = network->addPooling(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
     // assert(pool1 != nullptr);
     // pool1->setStride(DimsHW{2, 2});
     // pool1->setPadding(DimsHW{1, 1});

     auto pool1_1 = network->addPooling(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
     assert(pool1_1 != nullptr);
     pool1_1->setStride(DimsHW{2, 2});
     pool1_1->setPadding(DimsHW{1, 1});

     auto pool1_2 = network->addPooling(*pool1_1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
     assert(pool1_2 != nullptr);
     pool1_2->setStride(DimsHW{2, 2});
     pool1_2->setPadding(DimsHW{1, 1});

     auto fire2 = addFireLayer(network, *pool1_2->getOutput(0), 16, 64, 64,
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
     pool3->getOutput(0)->setName(CONV1_POOL3_NAME);
     network->markOutput(*pool3->getOutput(0));

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
     assert(pool5 != nullptr);
     pool5->setStride(DimsHW{2, 2});
     pool5->setPadding(DimsHW{1, 1});

     auto fire6 = addFireLayer(network, *pool5->getOutput(0), 48, 192, 192,
                               weightMap["fire6_squeeze1x1_kernels"],
                               weightMap["fire6_expand1x1_kernels"],
                               weightMap["fire6_expand3x3_kernels"],
                               weightMap["fire6_squeeze1x1_biases"],
                               weightMap["fire6_expand1x1_biases"],
                               weightMap["fire6_expand3x3_biases"]);
     fire6->getOutput(0)->setName(CONV1_FIRE6_NAME);
     network->markOutput(*fire6->getOutput(0));

     auto fire7 = addFireLayer(network, *fire6->getOutput(0), 48, 192, 192,
                               weightMap["fire7_squeeze1x1_kernels"],
                               weightMap["fire7_expand1x1_kernels"],
                               weightMap["fire7_expand3x3_kernels"],
                               weightMap["fire7_squeeze1x1_biases"],
                               weightMap["fire7_expand1x1_biases"],
                               weightMap["fire7_expand3x3_biases"]);
     fire7->getOutput(0)->setName(CONV1_FIRE7_NAME);
     network->markOutput(*fire7->getOutput(0));

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

     fire9->getOutput(0)->setName(CONV1_FIRE9_NAME);
     network->markOutput(*fire9->getOutput(0));

     // Build the engine
     builder->setMaxBatchSize(maxBatchSize);
     builder->setMaxWorkspaceSize(1 << 20);

     auto engine = builder->buildCudaEngine(*network);
     // we don't need the network any more
     // network->destroy();	// SIGSEGV, don't know why

     // Once we have built the cuda engine, we can release all of our held memory.
     for (auto &mem : weightMap)
     {
          sdt_free((void*)(mem.second.values));
     }
     return engine;
}

static ICudaEngine *
createConv2Engine(unsigned int maxBatchSize, IBuilder *builder, DataType dt, const char *wts)
{
     INetworkDefinition* network = builder->createNetwork();

     auto split1 = network->addInput(CONV2_SPLIT1_NAME, dt, DimsCHW{SPLIT_C, CONVOUT_H, CONVOUT_W});
     assert(split1 != nullptr);
     auto split2 = network->addInput(CONV2_SPLIT2_NAME, dt, DimsCHW{SPLIT_C, CONVOUT_H, CONVOUT_W});
     assert(split2 != nullptr);
     auto split3 = network->addInput(CONV2_SPLIT3_NAME, dt, DimsCHW{SPLIT_C, CONVOUT_H, CONVOUT_W});
     assert(split3 != nullptr);
     auto split4 = network->addInput(CONV2_SPLIT4_NAME, dt, DimsCHW{SPLIT_C, CONVOUT_H, CONVOUT_W});
     assert(split4 != nullptr);
     auto fire6 = network->addInput(CONV2_FIRE6_NAME, dt, DimsCHW{FIRE6_C, CONVOUT_H, CONVOUT_W});
     assert(fire6 != nullptr);
     auto fire7 = network->addInput(CONV2_FIRE7_NAME, dt, DimsCHW{FIRE7_C, CONVOUT_H, CONVOUT_W});
     assert(fire7 != nullptr);
     auto fire9 = network->addInput(CONV2_FIRE9_NAME, dt, DimsCHW{FIRE9_C, CONVOUT_H, CONVOUT_W});
     assert(fire9 != nullptr);
     auto consts = network->addInput(CONSTS_NAME, dt, DimsCHW{CONSTS_C, CONVOUT_H, CONVOUT_W});
     assert(consts != nullptr);

     std::map<std::string, Weights> weightMap = loadWeights(std::string(wts));

     auto elew6 = network->addElementWise(*fire6, *consts, ElementWiseOperation::kPROD);
     auto elew7 = network->addElementWise(*fire7, *consts, ElementWiseOperation::kPROD);
     ITensor *concat_input[7] = {split4, split3, split2, split1, elew7->getOutput(0), elew6->getOutput(0), fire9};
     auto concat = network->addConcatenation(concat_input, 7);

     auto fire10 = addFireLayer(network, *concat->getOutput(0), 96, 384, 384,
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
     // not need any more, because dropout probability is 1.0 during evaluation
     // ILayer *dropout11 = fire11;

     auto preds = network->addConvolution(*fire11->getOutput(0), CONVOUT_C, DimsHW{3, 3},
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
          sdt_free((void*)(mem.second.values));
     }
     return engine;
}

static ICudaEngine *
createInterpretEngine(unsigned int maxBatchSize, IBuilder *builder, DataType dt)
{
     INetworkDefinition* network = builder->createNetwork();

     auto class_tensor = network->addInput(CLASS_INPUT_NAME, dt, DimsNCHW{ANCHORS_PER_GRID, OUTPUT_CLS_SIZE, CONVOUT_H, CONVOUT_W});
     assert(class_tensor != nullptr);
     // auto confidence_tensor = network->addInput(CONF_INPUT_NAME, dt, DimsNCHW{INPUT_N, 1, 1, CONVOUT_W * CONVOUT_H * ANCHORS_PER_GRID});
     auto confidence_tensor = network->addInput(CONF_INPUT_NAME, dt, DimsCHW{1, 1, INPUT_N * CONVOUT_W * CONVOUT_H * ANCHORS_PER_GRID});
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

// maxBatch - NB must be at least as large as the batch we want to run with)
static void APIToModel(unsigned int maxBatchSize, IHostMemory **conv1ModelStream, IHostMemory **conv2ModelStream, IHostMemory **interpretModelStream, const char *wts)
{
     // create the builder
     IBuilder* builder = createInferBuilder(gLogger);

     // create the model to populate the network, then set the outputs and create an engine
     ICudaEngine* conv1Engine = createConv1Engine(maxBatchSize, builder, DataType::kFLOAT, wts);
     ICudaEngine* conv2Engine = createConv2Engine(maxBatchSize, builder, DataType::kFLOAT, wts);
     ICudaEngine* interpretEngine = createInterpretEngine(maxBatchSize, builder, DataType::kFLOAT);

     assert(conv1Engine != nullptr);
     assert(conv2Engine != nullptr);
     assert(interpretEngine != nullptr);

     // serialize the engine, then close everything down
     (*conv1ModelStream) = conv1Engine->serialize();
     (*conv2ModelStream) = conv2Engine->serialize();
     (*interpretModelStream) = interpretEngine->serialize();
     conv1Engine->destroy();
     conv2Engine->destroy();
     interpretEngine->destroy();
     builder->destroy();
}

// batch size is 1
static void setUpDevice(IExecutionContext *conv1Context, IExecutionContext *conv2Context, IExecutionContext *interpretContext, float* anchors, int batchSize)
{
     const ICudaEngine &conv1Engine = conv1Context->getEngine();
     const ICudaEngine &conv2Engine = conv2Context->getEngine();
     const ICudaEngine &interpretEngine = interpretContext->getEngine();

     assert(conv1Engine.getNbBindings() == 5);
     assert(conv2Engine.getNbBindings() == 9);
     assert(interpretEngine.getNbBindings() == 4);

     // In order to bind the buffers, we need to know the names of the input and output tensors.
     // note that indices are guaranteed to be less than IEngine::getNbBindings()
     inputIndex = conv1Engine.getBindingIndex(INPUT_NAME);
     pool3Index = conv1Engine.getBindingIndex(CONV1_POOL3_NAME);
     fire6OutIndex = conv1Engine.getBindingIndex(CONV1_FIRE6_NAME);
     fire7OutIndex = conv1Engine.getBindingIndex(CONV1_FIRE7_NAME);
     fire9OutIndex = conv1Engine.getBindingIndex(CONV1_FIRE9_NAME);
     split1Index = conv2Engine.getBindingIndex(CONV2_SPLIT1_NAME);
     split2Index = conv2Engine.getBindingIndex(CONV2_SPLIT2_NAME);
     split3Index = conv2Engine.getBindingIndex(CONV2_SPLIT3_NAME);
     split4Index = conv2Engine.getBindingIndex(CONV2_SPLIT4_NAME);
     fire6InIndex = conv2Engine.getBindingIndex(CONV2_FIRE6_NAME);
     fire7InIndex = conv2Engine.getBindingIndex(CONV2_FIRE7_NAME);
     fire9InIndex = conv2Engine.getBindingIndex(CONV2_FIRE9_NAME);
     constsIndex = conv2Engine.getBindingIndex(CONSTS_NAME);
     convoutIndex = conv2Engine.getBindingIndex(CONVOUT_NAME);
     classInputIndex = interpretEngine.getBindingIndex(CLASS_INPUT_NAME);
     confInputIndex = interpretEngine.getBindingIndex(CONF_INPUT_NAME);
     classOutputIndex = interpretEngine.getBindingIndex(CLASS_OUTPUT_NAME);
     confOutputIndex = interpretEngine.getBindingIndex(CONF_OUTPUT_NAME);

     // create GPU buffers and a stream
     anchorsNum = batchSize * CONVOUT_W * CONVOUT_H * ANCHORS_PER_GRID;
     size_t inputSize = batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float);
     size_t pool3Size = batchSize * SPLIT_C * CONVOUT_H * CONVOUT_W * 4 * sizeof(float);
     size_t fire9Size = batchSize * FIRE9_C * CONVOUT_H * CONVOUT_W * sizeof(float);
     size_t fire6Size = batchSize * FIRE6_C * CONVOUT_H * CONVOUT_W * sizeof(float);
     size_t fire7Size = batchSize * FIRE7_C * CONVOUT_H * CONVOUT_W * sizeof(float);
     size_t splitSize = batchSize * SPLIT_C * CONVOUT_H * CONVOUT_W * sizeof(float);
     size_t constsSize = batchSize * CONSTS_C * CONVOUT_H * CONVOUT_W * sizeof(float);
     size_t convoutSize = batchSize * CONVOUT_H * CONVOUT_W * CONVOUT_C * sizeof(float);
     size_t classInputSize = batchSize * CONVOUT_H * CONVOUT_W * CLASS_SLICE_C * sizeof(float);
     size_t confInputSize = batchSize * CONVOUT_H * CONVOUT_W * CONF_SLICE_C * sizeof(float);
     size_t bboxInputSize = batchSize * CONVOUT_H * CONVOUT_W * BBOX_SLICE_C * sizeof(float);
     size_t classOutputSize = batchSize * OUTPUT_CLS_SIZE * anchorsNum * sizeof(float);
     size_t confOutputSize = anchorsNum * sizeof(float);
     CHECK(cudaMalloc(&conv1Buffers[inputIndex], inputSize));
     CHECK(cudaMalloc(&conv1Buffers[pool3Index], pool3Size));
     CHECK(cudaMalloc(&conv1Buffers[fire6OutIndex], fire6Size));
     CHECK(cudaMalloc(&conv1Buffers[fire7OutIndex], fire7Size));
     CHECK(cudaMalloc(&conv1Buffers[fire9OutIndex], fire9Size));
     CHECK(cudaMalloc(&conv2Buffers[split1Index], splitSize));
     CHECK(cudaMalloc(&conv2Buffers[split2Index], splitSize));
     CHECK(cudaMalloc(&conv2Buffers[split3Index], splitSize));
     CHECK(cudaMalloc(&conv2Buffers[split4Index], splitSize));
     conv2Buffers[fire6InIndex] = conv1Buffers[fire6OutIndex];
     conv2Buffers[fire7InIndex] = conv1Buffers[fire7OutIndex];
     conv2Buffers[fire9InIndex] = conv1Buffers[fire9OutIndex];
     CHECK(cudaMalloc(&conv2Buffers[constsIndex], constsSize));
     // CHECK(cudaMalloc(&conv2Buffers[fire9InIndex], fire9Size));
     CHECK(cudaMalloc(&conv2Buffers[convoutIndex], convoutSize));
     CHECK(cudaMalloc(&interpretBuffers[classInputIndex], classInputSize));
     CHECK(cudaMalloc(&interpretBuffers[confInputIndex], confInputSize));
     CHECK(cudaMalloc(&bboxInput, bboxInputSize));
     CHECK(cudaMalloc(&interpretBuffers[classOutputIndex], classOutputSize));
     CHECK(cudaMalloc(&interpretBuffers[confOutputIndex], confOutputSize));

     int pool3_dims[] = {batchSize, SPLIT_C, CONVOUT_H*2, CONVOUT_W*2};
     int split_dims[] = {batchSize, SPLIT_C, CONVOUT_H, CONVOUT_W};
     int convout_dims[] = {batchSize, CONVOUT_C, CONVOUT_H, CONVOUT_W};
     int classInputDims[] = {batchSize, CLASS_SLICE_C, CONVOUT_H, CONVOUT_W};
     int confInputDims[] = {batchSize, CONF_SLICE_C, CONVOUT_H, CONVOUT_W};
     int bboxInputDims[] = {batchSize, BBOX_SLICE_C, CONVOUT_H, CONVOUT_W};
     int classOutputDims[] = {batchSize, ANCHORS_PER_GRID, OUTPUT_CLS_SIZE, CONVOUT_H, CONVOUT_W};
     int classTransDims[] = {batchSize, CONVOUT_H, CONVOUT_W, ANCHORS_PER_GRID, OUTPUT_CLS_SIZE};
     int confOutputDims[] = {batchSize, ANCHORS_PER_GRID, 1, CONVOUT_H, CONVOUT_W};
     int confTransDims[] = {batchSize, CONVOUT_H, CONVOUT_W, ANCHORS_PER_GRID, 1};
     int bboxOutputDims[] = {batchSize, ANCHORS_PER_GRID, OUTPUT_BBOX_SIZE, CONVOUT_H, CONVOUT_W};
     int bboxTransDims[] = {batchSize, CONVOUT_H, CONVOUT_W, ANCHORS_PER_GRID, OUTPUT_BBOX_SIZE};
     int transAxes[] = {0, 3, 4, 1, 2};
     transAxesDevice = (int *)cloneMem(transAxes, sizeof(int) * 5, H2D);
     pool3Tensor = createTensor((float *)conv1Buffers[pool3Index], 4, pool3_dims);
     split1Tensor = createTensor((float *)conv2Buffers[split1Index], 4, split_dims);
     split2Tensor = createTensor((float *)conv2Buffers[split2Index], 4, split_dims);
     split3Tensor = createTensor((float *)conv2Buffers[split3Index], 4, split_dims);
     split4Tensor = createTensor((float *)conv2Buffers[split4Index], 4, split_dims);
     convoutTensor = createTensor((float *)conv2Buffers[convoutIndex], 4, convout_dims);
     classInputTensor = createTensor((float *)interpretBuffers[classInputIndex], 4, classInputDims);
     confInputTensor = createTensor((float *)interpretBuffers[confInputIndex], 4, confInputDims);
     bboxInputTensor = createTensor(bboxInput, 4, bboxInputDims);
     classOutputTensor = createTensor((float *)interpretBuffers[classOutputIndex], 5, classOutputDims);
     confOutputTensor = createTensor((float *)interpretBuffers[confOutputIndex], 5, confOutputDims);
     bboxOutputTensor = reshapeTensor(bboxInputTensor, 5, bboxOutputDims);
     classTransTensor = mallocTensor(5, classTransDims, DEVICE);
     confTransTensor = mallocTensor(5, confTransDims, DEVICE);
     bboxTransTensor = mallocTensor(5, bboxTransDims, DEVICE);
     CHECK(cudaMalloc(&classTransWorkspace[0], sizeof(int) * classTransTensor->ndim * classTransTensor->len));
     CHECK(cudaMalloc(&classTransWorkspace[1], sizeof(int) * classTransTensor->ndim * classTransTensor->len));
     CHECK(cudaMalloc(&confTransWorkspace[0], sizeof(int) * confTransTensor->ndim * confTransTensor->len));
     CHECK(cudaMalloc(&confTransWorkspace[1], sizeof(int) * confTransTensor->ndim * confTransTensor->len));
     CHECK(cudaMalloc(&bboxTransWorkspace[0], sizeof(int) * bboxTransTensor->ndim * bboxTransTensor->len));
     CHECK(cudaMalloc(&bboxTransWorkspace[1], sizeof(int) * bboxTransTensor->ndim * bboxTransTensor->len));

     size_t anchorsDeviceSize = anchorsNum * ANCHOR_SIZE * sizeof(float);
     anchorsDevice = (float *)cloneMem(anchors, anchorsDeviceSize, H2D);

     int reduceMaxResDims[] = {batchSize, CONVOUT_H, CONVOUT_W, ANCHORS_PER_GRID, 1};
     int reduceArgResDims[] = {batchSize, CONVOUT_H, CONVOUT_W, ANCHORS_PER_GRID, 1};
     int mulResDims[] = {batchSize, CONVOUT_H, CONVOUT_W, ANCHORS_PER_GRID, 1};
     int bboxResDims[] = {batchSize, CONVOUT_H, CONVOUT_W, ANCHORS_PER_GRID, OUTPUT_BBOX_SIZE};
     // int anchorsDeviceDims[] = {batchSize, ANCHORS_PER_GRID, ANCHOR_SIZE, CONVOUT_H, CONVOUT_W};
     int anchorsDeviceDims[] = {batchSize, CONVOUT_H, CONVOUT_W, ANCHORS_PER_GRID, ANCHOR_SIZE};
     reduceMaxResTensor = mallocTensor(5, reduceMaxResDims, DEVICE);
     reduceArgResTensor = mallocTensor(5, reduceArgResDims, DEVICE);
     mulResTensor = mallocTensor(5, mulResDims, DEVICE);
     bboxResTensor = mallocTensor(5, bboxResDims, DEVICE);
     anchorsDeviceTensor = createTensor(anchorsDevice, 5, anchorsDeviceDims);

     int *orderHost = (int *)sdt_alloc(anchorsNum * sizeof(int));
     for (int i = 0; i < anchorsNum; i++)
          orderHost[i] = i;
     orderDevice = (int *)cloneMem(orderHost, anchorsNum * sizeof(int), H2D);
     orderDeviceTmp = (int *)cloneMem(orderHost, anchorsNum * sizeof(int), H2D);
     sdt_free(orderHost);

     int finalProbsDims[] = {batchSize, TOP_N_DETECTION, 1};
     int finalClassDims[] = {batchSize, TOP_N_DETECTION, 1};
     int finalBboxDims[] = {batchSize, TOP_N_DETECTION, OUTPUT_BBOX_SIZE};
     // finalProbsTensor shares data with mulResTensor
     finalProbsTensor = createTensor(mulResTensor->data, 3, finalProbsDims);
     // finalProbsTensor = mallocTensor(3, finalProbsDims, DEVICE);
     finalClassTensor = mallocTensor(3, finalClassDims, DEVICE);
     finalBboxTensor = mallocTensor(3, finalBboxDims, DEVICE);

     CHECK(cudaStreamCreate(&stream));
     CHECK(cudaEventCreate(&start_detect));
     CHECK(cudaEventCreate(&stop_detect));
     CHECK(cudaEventCreate(&start_misc));
     CHECK(cudaEventCreate(&stop_misc));
}

// batch size is 1
static void doInference(IExecutionContext *conv1Context, IExecutionContext *conv2Context, IExecutionContext *interpretContext, float* input, int inputSize, float *consts, size_t constsSize, int img_width, int img_height, int x_shift, int y_shift, struct predictions *preds, int batchSize)
{
     CHECK(cudaEventRecord(start_detect, 0));

     // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
     CHECK(cudaMemcpyAsync(conv1Buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice, stream));

     static int constsflag = 0;
     if (!constsflag) {
          CHECK(cudaMemcpyAsync(conv2Buffers[constsIndex], consts, constsSize, cudaMemcpyHostToDevice, stream));
          constsflag = 1;
     }

     conv1Context->enqueue(batchSize, conv1Buffers, stream, nullptr);
     splitTensor2x2(pool3Tensor, split1Tensor, split2Tensor, split3Tensor, split4Tensor);

     conv2Context->enqueue(batchSize, conv2Buffers, stream, nullptr);
     sliceTensor(convoutTensor, classInputTensor, 1, 0, CLASS_SLICE_C);
     sliceTensor(convoutTensor, confInputTensor, 1, CLASS_SLICE_C, CONF_SLICE_C);
     sliceTensor(convoutTensor, bboxInputTensor, 1, CLASS_SLICE_C + CONF_SLICE_C, BBOX_SLICE_C);

     interpretContext->enqueue(batchSize, interpretBuffers, stream, nullptr);
     transposeTensor(classOutputTensor, classTransTensor, transAxesDevice, classTransWorkspace);
     transposeTensor(confOutputTensor, confTransTensor, transAxesDevice, confTransWorkspace);
     transposeTensor(bboxOutputTensor, bboxTransTensor, transAxesDevice, bboxTransWorkspace);
     reduceArgMax(classTransTensor, reduceMaxResTensor, reduceArgResTensor, 4);
     multiplyElement(reduceMaxResTensor, confTransTensor, mulResTensor);
     transformBboxSQD(bboxTransTensor, anchorsDeviceTensor, bboxResTensor, INPUT_W, INPUT_H, img_width, img_height, x_shift, y_shift);

     CHECK(cudaEventRecord(stop_detect, 0));
     CHECK(cudaEventSynchronize(stop_detect));
     CHECK(cudaEventElapsedTime(&timeDetect, start_detect, stop_detect));

#ifdef DEBUG
     saveDeviceTensor("data/debug/convoutTensor.txt", convoutTensor, "%15.6e");
     saveDeviceTensor("data/debug/classInputTensor.txt", classInputTensor, "%15.6e");
     saveDeviceTensor("data/debug/confInputTensor.txt", confInputTensor, "%15.6e");
     saveDeviceTensor("data/debug/bboxInputTensor.txt", bboxInputTensor, "%15.6e");
     saveDeviceTensor("data/debug/classOutputTensor.txt", classOutputTensor, "%15.6e");
     saveDeviceTensor("data/debug/confOutputTensor.txt", confOutputTensor, "%15.6e");
     saveDeviceTensor("data/debug/bboxOutputTensor.txt", bboxOutputTensor, "%15.6e");
     saveDeviceTensor("data/debug/mulResTensor.txt", mulResTensor, "%15.6e");
     saveDeviceTensor("data/debug/reduceMaxResTensor.txt", reduceMaxResTensor, "%15.6e");
     saveDeviceTensor("data/debug/reduceArgResTensor.txt", reduceArgResTensor, "%15.6e");
     saveDeviceTensor("data/debug/bboxResTensor.txt", bboxResTensor, "%15.6e");
     saveDeviceTensor("data/debug/confTransTensor.txt", confTransTensor, "%15.6e");
     saveDeviceTensor("data/debug/classTransTensor.txt", classTransTensor, "%15.6e");
     saveDeviceTensor("data/debug/bboxTransTensor.txt", bboxTransTensor, "%15.6e");
     saveDeviceTensor("data/debug/anchorsDeviceTensor.txt", anchorsDeviceTensor, "%15.6e");

     // int classInputDims2[] = {batchSize, 9, 3, CONVOUT_W*CONVOUT_H};
     // int classInputDims3[] = {batchSize, 3, 9, CONVOUT_W*CONVOUT_H};
     // Tensor *classInput2 = reshapeTensor(classInputTensor, 4, classInputDims2);
     // Tensor *classInput3 = reshapeTensor(classInputTensor, 4, classInputDims3);
     // saveDeviceTensor("data/debug/classInputDims2.txt", classInput2, "%15.6e");
     // saveDeviceTensor("data/debug/classInputDims3.txt", classInput3, "%15.6e");
#endif
     // filter top-n-detection
     CHECK(cudaEventRecord(start_misc, 0));
     CHECK(cudaMemcpy(orderDeviceTmp, orderDevice, anchorsNum * sizeof(int), cudaMemcpyDeviceToDevice));
     // tensorIndexSort(mulResTensor, orderDeviceTmp);
     tensorIndexSort(confTransTensor, orderDeviceTmp);
     // already sort mulResTensor (sharing data with finalProbsTensor), so we can skip this
     // pickElements(mulResTensor->data, finalProbsTensor->data, 1, orderDeviceTmp, TOP_N_DETECTION);
     pickElements(mulResTensor->data, finalProbsTensor->data, 1, orderDeviceTmp, TOP_N_DETECTION);
     pickElements(reduceArgResTensor->data, finalClassTensor->data, 1, orderDeviceTmp, TOP_N_DETECTION);
     pickElements(bboxResTensor->data, finalBboxTensor->data, OUTPUT_BBOX_SIZE, orderDeviceTmp, TOP_N_DETECTION);

#ifdef DEBUG
     FILE * sort_file = fopen("data/debug/orderDevice.txt", "w");
     int *orderHost2 = (int *)cloneMem(orderDevice, anchorsNum * sizeof(int), D2H);
     for (int i = 0; i < anchorsNum; i++)
          fprintf(sort_file, "%d\n", orderHost2[i]);
     fclose(sort_file);
     sdt_free(orderHost2);
     saveDeviceTensor("data/debug/finalClassTensor.txt", finalClassTensor, "%15.6e");
     saveDeviceTensor("data/debug/finalProbsTensor.txt", finalProbsTensor, "%15.6e");
     saveDeviceTensor("data/debug/finalBboxTensor.txt", finalBboxTensor, "%15.6e");
#endif

     CHECK(cudaMemcpyAsync(preds->prob, finalProbsTensor->data, finalProbsTensor->len*sizeof(float), cudaMemcpyDeviceToHost, stream));
     CHECK(cudaMemcpyAsync(preds->klass, finalClassTensor->data, finalClassTensor->len*sizeof(float), cudaMemcpyDeviceToHost, stream));
     CHECK(cudaMemcpyAsync(preds->bbox, finalBboxTensor->data, finalBboxTensor->len*sizeof(float), cudaMemcpyDeviceToHost, stream));
     CHECK(cudaStreamSynchronize(stream));
}

static void cleanUp()
{
     // timer destroy
     CHECK(cudaEventDestroy(start_detect));
     CHECK(cudaEventDestroy(stop_detect));
     CHECK(cudaEventDestroy(start_misc));
     CHECK(cudaEventDestroy(stop_misc));

     // release the stream and the buffers
     CHECK(cudaStreamDestroy(stream));
     CHECK(cudaFree(conv1Buffers[inputIndex]));
     CHECK(cudaFree(conv1Buffers[pool3Index]));
     CHECK(cudaFree(conv1Buffers[fire6OutIndex]));
     CHECK(cudaFree(conv1Buffers[fire7OutIndex]));
     CHECK(cudaFree(conv1Buffers[fire9OutIndex]));
     CHECK(cudaFree(conv2Buffers[split1Index]));
     CHECK(cudaFree(conv2Buffers[split2Index]));
     CHECK(cudaFree(conv2Buffers[split3Index]));
     CHECK(cudaFree(conv2Buffers[split4Index]));
     CHECK(cudaFree(conv2Buffers[constsIndex]));
     // CHECK(cudaFree(conv2Buffers[fire9InIndex]));
     CHECK(cudaFree(conv2Buffers[convoutIndex]));
     CHECK(cudaFree(interpretBuffers[classInputIndex]));
     CHECK(cudaFree(interpretBuffers[confInputIndex]));
     CHECK(cudaFree(bboxInput));
     CHECK(cudaFree(interpretBuffers[classOutputIndex]));
     CHECK(cudaFree(interpretBuffers[confOutputIndex]));
     CHECK(cudaFree(classTransTensor->data));
     CHECK(cudaFree(confTransTensor->data));
     CHECK(cudaFree(bboxTransTensor->data));
     CHECK(cudaFree(classTransWorkspace[0]));
     CHECK(cudaFree(classTransWorkspace[1]));
     CHECK(cudaFree(confTransWorkspace[0]));
     CHECK(cudaFree(confTransWorkspace[1]));
     CHECK(cudaFree(bboxTransWorkspace[0]));
     CHECK(cudaFree(bboxTransWorkspace[1]));
     CHECK(cudaFree(reduceMaxResTensor->data));
     CHECK(cudaFree(reduceArgResTensor->data));
     CHECK(cudaFree(mulResTensor->data));
     CHECK(cudaFree(bboxResTensor->data));
     CHECK(cudaFree(transAxesDevice));
     CHECK(cudaFree(anchorsDevice));
     CHECK(cudaFree(orderDevice));
     CHECK(cudaFree(orderDeviceTmp));
     // already freed mulResTensor data (sharing data with finalProbsTensor), so skip this
     // CHECK(cudaFree(finalProbsTensor->data));
     CHECK(cudaFree(finalClassTensor->data));
     CHECK(cudaFree(finalBboxTensor->data));

     // free remaining tensor structure (already freed their data)
     freeTensor(pool3Tensor, 0);
     freeTensor(split1Tensor, 0);
     freeTensor(split2Tensor, 0);
     freeTensor(split3Tensor, 0);
     freeTensor(split4Tensor, 0);
     freeTensor(convoutTensor, 0);
     freeTensor(classInputTensor, 0);
     freeTensor(confInputTensor, 0);
     freeTensor(bboxInputTensor, 0);
     freeTensor(classOutputTensor, 0);
     freeTensor(confOutputTensor, 0);
     freeTensor(bboxOutputTensor, 0);
     freeTensor(reduceMaxResTensor, 0);
     freeTensor(reduceArgResTensor, 0);
     freeTensor(mulResTensor, 0);
     freeTensor(bboxResTensor, 0);
     freeTensor(anchorsDeviceTensor, 0);
     freeTensor(finalClassTensor, 0);
     freeTensor(finalProbsTensor, 0);
     freeTensor(finalBboxTensor, 0);
}

// rearrange image data to [N, C, H, W] order
static float *prepareData(float *data, cv::Mat &frame)
{
     assert(data && !frame.empty());
     unsigned int volChl = INPUT_H*INPUT_W;
     for (int c = 0; c < INPUT_C; ++c)
     {
          // the color image to input should be in BGR order
          for (unsigned j = 0; j < volChl; ++j)
               data[c*volChl + j] = float(frame.data[j*INPUT_C+c]) - PIXEL_MEAN[c];
     }

     return data;
}

static float *prepareAnchors(const float *anchor_shape, int width, int height, int N, int H, int W, int B)
{
     assert(anchor_shape);
     float center_x[W], center_y[H];
     float anchors[H * W * B * 4];
     int i, j, k;

     // for (i = 1; i <= W; i++)
     //      center_x[i-1] = i * width / (W + 1.0);
     // for (i = 1; i <= H; i++)
     //      center_y[i-1] = i * height / (H + 1.0);

     for (i = 1; i <= W; i++)
          center_x[i-1] = (2*i-1) * width / (W*2.0);
     for (i = 1; i <= H; i++)
          center_y[i-1] = (2*i-1) * height / (H*2.0);


     int h_vol = W * B * 4;
     int w_vol = B * 4;
     int b_vol = 4;
     for (i = 0; i < H; i++) {
          for (j = 0; j < W; j++) {
               for (k = 0; k < B; k++) {
                    anchors[i*h_vol+j*w_vol+k*b_vol] = center_x[j];
                    anchors[i*h_vol+j*w_vol+k*b_vol+1] = center_y[i];
                    anchors[i*h_vol+j*w_vol+k*b_vol+2] = anchor_shape[k*2];
                    anchors[i*h_vol+j*w_vol+k*b_vol+3] = anchor_shape[k*2+1];
               }
          }
     }

     float *ret = (float *)repeatMem(anchors, sizeof(float)*4*B*H*W, N, H2H);
     assert(ret);
     return ret;
}

static float *prepareConsts(float value, int len)
{
     assert(len > 0);
     float *array;
     int i;

     array = (float *)sdt_alloc(sizeof(float) * len);
     for (i = 0; i < len; i++)
          array[i] = value;

     return array;
}

static void detectionFilter(struct predictions *preds, float nms_thresh, float prob_thresh)
{
     assert(preds->bbox && preds->klass && preds->prob && preds->keep);

     // int j;
     int i, num = preds->num;
     int *keep = preds->keep;
     // float *klass = preds->klass;
     // float *bbox = preds->bbox;
     for (i = 0; i < num; i++)
          keep[i] = 0;
     keep[0] = 1;
     // for (i = 0; i < num; i++)
     //      keep[i] = 1;
     // keep[0] = 1;
     // for (i = 0; i < num; i++) {
     //      // keep[i] = 1;
     //      // if (probs[i] < prob_thresh) {
     //      //      keep[i] = 0;
     //      //      continue;
     //      // }
     //      // for (j = i - 1; j >= 0 ; j--) {
     //      for (j = i + 1; j < num; j++) {
     //           if (!keep[j] || klass[i] != klass[j])
     //                continue;
     //           if (computeIou(&bbox[i*OUTPUT_BBOX_SIZE],&bbox[j*OUTPUT_BBOX_SIZE]) > nms_thresh)
     //                     keep[j] = 0;
     //      }
     // }
     CHECK(cudaEventRecord(stop_misc, 0));
     CHECK(cudaEventSynchronize(stop_misc));
     CHECK(cudaEventElapsedTime(&timeMisc, start_misc, stop_misc));
}

static void sprintResult(char *buf, struct predictions *preds)
{
     assert(buf && preds->bbox && preds->klass && preds->prob && preds->keep);

     int i;
     float *bbox;
     buf[0] = '\0';
     for (i = 0; i < preds->num; i++) {
          // if (!preds->keep[i] || preds->prob[i] < PLOT_PROB_THRESH)
          if (!preds->keep[i])
               continue;
          bbox = &preds->bbox[i * OUTPUT_BBOX_SIZE];
          sprintf(buf+strlen(buf), "%s -1 -1 0.0 %.2f %.2f %.2f %.2f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %.3f\n",
                  CLASS_NAMES[(int)preds->klass[i]], bbox[0], bbox[1], bbox[2], bbox[3], preds->prob[i]);
     }
     // printf("%s", buf);
}

static void fprintResult(FILE *fp, struct predictions *preds)
{
     assert(fp && preds->bbox && preds->klass && preds->prob && preds->keep);

     int i;
     float *bbox;
     for (i = 0; i < preds->num; i++) {
          // if (!preds->keep[i] || preds->prob[i] < PLOT_PROB_THRESH)
          if (!preds->keep[i])
               continue;
          bbox = &preds->bbox[i * OUTPUT_BBOX_SIZE];
          fprintf(fp, "%s -1 -1 0.0 %.2f %.2f %.2f %.2f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %.3f\n",
                  CLASS_NAMES[(int)preds->klass[i]], bbox[0], bbox[1], bbox[2], bbox[3], preds->prob[i]);
     }
}

// static void drawBbox(cv::Mat &frame, struct predictions *preds)
// {
//      assert(!frame.empty() && preds->bbox && preds->klass && preds->prob && preds->keep);
//      int i;
//      char *prob_s = (char *)sdt_alloc(32);
//      float *bbox;
//      for (i = 0; i < preds->num; i++) {
//           // if (!preds->keep[i] || preds->prob[i] < PLOT_PROB_THRESH)
//           if (!preds->keep[i])
//                continue;
//           bbox = &preds->bbox[i * OUTPUT_BBOX_SIZE];
//           cv::rectangle(frame, cv::Point(bbox[0], bbox[1]), cv::Point(bbox[2], bbox[3]), cv::Scalar(0, 255, 0));
//           sprintf(prob_s, "%.2f", preds->prob[i]);
//           cv::putText(frame, std::string(CLASS_NAMES[(int)preds->klass[i]]) + ": " + std::string(prob_s), cv::Point(bbox[0], bbox[1]), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
//      }
//      sdt_free(prob_s);
// }

static size_t inputSize;
static size_t constsSize;
static float *data;
static float *anchors;
static float *consts;
static struct predictions preds;
static IHostMemory *conv1ModelStream{ nullptr };
static IHostMemory *conv2ModelStream{ nullptr };
static IHostMemory *interpretModelStream{ nullptr };
static IRuntime* runtime;
static ICudaEngine* conv1Engine;
static ICudaEngine* conv2Engine;
static ICudaEngine* interpretEngine;
static IExecutionContext *conv1Context;
static IExecutionContext *conv2Context;
static IExecutionContext *interpretContext;

void sdt_infer_init(const char *wts)
{
     // maloc host memory
     inputSize = sizeof(float) * INPUT_C * INPUT_H * INPUT_W;
     constsSize = sizeof(float) * CONSTS_C * CONVOUT_H * CONVOUT_W;
     data = (float *)sdt_alloc(inputSize);
     anchors = prepareAnchors(ANCHOR_SHAPE, INPUT_W, INPUT_H, INPUT_N, CONVOUT_H, CONVOUT_W, ANCHORS_PER_GRID);
     consts = prepareConsts(CONSTS_VALUE, CONSTS_C * CONVOUT_H * CONVOUT_W);
     preds.prob = (float *)sdt_alloc(sizeof(float) * TOP_N_DETECTION);
     preds.klass = (float *)sdt_alloc(sizeof(float) * TOP_N_DETECTION);
     preds.bbox = (float *)sdt_alloc(sizeof(float) * TOP_N_DETECTION * OUTPUT_BBOX_SIZE);
     preds.keep = (int *)sdt_alloc(sizeof(int) * TOP_N_DETECTION);
     preds.num = TOP_N_DETECTION;
     timeDetect = 0;
     timeMisc = 0;

     // create engines
     APIToModel(INPUT_N, &conv1ModelStream, &conv2ModelStream, &interpretModelStream, wts);

     // deserialize engines
     runtime = createInferRuntime(gLogger);
     conv1Engine = runtime->deserializeCudaEngine(conv1ModelStream->data(), conv1ModelStream->size(), nullptr);
     conv2Engine = runtime->deserializeCudaEngine(conv2ModelStream->data(), conv2ModelStream->size(), nullptr);
     interpretEngine = runtime->deserializeCudaEngine(interpretModelStream->data(), interpretModelStream->size(), nullptr);
     conv1Context = conv1Engine->createExecutionContext();
     conv2Context = conv2Engine->createExecutionContext();
     interpretContext = interpretEngine->createExecutionContext();

     // malloc device memory
     setUpDevice(conv1Context, conv2Context, interpretContext, anchors, INPUT_N);
}

void sdt_infer_detect(unsigned char *input, int height, int width, int x_shift, int y_shift,
                char *res_str, FILE *res_fp, struct predictions **res_preds)
{
     cv::Mat frame_origin, frame;
     float img_height, img_width;
     unsigned char *input_copy;
     size_t input_size;

     input_size = sizeof(char) * height * width * 3;
     input_copy = (unsigned char *)sdt_alloc(input_size);
     memcpy(input_copy, input, input_size);
     frame_origin = cv::Mat(height, width, CV_8UC(3), input_copy);
     frame = cv::Mat(INPUT_H, INPUT_W, CV_8UC(3));
     preprocessFrame(frame, frame_origin, INPUT_W, INPUT_H, &img_width, &img_height);
     prepareData(data, frame);
     doInference(conv1Context, conv2Context, interpretContext, data, inputSize, consts, constsSize, img_width, img_height, x_shift, y_shift, &preds, INPUT_N);
     detectionFilter(&preds, NMS_THRESH, PROB_THRESH);
     // drawBbox(frame_origin, &preds);
     // cv::imshow("detection", frame_origin);
     // int key = cv::waitKey(1000);
     // if (key == ' ') {
     //      cv::waitKey(0);
     // } else if (key == 'q' || key == 27) { // 27 is the ASCII code of ESC
     //      exit(1);
     // }
     // printf("new img\n");
     // fprintResult(stdout, &preds);
     if (res_str)
          sprintResult(res_str, &preds);
     if (res_fp)
          fprintResult(res_fp, &preds);
     if (res_preds)
          *res_preds = &preds;

     // char buf[1024];
     // sprintResult(buf, &preds);
     sdt_free(input_copy);
     frame_origin.release();
     frame.release();
}

void sdt_infer_cleanup(void)
{
     // clean up device memory
     cleanUp();

     // destroy the engine
     conv1Context->destroy();
     conv2Context->destroy();
     interpretContext->destroy();
     conv1Engine->destroy();
     conv2Engine->destroy();
     interpretEngine->destroy();
     runtime->destroy();

     // clean up host memory
     sdt_free(data);
     sdt_free(anchors);
     sdt_free(consts);
     sdt_free(preds.prob);
     sdt_free(preds.klass);
     sdt_free(preds.bbox);
     sdt_free(preds.keep);
}

float sdt_infer_get_time_detect(void)
{
     return timeDetect;
}

float sdt_infer_get_time_misc(void)
{
     return timeMisc;
}
