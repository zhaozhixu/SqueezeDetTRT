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

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "common.h"
#include "tensorUtil.h"
#include "trtUtil.h"

static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

static const int INPUT_N = 20;
static const int INPUT_C = 3;
static const int INPUT_H = 384;
static const int INPUT_W = 1248;
static const int CONVOUT_C = 72;
static const int CONVOUT_H = 24;
static const int CONVOUT_W = 78;
static const int CLASS_SLICE_C = 27;
static const int CONFIDENCE_SLICE_C = 9;
static const int BBOX_DELTA_SLICE_C = 36;
static const int ANCHORS_PER_GRID = 9;
static const int OUTPUT_CLS_SIZE = 3;
static const int OUTPUT_BBOX_SIZE = 4;
static const int TOP_N_DECTION = 64;

const char* INPUT_NAME0 = "data";
const char* CONVOUT__NAME0 = "conv_out";
const char* INTERPRET_INPUT_NAME0 = "class_slice";
const char* INTERPRET_INPUT_NAME1 = "confidence_slice";
const char* INTERPRET_INPUT_NAME2 = "bbox_slice";
const char* INTERPRET_OUTPUT_NAME0 = "bbox_delta";
const char* INTERPRET_OUTPUT_NAME1 = "pred_class_probs";
const char* INTERPRET_OUTPUT_NAME2 = "pred_confidence_score";

const float ANCHOR_SHAPE[] = {36, 37, 366, 174, 115, 59, /* w x h, 2 elements one group*/
                              162, 87, 38, 90, 258, 173,
                              224, 108, 78, 170, 72, 43};

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"samples/squeezeDetTrt/data/"};
    return locateFile(input, dirs);
}

IConvolutionLayer*
addFireLayer(INetworkDefinition* network, ITensor& input, int ns1x1, int ne1x1, int ne3x3,
             Weights wks1x1, Weights wke1x1, Weights wke3x3,
             Weights wbs1x1, Weights wbe1x1, Weights wbe3x3)
{
    auto sq1x1 = network->addConvolution(*input->getOutput(0), ns1x1, DimsHW{1, 1}, wks1x1, wbs1x1);
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

	auto data = network->addInput(INPUT_BLOB_NAME, dt, DimsCHW{INPUT_C, INPUT_H, INPUT_W});
	assert(data != nullptr);

    std::map<std::string, Weights> weightMap = loadWeights(locateFile("squeezedettrt.wts")); // ?
	auto conv1 = network->addConvolution(*data->getOutput(0), 64, DimsHW{3, 3},
										 weightMap["conv1filter"],
										 weightMap["conv1bias"]);
	assert(conv1 != nullptr);
	conv1->setStride(DimsHW{2, 2});
    auto relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1 != nullptr);

	auto pool1 = network->addPooling(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
	assert(pool1 != nullptr);
	pool1->setStride(DimsHW{2, 2});

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
    auto dropout11 = fire11;

    auto preds = network->addConvolution(*dropout11->getOutput(0), NUM_OUTPUT, DimsHW{3, 3},
										 weightMap["conv12_kernels"],
										 weightMap["conv12_biases"]);
    assert(preds != nullptr);
    preds->setStride(DimsHW{1, 1}); // what is xavier, stddev?

	prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	network->markOutput(*prob->getOutput(0));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);

	auto engine = builder->buildCudaEngine(*network);
	// we don't need the network any more
	network->destroy();

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

    auto class_tensor = network->addInput(CONVOUT_BLOB_NAME0, dt, DimsNCHW{INPUT_N, CLASS_SLICE_C, CONVOUT_H, CONVOUT_W});
    assert(class_tensor != nullptr);
    auto confidence_tensor = network->addInput(CONVOUT_BLOB_NAME1, dt, DimsNCHW{INPUT_N, CONFIDENCE_SLICE_C, CONVOUT_H, CONVOUT_W});
    assert(confidence_tensor != nullptr);
    auto bbox_delta_tensor = network->addInput(CONVOUT_BLOB_NAME2, dt, DimsNCHW{INPUT_N, BBOX_DELTA_SLICE_C, CONVOUT_H, CONVOUT_W});
    assert(bbox_delta_tensor != nullptr);

    Reshape *class_reshape1_plugin = new Reshape(DimsCHW{OUTPUT_CLS_SIZE, 1, 336960});

    auto class_reshape1 = network->addReshape(*class_tensor, );
    assert(class_reshape != nullptr);
    auto class_softmax = network->addSoftMax(*class_reshape->getOutput(0));
    assert(class_softmax != nullptr);
    auto class_reshape2 = network->addReshape(*class_softmax->getOutput(0), DimsNCHW{INPUT_N, OUTPUT_CLS_SIZE, 1, 16848});
    assert(class_reshape2 != nullptr);
    auto pred_class_probs = class_reshape2;

    auto conf_reshape = network->addReshape(*confidence_tensor, DimsNCHW{INPUT_N, 1, 1, 16848});
    assert(conf_reshape != nullptr);
    auto pred_conf = network->addActivation(*conf_reshape->getOutput(0), ActivationType::kSIGMOID);
    assert(pred_conf != nullptr);

    auto pred_bbox_delta = network->addReshape(*bbox_delta_tensor, DimsNCHW{INPUT_N, OUTPUT_BBOX_SIZE, 1, 16848});
    assert(pred_bbox_delta != nullptr);

    prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*prob->getOutput(0));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);

    auto engine = builder->buildCudaEngine(*network);
    // we don't need the network any more
    network->destroy();

    // Once we have built the cuda engine, we can release all of our held memory.
    for (auto &mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
    return engine;
}

void doInference(IExecutionContext& convContext, IExecutionContext& interpretContext, float* input, float *output, int batchSize)
{
	const ICudaEngine& convEngine = convContext.getEngine();
    const ICudaEngine& interpretEngine = interpretContext.getEngine();

	assert(convEngine.getNbBindings() == 2);
    assert(interpretEngine.getNbBindings() == 4);
	void* convBuffers[2];
    void* interpretBuffers[4];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex0 = convEngine.getBindingIndex(INPUT_NAME0),
		convoutIndex0 = convEngine.getBindingIndex(CONVOUT_NAME0),
        interpretInputIndex0 = interpretEngine.getBindingIndex(INTERPRET_INPUT_NAME0),
        interpretInputIndex1 = interpretEngine.getBindingIndex(INTERPRET_INPUT_NAME1),
        interpretOutputIndex0 = interpretEngine.getBindingIndex(INTERPRET_OUTPUT_NAME1),
        interpretOutputIndex1 = interpretEngine.getBindingIndex(INTERPRET_OUTPUT_NAME2);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&convBuffers[inputIndex0], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&convBuffers[convoutIndex0], batchSize * CONVOUT_H * CONVOUT_W * CONVOUT_C * sizeof(float)));
    CHECK(cudaMalloc(&interpretBuffers[interpretInputIndex0], batchSize * CONVOUT_H * CONVOUT_W * CLASS_SLICE_C * sizeof(float)));
    CHECK(cudaMalloc(&interpretBuffers[interpretInputIndex1], batchSize * CONVOUT_H * CONVOUT_W * CONFIDENCE_SLICE_C * sizeof(float)));
    CHECK(cudaMalloc(&interpretBuffers[interpretOutputIndex0], batchSize * CONVOUT_H * CONVOUT_W * CONFIDENCE_SLICE_C * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex0], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);

    int dims[] = {2, 3, 3};
    Tensor notsliced = createTensor(buffers[outputIndex0], 3, dims);
    Tensor *sliced;
    sliceTensor(notsliced, sliced, 2, 1, 2);


	CHECK(cudaMemcpyAsync(output, buffers[outputIndex0], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex0]));
	CHECK(cudaFree(buffers[outputIndex0]));
}

// maxBatch - NB must be at least as large as the batch we want to run with)
void APIToModel(unsigned int maxBatch, IHostMemory **convModelStream, IHostMemory **interpretModelStream)
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
	float* data = new float[N*INPUT_C*INPUT_H*INPUT_W];

	// srand(unsigned(time(nullptr))); // read a random sample image
	// std::random_shuffle(imageList.begin(), imageList.end(), [](int i) {return rand() % i; });
	assert(images.size() <= imageList.size());
	for (int i = 0; i < N; ++i)
		images.push_back(readImage(imageList[i]));

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

float *prepareAnchors(float *anchor_shape, int width, int height, int H, int W, int B)
{
    assert(anchor_shape);
    float center_x[W], center_y[H];
    float *anchors = new float[H*W*B*4];
    int anchors_dims[] = {W, H, B, 4};
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

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "usage: sqzdtrt IMAGE_DIR\n";
        exit(EXIT_SUCCESS);
    }

    std::vector<std::string> imageList = getImageList(argv[1]);
    float *data = prepareData(imgList);
    float *anchors = prepareAnchors(ANCHOR_SHAPE, INPUT_W, INPUT_H, CONVOUT_H, CONVOUT_W, ANCHORS_PER_GRID);
    float *out_probs = new float[INPUT_N*TOP_N_DECTION];
    float *out_class = new float[INPUT_N*TOP_N_DECTION];
    float *out_bbox = new float[INPUT_N*TOP_N_DECTION*4];

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
	doInference(*context, data, output, N);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("%.2f ", output[i]);
    }
    printf("\n");

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	// delete[] data;
	delete[] output;
	return 0;
}
