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

static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_C = 3;
static const int INPUT_H = 384;
static const int INPUT_W = 1248;
static const int IM_INFO_SIZE = 3; // ?
static const int OUTPUT_CLS_SIZE = 3;
static const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4; // ?
static const int ANCHOR_PER_GRID = 9;
static const int NUM_OUTPUT = ANCHOR_PER_GRID * (OUTPUT_CLS_SIZE + 1 + 4);

const std::string CLASSES[OUTPUT_CLS_SIZE]{ "car", "pedestrian", "cyclist" };

const char* INPUT_BLOB_NAME0 = "data";
// const char* INPUT_BLOB_NAME1 = "im_info";
const char* OUTPUT_BLOB_NAME0 = "bbox_delta";
const char* OUTPUT_BLOB_NAME1 = "pred_class_probs";
const char* OUTPUT_BLOB_NAME2 = "pred_confidence_score";

const int poolingH = 7;
const int poolingW = 7;
const int featureStride = 16;
const int preNmsTop = 6000;
const int nmsMaxOut = 300;
const int anchorsRatioCount = 3;
const int anchorsScaleCount = 3;
const float iouThreshold = 0.7f;
const float minBoxSize = 16;
const float spatialScale = 0.0625f;
const float anchorsRatios[anchorsRatioCount] = { 0.5f, 1.0f, 2.0f };
const float anchorsScales[anchorsScaleCount] = { 8.0f, 16.0f, 32.0f };

struct PPM
{
	std::string magic, fileName;
	int h, w, max;
	uint8_t buffer[INPUT_C*INPUT_H*INPUT_W];
};

struct BBox
{
	float x1, y1, x2, y2;
};

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
    while(count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t type, size;
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);
        if (wt.type == DataType::kFLOAT)
        {
            uint32_t *val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size)); // wrong sizeof oprand
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];

            }
            wt.values = val;
        } else if (wt.type == DataType::kHALF)
        {
            uint16_t *val = reinterpret_cast<uint16_t*>(malloc(sizeof(val) * size)); // wrong sizeof oprand
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

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/samples/faster-rcnn/", "data/faster-rcnn/"};
    return locateFile(input, dirs);
}

// simple PPM (portable pixel map) reader
void readPPMFile(const std::string& filename, PPM& ppm)
{
	ppm.fileName = filename;
	std::ifstream infile(locateFile(filename), std::ifstream::binary);
	infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
	infile.seekg(1, infile.cur);
	infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

void writePPMFileWithBBox(const std::string& filename, PPM ppm, BBox bbox)
{
	std::ofstream outfile("./" + filename, std::ofstream::binary);
	assert(!outfile.fail());
	outfile << "P6" << "\n" << ppm.w << " " << ppm.h << "\n" << ppm.max << "\n";
	auto round = [](float x)->int {return int(std::floor(x + 0.5f)); };
	for (int x = int(bbox.x1); x < int(bbox.x2); ++x)
	{
		// bbox top border
		ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3] = 255;
		ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3 + 1] = 0;
		ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3 + 2] = 0;
		// bbox bottom border
		ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3] = 255;
		ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3 + 1] = 0;
		ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3 + 2] = 0;
	}
	for (int y = int(bbox.y1); y < int(bbox.y2); ++y)
	{
		// bbox left border
		ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3] = 255;
		ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3 + 1] = 0;
		ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3 + 2] = 0;
		// bbox right border
		ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3] = 255;
		ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3 + 1] = 0;
		ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3 + 2] = 0;
	}
	outfile.write(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
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
createMNISTEngine(unsigned int maxBatchSize, IBuilder *builder, DataType dt)
{
	INetworkDefinition* network = builder->createNetwork();

	auto data = network->addInput(INPUT_BLOB_NAME, dt, DimsCHW{ 3, INPUT_H, INPUT_W});
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

void APIToModel(unsigned int maxBatchSize, // batch size - NB must be at least as large as the batch we want to run with)
		     IHostMemory **modelStream)
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// create the model to populate the network, then set the outputs and create an engine
	ICudaEngine* engine = createMNISTEngine(maxBatchSize, builder, DataType::kFLOAT);

	assert(engine != nullptr);

	// serialize the engine, then close everything down
	(*modelStream) = engine->serialize();
	engine->destroy();
	builder->destroy();
}

void doInference(IExecutionContext& context, float* inputData, float* inputImInfo, float* outputBboxPred, float* outputClsProb, float *outputRois, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly 1 inputs and 1 outputs.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex0 = engine.getBindingIndex(INPUT_BLOB_NAME0),
		outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex0], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));   // data
	CHECK(cudaMalloc(&buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float))); // bbox_pred
	CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float)));  // cls_prob
	CHECK(cudaMalloc(&buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float)));                // rois

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex0], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK(cudaMemcpyAsync(buffers[inputIndex1], inputImInfo, batchSize * IM_INFO_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(outputBboxPred, buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(outputClsProb, buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(outputRois, buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);


	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex0]));
	CHECK(cudaFree(buffers[inputIndex1]));
	CHECK(cudaFree(buffers[outputIndex0]));
	CHECK(cudaFree(buffers[outputIndex1]));
	CHECK(cudaFree(buffers[outputIndex2]));
}

void bboxTransformInvAndClip(float* rois, float* deltas, float* predBBoxes, float* imInfo,
	const int N, const int nmsMaxOut, const int numCls)
{
	float width, height, ctr_x, ctr_y;
	float dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
	float *deltas_offset, *predBBoxes_offset, *imInfo_offset;
	for (int i = 0; i < N * nmsMaxOut; ++i)
	{
		width = rois[i * 4 + 2] - rois[i * 4] + 1;
		height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
		ctr_x = rois[i * 4] + 0.5f * width;
		ctr_y = rois[i * 4 + 1] + 0.5f * height;
		deltas_offset = deltas + i * numCls * 4;
		predBBoxes_offset = predBBoxes + i * numCls * 4;
		imInfo_offset = imInfo + i / nmsMaxOut * 3;
		for (int j = 0; j < numCls; ++j)
		{
			dx = deltas_offset[j * 4];
			dy = deltas_offset[j * 4 + 1];
			dw = deltas_offset[j * 4 + 2];
			dh = deltas_offset[j * 4 + 3];
			pred_ctr_x = dx * width + ctr_x;
			pred_ctr_y = dy * height + ctr_y;
			pred_w = exp(dw) * width;
			pred_h = exp(dh) * height;
			predBBoxes_offset[j * 4] = std::max(std::min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 1] = std::max(std::min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 2] = std::max(std::min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 3] = std::max(std::min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
		}
	}
}

std::vector<int> nms(std::vector<std::pair<float, int> >& score_index, float* bbox, const int classNum, const int numClasses, const float nms_threshold)
{
	auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
		if (x1min > x2min) {
			std::swap(x1min, x2min);
			std::swap(x1max, x2max);
		}
		return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
	};
	auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
		float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
		float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
		float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
		float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
		float overlap2D = overlapX * overlapY;
		float u = area1 + area2 - overlap2D;
		return u == 0 ? 0 : overlap2D / u;
	};

	std::vector<int> indices;
	for (auto i : score_index)
	{
		const int idx = i.second;
		bool keep = true;
		for (unsigned k = 0; k < indices.size(); ++k)
		{
			if (keep)
			{
				const int kept_idx = indices[k];
				float overlap = computeIoU(&bbox[(idx*numClasses + classNum) * 4],
					&bbox[(kept_idx*numClasses + classNum) * 4]);
				keep = overlap <= nms_threshold;
			}
			else
				break;
		}
		if (keep) indices.push_back(idx);
	}
	return indices;
}

std::vector<std::string> getImageList(const char *pathname)
{
    struct stat statbuf;
    struct dirent *dirp;
    DIR *dp;
    std::vector<std::string> imgList;

    if (stat(pathname, &statbuf) < 0) {
        perror("stat failed");
        exit(EXIT_FAILURE);
    }
    if (S_ISDIR(statbuf.st_mode) == 0) {
        fprintf(stderr, "'%s' not a directory.\n", pathname);
        exit(EXIT_FAILURE);
    }
    if ((dp = opendir(pathname)) == NULL) {
        perror("cannot read directory");
        exit(EXIT_FAILURE);
    }

    while ((dirp = readdir(dp)) != NULL) {
        if (!strcmp(dirp->d_name, ".") || !strcmp(dirp->d_name, ".."))
            continue;
        // TODO: filter jpeg
        imgList.push_back(std::string(dirp->d_name));
    }
    return imgList;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "usage: sqzdtrt IMAGE_DIR\n";
        exit(EXIT_SUCCESS);
    }

    std::vector<std::string> imageList = getImageList(argv[1]);
	IHostMemory *gieModelStream{ nullptr };
	// batch size
	const int N = 20;
	// read a random sample image
	srand(unsigned(time(nullptr)));
	// available images
	std::vector<PPM> ppms(N);

	std::random_shuffle(imageList.begin(), imageList.end(), [](int i) {return rand() % i; });
	assert(ppms.size() <= imageList.size());
	for (int i = 0; i < N; ++i)
	{
		readPPMFile(imageList[i], ppms[i]);
		imInfo[i * 3] = float(ppms[i].h);   // number of rows
		imInfo[i * 3 + 1] = float(ppms[i].w); // number of columns
		imInfo[i * 3 + 2] = 1;         // image scale
	}

	float* data = new float[N*INPUT_C*INPUT_H*INPUT_W];
	// pixel mean used by the SqueezeDet's author
	float pixelMean[3]{ 103.939f, 116.779f, 123.68f }; // also in BGR order
	for (int i = 0, volImg = INPUT_C*INPUT_H*INPUT_W; i < N; ++i)
	{
		for (int c = 0; c < INPUT_C; ++c)
		{
			// the color image to input should be in BGR order
			for (unsigned j = 0, volChl = INPUT_H*INPUT_W; j < volChl; ++j)
				data[i*volImg + c*volChl + j] = float(ppms[i].buffer[j*INPUT_C + 2 - c]) - pixelMean[c];
		}
	}

	// deserialize the engine
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);

	IExecutionContext *context = engine->createExecutionContext();


	// host memory for outputs
	float* rois = new float[N * nmsMaxOut * 4];
	float* bboxPreds = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];
	float* clsProbs = new float[N * nmsMaxOut * OUTPUT_CLS_SIZE];

	// predicted bounding boxes
	float* predBBoxes = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];

	// run inference
	doInference(*context, data, imInfo, bboxPreds, clsProbs, rois, N);

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
	pluginFactory.destroyPlugin();

	// unscale back to raw image space
	for (int i = 0; i < N; ++i)
	{
		float * rois_offset = rois + i * nmsMaxOut * 4;
		for (int j = 0; j < nmsMaxOut * 4 && imInfo[i * 3 + 2] != 1; ++j)
			rois_offset[j] /= imInfo[i * 3 + 2];
	}

	bboxTransformInvAndClip(rois, bboxPreds, predBBoxes, imInfo, N, nmsMaxOut, OUTPUT_CLS_SIZE);

	const float nms_threshold = 0.3f;
	const float score_threshold = 0.8f;

	for (int i = 0; i < N; ++i)
	{
		float *bbox = predBBoxes + i * nmsMaxOut * OUTPUT_BBOX_SIZE;
		float *scores = clsProbs + i * nmsMaxOut * OUTPUT_CLS_SIZE;
		for (int c = 1; c < OUTPUT_CLS_SIZE; ++c) // skip the background
		{
			std::vector<std::pair<float, int> > score_index;
			for (int r = 0; r < nmsMaxOut; ++r)
			{
				if (scores[r*OUTPUT_CLS_SIZE + c] > score_threshold)
				{
					score_index.push_back(std::make_pair(scores[r*OUTPUT_CLS_SIZE + c], r));
					std::stable_sort(score_index.begin(), score_index.end(),
						[](const std::pair<float, int>& pair1,
							const std::pair<float, int>& pair2) {
						return pair1.first > pair2.first;
					});
				}
			}

			// apply NMS algorithm
			std::vector<int> indices = nms(score_index, bbox, c, OUTPUT_CLS_SIZE, nms_threshold);
			// Show results
			for (unsigned k = 0; k < indices.size(); ++k)
			{
				int idx = indices[k];
				std::string storeName = CLASSES[c] + "-" + std::to_string(scores[idx*OUTPUT_CLS_SIZE + c]) + ".ppm";
				std::cout << "Detected " << CLASSES[c] << " in " << ppms[i].fileName << " with confidence " << scores[idx*OUTPUT_CLS_SIZE + c] * 100.0f << "% "
					<< " (Result stored in " << storeName << ")." << std::endl;

				BBox b{ bbox[idx*OUTPUT_BBOX_SIZE + c * 4], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 1], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 2], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 3] };
				writePPMFileWithBBox(storeName, ppms[i], b);
			}
		}
	}


	delete[] data;
	delete[] rois;
	delete[] bboxPreds;
	delete[] clsProbs;
	delete[] predBBoxes;
	return 0;
}
