#include <stdlib.h>
#include <opencv2/opencv.hpp>

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
            uint32_t *val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size)); // wrong sizeof oprand
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];

            }
            wt.values = val;
        } else if (wt.type == DataType::kHALF) {
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

cv::Mat readImage(const std::string& filename)
{
    cv::Mat img = cv::imread(filename);
    cv::resize(img, img, Size(INPUT_W, INPUT_H));
    return img;
}


void APIToModel(unsigned int maxBatchSize, // batch size - NB must be at least as large as the batch we want to run with)
                IHostMemory **modelStream)
{
     // create the builder
     IBuilder* builder = createInferBuilder(gLogger);

     // create the model to populate the network, then set the outputs and create an engine
     ICudaEngine* engine = createEngine(maxBatchSize, builder, DataType::kFLOAT);

     assert(engine != nullptr);
     // serialize the engine, then close everything down
     (*modelStream) = engine->serialize();
     engine->destroy();
     builder->destroy();
}
