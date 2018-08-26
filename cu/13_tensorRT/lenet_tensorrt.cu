#include <iostream>
#include <map>
#include <fstream>
#include <sys/time.h>
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>

#include <cuda_runtime_api.h>
#include <NvCaffeParser.h>
#include <NvInfer.h>

#define getMillisecond(start, end) \
    (end.tv_sec-start.tv_sec)*1000 + \
    (end.tv_usec-start.tv_usec)/1000.0

#define checkCUDA(expression)                             \
{                                                         \
	cudaError_t status = (expression);                      \
	if (status != cudaSuccess) {                            \
		printf("Error on line %d: err code %d (%s)\n",        \
				__LINE__, status, cudaGetErrorString(status));    \
		exit(EXIT_FAILURE);                                   \
	}                                                       \
}

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;
using namespace nvcaffeparser1;

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
public:
    Logger(): Logger(Severity::kWARNING) {}
    Logger(Severity severity): reportableSeverity(severity) {}
    void log(Severity severity, const char* msg) override {
        //suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity) return;

        switch (severity) {
					case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: ";break;
					case Severity::kERROR: std::cerr << "ERROR: "; break;
					case Severity::kWARNING: std::cerr << "WARNING: "; break;
					case Severity::kINFO: std::cerr << "INFO: "; break;
					default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity{Severity::kWARNING};
};

Logger logger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);

    // Read number of weight blobs
    int32_t count;
    input >> count;

    while (count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t type, size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);

        // Load blob
        if (wt.type == DataType::kFLOAT) {
            uint32_t* val
							= reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x) {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        else if (wt.type == DataType::kHALF) {
            uint16_t* val =
							reinterpret_cast<uint16_t*>(malloc(sizeof(val) * size));
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

// Creat the engine using only the API and not any parser.
ICudaEngine* createMNISTEngine(unsigned int maxBatchSize, IBuilder* builder,
															 DataType dt) {
    INetworkDefinition* network = builder->createNetwork();

    // Create input tensor of shape { 1, 1, 28, 28 } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{1, INPUT_H,
				INPUT_W});

    // Create scale layer with default power/shift and specified scale parameter
    const float scaleParam = 0.0125f;
    const Weights power{DataType::kFLOAT, nullptr, 0};
    const Weights shift{DataType::kFLOAT, nullptr, 0};
    const Weights scale{DataType::kFLOAT, &scaleParam, 1};
    IScaleLayer* scale_1 = network->addScale(*data, ScaleMode::kUNIFORM,
				shift, scale, power);

    // Add convolution layer with 20 outputs and a 5x5 filter.
    std::map<std::string, Weights> weightMap =
			loadWeights("pretrained/mnistapi.wts");
	// TODO

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
	// TODO

    // Add second convolution layer with 50 outputs and a 5x5 filter.
	// TODO

    // Add second max pooling layer with stride of 2x2 and kernel size of 2x3>
	// TODO

    // Add fully connected layer with 500 outputs.
	// TODO

    // Add activation layer using the ReLU algorithm.
	// TODO

    // Add second fully connected layer with 20 outputs.
	// TODO

    // Add softmax layer to determine the probability.
	// TODO

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildCudaEngine(*network);

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
	IBuilder* builder = createInferBuilder(logger);

    // Create model to populate the network
		// then set the outputs and create an engine
    ICudaEngine* engine = createMNISTEngine(maxBatchSize, builder,
				DataType::kFLOAT);

    // Serialize the engine
	// TODO

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void ParserToModel(unsigned int maxBatchSize, 
					const std::vector<std::string>& outputs, // Names of network outputs
					IHostMemory** modelStream) {
    // Create builder
	IBuilder* builder = createInferBuilder(logger);

	// Parse caffe model 
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();

	const IBlobNameToTensor* blobNameToTensor = parser->parse("./data/mnist.prototxt", "./data/mnist.caffemodel", 
																*network, DataType::kFLOAT);

	for (auto& s : outputs)
	    network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
	ICudaEngine* engine = builder->buildCudaEngine(*network);

	// Serialize the engine
	(*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    network->destroy();
    builder->destroy();
    parser->destroy();
    shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, float* input, float* output,
		             int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of
		// the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    checkCUDA(cudaMalloc(&buffers[inputIndex],
					               batchSize * INPUT_H * INPUT_W * sizeof(float)));
    checkCUDA(cudaMalloc(&buffers[outputIndex],
					               batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously,
		// and DMA output back to host
    checkCUDA(cudaMemcpyAsync(buffers[inputIndex], input,
					                    batchSize * INPUT_H * INPUT_W * sizeof(float),
															cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    checkCUDA(cudaMemcpyAsync(output, buffers[outputIndex],
					                    batchSize * OUTPUT_SIZE * sizeof(float),
															cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    checkCUDA(cudaFree(buffers[inputIndex]));
    checkCUDA(cudaFree(buffers[outputIndex]));
}

void readPGMFile(const std::string& fileName, uint8_t* buffer) {
    std::ifstream infile(fileName, std::ifstream::binary);
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), INPUT_H * INPUT_W);
}

int main(int argc, char** argv) {

	//cudaSetDevice(1);

    // create a model using the API directly and serialize it to a stream
    IHostMemory* modelStream{nullptr};

    ParserToModel(1, std::vector<std::string>{OUTPUT_BLOB_NAME}, &modelStream);

    // Read random digit file
    srand(unsigned(time(nullptr)));
    uint8_t fileData[INPUT_H * INPUT_W];
    const int num = rand() % 10;
    readPGMFile("tensorrt_image/" + std::to_string(num) + ".pgm", fileData);

    // Print ASCII representation of digit image
    std::cout << "\nInput:\n"
              << std::endl;
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        std::cout << (" .:-=+*#%@"[fileData[i] / 26]) <<
					(((i + 1) % INPUT_W) ? "" : "\n");
		}

    // Parse mean file
    ICaffeParser* parser = createCaffeParser();
    IBinaryProtoBlob* meanBlob =
			 parser->parseBinaryProto("pretrained/mnist_mean.binaryproto");
    parser->destroy();
    const float* meanData = reinterpret_cast<const float*>(meanBlob->getData());

    // Subtract mean from image
    float data[INPUT_H * INPUT_W];
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = float(fileData[i]) - meanData[i];
		}
    meanBlob->destroy();

	// TODO create runtime & deserialize engine
	IRuntime* runtime = createInferRuntime(logger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream->data(),
			modelStream->size(), nullptr);
    modelStream->destroy();
	
	// TODO create execution context
	IExecutionContext* context = engine->createExecutionContext();

    // Run inference
    struct timeval start, end;
    float prob[OUTPUT_SIZE];

    gettimeofday(&start, NULL);
    doInference(*context, data, prob, 1);
    gettimeofday(&end, NULL);

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    float val{0.0f};
    int idx{0};
    for (unsigned int i = 0; i < 10; i++) {
        val = std::max(val, prob[i]);
        if (val == prob[i]) idx = i;
        std::cout << i << ": " <<
					std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
    }
    std::cout << std::endl;
    std::cout << "Inference duration: " << getMillisecond(start, end) <<
        " (ms)" << std::endl;

    return (idx == num && val > 0.9f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
