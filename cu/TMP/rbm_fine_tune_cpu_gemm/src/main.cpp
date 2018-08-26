#include <algorithm>
#include "mnist.h"
#include "rbm.h"
#include "Timer.h"

/** Algorithm from Earl F. Glynn's web page:
 * <a href="http://www.efg2.com/Lab/ScienceAndEngineering/Spectra.htm">Spectra Lab Report</a>
 * */

int main(int argc, const char *argv[])
{
    if (argc != 4) {
	std::cerr << "Usage: " << argv[0] << "<train-simple|train|train-cuda|fine-tune|fine-tune-cuda|test|test-simple> <image-file> <label-file>" << std::endl;
	return -1;
    }

    std::vector<Sample> samples;
    int n = mnist::read(argv[2], argv[3], samples);
    if (n <= 0) {
	std::cerr << "failed to read mnist data files: " << argv[2] << " ," << argv[3] << std::endl;
	return -1;
    }

    std::string command = argv[1];

    // initialize data
    int data_size = samples[0].data_.size();
    std::vector<Vector> inputs(n);
    std::vector<Vector> targets(n);
    for (size_t i=0; i< n; ++i) {
	const Sample& sample = samples[i];
	Vector& input = inputs[i];
	Vector& target = targets[i];

	input.resize(data_size); target.resize(10);
	for (size_t j=0; j<data_size; ++j) input[j] = sample.data_[j] / 255.0; // > 30 ? 1.0: 0.0; // binary 
	target[sample.label_] = 1.0;
    }

    // training and testing functions
    auto train_dbn_simple = [&](bool is_cuda) {
	DeepBeliefNet dbn;
	startTimer("# train-simple");

	LRBM::Conf conf;
	conf.max_epoch_ = 6; conf.max_batches_ = 100; conf.batch_size_ = 100;
//	conf.max_epoch_ = 1; conf.max_batches_ = 100; conf.batch_size_ = 100;

	dbn.build(std::vector<int>{data_size, 300, 300, 500}, std::vector<int>{0, 0, 10});
//	dbn.build(std::vector<int>{data_size, 500, 500, 500}, std::vector<int>{0, 0, 10});
//	dbn.build(std::vector<int>{data_size, 500, 500, 2000}, std::vector<int>{0, 0, 10});

	dbn.train(inputs, targets, dbn.max_layer(), conf, is_cuda);

	std::ofstream f("dbn-s.dat", std::ofstream::binary);
	dbn.store(f);
	if (is_cuda) std::cout << "is_cuda: true" << std::endl;
	endTimerp("# train-simple");
	getTimerp("## build");
	getTimerp("### RBM train()");
	getTimerp("#### RBM train()-delta");
	getTimerp("#### RBM train()-update");
	getTimerp("## DBN train()");
	getTimerp("##### activate_visible()");
	getTimerp("##### activate_hidden()");
	getTimerp("##### bernoulli()");
    };

    auto train_dbn = [&](bool is_cuda) {
	DeepBeliefNet dbn;

	startTimer("# train");
	dbn.build(std::vector<int>{data_size, 300, 300, 500, 10});
	auto& rbm = dbn.output_layer();
	rbm->type_ = RBM::Type::EXP;

	LRBM::Conf conf;

	bool resume = false;
	if (resume) {
	    std::ifstream f("dbn.dat", std::ifstream::binary);
	    dbn.load(f);
	    conf.max_epoch_ = 2; conf.max_batches_ = 300; conf.batch_size_ = 200;
	}
	else {
	    conf.max_epoch_ = 10; conf.max_batches_ = 300; conf.batch_size_ = 200;
	    dbn.pretrain(inputs, conf, is_cuda);
	}

	if (is_cuda) std::cout << "is_cuda: true" << std::endl;
	endTimerp("# train");
	getTimerp("## build");
	getTimerp("## DBN train()");
	getTimerp("### RBM train()");
	getTimerp("#### RBM train()-delta");
	getTimerp("#### RBM train()-update");
	getTimerp("##### activate_visible()");
	getTimerp("##### activate_hidden()");

	std::ofstream f("dbn.dat", std::ofstream::binary);
	dbn.store(f);
    };

    auto fine_tune_dbn = [&](bool is_cuda) {
	DeepBeliefNet dbn;
	std::string file = "dbn.dat";
	std::ifstream f(file, std::ifstream::binary);
	dbn.load(f);

	LRBM::Conf conf;

	conf.max_epoch_ = 10; conf.max_batches_ = 300; conf.batch_size_ = 200;
//	conf.max_epoch_ = 10; conf.max_batches_ /= 5; conf.batch_size_ *= 5;// takes 145m on XEON, 7hrs on TX1
	conf.max_epoch_ = 1; conf.max_batches_ /= 5; conf.batch_size_ *= 5; // takes 14m on XEON
	dbn.fine_tune(inputs, targets, conf, is_cuda);

	if (is_cuda) std::cout << "is_cuda: true" << std::endl;
	getTimerp("## fine_tune()");
	getTimerp("### minimize()");
	getTimerp("#### gradient()");
	getTimerp("##### gradient()-input forwarding");
	getTimerp("##### gradient()-calc gradient");
	getTimerp("##### activate_hidden()");
	std::ofstream fo("dbn.dat", std::ofstream::binary);
	dbn.store(fo);
    };

    auto test_dbn = [&](bool is_simple) {
	DeepBeliefNet rbm;
	std::string file = is_simple? "dbn-s.dat" : "dbn.dat";
	std::ifstream f(file, std::ifstream::binary);
	rbm.load(f);

	size_t correct = 0, second = 0;
	for (size_t i = 0; i < samples.size(); ++i) {
	    const Sample& sample = samples[i];

	    std::vector<int> idx(10);
	    for(int i=0; i<10; ++i) idx[i] = i;

	    static Vector nil;
	    Vector output(10);
	    if (is_simple)
		rbm.predict(inputs[i], output, nil);
	    else
		rbm.predict(inputs[i], nil, output);

	    std::sort(idx.begin(), idx.end(), [&output](int x, int y) { return output[x] > output[y]; });

	    if (idx[0] == (int)sample.label_) ++ correct;
	    else if (idx[1] == (int)sample.label_) ++ second;

	    if ((i + 1) % 100 == 0)	
		std::cout << "# " << correct << "/" << i + 1 
		    << " recognized. 1st: " << (correct * 100.0/ (i+1)) 
		    << "%, 1st+2nd: " << (correct + second) * 100.0/(i+1) << "%" 
		    << " image: " << i 
		    << " predicted/answer: " << idx[0] << "/" << (int) sample.label_
		    << std::endl;
	}

	std::cout << "# " << correct << " recognized." << std::endl;
    };

    cudaDeviceReset();
    // execute commands
    if (command == "train") train_dbn(false);
    else if (command == "train-cuda") train_dbn(true);
    else if (command == "fine-tune") fine_tune_dbn(false);
    else if (command == "fine-tune-cuda") fine_tune_dbn(true);
    else if (command == "train-simple") train_dbn_simple(false);
    else if (command == "train-simple-cuda") train_dbn_simple(true);
    else if (command == "test") test_dbn(false);
    else if (command == "test-simple") test_dbn(true);
    else {
	std::cerr << "unrecognized command: " << command << std::endl;	
    }

    return 0;
}

