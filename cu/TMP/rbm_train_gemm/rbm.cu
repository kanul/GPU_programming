#include "rbm.h"
#include "Timer.h"


float *d_v1;
float *d_v2;
float *d_h1;
float *d_h2;

float *d_w;
float *d_bias_hidden;
float *d_bias_visible;
float *d_rand;

float *d_weight_g;
float *d_weight_incs_g;
float *d_hidden_bias_g;
float *d_buf_g;
float *d_diff_g;

__device__ inline float _sigmoid(float x) 
{
    return (1.0 / (1.0 + exp(-x))); 
}

// dotproduct between matrix A's column x and matrix B's row y
__device__ inline float dotprod_tr(int x, int y, int len, float *A, float *B, float bias)
{
    float sum = 0;
    for(int i=0; i<len; i++) {
	sum += (A[x*len + i] * B[y*len + i]); 
    }

    return sum + bias;
}

float sigmoid(float x) 
{ 
    return (1.0 / (1.0 + exp(-x))); 
}
__global__ void matmulGPU_global(int nc, int nv, int nh, float *v, float *w, float *h, float* bias){}
		int c = blockIdx.y * blockDim.y + threadIdx.y;
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		float sum = 0;
		for (int j = 0; j < nv; ++j)
			sum += v[c * nv + j] * w[i * nv + j];
		sum += bias[i];

		sum = d_sigmoid(sum);
		h[c*nh + i] = sum;
	}
__global__ void matmulGPU_sigmoid_global(int nc, int nv, int nh, float *v,float *w, float *h, float* bias){}
__global__ void matmulGPU_exp_global(int nc, int nv, int nh, float *v,float *w, float *h, float* bias){}
__global__ void matmulGPU_addnoise_global(int nc, int nv, int nh, float *h,float *w, float *v, float* noise, float* bias){}
__global__ void matmulGPU_bernoulli_global(int nc, int nv, int nh, float *h,float *w, float *v, float* hs, float* bias){}
// GEMM_NT
__global__ void forward_gpu_sigmoid(int ns, int nh, int nv, float *v, float *w, float *h, float* bias){}
// GEMM_NT
__global__ void forward_gpu_exp(int ns, int nh, int nv, float *v, float *w, float *h, float* bias){}
// GEMM_NN
__global__ void gradient_non_linear_gpu(int ns, int nv, int nh, float *hd, float *w, float *vd, float *v){}
// GEMM_NN
__global__ void gradient_linear_gpu(int ns, int nv, int nh, float *hd, float *w, float *vd){}
// GEMM_TN
__global__ void calc_weight_incs_gpu(int nh, int nv, int ns, float *diff, float *buf, float *weight_incs){}
__global__ void weight_incs_add_diff_gpu(int ns, int nh, float *diff, float *weight_incs){}

const Vector& bernoulli(const Vector& input, Vector& output)
{ 
    startTimer("##### bernoulli()");
    static std::default_random_engine eng(::time(NULL));
    static std::uniform_real_distribution<float> rng(0.0, 1.0);

    for (size_t i=0; i<input.size(); ++i) { output[i] = (rng(eng) < input[i])? 1.0 : 0.0; } 
    endTimer("##### bernoulli()");
    return output;
}

float sigmoid(float x) 
{ 
    return (1.0 / (1.0 + exp(-x))); 
}

/* 
 * RBM
 */
int RBM::mirror(const RBM& rbm)
{
    size_t n_visible = bias_visible_.size(), n_hidden = bias_hidden_.size();
    if (n_hidden != rbm.num_visible() || n_visible != rbm.num_hidden()) { 
	std::cout << "not mirrorable" << std::endl;
	return -1;
    }

    bias_visible_ = rbm.bias_hidden_;
    bias_hidden_ = rbm.bias_visible_;
    for (size_t i = 0; i < n_visible; ++i) {
	for (size_t j = 0; j < n_hidden; ++j) {
	    weight_[j * n_visible + i] = rbm.weight_[i * n_hidden + j];
	}
    }
    return 0;  
}

const Vector& RBM::activate_visible(const Vector& hidden, Vector& visible) const
{
    startTimer("##### activate_visible()");
    size_t n_visible = bias_visible_.size(), n_hidden = bias_hidden_.size();

    std::fill(visible.begin(), visible.end(), 0);
    for (size_t i = 0; i < n_visible; ++i) {
	float s = 0;
	for (size_t j = 0; j < n_hidden; ++j) 
	    s += hidden[j] * weight_[j * n_visible+ i];
	s += bias_visible_[i];

	s = sigmoid(s);
	visible[i] = s;
    }
    endTimer("##### activate_visible()");

    return visible;
}

#define CPU_GEMM
float RBM::train(Batch inputs, const Conf& conf)
{
    size_t n_samples = inputs.size();
    size_t n_visible = bias_visible_.size(), n_hidden = bias_hidden_.size();
    float momentum = conf.momentum_, learning_rate = conf.learning_rate_, weight_cost = conf.weight_cost_;

    startTimer("### RBM train()");
    startTimer("#### RBM train()-delta");

    // temporary results
    Vector v1(n_visible), h1(n_hidden), v2(n_visible), h2(n_hidden), hs(n_hidden);

    //delta
    Vector gw(n_visible * n_hidden), gv(n_visible), gh(n_hidden);
#if defined CPU_GEMM
    static std::default_random_engine eng(::time(NULL));
    static std::normal_distribution<float> rng(0.0, 1.0);
    static std::uniform_real_distribution<float> uni(0.0, 1.0);
    float *h_v1 = new float[n_visible * n_samples];
    float *h_v2 = new float[n_visible * n_samples];
    float *h_h1 = new float[n_hidden * n_samples];
    float *h_h2 = new float[n_hidden * n_samples];

    float *d_v1;
	cudaMalloc(&d_v1, sizeof(float) * n_visible * n_samples);
    float *d_v2;
	cudaMalloc(&d_v2, sizeof(float) * n_visible * n_samples);
    float *d_h1;
	cudaMalloc(&d_h1, sizeof(float) * n_hidden * n_samples);
    float *d_h2;
	cudaMalloc(&d_h2, sizeof(float) * n_hidden * n_samples);
    float *d_w;
	cudaMalloc(&d_w, sizeof(float) * n_hidden * n_visible);
	float *d_bias;
	cudaMalloc(&d_bias, sizeof(float) * n_hidden);
	float *h_h1_result;
	cudaMalloc(&h_h1_result, sizeof(float) * n_hidden * n_samples);
	
	int ofs = 0;
    for (auto const& input: inputs) {
	std::copy (input.begin(), input.end(), h_v1 + ofs*n_visible);
	ofs++;
    }

	cudaMemcpy(d_v1, h_v1, sizeof(float) * n_visible * n_samples, cudaMemcpyHostToDevice);
	cudaMemcpy(d_w, weight_.data(), sizeof(float) * n_hidden * n_visible, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias, bias_hidden_.data(), sizeof(flaot) * n_hidden, cudaMemcpyHostToDevice);

	dim3 dim_threads(16, 16);
	//dim3 dim_threads;
	//dim_threads.x = 16;
	//dim_threads.y = 16;

	dim3 dim_grid;
	dim_grid.x = (n_hidden%dim_threads.x == 0) ? n_hidden/dim_threads.x : n_hidden/dim_threads.x + 1;
	dim_grid.y = (n_samples%dim_threads.y == 0) ? n_samples/dim_threads.y : n_samples/dim_threads.y + 1;

	if (type == Type:SIGMOID)
		matmulGPU_sigmoid_global<<<dim_grid, dim_threads>>>(n_samples, n_visible, n_hidden, d_v1, d_w, d_h1, d_bias);

	cudaMemcpy(h_h1_result, d_h, sizeof(float) * n_hidden * n_samples, cudaMemcpyHostToDevice);

    // activate_hidden(v1, h1)
    // (nc x n_vis) * (n_vis x n_hid) = (nc * n_hid)
    for (int c = 0 ; c < n_samples ; ++c)
    {
	for (int i =0 ; i < n_hidden ; ++i)
	{
	    float sum = 0;
	    for (int j = 0 ; j < n_visible ; ++j)
		sum += h_v1[c*n_visible +j] * weight_[i*n_visible + j];
	    sum += bias_hidden_[i];
	    if (type_ == Type::SIGMOID) sum = sigmoid(sum);
	    else if (type_ == Type::EXP) sum = exp(sum);
	    h_h1[c*n_hidden +i] = sum;
	}
    }

    // activate_visible(h1, v2)
    // (nc x n_hid) * (n_hid x n_vis) = (nc * n_vis)
    if (type_ == Type::LINEAR)
	for (int i = 0 ; i < n_samples*n_hidden ; ++i)
	    h_h1[i] += rng(eng);
    else
	for (int i = 0 ; i < n_samples*n_hidden ; ++i)
	    h_h1[i] = uni(eng) < h_h1[i]? 1.0 : 0.0;

    for (int c = 0 ; c < n_samples ; ++c)
    {
	for (int i = 0 ; i < n_visible ; ++i)
	{
	    float sum = 0;
	    for (int j = 0 ; j < n_hidden ; ++j)
		sum += h_h1[c*n_hidden +j] * weight_[j*n_visible + i];
	    sum += bias_visible_[i];
	    sum = sigmoid(sum);
	    h_v2[c*n_visible + i] = sum;
	}
    }
    
    // activate_hidden(v2, h2)
    // (nc x n_vis) * (n_vis x n_hid) = (nc * n_hid)
    for (int c = 0 ; c < n_samples ; ++c)
    {
	for (int i = 0 ; i < n_hidden ; ++i)
	{
	    float sum = 0;
	    for (int j = 0 ; j < n_visible ; ++j)
		sum += h_v2[c*n_visible + j] * weight_[i*n_visible + j];
	    sum += bias_hidden_[i];
	    if (type_ == Type::SIGMOID) sum = sigmoid(sum);
	    else if (type_ == Type::EXP) sum = exp(sum);
	    h_h2[c*n_hidden +i] = sum;
	}
    }

    for (int c = 0 ; c < n_samples ; ++c)
	for (size_t i = 0; i < n_visible; ++i)
	    for (size_t j = 0; j < n_hidden; ++j) 
		gw[j*n_visible + i] += h_h1[c*n_hidden + j] * h_v1[c*n_visible + i] - h_h2[c*n_hidden + j] * h_v2[c*n_visible + i];

    for (int c = 0 ; c < n_samples ; ++c)
	for (int i = 0 ; i < n_hidden ; ++i)
	    gh[i] += h_h1[c*n_hidden + i] - h_h2[c*n_hidden + i];

    for (int c = 0 ; c < n_samples ; ++c)
	for (int i = 0 ; i < n_visible ; ++i)
	    gv[i] += h_v1[c*n_visible + i] - h_v2[c*n_visible + i];

    delete[] h_v1;
    delete[] h_v2;
    delete[] h_h1;
    delete[] h_h2;
#else
    for (auto const& input: inputs) {
	v1 = input;
	this->activate_hidden(v1, h1);
	this->activate_visible((type_ == Type::LINEAR? add_noise(h1, hs): bernoulli(h1, hs)), v2);
	this->activate_hidden(v2, h2);

	for (size_t i = 0; i < n_visible; ++i) {
	    for (size_t j = 0; j < n_hidden; ++j) 
		gw[j * n_visible + i] += h1[j] * v1[i] - h2[j] * v2[i];
	}

	//      gh += (h1 - h2);
	//      gv += (v1 - v2);
	v::saxpy2(gh, 1.0, h1, -1.0, h2);
	v::saxpy2(gv, 1.0, v1, -1.0, v2);
    }
#endif // CPU_GEMM

    endTimer("#### RBM train()-delta");
    startTimer("#### RBM train()-update");

    //update
    //    gw /= float(n_samples);
    //    gw -= weight_ * weight_cost;
    v::saxpy(1.0/n_samples, gw, -weight_cost, weight_);
    //    weight_inc_ = weight_inc_ * momentum + gw * learning_rate;
    v::saxpy(momentum, weight_inc_, learning_rate, gw);

    //    weight_ += weight_inc_;
    v::saxpy(weight_, 1.0, weight_inc_);

    //    gh /= float(n_samples); 
    //    bias_hidden_inc_ = bias_hidden_inc_ * momentum + gh * learning_rate;
    v::saxpy(momentum, bias_hidden_inc_, learning_rate / n_samples, gh);
    //    bias_hidden_ += bias_hidden_inc_;
    v::saxpy(bias_hidden_, 1.0, bias_hidden_inc_);

    //    gv /= float(n_samples); 
    //    bias_visible_inc_ = bias_visible_inc_ * momentum + gv * learning_rate;
    v::saxpy(momentum, bias_visible_inc_, learning_rate / n_samples, gv);
    //    bias_visible_ += bias_visible_inc_;
    v::saxpy(bias_visible_, 1.0, bias_visible_inc_);

    //    float error = sqrt(gv.dot(gv) / n_visible);
    v::scale(gv, 1.0/n_samples);
    float error = sqrt(v::dot(gv, gv) / n_visible);
    //    std::cout << "error: " << error << ", energy: " << free_energy() << std::endl;

    endTimer("#### RBM train()-update");
    endTimer("### RBM train()");
    return error;
}

float RBM::train_gpu(Batch inputs, const Conf& conf)
{
    size_t n_samples = inputs.size();
    size_t n_visible = bias_visible_.size(), n_hidden = bias_hidden_.size();
    float momentum = conf.momentum_, learning_rate = conf.learning_rate_, weight_cost = conf.weight_cost_;

    startTimer("### RBM train()");
    startTimer("#### RBM train()-delta");

    // temporary results
    Vector v1(n_visible), h1(n_hidden), v2(n_visible), h2(n_hidden), hs(n_hidden);

    //delta
    Vector gw(n_visible * n_hidden), gv(n_visible), gh(n_hidden);
    static std::default_random_engine eng(::time(NULL));
    static std::normal_distribution<float> rng(0.0, 1.0);
    static std::uniform_real_distribution<float> uni(0.0, 1.0);
    float *h_v1 = new float[n_visible * n_samples];
    float *h_v2 = new float[n_visible * n_samples];
    float *h_h1 = new float[n_hidden * n_samples];
    float *h_h2 = new float[n_hidden * n_samples];
    float *h_rand = new float[n_hidden * n_samples];

    int ofs = 0;
    for (auto const& input: inputs) {
	std::copy (input.begin(), input.end(), h_v1 + ofs*n_visible);
	ofs++;
    }

  	/**********************************/
	/***** FILL CUDA CODE!!!! *****/
	/**********************************/

/* 
    if (type_ == Type::SIGMOID)
		matmulGPU_sigmoid_global<<<dim_grid, dim_threads>>>();
    else if (type_ == Type::EXP)
		matmulGPU_exp_global<<<dim_grid, dim_threads>>>();
    else
		matmulGPU_global<<<dim_grid, dim_threads>>>();

    //////////////////////////
	// generate random data //
	//////////////////////////

    if (type_ == Type::LINEAR)
		matmulGPU_addnoise_global<<<dim_grid, dim_threads>>>();
    else
		matmulGPU_bernoulli_global<<<dim_grid, dim_threads>>>();


    if (type_ == Type::SIGMOID)
		matmulGPU_sigmoid_global<<<dim_grid, dim_threads>>>();
    else if (type_ == Type::EXP)
		matmulGPU_exp_global<<<dim_grid, dim_threads>>>();
    else 
		matmulGPU_global<<<dim_grid, dim_threads>>>();
*/
	printf("Fill CUDA code in train_gpu()\n");
	exit(0);

    for (int c = 0 ; c < n_samples ; ++c)
	for (size_t i = 0; i < n_visible; ++i)
	    for (size_t j = 0; j < n_hidden; ++j) 
		gw[j*n_visible + i] += h_h1[c*n_hidden + j] * h_v1[c*n_visible + i] - h_h2[c*n_hidden + j] * h_v2[c*n_visible + i];

    for (int c = 0 ; c < n_samples ; ++c)
	for (int i = 0 ; i < n_hidden ; ++i)
	    gh[i] += h_h1[c*n_hidden + i] - h_h2[c*n_hidden + i];

    for (int c = 0 ; c < n_samples ; ++c)
	for (int i = 0 ; i < n_visible ; ++i)
	    gv[i] += h_v1[c*n_visible + i] - h_v2[c*n_visible + i];

    delete[] h_v1;
    delete[] h_v2;
    delete[] h_h1;
    delete[] h_h2;
    delete[] h_rand;


    endTimer("#### RBM train()-delta");
    startTimer("#### RBM train()-update");

    //update
    //    gw /= float(n_samples);
    //    gw -= weight_ * weight_cost;
    v::saxpy(1.0/n_samples, gw, -weight_cost, weight_);
    //    weight_inc_ = weight_inc_ * momentum + gw * learning_rate;
    v::saxpy(momentum, weight_inc_, learning_rate, gw);

    //    weight_ += weight_inc_;
    v::saxpy(weight_, 1.0, weight_inc_);

    //    gh /= float(n_samples); 
    //    bias_hidden_inc_ = bias_hidden_inc_ * momentum + gh * learning_rate;
    v::saxpy(momentum, bias_hidden_inc_, learning_rate / n_samples, gh);
    //    bias_hidden_ += bias_hidden_inc_;
    v::saxpy(bias_hidden_, 1.0, bias_hidden_inc_);

    //    gv /= float(n_samples); 
    //    bias_visible_inc_ = bias_visible_inc_ * momentum + gv * learning_rate;
    v::saxpy(momentum, bias_visible_inc_, learning_rate / n_samples, gv);
    //    bias_visible_ += bias_visible_inc_;
    v::saxpy(bias_visible_, 1.0, bias_visible_inc_);

    //    float error = sqrt(gv.dot(gv) / n_visible);
    v::scale(gv, 1.0/n_samples);
    float error = sqrt(v::dot(gv, gv) / n_visible);
    //    std::cout << "error: " << error << ", energy: " << free_energy() << std::endl;

    endTimer("#### RBM train()-update");
    endTimer("### RBM train()");
    return error;
}


/* 
 * Layered RBM
 */

int LRBM::build(const std::vector<int>& layers, const std::vector<int>& adjust /* = std::vector<int>() */)
{
    startTimer("## build");
    if (layers.size() <= 1) return -1;
    

    for (size_t i=0; i<layers.size() - 1; ++i) {
	int n_visible= layers[i] + (adjust.empty()? 0: adjust[i]);
	int n_hidden = layers[i+1];

	max_neurons = std::max(max_neurons, std::max(n_visible, n_hidden));
	max_n_visible = std::max(max_n_visible, n_visible);
	max_n_hidden = std::max(max_n_hidden, n_hidden);
	std::cout << "New RBM " << n_visible << " -> " << n_hidden << std::endl;
	rbms_.push_back(std::unique_ptr<RBM>(new RBM(n_visible, n_hidden)));
    }
    std::cout << "max_neurons: " << max_neurons << std::endl;
    endTimer("## build");

    return 0;
}

std::vector<int> LRBM::offsets(int start) const
{
    int n_layers = rbms_.size() - start;
    std::vector<int> dims(n_layers + 1);
    dims[0] = 0;
    int total = 0;
    for(size_t i=0; i<n_layers; ++i) {
	total += (rbms_[i + start]->num_visible() + 1) * rbms_[i + start]->num_hidden();
	dims[i+1] = total;
    }
    return dims;
}

void LRBM::to_image(Vector& image, int& width, int& height)
{
    width = 0; height = 0;
    auto& rbms = this->rbms_;
    for (auto& rbm: rbms) {
	if (width < rbm->num_hidden() + 1) width = rbm->num_hidden() + 1;
	height += (rbm->num_visible() + 2);  
    }
    image.resize(width * height);

    size_t y_offset = 0;
    for (auto& rbm: rbms) {
	size_t n_visible = rbm->num_visible();
	size_t n_hidden = rbm->num_hidden();
	size_t x_offset = (width - n_hidden) / 2;

	for (size_t j=0; j<n_hidden; ++j)
	    image[y_offset * width + x_offset + j] = rbm->bias_hidden_[j];
	for (size_t i=0; i<n_visible; ++i) {
	    for (size_t j=0; j<n_hidden; ++j)
		//          image[(y_offset + i) * width + x_offset + j] = rbm->weight_[i * n_hidden + j];
		image[(y_offset + i) * width + x_offset + j] = rbm->weight_[j * n_visible+ i];
	    image[(y_offset + i) * width + x_offset + n_hidden] = rbm->bias_visible_[i];
	}
	y_offset += n_visible + 2;
    }
}

void LRBM::store(std::ostream& os) const
{
    int32_t count = rbms_.size();
    os.write(reinterpret_cast<char *>(&count), sizeof(count));
    for (auto const& rbm: rbms_) rbm->store(os);
}

void LRBM::load(std::istream& is)
{
    int32_t count = 0;
    is.read(reinterpret_cast<char *>(&count), sizeof(count));

    rbms_.clear();
    for (size_t i = 0; i < count; ++i) 
    {
	RBMP rbm(new RBM());
	rbm->load(is);
	rbms_.push_back(std::move(rbm));
    }
}

/* 
 * Deep Belief Nets
 */

int DeepBeliefNet::train(std::vector<Vector>& inputs, std::vector<Vector>& labels, 
	int max_layer, LRBM::Conf& conf, bool is_cuda /*= false*/)
{
    startTimer("## DBN train()");
    int n_samples = inputs.size(), n_labels = labels.size();
    if (n_labels > 0 && n_samples != n_labels) {
	std::cerr << "# inputs does not match # labels" << std::endl;
	return -1;
    }

    int max_epoch = conf.max_epoch_, batch_size = conf.batch_size_; 
    int max_batches = std::min(conf.max_batches_, n_samples / batch_size); 

    cudaMalloc((void **) &d_v1, max_n_visible * conf.batch_size_ * sizeof(float) );
    cudaMalloc((void **) &d_v2, max_n_visible * conf.batch_size_ * sizeof(float) );
    cudaMalloc((void **) &d_h1, max_n_hidden * conf.batch_size_ * sizeof(float) );
    cudaMalloc((void **) &d_h2, max_n_hidden * conf.batch_size_ * sizeof(float));

    cudaMalloc((void **) &d_w, max_n_hidden * max_n_visible * sizeof(float) );

    cudaMalloc((void **) &d_bias_hidden, max_n_hidden * sizeof(float));
    cudaMalloc((void **) &d_bias_visible, max_n_visible * sizeof(float));

    cudaMalloc((void **) &d_rand, max_n_hidden * conf.batch_size_ * sizeof(float));

    std::vector<Vector> probs(n_samples);

    for(int layer = 0; layer < max_layer; ++layer) {
	auto& rbm = this->rbms_[layer];
	RBM::Conf conf;
	//XXX: more epochs and lower learning rate for linear rbm
	if (rbm->type_ == RBM::Type::LINEAR) { max_epoch = 100; conf.learning_rate_ = 0.001; }

	for (int epoch = 0; epoch < max_epoch; ++epoch) {

	    //XXX: update momentum
	    if (epoch > 5) conf.momentum_ = .9f;


	    for (size_t batch = 0; batch < max_batches; ++batch) {
		int start = batch * batch_size, end = std::min(start + batch_size, n_samples);

		Batch data;
		if (layer == 0) 
		    data = Batch{inputs.begin() + start, inputs.begin() + end};
		else 
		    data = Batch{probs.begin() + start, probs.begin() + end};

		float error;
		if (is_cuda)
		    error = rbm->train_gpu(data, conf);
		else
		    error = rbm->train(data, conf);

		if ((batch + 1) % 10 == 0) {
		    std::cout << "layer: " << layer << ", epoch: " << epoch << ", batch: " << batch + 1 
			<< ", error: " << error << ", energy: " << this->free_energy() << std::endl;
		}

		//save outputs to probs at last epoch
		if (epoch == max_epoch - 1) {
		    auto it = data.begin();
		    for(int i = start; i < end; ++i) {
			Vector output(rbm->num_hidden());
			rbm->activate_hidden(*it++, output);
			output.swap(probs[i]);
		    }

		    //attach labels for last layer
		    if (layer > 0 && layer + 1 == max_layer - 1 && !labels.empty()) {
			size_t input_size = probs[start].size(), label_size = labels.front().size();
			for (size_t i = start; i < end; ++i) {
			    const Vector& label = labels[i];
			    Vector& input = probs[i];
			    input.resize(input_size + label_size);
			    std::copy(label.begin(), label.end(), input.begin() + input_size);
			}
		    }
		} // save output 
	    } // batch  
	} // epoch
    } // layer

    endTimer("## DBN train()");
    return 0;
}

int DeepBeliefNet::predict(const Vector& sample, Vector& output, Vector& probs)
{
    static std::default_random_engine eng(::time(NULL));
    std::uniform_real_distribution<float> rng(0.0, 1.0);

    Vector input(sample);
    int n_layers = rbms_.size();
    for (int i =0; i<n_layers - 1; ++i) {
	const RBMP& rbm = rbms_[i];
	size_t n_visible = rbm->num_visible(), n_hidden = rbm->num_hidden();

	Vector next(n_hidden);
	rbm->activate_hidden(input, next); 
	input.swap(next);  
    }

    RBMP& rbm = rbms_[n_layers - 1];
    size_t n_visible = rbm->num_visible();
    size_t n_hidden = rbm->num_hidden();
    size_t n_input = input.size();
    if (n_input  + output.size() != n_visible) {
	return -1;
    }

    // attach zero-ed labels
    if (n_visible > n_input) input.resize(n_visible);

    Vector h1(n_hidden);
    rbm->activate_hidden(input, h1);

    if (! probs.empty()) 
	probs = h1;

    if (! output.empty()) {
	Vector hs(n_hidden), v2(n_visible);
	rbm->activate_visible(bernoulli(h1, hs), v2);
	std::copy(v2.begin() + n_input, v2.end(), output.begin());
    }

    return 0;    
}

int DeepBeliefNet::gradient(GradientContext& ctx, const Vector& weights, Vector& weight_incs, float& cost)
{
    Batch& inputs = ctx.inputs_;
    std::vector<std::vector<Vector>>& probs = ctx.probs_; 
    bool has_targets = !ctx.targets_.empty();

    int max_layer = this->rbms_.size();

    size_t n_hidden = rbms_.back()->num_hidden();
    size_t n_samples = inputs.size();
    std::vector<Vector> diffs(n_samples);

    auto cstart = std::chrono::high_resolution_clock::now();
    auto dims = this->offsets(ctx.start_layer_);

    startTimer("#### gradient()");
    startTimer("##### gradient()-input forwarding");
    std::fill(weight_incs.begin(), weight_incs.end(), 0);
    cost = 0;
    float error = 0;
#if defined CPU_GEMM

	/**********************************/
	/***** FILL CPU GEMM CODE!!!! *****/
	/**********************************/
	printf("Fill CPU GEMM code in gradient()\n");
	exit(0);
/*
    if (type_ == Type::SIGMOID)
		matmulGPU_sigmoid_global<<<dim_grid, dim_threads>>>();
    else if (type_ == Type::EXP)
		matmulGPU_exp_global<<<dim_grid, dim_threads>>>();
    else
		matmulGPU_global<<<dim_grid, dim_threads>>>();

    //////////////////////////
	// generate random data //
	//////////////////////////

    if (type_ == Type::LINEAR)
		matmulGPU_addnoise_global<<<dim_grid, dim_threads>>>();
    else
		matmulGPU_bernoulli_global<<<dim_grid, dim_threads>>>();


    if (type_ == Type::SIGMOID)
		matmulGPU_sigmoid_global<<<dim_grid, dim_threads>>>();
    else if (type_ == Type::EXP)
		matmulGPU_exp_global<<<dim_grid, dim_threads>>>();
    else 
		matmulGPU_global<<<dim_grid, dim_threads>>>();
*/

#else
    for (size_t sample = 0; sample < n_samples; ++sample) { 
	const Vector& input = inputs[sample];
	v::LightVector bias_hidden, weight;
	for (int layer=0; layer < max_layer; ++layer) {
	    const RBMP& rbm = this->rbms_[layer];
	    if (layer < ctx.start_layer_ || weights.empty()) { 
		float *start = const_cast<float *>(rbm->weight_.data()), *end = start + rbm->num_weight();
		weight = v::LightVector(start, end);
		start = const_cast<float *>(rbm->bias_hidden_.data()); end = start + rbm->num_hidden();
		bias_hidden = v::LightVector(start, end);
	    } else { 
		float *start = const_cast<float *>(weights.data()) + dims[layer - ctx.start_layer_], *end = start + rbm->num_weight();
		weight = v::LightVector(start, end);
		start = end; end = start + rbm->num_hidden();
		bias_hidden = v::LightVector(start, end);
	    }

	    Vector& output = probs[layer][sample];
	    const Vector& _input = (layer == 0? input: probs[layer - 1][sample]);
	    RBM::activate_hidden(_input, output, bias_hidden, weight, rbm->type_);
	}

	//output
	Vector& result = probs[max_layer - 1][sample]; 
	Vector& diff = diffs[sample];
	diff.resize(n_hidden);

	if (has_targets) {
	    float s = std::accumulate(result.begin(), result.end(), 0.0);
	    v::scale(result, 1.0/s);

	    const Vector& target = ctx.targets_[sample];
	    for(size_t i=0; i<n_hidden; ++i) {
		diff[i] = (result[i] - target[i]);
		cost += target[i] * log(result[i]); 
		error += diff[i] * diff[i];
	    }
	} else {
	    for(size_t i=0; i<n_hidden; ++i) {
		diff[i] = (result[i] - input[i]) / n_samples;
		cost += input[i] * log(result[i]) + (1 - input[i]) * log(1 - result[i]);  
		error += (result[i] - input[i]) * (result[i] - input[i]);
	    }
	}
    }
#endif

    cost = -cost;
    if (! has_targets) cost *= 1/ n_samples;

    endTimer("##### gradient()-input forwarding");
    startTimer("##### gradient()-calc gradient");

    //calculate gradient
#if defined CPU_GEMM
    // weight_incs.size() comes from dims which is created from LRBM::offsets()
    // (accumulation of (num_visible+1)*num_hidden from each layer)

	/**********************************/
	/***** FILL CPU GEMM CODE!!!! *****/
	/**********************************/

#else
    for (int layer=max_layer - 1; layer >= 0; --layer) {
	if (layer < ctx.start_layer_) 
	    break;

	if (layer != max_layer - 1) {
	    const RBMP& rbm = this->rbms_[layer + 1];
	    //        const Vector& weight = rbm->weight_; 
	    size_t n_visible = rbm->num_visible(), n_hidden = rbm->num_hidden();
	    size_t offset = dims[layer + 1 - ctx.start_layer_];
	    v::LightVector weight(const_cast<float *>(weights.data()) + offset, const_cast<float *>(weights.data()) + offset + rbm->num_weight());
	    for (size_t sample = 0; sample < n_samples; ++sample) { 
		Vector diff(n_visible);
		for (size_t j=0; j<n_visible; ++j) {
		    float s = 0;
		    for (size_t k=0; k<n_hidden; ++k) {
			//              s += diffs[sample][k] * weight[j * n_hidden + k];
			s += diffs[sample][k] * weight[k * n_visible + j];
		    }
		    if (rbms_[layer]->type_ != RBM::Type::LINEAR)
			s *= probs[layer][sample][j] * (1.0 - probs[layer][sample][j]);
		    diff[j] = s;
		}
		diffs[sample].swap(diff);
	    } 
	}

	RBMP& rbm = this->rbms_[layer];
	size_t n_visible = rbm->num_visible(), n_hidden = rbm->num_hidden();
	size_t offset = dims[layer - ctx.start_layer_];

	for (size_t sample = 0; sample < n_samples; ++sample) { 
	    const auto& v = (layer > 0? probs[layer-1][sample] : inputs[sample]);
	    const auto& d = diffs[sample];
	    for (size_t j=0; j<n_visible; ++j) {
		for (size_t k=0; k<n_hidden; ++k) {
		    weight_incs[offset + k * n_visible + j] += v[j] * d[k];
		}
	    }
	    for (size_t k=0; k<n_hidden; ++k) {
		weight_incs[offset + n_visible * n_hidden + k] += d[k];
	    }
	}
    }
#endif

    auto cend = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000.0;
    std::cout << "evaluating: cost=" << cost << ", error=" << error / n_samples << " in " << duration << "ms" << std::endl;

    endTimer("##### gradient()-calc gradient");
    endTimer("#### gradient()");
    return 0;
}

    // translate into C++ from matlab code
    //    http://learning.eng.cam.ac.uk/carl/code/minimize/minimize.m
int DeepBeliefNet::minimize(GradientContext& ctx, bool is_cuda)
{
    const float INT = 0.1, EXT = 3.0;
    const float SIG = 0.1, RHO = SIG / 2.0, RATIO = 10;
    const int max_iteration = ctx.max_iteration_;

    startTimer("### minimize()");
    // initialize
    float cost = 0;
    auto dims = this->offsets(ctx.start_layer_); 
    Vector weights(dims.back()), weight_incs(dims.back()); 

    {
	auto offset = weights.begin();
	for (size_t i=ctx.start_layer_; i<this->rbms_.size(); ++i) {
	    const RBMP& rbm = this->rbms_[i];
	    std::copy(rbm->weight_.begin(), rbm->weight_.end(), offset);
	    offset += rbm->num_weight();
	    std::copy(rbm->bias_hidden_.begin(), rbm->bias_hidden_.end(), offset);
	    offset += rbm->num_hidden();
	}
    }

	if (is_cuda)
	    this->gradient_gpu(ctx, weights, weight_incs, cost);
	else
	    this->gradient(ctx, weights, weight_incs, cost);

    Vector df0(weight_incs);
    Vector s(df0); v::scale(s, -1.0);
    float d0 = -v::dot(s, s), f0 = cost;
    float d3 = 0, x3 = 1.0 / (1 - d0);

    bool failed = false;
    // line search
    for (int i=0; i<max_iteration; ++i) {
	// extrapolation
	float best_cost = f0;
	Vector best_weights(weights), best_weight_incs(weight_incs);

	float f3 = 0;
	Vector df3(weights.size());

	int M = 20;
	float f1 = 0, x1 = 0, d1 = 0;
	float f2 = 0, x2 = 0, d2 = 0;
	while (true) {
	    x2 = 0; f2 = f0; d2 = d0; 
	    f3 = f0; df3 = df0;

	    while (true) {
		if (M -- < 0) break;

		Vector tmp_weights(weights);
		//          tmp_weights += s * x3;
		v::saxpy(tmp_weights, x3, s);
		if (is_cuda)
			this->gradient_gpu(ctx, tmp_weights, weight_incs, cost);
		else
			this->gradient(ctx, tmp_weights, weight_incs, cost);
		f3 = cost; df3 = weight_incs;
		if (std::isfinite(cost) && v::isfinite(weight_incs)) {
		    //found one and save best result if available
		    if (f3 < best_cost) {
			best_cost = f3;
			best_weights = tmp_weights;
			best_weight_incs = weight_incs;
		    }
		    break;
		}

		//back off and retry
		x3 = (x2 + x3) / 2.0;
	    }

	    // check slope and done extrapolation?
	    d3 = v::dot(df3,s);
	    if (d3 > SIG*d0 || f3 > f0 + x3*RHO*d0 || M <= 0) break;

	    x1 = x2; f1 = f2; d1 = d2;
	    x2 = x3; f2 = f3; d2 = d3;  

	    // cubic extrapolation
	    float dx = x2-x1;
	    float A = 6.0*(f1-f2) + 3.0*(d2+d1)*dx;
	    float B = 3.0*(f2-f1) - (2.0*d1+d2)*dx;
	    x3 = x1-d1*dx*dx/(B+sqrt(B*B-A*d1*dx));

	    // keep it in range
	    float upper = x2 * EXT, lower = x2 + INT * dx;
	    if (!std::isfinite(x3) || x3 < 0 || x3 > upper) x3 = upper;
	    else if (x3 < lower) x3 = lower;
	}

	// interpolation
	float f4 = 0, x4 = 0, d4 = 0;
	while ((std::abs(d3) > -SIG*d0 || f3 > f0 + x3*RHO*d0) && M > 0) {
	    if (d3 > 0 || f3 > f0+x3*RHO*d0) {
		x4 = x3; f4 = f3; d4 = d3;        
	    } else {
		x2 = x3; f2 = f3; d2 = d3;
	    }

	    float dx = x4 - x2;
	    if (f4 > f0) {
		x3 = x2-(0.5*d2*dx*dx)/(f4-f2-d2*dx);  // quadratic interpolation
	    } else {
		float A = 6*(f2-f4)/dx+3*(d4+d2);     // cubic interpolation
		float B = 3*(f4-f2)-(2*d2+d4)*dx;
		x3 = x2+(sqrt(B*B-A*d2*dx*dx)-B)/A; 
	    }

	    if (! std::isfinite(x3)) {
		x3 = (x2 + x4) / 2;
	    }

	    // keep it in range
	    x3 = std::max(std::min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));

	    Vector tmp_weights(weights);
	    //        tmp_weights += s * x3;
	    v::saxpy(tmp_weights, x3, s);
		if (is_cuda)
		    this->gradient_gpu(ctx, tmp_weights, weight_incs, cost);
		else
		    this->gradient(ctx, tmp_weights, weight_incs, cost);
	    f3 = cost; df3 = weight_incs;
	    if (f3 < best_cost) {
		best_cost = f3;
		best_weights = tmp_weights;
		best_weight_incs = weight_incs;
	    }

	    --M;
	    //        d3 = df3.dot(s);
	    d3 = v::dot(df3,s);
	}

	if (std::abs(d3) < -SIG*d0 && f3 < f0 + x3*RHO*d0) { // succeeded
	    //        weights += s * x3; 
	    v::saxpy(weights, x3, s);
	    f0 = f3; 
	    //        s *= (df3.dot(df3) - df3.dot(df0)) / df0.dot(df0); s -= df3; // Polack-Ribiere CG direction
	    float g = (v::dot(df3,df3) - v::dot(df3, df0)) / v::dot(df0, df0);
	    v::saxpy(g, s, -1.0, df3); // Polack-Ribiere CG direction
	    //        d3 = d0; d0  = df3.dot(s); df0 = df3; 
	    d3 = d0; d0  = v::dot(df3, s); df0 = df3; 
	    if (d0 > 0) {
		//          s = -df0; d0 = -df0.dot(df0);
		s = df0; v::scale(s, -1.0); d0 = -v::dot(df0, df0);
	    }

	    x3 = x3 * std::min(RATIO, float(d3 / (d0 - 1e-37)));
	    failed = false;
	    std::cout << "found: iteration i=" << i << ", cost=" << f3 << std::endl;
	} else { // failed
	    std::cout << "x3 = " << x3 << " failed" << std::endl;
	    weights = best_weights; f0 = best_cost; df0 = best_weight_incs; 
	    if (failed) break;  

	    //        s = -df0; d0 = - s.dot(s); x3 = 1.0/(1.0-d0);
	    s = df0; v::scale(s, -1.0); d0 = -v::dot(s, s); x3 = 1.0/(1.0-d0);
	    failed = true;
	}
    }

    //apply the new weights
    {
	auto offset = weights.begin();
	for (size_t i=ctx.start_layer_; i<this->rbms_.size(); ++i) {
	    const RBMP& rbm = this->rbms_[i];
	    std::copy(offset, offset + rbm->num_weight(), rbm->weight_.begin());
	    offset += rbm->num_weight();
	    std::copy(offset, offset + rbm->num_hidden(), rbm->bias_hidden_.begin());
	    offset += rbm->num_hidden();
	}
    }

    std::cout << "applying new weights to " << ctx.start_layer_ << "+" << std::endl;
    endTimer("### minimize()");
    return 0;
}

int DeepBeliefNet::gradient_gpu(GradientContext& ctx, const Vector& weights, Vector& weight_incs, float& cost)
{
    Batch& inputs = ctx.inputs_;
    std::vector<std::vector<Vector>>& probs = ctx.probs_; 
    bool has_targets = !ctx.targets_.empty();

    int max_layer = this->rbms_.size();

    size_t n_hidden = rbms_.back()->num_hidden();
    size_t n_samples = inputs.size();
    std::vector<Vector> diffs(n_samples);

    auto cstart = std::chrono::high_resolution_clock::now();
    auto dims = this->offsets(ctx.start_layer_);

    startTimer("#### gradient()");
    startTimer("##### gradient()-input forwarding");
    std::fill(weight_incs.begin(), weight_incs.end(), 0);
    cost = 0;
    float error = 0;
	/**************************************************************************/
	/************************** FILL CUDA CODE!!!! ****************************/
	/**************************************************************************/
	/*
	HINT 1. Calculate offset of each layer of the data (input & probs, weight, bias, etc.)
	HINT 2. Create host buffer to copy 2D-vector data to 1D-array
	HINT 3. In most mase, compution is GEMM. Like (n_samples, n_visible) x (n_visible * n_hidden)
		Some are transposed (GEMM_TN)

	if (rbm->type_ == RBM::Type::SIGMOID)
		forward_gpu_sigmoid<<<dim_grid, dim_threads>>>();
	else if (rbm->type_ == RBM::Type::EXP)
		forward_gpu_exp<<<dim_grid, dim_threads>>>();

	// Calculate last layer's diff (= output - target), cost, error using CPU

    for (int layer=max_layer - 1; layer >= 0; --layer) {
		if (layer < ctx.start_layer_) 
		    break;
		if (layer != max_layer - 1) {
			if (rbms_[layer]->type_ != RBM::Type::LINEAR)
				gradient_non_linear_gpu<<<dim_grid, dim_threads>>>(); 
			else
				gradient_linear_gpu<<<dim_grid, dim_threads>>>();
		}
		calc_weight_incs_gpu<<<dim_grid, dim_threads>>>();
		weight_incs_add_diff_gpu<<<dim_grid, dim_threads>>>();
    }
	*/
	printf("Fill CUDA CODE in gradient_gpu()\n");
	exit(0);

    auto cend = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000.0;
    std::cout << "evaluating: cost=" << cost << ", error=" << error / n_samples << " in " << duration << "ms" << std::endl;

    endTimer("##### gradient()-calc gradient");
    endTimer("#### gradient()");
    return 0;
}

int DeepBeliefNet::fine_tune(std::vector<Vector>& inputs, std::vector<Vector>& targets, 
	LRBM::Conf& conf, bool is_cuda /* = false */) 
{
    int batch_size = conf.batch_size_, max_epoch = conf.max_epoch_, max_batches = conf.max_batches_; 
    int max_layer = this->rbms_.size();

    startTimer("## fine_tune()");
    std::vector<std::vector<Vector>> probs(max_layer);
    for (int i = 0; i < max_layer; ++i) {
	const RBMP& rbm = this->rbms_[i];
	probs[i].resize(batch_size);
	for (auto &v: probs[i]) { v.resize(rbm->num_hidden()); }
    }

    for (int epoch = 0; epoch < max_epoch; ++epoch) {
	for (int j = 0; j < max_batches; ++j) {
	    int start = j * batch_size, end = start + std::min(batch_size, int(inputs.size()) - j * batch_size);
	    std::cout << "epoch: " << epoch << ", batch: " << j << ", samples: "<< (end - start) << std::endl;
	    GradientContext ctx(Batch(inputs.begin() + start, inputs.begin() + end), probs, epoch);
	    //        ctx.start_layer_ = (epoch > std::min(6, max_epoch / 2)? 0: this->rbms_.size() - 1);
	    if (! targets.empty())
		ctx.targets_ = Batch(targets.begin() + start, targets.begin() + end);
		this->minimize(ctx, is_cuda);
	}
    }

    endTimer("## fine_tune()");
    return 0;
}

template <class Vector1, class Vector2, class Vector3>
const Vector2& RBM::activate_hidden_gpu(const Vector1& visible, Vector2& hidden, const Vector3& bias_hidden, const Vector3& weight, Type type)
{
	startTimer("##### activate_hidden()");
	size_t n_visible = visible.size(), n_hidden = hidden.size();
	std::fill(hidden.begin(), hidden.end(), 0);

	float *d_visible, *d_weight, *d_hidden, *d_bias;
	cudaMalloc((void **)&d_visible, n_visible * sizeof(float));
	cudaMalloc((void **)&d_weight, n_hidden * n_visible * sizeof(float));
	cudaMalloc((void **)&d_bias, n_hidden * sizeof(float));
	cudaMalloc((void **)&d_hidden, n_hidden * sizeof(float));
	cudaMemcpy(d_visible, &visible[0], n_visible * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, &weight[0], n_hidden * n_visible * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias, &bias_hidden[0], n_hidden * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(d_hidden, 0, n_hidden * sizeof(float));

	int threads_width=512;
	int d_x = (n_hidden % threads_width) ? (n_hidden/threads_width+1) : (n_hidden/threads_width);
	int d_y = 1;
	dim3 dim_threads(threads_width, 1);
	dim3 dim_grid(d_x, d_y);

	if (type == Type::SIGMOID)
		dot_gpu<<<dim_grid, dim_threads>>>(n_hidden, n_visible, d_weight, d_visible, d_hidden, d_bias, 1);
	else if (type == Type::EXP)
		dot_gpu<<<dim_grid, dim_threads>>>(n_hidden, n_visible, d_weight, d_visible, d_hidden, d_bias, 0);
	cudaThreadSynchronize();
	cudaMemcpy(&hidden[0], d_hidden, n_hidden * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_visible);
	cudaFree(d_weight);
	cudaFree(d_hidden);
	cudaFree(d_bias);

	endTimer("##### activate_hidden()");
	return hidden;
}
