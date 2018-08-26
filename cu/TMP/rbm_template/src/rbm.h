/*
   Copyright (c) 2013, jackdeng@gmail.com
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Algorithm is based on matlab code from http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html

// Conjugate Gradient implementation is based on matlab code at http://learning.eng.cam.ac.uk/carl/code/minimize/minimize.m
// % Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).

#pragma once

#include <iostream>
#include <numeric>
#include <vector>
#include <random>
#include <memory>
#include <fstream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#include "v.h"
#include "Timer.h"

struct Batch: public std::pair<std::vector<Vector>::iterator, std::vector<Vector>::iterator>
{
	using Iterator = std::vector<Vector>::iterator;
	using Parent = std::pair<Iterator, Iterator>;
	template <typename... Arg> Batch(Arg&& ... arg): Parent(std::forward<Arg>(arg) ...) {}

	Iterator begin() const { return first; }
	Iterator end() const { return second; }
	size_t size() const { return std::distance(first, second); }
	bool empty() const  { return first == second; }

	Vector& operator[](size_t i) { return *(first + i); }
	const Vector& operator[](size_t i) const { return *(first + i); }
};

const Vector& bernoulli(const Vector& input, Vector& output);
float sigmoid(float x);

__device__ inline float _sigmoid(float x) ;
__device__ inline float dopprod_tr(int x, int y, int len, float *A, float *B, float bias);
__global__ void matmulGPU_global(int nc, int nv, int nh, float *v, float *w, float *h, float* bias);
__global__ void matmulGPU_exp_global(int nc, int nv, int nh, float *v,float *w, float *h, float* bias);
__global__ void matmulGPU_addnoise_global(int nc, int nv, int nh, float *h,float *w, float *v, float* noise, float* bias);
__global__ void matmulGPU_bernoulli_global(int nc, int nv, int nh, float *h,float *w, float *v, float* hs, float* bias);
__global__ void forward_gpu_sigmoid(int ns, int nh, int nv, float *v, float *w, float *h, float* bias);
__global__ void forward_gpu_exp(int ns, int nh, int nv, float *v, float *w, float *h, float* bias);
__global__ void gradient_non_linear_gpu(int ns, int nv, int nh, float *hd, float *w, float *vd, float *v);
__global__ void gradient_linear_gpu(int ns, int nv, int nh, float *hd, float *w, float *vd);
__global__ void calc_weight_incs_gpu(int nh, int nv, int ns, float *diff, float *buf, float *weight_incs);
__global__ void weight_incs_add_diff_gpu(int ns, int nh, float *diff, float *weight_incs);
__global__ void dot_gpu(int M, int N, float *a, float *b, float *r, float *bias, bool type);

struct RBM
{
	enum class Type 
	{
		SIGMOID,
		LINEAR,
		EXP
	};

	Type type_ = Type::SIGMOID;
	Vector bias_visible_, bias_hidden_, bias_visible_inc_, bias_hidden_inc_;
	Vector weight_, weight_inc_;

	struct Conf 
	{
		float momentum_ = 0.5, weight_cost_ = 0.0002, learning_rate_ = 0.1;
	};


	static const Vector& add_noise(const Vector& input, Vector& output)
	{ 
		static std::default_random_engine eng(::time(NULL));
		static std::normal_distribution<float> rng(0.0, 1.0);

		for (size_t i=0; i<input.size(); ++i) { output[i] = input[i] + rng(eng); }
		return output;
	}

	RBM() {}

	RBM(size_t visible, size_t hidden)
		: bias_visible_(visible), bias_hidden_(hidden), weight_(visible * hidden)
		  , bias_visible_inc_(visible), bias_hidden_inc_(hidden), weight_inc_(visible * hidden)
	{
		static std::default_random_engine eng(::time(NULL));
		static std::normal_distribution<float> rng(0.0, 1.0);
		for (auto& x: weight_) x = rng(eng) * .1;
	}

	size_t num_hidden() const { return bias_hidden_.size(); }
	size_t num_visible() const { return bias_visible_.size(); }
	size_t num_weight() const { return weight_.size(); }

	// copy RBM to RBM
	int mirror(const RBM& rbm);

	const Vector& activate_hidden(const Vector& visible, Vector& hidden) const {
		return RBM::activate_hidden(visible, hidden, bias_hidden_, weight_, type_);
	}

	const Vector& activate_hidden_gpu(const Vector& visible, Vector& hidden) const {
		return RBM::activate_hidden_gpu(visible, hidden, bias_hidden_, weight_, type_);
	}

	template <class Vector1, class Vector2, class Vector3>
		static const Vector2& activate_hidden(const Vector1& visible, Vector2& hidden, const Vector3& bias_hidden, const Vector3& weight, Type type)
		{
			startTimer("##### activate_hidden()");
			size_t n_visible = visible.size(), n_hidden = hidden.size();

			std::fill(hidden.begin(), hidden.end(), 0);
			for (size_t i = 0; i < n_hidden; ++i) {
				float *xd = const_cast<float *>(weight.data() + i * n_visible);
				float s = v::dot(visible, v::LightVector(xd, xd + n_visible));
				s += bias_hidden[i];

				if (type == Type::SIGMOID) s = sigmoid(s);
				else if (type == Type::EXP) s = exp(s);

				hidden[i] = s;
			}

			endTimer("##### activate_hidden()");
			return hidden;
		}

	template <class Vector1, class Vector2, class Vector3>
		static const Vector2& activate_hidden_gpu(const Vector1& visible, Vector2& hidden, const Vector3& bias_hidden, const Vector3& weight, Type type);

	const Vector& activate_visible(const Vector& hidden, Vector& visible) const;

	float train(Batch inputs, const Conf& conf);
	float train_gpu(Batch inputs, const Conf& conf);

	virtual float free_energy() const
	{
		size_t n_visible = bias_visible_.size(), n_hidden = bias_hidden_.size();
		float s = 0;
		for (size_t i = 0; i < n_visible; ++i) {
			for (size_t j = 0; j < n_hidden; ++j) 
				s += weight_[j * n_visible+ i] * bias_hidden_[j] * bias_visible_[i];
		}
		return -s;
	}

	template <typename T> static void _write(std::ostream& os, const T& v) { os.write(reinterpret_cast<const char *>(&v), sizeof(v)); }
	void store(std::ostream& os) const
	{
		int type = (int)type_;
		size_t n_visible = bias_visible_.size();
		size_t n_hidden = bias_hidden_.size();

		_write(os, type); _write(os, n_visible); _write(os, n_hidden);
		for (float v: bias_visible_) _write(os, v);
		for (float v: bias_hidden_) _write(os, v);
		for (float v: weight_) _write(os, v);
	}

	template <typename T> static void _read(std::istream& is, T& v) { is.read(reinterpret_cast<char *>(&v), sizeof(v)); }
	void load(std::istream& is)
	{
		int type = 0;
		size_t n_visible = 0, n_hidden = 0;
		_read(is, type); _read(is, n_visible); _read(is, n_hidden);

		type_ = (Type)type;
		bias_visible_.resize(n_visible);
		bias_hidden_.resize(n_hidden);
		weight_.resize(n_visible * n_hidden);

		for (float& v: bias_visible_) _read(is, v);
		for (float& v: bias_hidden_) _read(is, v);
		for (float& v: weight_) _read(is, v);
	}
};

using RBMP = std::unique_ptr<RBM>;
struct LRBM // layered RBM
{
	int max_neurons = 0;
	int max_n_visible = 0;
	int max_n_hidden = 0;

	struct Conf
	{
		int max_epoch_ = 20, max_batches_ = 1000, batch_size_ = 30;
	};

	std::vector<RBMP> rbms_;

	RBMP& output_layer() { return rbms_[rbms_.size() - 1]; }
	size_t max_layer() const { return rbms_.size(); }

	int build(const std::vector<int>& layers, const std::vector<int>& adjust = std::vector<int>());
	std::vector<int> offsets(int start) const;
	void to_image(Vector& image, int& width, int& height);
	void store(std::ostream& os) const;
	void load(std::istream& is);
};

struct DeepBeliefNet : public LRBM
{
	struct GradientContext
	{
		int max_iteration_;

		int epoch_;
		Batch inputs_;
		Batch targets_;
		int start_layer_;

		std::vector<std::vector<Vector>>& probs_;

		GradientContext(Batch inputs, std::vector<std::vector<Vector>> & probs, int epoch)
			: max_iteration_(3), epoch_(epoch), inputs_(inputs), start_layer_(0), probs_(probs)
		{}
	};

	float free_energy() const {
		return std::accumulate(rbms_.begin(), rbms_.end(), 
				0.0f, [](float x, const RBMP& rbm) { return x + rbm->free_energy(); });
	}

	virtual int pretrain(std::vector<Vector>& inputs, LRBM::Conf& conf, bool is_cuda = false)
	{
		static std::vector<Vector> nil;
		return train(inputs, nil, this->rbms_.size() - 1, conf, is_cuda);
	}

	int train(std::vector<Vector>& inputs, std::vector<Vector>& labels, 
			int max_layer, LRBM::Conf& conf, bool is_cuda = false);
	int predict(const Vector& sample, Vector& output, Vector& probs);
	int gradient(GradientContext& ctx, const Vector& weights, Vector& weight_incs, float& cost);
	int minimize(GradientContext& ctx, bool is_cuda = false);
	int fine_tune(std::vector<Vector>& inputs, std::vector<Vector>& targets, 
			LRBM::Conf& conf, bool is_cuda = false);

	// added by kyungchul
	int gradient_gpu(GradientContext& ctx, const Vector& weights, Vector& weight_incs, float& cost);
};

