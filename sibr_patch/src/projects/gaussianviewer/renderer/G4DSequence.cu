/*
 * SIBR-4D - derivative work of the SIBR "gaussianviewer" project
 * (Copyright (C) 2023, Inria, GRAPHDECO research group, https://team.inria.fr/graphdeco).
 * Modified / added in 2026 by the SIBR-4D contributors: 4D sequence playback, comparison viewer.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE_GAUSSIAN_SPLATTING.md file.
 */
// SIBR-4D (11-23): CUDA helpers for the 4D sequence viewer.
// Frame layout of sequence.g4d (see viewer_4d/export_g4d.py):
//   pos float32[N*3] | rot fp16[N*4] | scale fp16[N*3] | opacity fp16[N] | shs fp16[N*48]
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void g4d_unpack_kernel(const float* __restrict__ fpos, const __half* __restrict__ h, int N,
	float* pos, float* rot, float* scale, float* opacity, float* shs)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const __half* hrot = h;
	const __half* hscale = hrot + (size_t)N * 4;
	const __half* hop = hscale + (size_t)N * 3;
	const __half* hshs = hop + (size_t)N;
	for (int k = 0; k < 3; k++) pos[i * 3 + k] = fpos[(size_t)i * 3 + k];
	for (int k = 0; k < 4; k++) rot[i * 4 + k] = __half2float(hrot[(size_t)i * 4 + k]);
	for (int k = 0; k < 3; k++) scale[i * 3 + k] = __half2float(hscale[(size_t)i * 3 + k]);
	opacity[i] = __half2float(hop[i]);
	for (int k = 0; k < 48; k++) shs[(size_t)i * 48 + k] = __half2float(hshs[(size_t)i * 48 + k]);
}

size_t g4dFrameBytes(int N)
{
	return (size_t)N * 3 * sizeof(float) + (size_t)N * (4 + 3 + 1 + 48) * sizeof(__half);
}

static inline int nblk(size_t n) { return int((n + 255) / 256); }

void g4dUnpackFrame(const void* frame, int N, float* pos, float* rot, float* scale, float* opacity, float* shs)
{
	const float* fpos = (const float*)frame;
	const __half* h = (const __half*)((const char*)frame + (size_t)N * 3 * sizeof(float));
	g4d_unpack_kernel<<<nblk(N), 256>>>(fpos, h, N, pos, rot, scale, opacity, shs);
}

// ---------------------------------------------------------------- per-Gaussian false colours
__device__ float3 turbo(float x)
{
	x = fminf(fmaxf(x, 0.f), 1.f);
	// polynomial fit of Google's Turbo colormap
	float r = 0.13572138f + x * (4.61539260f + x * (-42.66032258f + x * (132.13108234f + x * (-152.94239396f + x * 59.28637943f))));
	float g = 0.09140261f + x * (2.19418839f + x * (4.84296658f + x * (-14.18503333f + x * (4.27729857f + x * 2.82956604f))));
	float b = 0.10667330f + x * (12.64194608f + x * (-60.58204836f + x * (110.36276771f + x * (-89.90310912f + x * 27.34824973f))));
	return make_float3(fminf(fmaxf(r, 0.f), 1.f), fminf(fmaxf(g, 0.f), 1.f), fminf(fmaxf(b, 0.f), 1.f));
}

// mode: 0 = SH DC colour, 1 = |pos - ref| (motion / velocity), 2 = opacity, 3 = log max scale
__global__ void g4d_colors_kernel(int mode, int N, const float* pos, const float* ref, const float* opacity,
	const float* scale, const float* shs, float vmin, float vmax, float* out)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	float3 c;
	if (mode == 0)
	{
		const float C0 = 0.28209479177387814f;
		c = make_float3(fminf(fmaxf(0.5f + C0 * shs[i * 48 + 0], 0.f), 1.f),
			fminf(fmaxf(0.5f + C0 * shs[i * 48 + 1], 0.f), 1.f),
			fminf(fmaxf(0.5f + C0 * shs[i * 48 + 2], 0.f), 1.f));
	}
	else
	{
		float v = 0.f;
		if (mode == 1)
		{
			float dx = pos[i * 3] - ref[i * 3], dy = pos[i * 3 + 1] - ref[i * 3 + 1], dz = pos[i * 3 + 2] - ref[i * 3 + 2];
			v = sqrtf(dx * dx + dy * dy + dz * dz);
		}
		else if (mode == 2) v = opacity[i];
		else v = log10f(fmaxf(fmaxf(scale[i * 3], scale[i * 3 + 1]), fmaxf(scale[i * 3 + 2], 1e-8f)));
		c = turbo((v - vmin) / fmaxf(vmax - vmin, 1e-12f));
	}
	out[i * 3] = c.x; out[i * 3 + 1] = c.y; out[i * 3 + 2] = c.z;
}

void g4dColors(int mode, int N, const float* pos, const float* ref, const float* opacity, const float* scale,
	const float* shs, float vmin, float vmax, float* out)
{
	g4d_colors_kernel<<<nblk(N), 256>>>(mode, N, pos, ref, opacity, scale, shs, vmin, vmax, out);
}

__global__ void g4d_fill_kernel(int n, float v, float* out) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = v; }
void g4dFill(int n, float v, float* out) { g4d_fill_kernel<<<nblk(n), 256>>>(n, v, out); }

// ---------------------------------------------------------------- compose two planar CHW renders into RGBA8
// cmp: 0 = A only, 1 = side by side, 2 = wipe at column split, 3 = |A-B| * gain, 4 = B only, 5 = blend (1-gain)A + gain B
__device__ float pix(const float* img, int w, int h, int x, int y, int c) { return img[(size_t)c * w * h + (size_t)y * w + x]; }

__global__ void g4d_compose_kernel(int cmp, const float* A, int wa, int ha, const float* B, int wb, int hb,
	int W, int H, int split, float gain, uint32_t* out)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	float r = 0.1f, g = 0.1f, b = 0.1f;
	if (cmp == 0 || cmp == 4)
	{
		const float* I = cmp == 0 ? A : B; int w = cmp == 0 ? wa : wb, h = cmp == 0 ? ha : hb;
		if (I && x < w && y < h) { r = pix(I, w, h, x, y, 0); g = pix(I, w, h, x, y, 1); b = pix(I, w, h, x, y, 2); }
	}
	else if (cmp == 1)
	{
		int half = W / 2;
		if (x < half) { if (A && x < wa && y < ha) { r = pix(A, wa, ha, x, y, 0); g = pix(A, wa, ha, x, y, 1); b = pix(A, wa, ha, x, y, 2); } }
		else if (x == half) { r = g = b = 1.f; }
		else { int xb = x - half - 1; if (B && xb < wb && y < hb) { r = pix(B, wb, hb, xb, y, 0); g = pix(B, wb, hb, xb, y, 1); b = pix(B, wb, hb, xb, y, 2); } }
	}
	else if (cmp == 2)
	{
		const float* I = x < split ? A : B; int w = x < split ? wa : wb, h = x < split ? ha : hb;
		if (I && x < w && y < h) { r = pix(I, w, h, x, y, 0); g = pix(I, w, h, x, y, 1); b = pix(I, w, h, x, y, 2); }
		if (x == split || x == split + 1) { r = 1.f; g = 1.f; b = 0.f; }
	}
	else if (cmp == 5)   // overlay: (1 - gain) * A + gain * B   (gain = blend weight 0..1)
	{
		if (A && B && x < wa && y < ha && x < wb && y < hb)
		{
			r = (1.f - gain) * pix(A, wa, ha, x, y, 0) + gain * pix(B, wb, hb, x, y, 0);
			g = (1.f - gain) * pix(A, wa, ha, x, y, 1) + gain * pix(B, wb, hb, x, y, 1);
			b = (1.f - gain) * pix(A, wa, ha, x, y, 2) + gain * pix(B, wb, hb, x, y, 2);
		}
	}
	else if (cmp == 3)
	{
		if (A && B && x < wa && y < ha && x < wb && y < hb)
		{
			r = fabsf(pix(A, wa, ha, x, y, 0) - pix(B, wb, hb, x, y, 0)) * gain;
			g = fabsf(pix(A, wa, ha, x, y, 1) - pix(B, wb, hb, x, y, 1)) * gain;
			b = fabsf(pix(A, wa, ha, x, y, 2) - pix(B, wb, hb, x, y, 2)) * gain;
		}
	}
	auto q = [](float v) { return (uint32_t)(fminf(fmaxf(v, 0.f), 1.f) * 255.f + 0.5f); };
	out[(size_t)y * W + x] = q(r) | (q(g) << 8) | (q(b) << 16) | (255u << 24);
}

void g4dCompose(int cmp, const float* A, int wa, int ha, const float* B, int wb, int hb, int W, int H, int split, float gain, uint32_t* out)
{
	dim3 t(16, 16), g((W + 15) / 16, (H + 15) / 16);
	g4d_compose_kernel<<<g, t>>>(cmp, A, wa, ha, B, wb, hb, W, H, split, gain, out);
}

// ---------------------------------------------------------------- sum of squared error (for PSNR)
__global__ void g4d_sse_kernel(const float* A, const float* B, size_t n, double* acc)
{
	__shared__ double s[256];
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	double v = 0.0;
	if (i < n) { float a = fminf(fmaxf(A[i], 0.f), 1.f), b = fminf(fmaxf(B[i], 0.f), 1.f); v = double(a - b) * double(a - b); }
	s[threadIdx.x] = v;
	__syncthreads();
	for (int st = 128; st > 0; st >>= 1) { if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st]; __syncthreads(); }
	if (threadIdx.x == 0) atomicAdd(acc, s[0]);
}

double g4dMSE(const float* A, const float* B, size_t n, double* accCuda)
{
	cudaMemset(accCuda, 0, sizeof(double));
	g4d_sse_kernel<<<nblk(n), 256>>>(A, B, n, accCuda);
	double h = 0;
	cudaMemcpy(&h, accCuda, sizeof(double), cudaMemcpyDeviceToHost);
	return h / double(n);
}

__global__ void g4d_scale_kernel(int n, float k, float* buf) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) buf[i] *= k; }
void g4dScaleBuf(int n, float k, float* buf) { g4d_scale_kernel<<<nblk(n), 256>>>(n, k, buf); }
