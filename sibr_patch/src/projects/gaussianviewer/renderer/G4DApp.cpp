/*
 * SIBR-4D - derivative work of the SIBR "gaussianviewer" project
 * (Copyright (C) 2023, Inria, GRAPHDECO research group, https://team.inria.fr/graphdeco).
 * Modified / added in 2026 by the SIBR-4D contributors: 4D sequence playback, comparison viewer.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE_GAUSSIAN_SPLATTING.md file.
 */
// SIBR-4D viewer (11-23) — YUView-style comparison viewer built on the original 3DGS SIBR libraries.
//   menu bar (File / View / Compare / Camera / Frame / Panels), playlist (left), resizable centre view,
//   timeline (bottom), info + pixel inspector + PSNR (right).
//   Rendering = the original 3DGS CUDA rasterizer (CudaRasterizer::Rasterizer::forward) and SIBR's ellipsoid renderer.
//   A model folder = 3DGS layout (cfg_args, cameras.json, point_cloud/iteration_N/point_cloud.ply) + optional
//   sequence.g4d written by viewer_4d/export_g4d.py (every frame's deformed Gaussians, kept on the GPU).
#include "Config.hpp"
#include "GaussianSurfaceRenderer.hpp"
#include <core/graphics/Window.hpp>
#include <core/graphics/GUI.hpp>
#include <core/graphics/Input.hpp>
#include <core/graphics/RenderTarget.hpp>
#include <core/view/InteractiveCameraHandler.hpp>
#include <core/assets/InputCamera.hpp>
#include <core/system/Utils.hpp>
#include <core/system/CommandLineArgs.hpp>
#include <rasterizer.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <imgui/imgui.h>
#include <imgui/examples/opengl3_example/imgui_impl_glfw_gl3.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

size_t g4dFrameBytes(int N);
void g4dUnpackFrame(const void* frame, int N, float* pos, float* rot, float* scale, float* opacity, float* shs);
void g4dColors(int mode, int N, const float* pos, const float* ref, const float* opacity, const float* scale,
	const float* shs, float vmin, float vmax, float* out);
void g4dFill(int n, float v, float* out);
void g4dCompose(int cmp, const float* A, int wa, int ha, const float* B, int wb, int hb, int W, int H, int split, float gain, uint32_t* out);
double g4dMSE(const float* A, const float* B, size_t n, double* accCuda);
void g4dScaleBuf(int n, float k, float* buf);

namespace {

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { SIBR_WRG << "CUDA " << cudaGetErrorString(e_) << " at " << #x << std::endl; } } while (0)

std::function<char* (size_t)> growFn(void** ptr, size_t& size)
{
	return [ptr, &size](size_t N) {
		if (N > size) {
			if (*ptr) cudaFree(*ptr);
			cudaMalloc(ptr, 2 * N);
			size = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
	};
}

enum Mode { M_SPLATS, M_SH0, M_POINTS, M_MOTION, M_VELOCITY, M_OPACITY, M_SCALE, M_ELLIPSOIDS, M_COUNT };
const char* kModeNames[M_COUNT] = { "Splats (SH3)", "Splats (DC colour only)", "Points (centres)", "Motion heat (vs frame 0)",
	"Velocity heat (vs previous frame)", "Opacity heat", "Scale heat (log10 max axis)", "Ellipsoids" };
enum Cmp { C_SINGLE, C_SIDE, C_WIPE, C_DIFF, C_OVERLAY, C_COUNT };
const char* kCmpNames[C_COUNT] = { "Single view", "Grid: 1-4 views (synced camera)", "Wipe (view 1 / view 2 split)", "Difference |1 - 2|", "Overlay (1 blended over 2)" };
const char* kLayoutNames[4] = { "Auto", "Row (side by side)", "Column (stacked)", "2 x 2" };

struct Model
{
	std::string folder, name, source;
	int N = 0, F = 1, shDeg = 3, shown = -1, glShown = -1;
	float tLast = 0.f;
	bool whiteBg = false;
	sibr::Vector3f center = sibr::Vector3f(0, 0, 0);   // median of Gaussian centres (orbit pivot)
	float radius = 1.f;                                // 90th percentile distance from centre
	std::vector<sibr::InputCamera::Ptr> cams;
	size_t frameBytes = 0, vram = 0;
	bool inRam = false;
	char* seq = nullptr;        // all frames on the GPU, or
	char* seqHost = nullptr;    // all frames in pinned RAM (when the GPU is full); the current frame is staged
	char* stage = nullptr;
	float *pos0 = nullptr, *posRef = nullptr; int posRefFrame = -1;
	float *pos = nullptr, *rot = nullptr, *scale = nullptr, *opac = nullptr, *shs = nullptr, *colors = nullptr, *ones = nullptr;
	int* rects = nullptr;
	void *geomP = nullptr, *binP = nullptr, *imgP = nullptr;
	size_t geomS = 0, binS = 0, imgS = 0;
	std::function<char* (size_t)> geomF, binF, imgF;
	std::unique_ptr<sibr::GaussianData> gl;
	cudaGraphicsResource_t glRes[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };
	float lastMs = 0.f;

	~Model()
	{
		for (auto r : glRes) if (r) cudaGraphicsUnregisterResource(r);
		if (seqHost) cudaFreeHost(seqHost);
		for (void* p : { (void*)seq, (void*)stage, (void*)pos0, (void*)posRef, (void*)pos, (void*)rot, (void*)scale, (void*)opac, (void*)shs, (void*)colors, (void*)ones, (void*)rects, geomP, binP, imgP })
			if (p) cudaFree(p);
	}

	bool animated() const { return seq || seqHost; }
	const float* framePos(int f)   // centre positions of frame f (device pointer)
	{
		f = std::max(0, std::min(F - 1, f));
		if (seq) return (const float*)(seq + frameBytes * (size_t)f);
		if (!seqHost) return pos;
		if (f == 0) return pos0;
		if (f != posRefFrame) { CK(cudaMemcpy(posRef, seqHost + frameBytes * (size_t)f, (size_t)N * 12, cudaMemcpyHostToDevice)); posRefFrame = f; }
		return posRef;
	}

	void showFrame(int f)
	{
		f = std::max(0, std::min(F - 1, f));
		if (!animated() || f == shown) return;
		if (seq) g4dUnpackFrame(seq + frameBytes * (size_t)f, N, pos, rot, scale, opac, shs);
		else
		{
			CK(cudaMemcpy(stage, seqHost + frameBytes * (size_t)f, frameBytes, cudaMemcpyHostToDevice));
			g4dUnpackFrame(stage, N, pos, rot, scale, opac, shs);
		}
		shown = f;
	}

	float glScale = 1.f;
	void syncGL(float scaleMul = 1.f)   // copy the current frame into the ellipsoid renderer's GL buffers
	{
		if (!gl)
		{
			gl.reset(new sibr::GaussianData(N, nullptr, nullptr, nullptr, nullptr, nullptr));
			for (int i = 0; i < 5; i++)
				CK(cudaGraphicsGLRegisterBuffer(&glRes[i], gl->buffer(i), cudaGraphicsRegisterFlagsWriteDiscard));
			glShown = -2;
		}
		int cur = animated() ? shown : 0;
		if (cur == glShown && scaleMul == glScale) return;
		const float* src[5] = { pos, rot, scale, opac, shs };
		size_t bytes[5] = { N * 12ull, N * 16ull, N * 12ull, N * 4ull, N * 192ull };
		CK(cudaGraphicsMapResources(5, glRes));
		for (int i = 0; i < 5; i++)
		{
			void* dst = nullptr; size_t sz = 0;
			CK(cudaGraphicsResourceGetMappedPointer(&dst, &sz, glRes[i]));
			CK(cudaMemcpy(dst, src[i], std::min(sz, bytes[i]), cudaMemcpyDeviceToDevice));
			if (i == 2 && scaleMul != 1.f) g4dScaleBuf(N * 3, scaleMul, (float*)dst);
		}
		CK(cudaGraphicsUnmapResources(5, glRes));
		glShown = cur; glScale = scaleMul;
	}
};

std::string readText(const std::string& p) { std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str(); }

std::string cfgValue(const std::string& cfg, const std::string& key)
{
	size_t s = cfg.find(key + "=");
	if (s == std::string::npos) return "";
	s += key.size() + 1;
	size_t e = cfg.find_first_of(",)", s);
	std::string v = cfg.substr(s, e - s);
	v.erase(std::remove(v.begin(), v.end(), '\''), v.end());
	return v;
}

std::string latestPly(const std::string& folder)
{
	std::string pc = folder + "/point_cloud";
	int best = -1; std::string bestDir;
	for (const auto& d : sibr::listSubdirectories(pc))
	{
		if (d.rfind("iteration_", 0) != 0) continue;
		int it = std::atoi(d.c_str() + 10);
		if (it > best) { best = it; bestDir = d; }
	}
	return best < 0 ? "" : pc + "/" + bestDir + "/point_cloud.ply";
}

// Standard 3DGS ply (x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3, float32, binary LE)
bool readPly(const std::string& path, std::vector<float>& P, std::vector<float>& R, std::vector<float>& S,
	std::vector<float>& O, std::vector<float>& SH, int& N, std::string& err)
{
	std::ifstream f(path, std::ios::binary);
	if (!f) { err = "cannot open " + path; return false; }
	std::string line; std::vector<std::string> props; N = 0;
	while (std::getline(f, line))
	{
		if (!line.empty() && line.back() == '\r') line.pop_back();
		std::istringstream ls(line); std::string a, b, c; ls >> a >> b >> c;
		if (a == "element" && b == "vertex") N = std::atoi(c.c_str());
		else if (a == "property") { if (b != "float") { err = "ply property type " + b + " not supported"; return false; } props.push_back(c); }
		else if (a == "end_header") break;
	}
	std::map<std::string, int> idx; for (int i = 0; i < (int)props.size(); i++) idx[props[i]] = i;
	for (const char* k : { "x", "f_dc_0", "opacity", "scale_0", "rot_0" }) if (!idx.count(k)) { err = std::string("ply lacks ") + k; return false; }
	int nRest = 0; while (idx.count("f_rest_" + std::to_string(nRest))) nRest++;
	int D = nRest / 3 + 1;   // coefficients per channel
	std::vector<float> raw((size_t)N * props.size());
	f.read((char*)raw.data(), raw.size() * sizeof(float));
	if (!f) { err = "ply truncated"; return false; }
	P.resize(N * 3); R.resize(N * 4); S.resize(N * 3); O.resize(N); SH.assign((size_t)N * 48, 0.f);
	const int C = (int)props.size();
	for (int i = 0; i < N; i++)
	{
		const float* v = &raw[(size_t)i * C];
		for (int k = 0; k < 3; k++) P[i * 3 + k] = v[idx["x"] + k];
		float q[4], l = 0; for (int k = 0; k < 4; k++) { q[k] = v[idx["rot_0"] + k]; l += q[k] * q[k]; }
		l = std::sqrt(std::max(l, 1e-20f)); for (int k = 0; k < 4; k++) R[i * 4 + k] = q[k] / l;
		for (int k = 0; k < 3; k++) S[i * 3 + k] = std::exp(v[idx["scale_0"] + k]);
		O[i] = 1.f / (1.f + std::exp(-v[idx["opacity"]]));
		for (int c = 0; c < 3; c++) SH[(size_t)i * 48 + c] = v[idx["f_dc_0"] + c];
		for (int j = 1; j < std::min(D, 16); j++)
			for (int c = 0; c < 3; c++) SH[(size_t)i * 48 + j * 3 + c] = v[idx["f_rest_0"] + c * (D - 1) + (j - 1)];
	}
	return true;
}

std::unique_ptr<Model> loadModel(const std::string& folderIn, std::string& err)
{
	std::string folder = folderIn; for (auto& c : folder) if (c == '\\') c = '/';
	while (!folder.empty() && folder.back() == '/') folder.pop_back();
	auto m = std::make_unique<Model>();
	m->folder = folder;
	m->name = folder.substr(folder.find_last_of('/') + 1);
	std::string cfg = readText(folder + "/cfg_args");
	m->whiteBg = cfgValue(cfg, "white_background") == "True";
	m->source = cfgValue(cfg, "source_path");
	std::string ply = latestPly(folder);
	if (ply.empty()) { err = "no point_cloud/iteration_*/point_cloud.ply in " + folder; return nullptr; }
	if (sibr::fileExists(folder + "/cameras.json"))
		m->cams = sibr::InputCamera::loadJSON(folder + "/cameras.json", 0.01f, 1000.f);

	std::vector<float> P, R, S, O, SH; int N = 0;
	if (!readPly(ply, P, R, S, O, SH, N, err)) return nullptr;
	m->N = N;
	{
		std::vector<float> ax(N);
		for (int k = 0; k < 3; k++)
		{
			for (int i = 0; i < N; i++) ax[i] = P[i * 3 + k];
			std::nth_element(ax.begin(), ax.begin() + N / 2, ax.end());
			m->center[k] = ax[N / 2];
		}
		std::vector<float> d(N);
		for (int i = 0; i < N; i++) d[i] = (sibr::Vector3f(P[i * 3], P[i * 3 + 1], P[i * 3 + 2]) - m->center).norm();
		std::nth_element(d.begin(), d.begin() + N * 9 / 10, d.end());
		m->radius = std::max(d[N * 9 / 10], 1e-3f);
	}
	size_t need = (size_t)N * (3 + 4 + 3 + 1 + 48 + 3 + 1) * 4 + (size_t)N * 8;
	std::string seqPath = folder + "/sequence.g4d";
	int F = 0; float tLast = 0.f;
	if (sibr::fileExists(seqPath))
	{
		std::ifstream f(seqPath, std::ios::binary);
		char magic[4]; int sN = 0, sF = 0, shc = 0; char pad[12];
		f.read(magic, 4); f.read((char*)&sN, 4); f.read((char*)&sF, 4); f.read((char*)&shc, 4); f.read((char*)&tLast, 4); f.read(pad, 12);
		if (std::string(magic, 4) == "G4D1" && sN == N && shc == 16 && sF > 0) F = sF;
		else SIBR_WRG << seqPath << ": header mismatch (N " << sN << " vs ply " << N << "), loading static ply" << std::endl;
		if (F > 0)
		{
			m->frameBytes = g4dFrameBytes(N);
			need += m->frameBytes * (size_t)F;
		}
	}
	size_t freeB = 0, totalB = 0; cudaMemGetInfo(&freeB, &totalB);
	const size_t reserve = 3072ull << 20;   // render buffers for up to 4 tiles + other apps on the GPU
	bool inRam = false;
	if (F > 0 && need + reserve > freeB)
	{
		// GPU full: keep the frames in pinned RAM and stage only the current frame
		inRam = true;
		need -= m->frameBytes * (size_t)F;
		need += m->frameBytes + (size_t)N * 24;
	}
	if (need + (256ull << 20) > freeB)
	{
		err = "not enough GPU memory: need " + std::to_string(need >> 20) + " MB, free " + std::to_string(freeB >> 20) + " MB (close another model first)";
		return nullptr;
	}
	CK(cudaMalloc((void**)&m->pos, N * 12ull)); CK(cudaMalloc((void**)&m->rot, N * 16ull)); CK(cudaMalloc((void**)&m->scale, N * 12ull));
	CK(cudaMalloc((void**)&m->opac, N * 4ull)); CK(cudaMalloc((void**)&m->shs, N * 192ull)); CK(cudaMalloc((void**)&m->colors, N * 12ull));
	CK(cudaMalloc((void**)&m->ones, N * 4ull)); CK(cudaMalloc((void**)&m->rects, N * 8ull));
	CK(cudaMemcpy(m->pos, P.data(), N * 12ull, cudaMemcpyHostToDevice)); CK(cudaMemcpy(m->rot, R.data(), N * 16ull, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(m->scale, S.data(), N * 12ull, cudaMemcpyHostToDevice)); CK(cudaMemcpy(m->opac, O.data(), N * 4ull, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(m->shs, SH.data(), N * 192ull, cudaMemcpyHostToDevice));
	g4dFill(N, 1.f, m->ones);
	if (F > 0)
	{
		std::ifstream f(seqPath, std::ios::binary); f.seekg(32);
		if (!inRam)
		{
			if (cudaMalloc((void**)&m->seq, m->frameBytes * (size_t)F) != cudaSuccess) { cudaGetLastError(); err = "cudaMalloc failed for sequence"; return nullptr; }
			std::vector<char> buf(m->frameBytes);
			for (int k = 0; k < F; k++)
			{
				f.read(buf.data(), m->frameBytes);
				if (!f) { SIBR_WRG << "sequence truncated at frame " << k << std::endl; F = k; break; }
				CK(cudaMemcpy(m->seq + m->frameBytes * k, buf.data(), m->frameBytes, cudaMemcpyHostToDevice));
			}
		}
		else
		{
			if (cudaMallocHost((void**)&m->seqHost, m->frameBytes * (size_t)F) != cudaSuccess)
			{ cudaGetLastError(); err = "not enough RAM for the sequence (" + std::to_string((m->frameBytes * (size_t)F) >> 20) + " MB)"; return nullptr; }
			f.read(m->seqHost, m->frameBytes * (size_t)F);
			if (!f) { F = int(f.gcount() / m->frameBytes); SIBR_WRG << "sequence truncated at frame " << F << std::endl; }
			CK(cudaMalloc((void**)&m->stage, m->frameBytes));
			CK(cudaMalloc((void**)&m->pos0, (size_t)N * 12)); CK(cudaMalloc((void**)&m->posRef, (size_t)N * 12));
			CK(cudaMemcpy(m->pos0, m->seqHost, (size_t)N * 12, cudaMemcpyHostToDevice));
			m->inRam = true;
		}
		m->F = std::max(F, 1); m->tLast = tLast;
		m->showFrame(0);
	}
	m->vram = need;
	m->geomF = growFn(&m->geomP, m->geomS); m->binF = growFn(&m->binP, m->binS); m->imgF = growFn(&m->imgP, m->imgS);
	SIBR_LOG << "SIBR-4D loaded " << m->name << ": N=" << N << " frames=" << m->F << " cams=" << m->cams.size() << " VRAM " << (need >> 20) << " MB" << (inRam ? " (frames in RAM)" : "") << std::endl;
	return m;
}

struct Settings
{
	int mode = M_SPLATS, cmp = C_SINGLE, bg = 0;   // bg: 0 model default, 1 black, 2 white
	float scaleMod = 1.f, heatMin = 0.f, heatMax = 0.f, diffGain = 4.f, wipe = 0.5f;
	bool aa = false, autoHeat = true;
	int views = 4, layout = 0;                 // grid: number of views (1-4) and arrangement
	float overlay = 0.5f, flickerHz = 2.f; bool flicker = false;
	int selfOffset = 1;                        // wipe / diff / overlay with one model: view 2 = same model at frame + offset
	float ellAlpha = 0.2f, ellScale = 1.f; bool ellWire = false;
};

void heatRange(int mode, const Model& m, float& lo, float& hi)
{
	float diag = 1.f;
	if (!m.cams.empty())
	{
		sibr::Vector3f a = m.cams[0]->position(), b = a;
		for (auto& c : m.cams) { a = a.cwiseMin(c->position()); b = b.cwiseMax(c->position()); }
		diag = std::max((b - a).norm(), 1e-3f);
	}
	switch (mode)
	{
	case M_MOTION: lo = 0.f; hi = 0.05f * diag; break;
	case M_VELOCITY: lo = 0.f; hi = 0.005f * diag; break;
	case M_OPACITY: lo = 0.f; hi = 1.f; break;
	case M_SCALE: lo = -4.f; hi = 0.f; break;
	default: lo = 0.f; hi = 1.f;
	}
}

// render one model into a planar CHW float image
void renderSplat(Model& m, const Settings& s, int frame, sibr::Camera eye, int w, int h, float* out, float* viewC, float* projC, float* camC, float* bgC)
{
	m.showFrame(frame);
	eye.aspect(float(w) / float(h));
	auto view_mat = eye.view();
	auto proj_mat = eye.viewproj();
	view_mat.row(1) *= -1; view_mat.row(2) *= -1; proj_mat.row(1) *= -1;
	float tan_fovy = std::tan(eye.fovy() * 0.5f), tan_fovx = tan_fovy * eye.aspect();
	CK(cudaMemcpy(viewC, view_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
	CK(cudaMemcpy(projC, proj_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
	CK(cudaMemcpy(camC, &eye.position(), sizeof(float) * 3, cudaMemcpyHostToDevice));
	bool white = s.bg == 0 ? m.whiteBg : s.bg == 2;
	float bg[3] = { white ? 1.f : 0.f, white ? 1.f : 0.f, white ? 1.f : 0.f };
	CK(cudaMemcpy(bgC, bg, sizeof(bg), cudaMemcpyHostToDevice));

	const float* colors = nullptr; const float* opac = m.opac; float smod = s.scaleMod; int deg = m.shDeg;
	if (s.mode != M_SPLATS)
	{
		float lo = s.heatMin, hi = s.heatMax;
		if (s.autoHeat) heatRange(s.mode, m, lo, hi);
		int cm = 0; const float* ref = m.pos;
		if (s.mode == M_MOTION) { cm = 1; ref = m.framePos(0); }
		else if (s.mode == M_VELOCITY) { cm = 1; ref = m.framePos(std::max(0, std::min(m.F - 1, frame) - 1)); }
		else if (s.mode == M_OPACITY) cm = 2;
		else if (s.mode == M_SCALE) cm = 3;
		g4dColors(cm, m.N, m.pos, ref, m.opac, m.scale, m.shs, lo, hi, m.colors);
		colors = m.colors;
		if (s.mode == M_POINTS) { opac = m.ones; smod = 1e-4f; }
	}
	auto t0 = std::chrono::high_resolution_clock::now();
	CudaRasterizer::Rasterizer::forward(m.geomF, m.binF, m.imgF, m.N, deg, 16, bgC, w, h,
		m.pos, colors ? nullptr : m.shs, colors, opac, m.scale, smod, m.rot, nullptr,
		viewC, projC, camC, tan_fovx, tan_fovy, false, out, s.aa, nullptr, m.rects, nullptr, nullptr);
	cudaDeviceSynchronize();
	m.lastMs = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}

struct Tex   // GL texture filled from CUDA
{
	GLuint id = 0; int w = 0, h = 0; cudaGraphicsResource_t res = nullptr; uint32_t* lin = nullptr;
	void ensure(int W, int H)
	{
		if (W == w && H == h && id) return;
		if (res) cudaGraphicsUnregisterResource(res);
		if (id) glDeleteTextures(1, &id);
		if (lin) cudaFree(lin);
		w = W; h = H;
		glGenTextures(1, &id);
		glBindTexture(GL_TEXTURE_2D, id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		glBindTexture(GL_TEXTURE_2D, 0);
		CK(cudaGraphicsGLRegisterImage(&res, id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
		CK(cudaMalloc((void**)&lin, (size_t)w * h * 4));
	}
	void upload()
	{
		cudaArray_t arr;
		CK(cudaGraphicsMapResources(1, &res));
		CK(cudaGraphicsSubResourceGetMappedArray(&arr, res, 0, 0));
		CK(cudaMemcpy2DToArray(arr, 0, 0, lin, w * 4, w * 4, h, cudaMemcpyDeviceToDevice));
		CK(cudaGraphicsUnmapResources(1, &res));
	}
};

// ---------------------------------------------------------------- orbit camera (mouse: L orbit, R/M pan, wheel dolly, Ctrl+wheel FOV)
struct OrbitCam
{
	std::unique_ptr<sibr::InputCamera> base;
	sibr::Vector3f pivot = sibr::Vector3f(0, 0, 0), up = sibr::Vector3f(0, 1, 0);
	float dist = 1.f, yaw = 0.f, pitch = 0.f, fovy = 0.8f;
	bool exact = false; sibr::Vector3f exPos; sibr::Quaternionf exRot;   // exact training-camera pose until the user moves
	float animT = 1.f; sibr::Vector3f aP0, aP1; sibr::Quaternionf aR0, aR1;
	int camIdx = 0;

	void basis(sibr::Vector3f& e1, sibr::Vector3f& e2) const
	{
		sibr::Vector3f a = std::abs(up.x()) < 0.9f ? sibr::Vector3f(1, 0, 0) : sibr::Vector3f(0, 0, 1);
		e1 = (a - up * a.dot(up)).normalized(); e2 = up.cross(e1);
	}
	sibr::Vector3f eyePos() const
	{
		sibr::Vector3f e1, e2; basis(e1, e2);
		return pivot + dist * (std::cos(pitch) * (std::cos(yaw) * e1 + std::sin(yaw) * e2) + std::sin(pitch) * up);
	}
	void fromPose(const sibr::Vector3f& pos, const sibr::Vector3f& dir, float d)
	{
		dist = std::max(d, 1e-3f); pivot = pos + dir.normalized() * dist;
		sibr::Vector3f e1, e2; basis(e1, e2);
		sibr::Vector3f v = (pos - pivot) / dist;
		pitch = std::asin(std::max(-0.999f, std::min(0.999f, v.dot(up))));
		yaw = std::atan2(v.dot(e2), v.dot(e1));
	}
	sibr::InputCamera current() const
	{
		sibr::InputCamera c = *base;
		sibr::Vector3f p; sibr::Quaternionf q;
		if (exact) { p = exPos; q = exRot; }
		else { c.setLookAt(eyePos(), pivot, up); p = c.position(); q = c.rotation(); }
		if (animT < 1.f)
		{
			float t = animT * animT * (3.f - 2.f * animT);
			p = aP0 + (p - aP0) * t; q = aR0.slerp(t, q);
		}
		c.set(p, q);
		c.fovy(fovy);
		return c;
	}
	void beginAnim() { sibr::InputCamera c = current(); aP0 = c.position(); aR0 = c.rotation(); animT = 0.f; }
	void setup(const Model& m)
	{
		if (!m.cams.empty()) base.reset(new sibr::InputCamera(*m.cams[0]));
		else base.reset(new sibr::InputCamera(1000.f, 0.f, 0.f, 1280, 720, 0));
		sibr::Vector3f u(0, 0, 0);
		for (auto& c : m.cams) u += c->up();
		up = u.norm() > 1e-6f ? u.normalized() : sibr::Vector3f(0, 1, 0);
		camIdx = 0;
		if (!m.cams.empty()) snap(m, 0, false);
		else { pivot = m.center; dist = m.radius * 2.5f; yaw = 0; pitch = 0.3f; exact = false; fovy = base->fovy(); }
	}
	void snap(const Model& m, int k, bool animate = true)
	{
		if (m.cams.empty()) return;
		k = (k % (int)m.cams.size() + (int)m.cams.size()) % (int)m.cams.size();
		if (animate) beginAnim();
		camIdx = k;
		const auto& c = *m.cams[k];
		exact = true; exPos = c.position(); exRot = c.rotation(); fovy = c.fovy();
		float d = (m.center - c.position()).dot(c.dir());
		fromPose(c.position(), c.dir(), d > 0.05f * m.radius ? d : m.radius);
	}
	void frameScene(const Model& m) { beginAnim(); exact = false; pivot = m.center; dist = m.radius * 2.2f; }
};

// ---------------------------------------------------------------- UI font / settings
std::string settingsPath() { return "sibr4d_settings.txt"; }
struct UiPrefs { float fontPx = 22.f; bool playlist = true, info = true, timeline = true; float orbitSpeed = 1.f, panSpeed = 1.f, flySpeed = 1.f; float leftW = 0.f, rightW = 0.f, bottomH = 0.f; };
void loadPrefs(UiPrefs& p)
{
	std::ifstream f(settingsPath()); std::string k;
	while (f >> k)
	{
		if (k == "fontPx") f >> p.fontPx; else if (k == "playlist") f >> p.playlist; else if (k == "info") f >> p.info;
		else if (k == "timeline") f >> p.timeline; else if (k == "orbitSpeed") f >> p.orbitSpeed; else if (k == "panSpeed") f >> p.panSpeed;
		else if (k == "flySpeed") f >> p.flySpeed;
		else if (k == "leftW") f >> p.leftW; else if (k == "rightW") f >> p.rightW; else if (k == "bottomH") f >> p.bottomH;
	}
	p.fontPx = std::max(12.f, std::min(48.f, p.fontPx));
}
void savePrefs(const UiPrefs& p)
{
	std::ofstream f(settingsPath());
	f << "fontPx " << p.fontPx << "\nplaylist " << p.playlist << "\ninfo " << p.info << "\ntimeline " << p.timeline
		<< "\norbitSpeed " << p.orbitSpeed << "\npanSpeed " << p.panSpeed << "\nflySpeed " << p.flySpeed
		<< "\nleftW " << p.leftW << "\nrightW " << p.rightW << "\nbottomH " << p.bottomH << "\n";
}
ImGuiStyle g_baseStyle;
void applyFont(float px)   // call at the top of a frame, before any widget
{
	ImGuiIO& io = ImGui::GetIO();
	ImGui::EndFrame();
	io.Fonts->Clear();
	ImFontConfig cfg; cfg.OversampleH = 3; cfg.OversampleV = 1;
	const char* ttf = "C:/Windows/Fonts/segoeui.ttf";
	if (sibr::fileExists(ttf)) io.Fonts->AddFontFromFileTTF(ttf, px, &cfg);
	else { cfg.SizePixels = px; io.Fonts->AddFontDefault(&cfg); }
	io.FontGlobalScale = 1.f;
	ImGui::GetStyle() = g_baseStyle;
	ImGui::GetStyle().ScaleAllSizes(px / 15.f);
	ImGui_ImplGlfwGL3_InvalidateDeviceObjects();
	ImGui_ImplGlfwGL3_NewFrame();
}

// ---------------------------------------------------------------- drag & drop of model folders (or any file inside one)
std::vector<std::string> g_dropped;
void onDrop(GLFWwindow*, int count, const char** paths) { for (int i = 0; i < count; i++) g_dropped.push_back(paths[i]); }
bool isModelDir(const std::string& d)
{
	return sibr::fileExists(d + "/cfg_args") || sibr::fileExists(d + "/sequence.g4d") || sibr::directoryExists(d + "/point_cloud");
}
std::vector<std::string> modelDirsFromDrop(std::string p)
{
	for (auto& c : p) if (c == '\\') c = '/';
	while (!p.empty() && p.back() == '/') p.pop_back();
	std::string d = sibr::directoryExists(p) ? p : p.substr(0, p.find_last_of('/'));
	for (int up = 0; up < 4 && !d.empty(); up++)   // dropped a file or sub-folder of a model
	{
		if (isModelDir(d)) return { d };
		size_t s = d.find_last_of('/'); if (s == std::string::npos) break; d = d.substr(0, s);
	}
	std::vector<std::string> out;   // dropped a folder that holds several model folders
	if (sibr::directoryExists(p))
		for (const auto& sub : sibr::listSubdirectories(p)) if (isModelDir(p + "/" + sub)) out.push_back(p + "/" + sub);
	return out;
}

struct FloatImg { float* d = nullptr; int w = 0, h = 0; void ensure(int W, int H) { if (W * H > w * h || !d) { if (d) cudaFree(d); CK(cudaMalloc((void**)&d, (size_t)W * H * 3 * 4)); } w = W; h = H; } };

} // namespace


SIBR_EXP_ULR_EXPORT int sibr4dMain(int ac, char** av)
{
	sibr::CommandLineArgs::parseMainArgs(ac, av);
	sibr::WindowArgs wargs;
	sibr::Window window("SIBR-4D (4DGS viewer)", sibr::Vector2i(40, 40), wargs);
	glfwSetDropCallback(window.GLFW(), onDrop);

	std::vector<std::unique_ptr<Model>> models;
	int slot[4] = { -1, -1, -1, -1 };   // up to 4 compared models; slot 0 = reference (camera, wipe/diff "1"), slot 1 = "2"
	std::string status;
	auto nSel = [&]() { int n = 0; for (int k = 0; k < 4; k++) n += slot[k] >= 0; return n; };
	auto compact = [&]() { int t[4] = { -1, -1, -1, -1 }, n = 0; for (int k = 0; k < 4; k++) if (slot[k] >= 0) t[n++] = slot[k]; for (int k = 0; k < 4; k++) slot[k] = t[k]; };
	auto slotOf = [&](int i) { for (int k = 0; k < 4; k++) if (slot[k] == i) return k; return -1; };
	auto addSlot = [&](int i) { if (slotOf(i) >= 0) return true; for (int k = 0; k < 4; k++) if (slot[k] < 0) { slot[k] = i; return true; } return false; };
	auto makeRef = [&](int i) {
		int k = slotOf(i);
		if (k >= 0) std::swap(slot[0], slot[k]);
		else { for (int j = 3; j > 0; j--) slot[j] = slot[j - 1]; slot[0] = i; }
		compact();
	};
	auto open = [&](const std::string& p) {
		std::string err;
		auto m = loadModel(p, err);
		if (m)
		{
			models.push_back(std::move(m));
			status = "loaded " + models.back()->name + (models.back()->inRam ? " (frames in RAM)" : "");
			if (!addSlot((int)models.size() - 1)) status += "  - 4 models already compared, tick it in the playlist to swap";
		}
		else { status = "load failed: " + err; SIBR_WRG << status << std::endl; }
	};
	for (int i = 1; i < ac; i++)
	{
		std::string a = av[i];
		if (a.rfind("--", 0) == 0) { i++; continue; }
		if (a == "-m" && i + 1 < ac) { open(av[++i]); continue; }
		if (!a.empty() && a[0] != '-') open(a);
	}

	Settings s;
	int startFrame = 0;
	if (nSel() > 1) s.cmp = C_SIDE;
	for (int i = 1; i + 1 < ac; i++)   // --mode 1..8 (view mode), --cmp 0..4 (compare mode), --frame N, --views 1..4
	{
		if (std::string(av[i]) == "--mode") s.mode = std::max(0, std::min(M_COUNT - 1, std::atoi(av[i + 1]) - 1));
		if (std::string(av[i]) == "--cmp") s.cmp = std::max(0, std::min(C_COUNT - 1, std::atoi(av[i + 1])));
		if (std::string(av[i]) == "--frame") startFrame = std::max(0, std::atoi(av[i + 1]));
		if (std::string(av[i]) == "--views") s.views = std::max(1, std::min(4, std::atoi(av[i + 1])));
	}
	if (s.views == 4 && nSel() < 4) s.views = std::max(1, nSel());
	int frame = startFrame; bool play = false, loop = true; float fps = 30.f; double accum = 0.0;
	UiPrefs prefs; loadPrefs(prefs);
	bool& showPlaylist = prefs.playlist; bool& showInfo = prefs.info; bool& showTimeline = prefs.timeline;
	OrbitCam cam; const Model* camModel = nullptr;
	g_baseStyle = ImGui::GetStyle();
	float fontWanted = prefs.fontPx, fontApplied = -1.f;
	std::unique_ptr<sibr::GaussianSurfaceRenderer> ellR;
	sibr::RenderTargetRGBA::Ptr ellRT[4];
	Tex tex[4]; FloatImg img[4];
	float *viewC, *projC, *camC, *bgC; double* accC;
	CK(cudaMalloc((void**)&viewC, 64)); CK(cudaMalloc((void**)&projC, 64)); CK(cudaMalloc((void**)&camC, 12)); CK(cudaMalloc((void**)&bgC, 12)); CK(cudaMalloc((void**)&accC, 8));
	double psnr[4] = { -1, -1, -1, -1 };
	float pix[4][3] = {}; int pixX = -1, pixY = -1;
	bool sizesDirty = false;
	std::string toOpen; int closeIdx = -1;   // requests from menus / playlist, handled at the top of the next frame
	auto lastT = std::chrono::steady_clock::now();
	auto maxFrames = [&]() { int F = 1; for (int k = 0; k < 4; k++) if (slot[k] >= 0) F = std::max(F, models[slot[k]]->F); return F; };

	while (window.isOpened())
	{
		sibr::Input::poll();
		window.makeContextCurrent();
		if (fontWanted != fontApplied) { applyFont(fontWanted); fontApplied = fontWanted; prefs.fontPx = fontWanted; savePrefs(prefs); }
		auto now = std::chrono::steady_clock::now();
		float dt = std::chrono::duration<float>(now - lastT).count(); lastT = now;
		const sibr::Input& in = sibr::Input::global();
		ImGuiIO& io = ImGui::GetIO();
		auto firstFilled = [&]() { for (int k = 0; k < 4; k++) if (slot[k] >= 0) return slot[k]; return -1; };
		int A = firstFilled();
		int F = maxFrames();

		// ---------------- menu bar
		if (ImGui::BeginMainMenuBar())
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Open model folder...", nullptr)) { std::string p; if (sibr::showFilePicker(p, sibr::Directory)) toOpen = p; }
				if (ImGui::BeginMenu("Close model", !models.empty()))
				{
					for (int i = 0; i < (int)models.size(); i++) if (ImGui::MenuItem(models[i]->name.c_str())) closeIdx = i;
					ImGui::EndMenu();
				}
				ImGui::Separator();
				if (ImGui::MenuItem("Exit", "Esc")) window.close();
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("View"))
			{
				for (int i = 0; i < M_COUNT; i++) if (ImGui::MenuItem(kModeNames[i], std::to_string(i + 1).c_str(), s.mode == i)) s.mode = i;
				ImGui::Separator();
				ImGui::MenuItem("Antialiasing", nullptr, &s.aa);
				const char* bgN[3] = { "Background: model default", "Background: black", "Background: white" };
				for (int i = 0; i < 3; i++) if (ImGui::MenuItem(bgN[i], nullptr, s.bg == i)) s.bg = i;
				ImGui::SliderFloat("Scaling modifier", &s.scaleMod, 0.01f, 1.f);
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Compare"))
			{
				for (int i = 0; i < C_COUNT; i++) if (ImGui::MenuItem(kCmpNames[i], nullptr, s.cmp == i, i == C_SINGLE || nSel() >= 2)) s.cmp = i;
				ImGui::Separator();
				if (ImGui::MenuItem("Next model as reference (1)", "Tab", false, nSel() >= 2)) { int t = slot[0]; for (int k = 0; k < 3; k++) slot[k] = slot[k + 1]; slot[3] = t; compact(); }
				ImGui::SliderFloat("Difference gain", &s.diffGain, 1.f, 32.f);
				ImGui::SliderFloat("Wipe position", &s.wipe, 0.f, 1.f);
				ImGui::TextDisabled("Tick up to 4 models in the playlist; click a name = reference (1)");
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Camera"))
			{
				if (A >= 0 && camModel)
				{
					Model& m = *models[A];
					if (ImGui::MenuItem("Previous training camera", "[", false, !m.cams.empty())) cam.snap(m, cam.camIdx - 1);
					if (ImGui::MenuItem("Next training camera", "]", false, !m.cams.empty())) cam.snap(m, cam.camIdx + 1);
					if (ImGui::MenuItem("Back to current training camera", "R", false, !m.cams.empty())) cam.snap(m, cam.camIdx);
					if (ImGui::MenuItem("Frame whole scene", "F")) cam.frameScene(m);
					if (!m.cams.empty() && ImGui::BeginMenu("Go to training camera"))
					{
						for (int i = 0; i < (int)m.cams.size(); i++)
							if (ImGui::MenuItem(("cam " + std::to_string(i) + "   " + m.cams[i]->name()).c_str(), nullptr, cam.camIdx == i)) cam.snap(m, i);
						ImGui::EndMenu();
					}
					ImGui::Separator();
					float deg = cam.fovy * 57.2958f;
					if (ImGui::SliderFloat("Vertical FOV (deg)", &deg, 3.f, 120.f)) cam.fovy = deg / 57.2958f;
				}
				ImGui::SliderFloat("Orbit speed", &prefs.orbitSpeed, 0.2f, 3.f);
				ImGui::SliderFloat("Pan speed", &prefs.panSpeed, 0.2f, 3.f);
				ImGui::SliderFloat("Fly speed (WASD QE)", &prefs.flySpeed, 0.1f, 5.f);
				ImGui::TextDisabled("Mouse: left drag = orbit, right/middle drag = pan,");
				ImGui::TextDisabled("wheel = zoom (dolly), Ctrl+wheel = FOV zoom");
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Frame"))
			{
				if (ImGui::MenuItem(play ? "Pause" : "Play", "Space")) play = !play;
				if (ImGui::MenuItem("First", "Home")) frame = 0;
				if (ImGui::MenuItem("Previous", "Left")) { play = false; frame = std::max(0, frame - 1); }
				if (ImGui::MenuItem("Next", "Right")) { play = false; frame = std::min(F - 1, frame + 1); }
				if (ImGui::MenuItem("Last", "End")) frame = F - 1;
				ImGui::MenuItem("Loop", nullptr, &loop);
				ImGui::InputInt("Go to frame", &frame);
				ImGui::SliderFloat("Playback FPS", &fps, 1.f, 60.f);
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Panels"))
			{
				ImGui::MenuItem("Playlist", nullptr, &showPlaylist);
				ImGui::MenuItem("Info / pixel inspector", nullptr, &showInfo);
				ImGui::MenuItem("Timeline", nullptr, &showTimeline);
				if (ImGui::MenuItem("Reset panel sizes")) { prefs.leftW = prefs.rightW = prefs.bottomH = 0.f; sizesDirty = true; }
				ImGui::Separator();
				ImGui::Text("UI text size: %.0f px", fontWanted);
				if (ImGui::MenuItem("Larger text", "Ctrl +")) fontWanted = std::min(48.f, fontWanted + 2.f);
				if (ImGui::MenuItem("Smaller text", "Ctrl -")) fontWanted = std::max(12.f, fontWanted - 2.f);
				if (ImGui::MenuItem("Reset text size")) fontWanted = 22.f;
				ImGui::EndMenu();
			}
			ImGui::TextDisabled("   %s", status.c_str());
			ImGui::EndMainMenuBar();
		}
		std::vector<std::string> opens;
		if (!toOpen.empty()) { opens.push_back(toOpen); toOpen.clear(); }
		for (const auto& d : g_dropped)
		{
			auto dirs = modelDirsFromDrop(d);
			if (dirs.empty()) status = "dropped item is not a model folder: " + d;
			for (const auto& x : dirs) opens.push_back(x);
		}
		g_dropped.clear();
		for (const auto& o : opens)
		{
			open(o);
			int kk = slotOf((int)models.size() - 1);
			if (kk >= 0 && kk + 1 > s.views) s.views = kk + 1;
			if (nSel() > 1 && s.cmp == C_SINGLE) s.cmp = C_SIDE;
		}
		if (closeIdx >= 0)
		{
			if (models[closeIdx].get() == camModel) camModel = nullptr;
			models.erase(models.begin() + closeIdx);
			for (int k = 0; k < 4; k++) { if (slot[k] == closeIdx) slot[k] = -1; else if (slot[k] > closeIdx) slot[k]--; }
			compact();
			if (slot[0] < 0 && !models.empty()) slot[0] = 0;
			closeIdx = -1;
		}
		compact();   // no gaps: view 1 is always the first filled one
		if ((s.cmp == C_WIPE || s.cmp == C_DIFF || s.cmp == C_OVERLAY) && slot[1] < 0)   // needs a second model: take the next open one
			for (int i = 0; i < (int)models.size(); i++) if (slotOf(i) < 0) { slot[1] = i; break; }
		A = firstFilled();
		F = maxFrames();
		if (A >= 0 && camModel != models[A].get()) { cam.setup(*models[A]); camModel = models[A].get(); }
		if (A < 0) camModel = nullptr;
		bool camReady = camModel != nullptr;

		// ---------------- keyboard (only when ImGui does not want it)
		if (!io.WantCaptureKeyboard)
		{
			if (in.key().isPressed(sibr::Key::Escape)) window.close();
			if (in.key().isPressed(sibr::Key::Space)) play = !play;
			if (in.key().isPressed(sibr::Key::Right)) { play = false; frame = std::min(F - 1, frame + 1); }
			if (in.key().isPressed(sibr::Key::Left)) { play = false; frame = std::max(0, frame - 1); }
			if (in.key().isPressed(sibr::Key::Home)) frame = 0;
			if (in.key().isPressed(sibr::Key::End)) frame = F - 1;
			if (in.key().isPressed(sibr::Key::Tab) && nSel() >= 2) { int t = slot[0]; for (int k = 0; k < 3; k++) slot[k] = slot[k + 1]; slot[3] = t; compact(); }
			for (int i = 0; i < M_COUNT; i++) if (in.key().isPressed(sibr::Key::Code(int(sibr::Key::Num1) + i))) s.mode = i;
			bool ctrl = in.key().isActivated(sibr::Key::LeftControl);
			if (ctrl && (in.key().isPressed(sibr::Key::Equal) || in.key().isPressed(sibr::Key::KPAdd))) fontWanted = std::min(48.f, fontWanted + 2.f);
			if (ctrl && (in.key().isPressed(sibr::Key::Minus) || in.key().isPressed(sibr::Key::KPSubtract))) fontWanted = std::max(12.f, fontWanted - 2.f);
			if (camReady)
			{
				if (in.key().isPressed(sibr::Key::LeftBracket)) cam.snap(*models[A], cam.camIdx - 1);
				if (in.key().isPressed(sibr::Key::RightBracket)) cam.snap(*models[A], cam.camIdx + 1);
				if (!ctrl && in.key().isPressed(sibr::Key::R)) cam.snap(*models[A], cam.camIdx);
				if (!ctrl && in.key().isPressed(sibr::Key::F)) cam.frameScene(*models[A]);
			}
		}
		if (play && F > 1)
		{
			accum += dt * fps; int step = int(accum); accum -= step;
			frame += step;
			if (frame >= F) frame = loop ? frame % F : F - 1;
		}
		frame = std::max(0, std::min(F - 1, frame));

		// ---------------- layout (panel sizes are dragged with the splitters around the view, saved in sibr4d_settings.txt)
		sibr::Vector2i ws(int(io.DisplaySize.x), int(io.DisplaySize.y));
		float mh = ImGui::GetFrameHeight(), fs = ImGui::GetFontSize();
		const ImGuiStyle& st = ImGui::GetStyle();
		float minTh = ImGui::GetFrameHeight() + 2.f * ImGui::GetFrameHeightWithSpacing() + 2.f * st.WindowPadding.y + st.ItemSpacing.y;
		if (prefs.leftW <= 0.f) prefs.leftW = 15.f * fs;
		if (prefs.rightW <= 0.f) prefs.rightW = 17.f * fs;
		if (prefs.bottomH <= 0.f) prefs.bottomH = minTh;
		prefs.leftW = std::max(6.f * fs, std::min(prefs.leftW, 0.45f * ws.x()));
		prefs.rightW = std::max(6.f * fs, std::min(prefs.rightW, 0.45f * ws.x()));
		prefs.bottomH = std::max(minTh, std::min(prefs.bottomH, 0.5f * ws.y()));
		float lw = showPlaylist ? prefs.leftW : 0.f, rw = showInfo ? prefs.rightW : 0.f, th = showTimeline ? prefs.bottomH : 0.f;
		ImVec2 cPos(lw, mh), cSize(std::max(64.f, ws.x() - lw - rw), std::max(64.f, ws.y() - mh - th));
		const ImGuiWindowFlags fixed = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings;

		if (showPlaylist)
		{
			ImGui::SetNextWindowPos(ImVec2(0, mh)); ImGui::SetNextWindowSize(ImVec2(lw, ws.y() - mh));
			ImGui::Begin("Playlist", nullptr, fixed);
			if (ImGui::Button("+ Open model folder")) { std::string p; if (sibr::showFilePicker(p, sibr::Directory)) toOpen = p; }
			ImGui::TextDisabled("tick = compare (max 4), click = reference 1");
			ImGui::Separator();
			for (int i = 0; i < (int)models.size(); i++)
			{
				Model& m = *models[i];
				int k = slotOf(i);
				bool on = k >= 0;
				if (ImGui::Checkbox(("##sel" + std::to_string(i)).c_str(), &on))
				{
					if (on) { if (!addSlot(i)) status = "max 4 models in compare - untick one first"; else { int kk = slotOf(i); if (kk + 1 > s.views) s.views = kk + 1; if (nSel() > 1 && s.cmp == C_SINGLE) s.cmp = C_SIDE; } }
					else slot[k] = -1;
				}
				ImGui::SameLine();
				std::string tag = k >= 0 ? "[" + std::to_string(k + 1) + "] " : "";
				if (ImGui::Selectable((tag + m.name + "##" + std::to_string(i)).c_str(), k == 0)) makeRef(i);
				if (ImGui::BeginPopupContextItem(("ctx" + std::to_string(i)).c_str()))
				{
					if (ImGui::MenuItem("Make reference (1)")) makeRef(i);
					if (ImGui::MenuItem("Close")) closeIdx = i;
					ImGui::EndPopup();
				}
				if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", m.folder.c_str());
				if (m.inRam) ImGui::TextDisabled("     N %d | %d frames | GPU %d MB + RAM %d MB", m.N, m.F, int(m.vram >> 20), int((m.frameBytes * (size_t)m.F) >> 20));
				else ImGui::TextDisabled("     N %d | %d frames | GPU %d MB", m.N, m.F, int(m.vram >> 20));
			}
			if (camReady && !models[A]->cams.empty())
			{
				Model& m = *models[A];
				ImGui::Spacing(); ImGui::Separator();
				ImGui::Text("Training cameras (1: %s)", m.name.c_str());
				if (ImGui::Button(" < ")) cam.snap(m, cam.camIdx - 1);
				ImGui::SameLine(); if (ImGui::Button(" > ")) cam.snap(m, cam.camIdx + 1);
				ImGui::SameLine(); if (ImGui::Button("Back")) cam.snap(m, cam.camIdx);
				ImGui::SameLine(); if (ImGui::Button("Fit")) cam.frameScene(m);
				ImGui::BeginChild("##camlist", ImVec2(0, 0), true);
				for (int i = 0; i < (int)m.cams.size(); i++)
					if (ImGui::Selectable(("cam " + std::to_string(i) + "   " + m.cams[i]->name() + "##c" + std::to_string(i)).c_str(), cam.camIdx == i && cam.exact))
						cam.snap(m, i);
				ImGui::EndChild();
			}
			ImGui::End();
		}

		// ---------------- centre view + splitters
		ImGui::SetNextWindowPos(cPos); ImGui::SetNextWindowSize(cSize);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
		ImGui::Begin("##view", nullptr, fixed | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBringToFrontOnFocus);
		const float sp = std::max(10.f, 0.6f * fs);   // splitter thickness
		ImVec2 wp = ImGui::GetCursorScreenPos();
		auto splitter = [&](const char* id, ImVec2 pos, ImVec2 size, bool horiz, float& val, float sign) {
			ImGui::SetCursorScreenPos(pos);
			ImGui::InvisibleButton(id, size);
			bool hov = ImGui::IsItemHovered(), act = ImGui::IsItemActive();
			if (hov || act) ImGui::SetMouseCursor(horiz ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeNS);
			if (act) { val += sign * (horiz ? io.MouseDelta.x : io.MouseDelta.y); sizesDirty = true; }
			ImGui::GetWindowDrawList()->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y),
				act ? IM_COL32(120, 140, 220, 255) : hov ? IM_COL32(100, 110, 170, 255) : IM_COL32(70, 70, 88, 255));
			ImVec2 c(pos.x + size.x * 0.5f, pos.y + size.y * 0.5f);   // grip dots
			for (int d = -2; d <= 2; d++)
				ImGui::GetWindowDrawList()->AddCircleFilled(horiz ? ImVec2(c.x, c.y + d * 6.f) : ImVec2(c.x + d * 6.f, c.y), 1.6f, IM_COL32(200, 200, 210, 255));
		};
		float vx = wp.x + (showPlaylist ? sp : 0.f), vy = wp.y;
		int W = std::max(16, int(cSize.x - (showPlaylist ? sp : 0.f) - (showInfo ? sp : 0.f)));
		int H = std::max(16, int(cSize.y - (showTimeline ? sp : 0.f)));
		if (showPlaylist) splitter("##splitL", wp, ImVec2(sp, cSize.y), true, prefs.leftW, +1.f);
		if (showInfo) splitter("##splitR", ImVec2(wp.x + cSize.x - sp, wp.y), ImVec2(sp, cSize.y), true, prefs.rightW, -1.f);
		if (showTimeline) splitter("##splitB", ImVec2(vx, wp.y + H), ImVec2((float)W, sp), false, prefs.bottomH, -1.f);
		if (sizesDirty && !io.MouseDown[0]) { savePrefs(prefs); sizesDirty = false; }

		ImVec2 p0(vx, vy);
		ImGui::SetCursorScreenPos(p0);
		ImGui::InvisibleButton("##viewbtn", ImVec2((float)W, (float)H));
		bool hovered = ImGui::IsItemHovered();
		if (hovered) { ImGui::CaptureKeyboardFromApp(false); ImGui::CaptureMouseFromApp(false); }

		// wipe line: drag it directly in the view (takes precedence over camera orbit)
		static bool wipeDrag = false;
		if (s.mode != M_ELLIPSOIDS && s.cmp == C_WIPE && hovered && io.MouseClicked[0] && std::abs(io.MousePos.x - (p0.x + s.wipe * W)) < 0.8f * fs) wipeDrag = true;
		if (!io.MouseDown[0]) wipeDrag = false;
		if (wipeDrag) s.wipe = std::max(0.f, std::min(1.f, (io.MousePos.x - p0.x) / float(W)));

		// camera interaction
		if (camReady)
		{
			if (cam.animT < 1.f) cam.animT = std::min(1.f, cam.animT + dt / 0.35f);
			bool shift = in.key().isActivated(sibr::Key::LeftShift), ctrl = in.key().isActivated(sibr::Key::LeftControl);
			float dx = io.MouseDelta.x, dy = io.MouseDelta.y;
			bool moved = false;
			static bool dragging = false;
			if (hovered && !wipeDrag && (io.MouseClicked[0] || io.MouseClicked[1] || io.MouseClicked[2])) dragging = true;
			if (!io.MouseDown[0] && !io.MouseDown[1] && !io.MouseDown[2]) dragging = false;
			if (dragging && (dx != 0.f || dy != 0.f))
			{
				if (io.MouseDown[0] && !shift)
				{
					cam.yaw -= dx * 0.005f * prefs.orbitSpeed;
					cam.pitch = std::max(-1.55f, std::min(1.55f, cam.pitch + dy * 0.005f * prefs.orbitSpeed));
				}
				else
				{
					sibr::InputCamera c = cam.current();
					float k = cam.dist * 2.f * std::tan(cam.fovy * 0.5f) / std::max(1, H) * prefs.panSpeed;
					cam.pivot += (-dx * c.right() + dy * c.up()) * k;
				}
				moved = true;
			}
			double wheel = hovered ? in.mouseScroll() : 0.0;
			if (wheel != 0.0)
			{
				if (ctrl) cam.fovy = std::max(0.02f, std::min(2.6f, cam.fovy * std::pow(0.93f, float(wheel))));
				else cam.dist = std::max(1e-3f, cam.dist * std::pow(0.88f, float(wheel)));
				moved = true;
			}
			if (hovered && !io.WantCaptureKeyboard && !ctrl)
			{
				sibr::InputCamera c = cam.current();
				sibr::Vector3f mv(0, 0, 0);
				if (in.key().isActivated(sibr::Key::W)) mv += c.dir();
				if (in.key().isActivated(sibr::Key::S)) mv -= c.dir();
				if (in.key().isActivated(sibr::Key::D)) mv += c.right();
				if (in.key().isActivated(sibr::Key::A)) mv -= c.right();
				if (in.key().isActivated(sibr::Key::E)) mv += cam.up;
				if (in.key().isActivated(sibr::Key::Q)) mv -= cam.up;
				if (mv.norm() > 0.f) { cam.pivot += mv * (cam.dist * dt * prefs.flySpeed * (shift ? 3.f : 1.f)); moved = true; }
			}
			if (moved) { cam.exact = false; cam.animT = 1.f; }
		}

		// rendering ------------------------------------------------------------
		// view k shows model slot[k] (-1 = empty); the first filled view is the reference ("1") for PSNR / camera
		int n = nSel();
		int ref = -1; for (int k = 0; k < 4; k++) if (slot[k] >= 0) { ref = k; break; }
		int two[2] = { -1, -1 }; { int t = 0; for (int k = 0; k < 4 && t < 2; k++) if (slot[k] >= 0) two[t++] = k; }
		int cmp = s.cmp;

		for (int k = 0; k < 4; k++) psnr[k] = -1;
		pixX = pixY = -1;
		bool pixOk[4] = { false, false, false, false };
		auto dl = ImGui::GetWindowDrawList();
		if (ref >= 0 && camReady)
		{
			sibr::InputCamera eye = cam.current();
			int V = cmp == C_SIDE ? std::max(1, std::min(4, s.views)) : 1;
			int cols = 1, rows = 1;
			if (V > 1)
			{
				int lay = s.layout;
				if (lay == 0) lay = (V == 4) ? 3 : 1;          // auto: 2-3 views in a row, 4 views 2x2
				if (lay == 1) { cols = V; rows = 1; }
				else if (lay == 2) { cols = 1; rows = V; }
				else { cols = 2; rows = (V + 1) / 2; }
			}
			const int gap = 2;
			int tw = std::max(8, (W - (cols - 1) * gap) / cols), tht = std::max(8, (H - (rows - 1) * gap) / rows);
			auto tileOrigin = [&](int k) { return ImVec2(p0.x + (k % cols) * (tw + gap), p0.y + (k / cols) * (tht + gap)); };
			ImU32 colU[4] = { IM_COL32(255, 255, 0, 255), IM_COL32(0, 255, 255, 255), IM_COL32(255, 128, 255, 255), IM_COL32(128, 255, 128, 255) };
			auto label = [&](int k, ImVec2 at, const char* extra = nullptr) {
				std::string t = std::to_string(k + 1) + ": " + (slot[k] >= 0 ? models[slot[k]]->name : std::string("(empty - pick a model in Compare / Info)"));
				if (k != ref && psnr[k] >= 0) { char b[64]; snprintf(b, sizeof(b), "   PSNR vs %d: %.2f dB", ref + 1, psnr[k]); t += b; }
				if (extra) t += extra;
				ImVec2 sz = ImGui::CalcTextSize(t.c_str());
				dl->AddRectFilled(ImVec2(at.x + 4, at.y + 4), ImVec2(at.x + 12 + sz.x, at.y + 8 + sz.y), IM_COL32(0, 0, 0, 150));
				dl->AddText(ImVec2(at.x + 8, at.y + 6), colU[k], t.c_str());
			};
			auto readPix = [&](int k, int w, int h, int x, int y) {
				for (int ch = 0; ch < 3; ch++) cudaMemcpy(&pix[k][ch], img[k].d + (size_t)ch * w * h + (size_t)y * w + x, 4, cudaMemcpyDeviceToHost);
				pixOk[k] = true;
			};

			if (s.mode == M_ELLIPSOIDS)
			{
				if (!ellR) ellR.reset(new sibr::GaussianSurfaceRenderer());
				int nt = cmp == C_SIDE ? V : 1;
				int ew = cmp == C_SIDE ? tw : W, eh = cmp == C_SIDE ? tht : H;
				for (int k = 0; k < nt; k++)
				{
					int vk = cmp == C_SIDE ? k : ref;
					ImVec2 a = cmp == C_SIDE ? tileOrigin(k) : p0;
					if (slot[vk] < 0) { label(vk, a); continue; }
					Model& m = *models[slot[vk]];
					m.showFrame(frame); m.syncGL(s.ellScale);
					if (!ellRT[k] || ellRT[k]->w() != (uint)ew || ellRT[k]->h() != (uint)eh) ellRT[k].reset(new sibr::RenderTargetRGBA(ew, eh));
					sibr::InputCamera e = eye; e.aspect(float(ew) / eh);
					ellRT[k]->clear();
					auto t0 = std::chrono::high_resolution_clock::now();
					glViewport(0, 0, ew, eh);   // the renderer draws into its own FBO at the target size
					if (s.ellWire) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
					ellR->process(m.N, *m.gl, e, *ellRT[k], s.ellAlpha);
					if (s.ellWire) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
					glBindFramebuffer(GL_FRAMEBUFFER, 0);
					m.lastMs = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
					dl->AddImage((void*)(intptr_t)ellRT[k]->handle(), a, ImVec2(a.x + ew, a.y + eh), ImVec2(0, 1), ImVec2(1, 0));
					label(vk, a);
				}
			}
			else if (cmp == C_SIDE || cmp == C_SINGLE)
			{
				for (int k = 0; k < V; k++)
				{
					int vk = cmp == C_SIDE ? k : ref;
					ImVec2 a = cmp == C_SIDE ? tileOrigin(k) : p0;
					if (slot[vk] < 0) continue;
					img[vk].ensure(tw, tht);
					renderSplat(*models[slot[vk]], s, frame, eye, tw, tht, img[vk].d, viewC, projC, camC, bgC);
					tex[k].ensure(tw, tht);
					g4dCompose(C_SINGLE, img[vk].d, tw, tht, nullptr, tw, tht, tw, tht, 0, 1.f, tex[k].lin);
					tex[k].upload();
					dl->AddImage((void*)(intptr_t)tex[k].id, a, ImVec2(a.x + tw, a.y + tht));
				}
				if (cmp == C_SIDE)
					for (int k = 0; k < V; k++)
						if (k != ref && slot[k] >= 0 && slot[ref] >= 0)
							psnr[k] = 10.0 * std::log10(1.0 / std::max(1e-12, g4dMSE(img[ref].d, img[k].d, (size_t)tw * tht * 3, accC)));
				for (int k = 0; k < V; k++)
				{
					int vk = cmp == C_SIDE ? k : ref;
					ImVec2 a = cmp == C_SIDE ? tileOrigin(k) : p0;
					if (slot[vk] < 0) dl->AddRectFilled(a, ImVec2(a.x + tw, a.y + tht), IM_COL32(30, 30, 34, 255));
					label(vk, a);
				}
				if (hovered)
				{
					int mx = int(io.MousePos.x - p0.x), my = int(io.MousePos.y - p0.y);
					int c = std::min(cols - 1, mx / (tw + gap)), r = std::min(rows - 1, my / (tht + gap));
					int lx = mx - c * (tw + gap), ly = my - r * (tht + gap);
					if (lx >= 0 && lx < tw && ly >= 0 && ly < tht)
					{
						pixX = lx; pixY = ly;
						for (int k = 0; k < V; k++) { int vk = cmp == C_SIDE ? k : ref; if (slot[vk] >= 0) readPix(vk, tw, tht, lx, ly); }
					}
				}
			}
			else   // wipe / difference / overlay: view 1 vs view 2, or view 1 vs itself at another frame when only one model is open
			{
				bool self = slot[1] < 0;
				int f2 = self ? std::max(0, std::min(models[slot[0]]->F - 1, frame + s.selfOffset)) : frame;
				img[0].ensure(W, H); renderSplat(*models[slot[0]], s, frame, eye, W, H, img[0].d, viewC, projC, camC, bgC);
				img[1].ensure(W, H); renderSplat(*models[slot[self ? 0 : 1]], s, f2, eye, W, H, img[1].d, viewC, projC, camC, bgC);
				psnr[1] = 10.0 * std::log10(1.0 / std::max(1e-12, g4dMSE(img[0].d, img[1].d, (size_t)W * H * 3, accC)));
				int kc = cmp; float gain = s.diffGain;
				bool showB = false;
				if (cmp == C_OVERLAY)
				{
					if (s.flicker)
					{
						double tsec = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
						showB = std::fmod(tsec * s.flickerHz, 1.0) >= 0.5;
						kc = showB ? 4 : 0;
					}
					else { kc = 5; gain = s.overlay; }
				}
				tex[0].ensure(W, H);
				g4dCompose(kc, img[0].d, W, H, img[1].d, W, H, W, H, int(s.wipe * W), gain, tex[0].lin);
				tex[0].upload();
				dl->AddImage((void*)(intptr_t)tex[0].id, p0, ImVec2(p0.x + W, p0.y + H));
				char ex[96] = "";
				if (self) snprintf(ex, sizeof(ex), "   @ frame %d (self, offset %+d)", f2, s.selfOffset);
				if (cmp == C_OVERLAY && !s.flicker) { size_t l = strlen(ex); snprintf(ex + l, sizeof(ex) - l, "   (blend %.0f%%)", s.overlay * 100.f); }
				auto label2 = [&](ImVec2 at) {
					std::string t = std::string("2: ") + models[slot[self ? 0 : 1]]->name + ex;
					char b[48]; snprintf(b, sizeof(b), "   PSNR vs 1: %.2f dB", psnr[1]); t += b;
					ImVec2 sz = ImGui::CalcTextSize(t.c_str());
					dl->AddRectFilled(ImVec2(at.x + 4, at.y + 4), ImVec2(at.x + 12 + sz.x, at.y + 8 + sz.y), IM_COL32(0, 0, 0, 150));
					dl->AddText(ImVec2(at.x + 8, at.y + 6), colU[1], t.c_str());
				};
				if (cmp == C_OVERLAY && s.flicker) { if (showB) label2(p0); else label(0, p0, "   (flicker)"); }
				else
				{
					label(0, p0);
					label2(cmp == C_WIPE ? ImVec2(p0.x + s.wipe * W, p0.y) : ImVec2(p0.x, p0.y + 1.6f * fs));
				}
				if (cmp == C_WIPE)   // drag the split line directly in the view
				{
					float sx = p0.x + s.wipe * W;
					if (hovered && std::abs(io.MousePos.x - sx) < 0.6f * fs) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
				}
				if (hovered)
				{
					int mx = int(io.MousePos.x - p0.x), my = int(io.MousePos.y - p0.y);
					if (mx >= 0 && mx < W && my >= 0 && my < H) { pixX = mx; pixY = my; readPix(0, W, H, mx, my); readPix(1, W, H, mx, my); }
				}
			}
			if (s.mode >= M_MOTION && s.mode <= M_SCALE)
				dl->AddText(ImVec2(p0.x + 8, p0.y + H - 1.6f * fs), IM_COL32(255, 255, 255, 255), kModeNames[s.mode]);
		}
		else
		{
			dl->AddText(ImVec2(p0.x + 20, p0.y + 20), IM_COL32(220, 220, 220, 255), "Drag & drop model folders here   (or File > Open model folder...)");
		}
		ImGui::End();
		ImGui::PopStyleVar(2);

		// ---------------- timeline
		if (showTimeline)
		{
			ImGui::SetNextWindowPos(ImVec2(lw, ws.y() - th)); ImGui::SetNextWindowSize(ImVec2(ws.x() - lw - rw, th));
			ImGui::Begin("Timeline", nullptr, fixed);
			if (ImGui::Button(play ? " Pause " : " Play  ")) { play = !play; accum = 0; }
			ImGui::SameLine(); if (ImGui::Button("|<")) frame = 0;
			ImGui::SameLine(); if (ImGui::Button("<")) { play = false; frame = std::max(0, frame - 1); }
			ImGui::SameLine(); if (ImGui::Button(">")) { play = false; frame = std::min(F - 1, frame + 1); }
			ImGui::SameLine(); if (ImGui::Button(">|")) frame = F - 1;
			ImGui::SameLine(); ImGui::Checkbox("Loop", &loop);
			ImGui::SameLine(); ImGui::PushItemWidth(8.f * fs); ImGui::SliderFloat("fps", &fps, 1.f, 60.f, "%.0f"); ImGui::PopItemWidth();
			float tA = (A >= 0 && models[A]->F > 1) ? models[A]->tLast * std::min(frame, models[A]->F - 1) / (models[A]->F - 1) : 0.f;
			ImGui::SameLine(); ImGui::Text("   frame %d / %d   t = %.4f", frame, F - 1, tA);
			ImGui::PushItemWidth(-1);
			ImGui::SliderInt("##frame", &frame, 0, std::max(0, F - 1));
			ImGui::PopItemWidth();
			ImGui::End();
		}

		// ---------------- info
		if (showInfo)
		{
			ImGui::SetNextWindowPos(ImVec2(ws.x() - rw, mh)); ImGui::SetNextWindowSize(ImVec2(rw, ws.y() - mh));
			ImGui::Begin("Info", nullptr, fixed);
			ImVec4 col[4] = { ImVec4(1, 1, 0, 1), ImVec4(0, 1, 1, 1), ImVec4(1, .5f, 1, 1), ImVec4(.5f, 1, .5f, 1) };
			if (ImGui::CollapsingHeader("Compare setup", ImGuiTreeNodeFlags_DefaultOpen))
			{
				ImGui::PushItemWidth(-1);
				if (ImGui::BeginCombo("##cmpmode", kCmpNames[s.cmp])) { for (int i = 0; i < C_COUNT; i++) if (ImGui::Selectable(kCmpNames[i], s.cmp == i)) s.cmp = i; ImGui::EndCombo(); }
				ImGui::PopItemWidth();
				if (s.cmp == C_SIDE)
				{
					ImGui::Text("Views:"); for (int v = 1; v <= 4; v++) { ImGui::SameLine(); if (ImGui::RadioButton(std::to_string(v).c_str(), s.views == v)) s.views = v; }
					ImGui::Text("Layout:"); ImGui::SameLine(); ImGui::PushItemWidth(-1);
					if (ImGui::BeginCombo("##layout", kLayoutNames[s.layout])) { for (int i = 0; i < 4; i++) if (ImGui::Selectable(kLayoutNames[i], s.layout == i)) s.layout = i; ImGui::EndCombo(); }
					ImGui::PopItemWidth();
				}
				int nv = s.cmp == C_SIDE ? s.views : (s.cmp == C_SINGLE ? 1 : 2);
				for (int k = 0; k < nv; k++)
				{
					ImGui::TextColored(col[k], "View %d", k + 1); ImGui::SameLine(); ImGui::PushItemWidth(-1);
					std::string cur = slot[k] >= 0 ? models[slot[k]]->name : std::string("(empty)");
					if (ImGui::BeginCombo(("##view" + std::to_string(k)).c_str(), cur.c_str()))
					{
						if (ImGui::Selectable("(empty)", slot[k] < 0)) slot[k] = -1;
						for (int i = 0; i < (int)models.size(); i++) if (ImGui::Selectable((models[i]->name + "##m" + std::to_string(i)).c_str(), slot[k] == i)) slot[k] = i;
						ImGui::EndCombo();
					}
					ImGui::PopItemWidth();
				}
				if (s.cmp == C_WIPE) ImGui::SliderFloat("Wipe", &s.wipe, 0.f, 1.f);
				if (s.cmp == C_DIFF) ImGui::SliderFloat("Diff gain", &s.diffGain, 1.f, 32.f);
				if (s.cmp == C_OVERLAY)
				{
					ImGui::SliderFloat("Blend 1 -> 2", &s.overlay, 0.f, 1.f, "%.2f");
					ImGui::Checkbox("Flicker 1 / 2", &s.flicker);
					if (s.flicker) ImGui::SliderFloat("Flicker Hz", &s.flickerHz, 0.5f, 10.f, "%.1f");
				}
				if (s.cmp != C_SIDE && s.cmp != C_SINGLE)
				{
					if (slot[1] < 0) { ImGui::SliderInt("Frame offset", &s.selfOffset, -30, 30); ImGui::TextDisabled("one model open: view 2 = same model at frame + offset"); }
					else ImGui::TextDisabled("view 1 vs view 2 (pick models above)");
				}
			}
			if (ImGui::CollapsingHeader("Render settings", ImGuiTreeNodeFlags_DefaultOpen))
			{
				ImGui::PushItemWidth(-1);
				if (ImGui::BeginCombo("##rmode", kModeNames[s.mode])) { for (int i = 0; i < M_COUNT; i++) if (ImGui::Selectable(kModeNames[i], s.mode == i)) s.mode = i; ImGui::EndCombo(); }
				ImGui::PopItemWidth();
				if (s.mode == M_ELLIPSOIDS)
				{
					ImGui::SliderFloat("Opacity cut", &s.ellAlpha, 0.f, 1.f, "%.2f");
					ImGui::SliderFloat("Size scale", &s.ellScale, 0.1f, 3.f, "%.2f");
					ImGui::Checkbox("Wireframe", &s.ellWire);
					ImGui::TextDisabled("solid: opacity >= cut, the rest blended");
				}
				else
				{
					ImGui::SliderFloat("Scaling modifier", &s.scaleMod, 0.01f, 1.f);
					ImGui::Checkbox("Antialiasing", &s.aa);
				}
				if (s.mode >= M_MOTION && s.mode <= M_SCALE)
				{
					ImGui::Checkbox("Auto heat range", &s.autoHeat);
					if (!s.autoHeat) { ImGui::InputFloat("min", &s.heatMin, 0, 0, 5); ImGui::InputFloat("max", &s.heatMax, 0, 0, 5); }
					else if (A >= 0) { heatRange(s.mode, *models[A], s.heatMin, s.heatMax); ImGui::Text("range %.5f .. %.5f", s.heatMin, s.heatMax); }
					ImGui::TextDisabled("turbo colormap: blue = low, red = high");
				}
			}
			if (ImGui::CollapsingHeader("Statistics", ImGuiTreeNodeFlags_DefaultOpen))
			{
				ImGui::Text("Viewport %d x %d   %.0f fps", W, H, io.Framerate);
				for (int k = 0; k < 4; k++)
				{
					if (slot[k] < 0) continue;
					Model& m = *models[slot[k]];
					ImGui::TextColored(col[k], "%d: %s", k + 1, m.name.c_str());
					ImGui::Text("   N %d  frames %d  render %.1f ms", m.N, m.F, m.lastMs);
					if (psnr[k] >= 0) ImGui::Text("   PSNR vs %d (this view): %.2f dB", ref + 1, psnr[k]);
					ImGui::TextDisabled("   %s%s", m.source.c_str(), m.inRam ? "  [frames in RAM]" : "");
				}
				if (pixX >= 0)
				{
					ImGui::Text("Pixel (%d, %d)", pixX, pixY);
					for (int k = 0; k < 4; k++)
						if (pixOk[k]) ImGui::TextColored(col[k], "   %d   R %3d  G %3d  B %3d", k + 1, int(pix[k][0] * 255 + .5f), int(pix[k][1] * 255 + .5f), int(pix[k][2] * 255 + .5f));
				}
			}
			ImGui::Separator();
			ImGui::TextWrapped("Keys: Space play | Left/Right step | Home/End | 1-8 view modes | Tab next reference");
			ImGui::TextWrapped("Mouse: L-drag orbit | R/M-drag pan | wheel dolly | Ctrl+wheel FOV zoom | WASD QE fly (Shift fast)");
			ImGui::TextWrapped("[ ] prev/next train cam | R back | F fit | Ctrl +/- text size | drag panel edges to resize");
			ImGui::Text("%.1f fps", io.Framerate);
			ImGui::End();
		}

		window.viewport().bind();
		glClearColor(37.f / 255.f, 37.f / 255.f, 38.f / 255.f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT);
		window.swapBuffer();
	}
	savePrefs(prefs);
	models.clear();
	return 0;
}
