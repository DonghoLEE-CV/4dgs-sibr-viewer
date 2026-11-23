// Modified in 2026 by the SIBR-4D contributors (marked "SIBR-4D" below).
/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */


#pragma once

# include <core/graphics/Shader.hpp>
# include <core/graphics/Texture.hpp>
# include <core/graphics/Mesh.hpp>
# include <core/graphics/Camera.hpp>

# include <core/renderer/Config.hpp>

namespace sibr { 

	class GaussianData
	{
	public:
		typedef std::shared_ptr<GaussianData>	Ptr;

	public:

		/// Constructor.
		GaussianData(int num_gaussians, float* mean_data, float* rot_data, float* scale_data, float* alpha_data, float* color_data);

		void render(int G) const;

		/// SIBR-4D: GL buffer ids (0 mean, 1 rot, 2 scale, 3 alpha, 4 sh colors) for CUDA-GL interop updates
		GLuint buffer(int i) const { return i == 0 ? meanBuffer : i == 1 ? rotBuffer : i == 2 ? scaleBuffer : i == 3 ? alphaBuffer : colorBuffer; }

	private:

		int _num_gaussians;
		GLuint meanBuffer;
		GLuint rotBuffer;
		GLuint scaleBuffer;
		GLuint alphaBuffer;
		GLuint colorBuffer;
	};

	/** Render a mesh colored using the per-vertex color attribute.
	\ingroup sibr_renderer
	*/
	class GaussianSurfaceRenderer
	{
	public:
		typedef std::shared_ptr<GaussianSurfaceRenderer>	Ptr;

	public:

		/// Constructor.
		GaussianSurfaceRenderer( void );

		/** Render the mesh using its vertices colors, interpolated over triangles.
		\param mesh the mesh to render
		\param eye the viewpoint to use
		\param dst the destination rendertarget
		\param mode the rendering mode of the mesh
		\param backFaceCulling should backface culling be performed
		*/
		int	process(
			int G,
			/*input*/	const GaussianData& mesh,
			/*input*/	const Camera& eye,
			/*output*/	IRenderTarget& dst,
			float alphaLimit,
			/*mode*/    sibr::Mesh::RenderMode mode = sibr::Mesh::FillRenderMode,
			/*BFC*/     bool backFaceCulling = true);

		void makeFBO(int w, int h);

	private:

		GLuint idTexture;
		GLuint colorTexture;
		GLuint depthBuffer;
		GLuint fbo;
		int resX, resY;

		GLShader			_shader; ///< Color shader.
		GLParameter			_paramMVP; ///< MVP uniform.
		GLParameter			_paramCamPos;
		GLParameter			_paramLimit;
		GLParameter			_paramStage;
		GLuint clearProg;
		GLuint clearShader;
	};

} /*namespace sibr*/ 
