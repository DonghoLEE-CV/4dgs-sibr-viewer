/*
 * SIBR-4D - derivative work of the SIBR "gaussianviewer" project
 * (Copyright (C) 2023, Inria, GRAPHDECO research group, https://team.inria.fr/graphdeco).
 * Modified / added in 2026 by the SIBR-4D contributors: 4D sequence playback, comparison viewer.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE_GAUSSIAN_SPLATTING.md file.
 */
// SIBR-4D viewer entry point (11-23). usage: SIBR_gaussian4D_app <model folder> [<model folder 2> ...]
#include <projects/gaussianviewer/renderer/Config.hpp>

SIBR_EXP_ULR_EXPORT int sibr4dMain(int ac, char** av);

int main(int ac, char** av)
{
	return sibr4dMain(ac, av);
}
