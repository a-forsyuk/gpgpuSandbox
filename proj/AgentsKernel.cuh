#ifndef _AGENT_KERNEL_CUH_
#define _AGENT_KERNEL_CUH_

//#include <cuda_texture_types.h>

#ifdef CUDASystems_EXPORTS
    #define CUDASystems_API __declspec(dllexport)
#else
    #define CUDASystems_API __declspec(dllimport)
#endif

namespace CUDASystems
{
    CUDASystems_API void Init(unsigned agentsCount);
    CUDASystems_API void Release();

    CUDASystems_API void Update(float dt);

    CUDASystems_API void GetMapDimensions(float* width, float* height);
    CUDASystems_API void GetMapNodesDimensions(unsigned* width, unsigned* height);

    CUDASystems_API void MapPositions(float** data);
    CUDASystems_API void MapColors(float** data);
}

#endif