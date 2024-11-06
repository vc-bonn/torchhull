#pragma once

#include <cmath>

#include <c10/macros/Macros.h>

namespace torchhull
{

inline C10_HOST_DEVICE float
lerp(const float v0, const float v1, const float t)
{
    return fmaf(t, v1, fmaf(-t, v0, v0));
}

inline C10_HOST_DEVICE float
bilerp(const float v00, const float v10, const float v01, const float v11, const float ty, const float tx)
{
    float v0 = lerp(v00, v10, ty);
    float v1 = lerp(v01, v11, ty);

    return lerp(v0, v1, tx);
}

} // namespace torchhull
