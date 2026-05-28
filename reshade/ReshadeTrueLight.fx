//===========================================================================
// SKYRIM REALISTIC PIPELINE — ReshadeTrueLight
// Author: Obewan (https://github.com/obewan)
// Co-authors: Microsoft Copilot, OpenAI ChatGPT, Google Gemini.
// Version: 0.0.1
// Requirement: 
//     	- bluenoise.png texture in the reshade-shaders/textures folder
// 		- a depth buffer correctly setted (check it using the DisplayDepth shader)
//================================================================================

#include "ReshadeTrueLight.fxh"


// ============================
// HELPERS
// ============================
float Hash11(float n) { return frac(sin(n) * 43758.5453); }
float Hash21(float2 p)
{
    p = frac(p * float2(127.1, 311.7));
    p += dot(p, p + 34.345);
    return frac(p.x * p.y);
}
float2 BlueNoiseJitterUV(float2 uv)
{
    if (UseBlueNoise)
    {
        float2 bn = tex2D(BlueNoiseSampler, frac(uv * float2(256.0,256.0) + float2(FrameIndex, FrameIndex))).rg;
        return (bn - 0.5) * 1.0;
    }
    else
    {
        float n = Hash21(uv + float2(FrameIndex, FrameIndex));
        return float2(n - 0.5, Hash11(n) - 0.5) * 0.5 * (1.0 / max(1.0, float(BUFFER_WIDTH)/1920.0));
    }
}

float Luma(float3 c) { return dot(c, float3(0.2126,0.7152,0.0722)); }


// Reconstruct view-space position from depth texture sample
float3 ReconstructViewPos(float2 uv)
{
    float depth = ReShade::GetLinearizedDepth(uv);

    // FIX: no projection matrix usage in ReShade
    float2 ndc = uv * 2.0 - 1.0;

    float3 view;
    view.xy = ndc * depth;   // approximate view-space reconstruction
    view.z = depth;

    return view;
}

// Project view-space position into UV using provided matrix
float2 ProjectToUV(float3 viewPos)
{
    // assumes symmetric projection approximation
    float2 uv = viewPos.xy / max(viewPos.z, 1e-5);
    return uv * 0.5 + 0.5;
}

// Normal estimation (bilateral) - flattened and derivative-safe sampling
float3 EstimateNormalBilateral(float2 uv)
{
     float2 t = ReShade::PixelSize;

    float centerD = ReShade::GetLinearizedDepth(uv);
    float3 centerP = ReconstructViewPos(uv);

    float3 sum = float3(0,0,0);
    float wsum = 0.0;

    static const int OFFCNT = 8;

    static const float2 offsets[OFFCNT] =
    {
        float2( t.x, 0), float2(-t.x, 0), float2(0,  t.y), float2(0, -t.y),
        float2( t.x, t.y), float2(-t.x, t.y), float2( t.x, -t.y), float2(-t.x, -t.y)
    };

    [loop]
    for (int i = 0; i < OFFCNT; ++i)
    {
        float2 o = offsets[i];

        float sd = ReShade::GetLinearizedDepth(uv + o);

        float3 sp = ReconstructViewPos(uv + o);

        float depthDiff = abs(centerD - sd);
        float w = exp(-depthDiff * 200.0);

        sum += sp * w;
        wsum += w;
    }

    float3 avgP = sum / max(wsum, 1e-6);

    float3 px = ReconstructViewPos(uv + float2(t.x, 0));
    float3 py = ReconstructViewPos(uv + float2(0, t.y));

    float3 dx = px - centerP;
    float3 dy = py - centerP;

    float3 n = normalize(cross(dx, dy));

    if (any(isnan(n)) || length(n) < 1e-3)
        n = float3(0,0,1);

    return n;
}
float3 EstimateNormal(float2 uv) { return EstimateNormalBilateral(uv); }

// Motion estimate (reprojection-based)
float2 EstimateMotionUV(float2 uv)
{
    float depth = ReShade::GetLinearizedDepth(uv);

    // far plane / sky
    if (depth <= 0.0 || depth >= 1.0)
        return float2(0.0, 0.0);

    float3 viewPos = ReconstructViewPos(uv);

    // previous frame projection
    float4 prevClip = mul(PrevViewProj, float4(viewPos, 1.0));

    if (abs(prevClip.w) < 1e-6)
        return float2(0.0, 0.0);

    prevClip /= prevClip.w;

    float2 prevUV = prevClip.xy * 0.5 + 0.5;

    // motion vector = previous - current
    return prevUV - uv;
}

// Motion-aware temporal blend helpers
float MotionAwareBlendScalar(float curr, float hist, float2 motionUV, float baseBlend, float clampRange)
{
    float motionLen = length(motionUV) * float(BUFFER_WIDTH);
    float motionFactor = saturate(1.0 - smoothstep(0.5, 8.0, motionLen));
    float blend = lerp(baseBlend * 0.25, baseBlend, motionFactor);
    float minAllowed = max(0.0, curr - clampRange);
    float maxAllowed = min(1.0, curr + clampRange);
    float clampedHist = clamp(hist, minAllowed, maxAllowed);
    return lerp(curr, clampedHist, blend);
}

float4 MotionAwareBlendVec4(float4 curr, float4 hist, float2 motionUV, float baseBlend, float clampRange)
{
    float motionLen = length(motionUV) * float(BUFFER_WIDTH);
    float motionFactor = saturate(1.0 - smoothstep(0.5, 8.0, motionLen));
    float blend = lerp(baseBlend * 0.25, baseBlend, motionFactor);
    float4 clamped;
    clamped.r = clamp(hist.r, curr.r - clampRange, curr.r + clampRange);
    clamped.g = clamp(hist.g, curr.g - clampRange, curr.g + clampRange);
    clamped.b = clamp(hist.b, curr.b - clampRange, curr.b + clampRange);
    clamped.a = clamp(hist.a, curr.a - clampRange, curr.a + clampRange);
    return lerp(curr, clamped, blend);
}

// Edge-aware temporal rejection
bool TemporalReject(float2 uv, float2 prevUV)
{
    // out of bounds
    if (prevUV.x < 0.0 || prevUV.x > 1.0 ||
        prevUV.y < 0.0 || prevUV.y > 1.0)
        return true;

    // normals
    float3 nCurr = EstimateNormal(uv);
    float3 nPrev = EstimateNormal(prevUV);

    float ndiff = length(nCurr - nPrev);
    if (ndiff > 0.35)
        return true;

    // luminance (safe)
    float lCurr = Luma(tex2D(BackBuffer, uv).rgb);
    float lPrev = Luma(tex2D(BackBuffer, prevUV).rgb);

    if (abs(lCurr - lPrev) > 0.35)
        return true;

    // depth (ReShade-safe replacement)
    float dCurr = ReShade::GetLinearizedDepth(uv);
    float dPrev = ReShade::GetLinearizedDepth(prevUV);

    float diff = abs(dCurr - dPrev);
    float threshold = max(0.1, dCurr * 0.05);

    if (diff > threshold)
        return true;

    return false;
}

// ============================
// AO: view-space radius conversion and compute
// ============================
float2 ViewRadiusToUV(float3 viewPos, float viewRadius)
{
    // Convert view-space radius to screen-space scale using depth
    float depth = max(viewPos.z, 1e-5);

    // approximate perspective scaling
    float2 uvScale = viewRadius / depth;

    return uvScale;
}

// AO compute (dynamic loop, precomputed directions)
float ComputeAO_Current(float2 uv)
{
    float centerDepth = ReShade::GetLinearizedDepth(uv);

    if (centerDepth <= 0.0 || centerDepth >= 1.0)
        return 1.0;

    float3 p = ReconstructViewPos(uv);
    float3 n = EstimateNormal(uv);

    float occlusion = 0.0;

    int samples = min(max(AOSamples, 4), AO_MAX_SAMPLES);

    float2 uvRadius = ViewRadiusToUV(p, AORadiusMeters);
    float2 jitter = BlueNoiseJitterUV(uv) * 0.5;

    [loop]
    for (int i = 0; i < AO_MAX_SAMPLES; ++i)
    {
        if (i >= samples)
            break;

        float2 dir = AO_Dirs[i];

        float2 sampleUV =
            uv +
            dir * uvRadius +
            jitter * ReShade::PixelSize;

        float sd = ReShade::GetLinearizedDepth(sampleUV);

        if (sd <= 0.0 || sd >= 1.0)
            continue;

        float3 sp = ReconstructViewPos(sampleUV);

        float3 v = sp - p;
        float dist = length(v);

        float nd = dot(n, normalize(v));

        float rangeAtten =
            saturate(1.0 - dist / (AORadiusMeters * 3.0));

        float nTerm =
            saturate((nd + 1.0) * 0.5);

        occlusion += (1.0 - nTerm) * rangeAtten;
    }

    occlusion /= max(samples, 1);

    float ao = saturate(1.0 - occlusion * AOStrength);

    return pow(ao, 1.2);
}

// Sample history AO (reads Prev)
float SampleHistoryAO(float2 uv, float3 viewPos)
{
    if (UseReprojection)
    {
        float2 motion = EstimateMotionUV(uv);
        float2 prevUV = uv + motion;

        if (TemporalReject(uv, prevUV))
            return tex2D(AOHistoryPrevSampler, uv).r;

        if (prevUV.x < 0.0 || prevUV.x > 1.0 ||
            prevUV.y < 0.0 || prevUV.y > 1.0)
            return tex2D(AOHistoryPrevSampler, uv).r;

        return tex2D(AOHistoryPrevSampler, prevUV).r;
    }
    else
    {
        return tex2D(AOHistoryPrevSampler, uv).r;
    }
}

float TemporalBlendAO_Motion(float currAO, float histAO, float2 motionUV)
{
    return MotionAwareBlendScalar(currAO, histAO, motionUV, TemporalBlend, TemporalClamp);
}

// Bilateral blur for AO history (flattened 5x5 kernel)
float BlurAO_Bilateral(float2 uv)
{
    float centerD = ReShade::GetLinearizedDepth(uv);

    float sum = 0.0;
    float wsum = 0.0;

    float2 px = ReShade::PixelSize;

    [loop]
    for (int y = -2; y <= 2; ++y)
    {
        [loop]
        for (int x = -2; x <= 2; ++x)
        {
            float2 offset = float2(x, y) * px;
            float2 sUV = uv + offset;

            float v = tex2D(AOHistoryCurrSampler, sUV).r;

            float sd = ReShade::GetLinearizedDepth(sUV);

            float depthDiff = abs(centerD - sd);

            float w = exp(-depthDiff * 50.0);

            sum += v * w;
            wsum += w;
        }
    }

    return sum / max(1e-6, wsum);
}

// ============================
// CONTACT SHADOWS (temporal + bilateral blur)
// ============================
float ComputeContact_Current(float2 uv)
{
    float depth = ReShade::GetLinearizedDepth(uv);

    if (depth <= 0.0 || depth >= 1.0)
        return 1.0;

    float3 p = ReconstructViewPos(uv);
    float3 n = EstimateNormal(uv);

    float3 L = normalize(float3(0.0, -0.7, -0.7));

    float ndotl = saturate(dot(n, -L));

    if (ndotl < 0.05)
        return 1.0;

    float stepLen = 0.008 / 8.0;

    float occlusion = 0.0;

    float3 samplePos = p;

    float2 jitter = BlueNoiseJitterUV(uv) * 0.25;

    [loop]
    for (int i = 1; i <= 8; ++i)
    {
        samplePos += -L * stepLen;

        float2 sampleUV =
            ProjectToUV(samplePos) + jitter;

        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 ||
            sampleUV.y < 0.0 || sampleUV.y > 1.0)
            continue;

        float sampleDepth = ReShade::GetLinearizedDepth(sampleUV);

        if (sampleDepth <= 0.0 || sampleDepth >= 1.0)
            continue;

        float3 sp = ReconstructViewPos(sampleUV);

        float distSample = length(sp - p);
        float distRay = length(samplePos - p);

        if (distSample + 0.002 < distRay * 0.98)
        {
            float w = saturate(1.0 - (distRay / (0.008 * 1.05)));
            occlusion += w;
        }
    }

    occlusion = saturate(occlusion / 8.0);

    float shadow = lerp(1.0, 1.0 - occlusion, 0.65 * ndotl);

    return saturate(shadow);
}

// Sample history contact (reads Prev)
float SampleHistoryContact(float2 uv, float3 viewPos)
{
    float2 motion = EstimateMotionUV(uv);
    float2 prevUV = uv + motion;

    if (prevUV.x < 0.0 || prevUV.x > 1.0 ||
        prevUV.y < 0.0 || prevUV.y > 1.0)
        return tex2D(ContactHistoryPrevSampler, uv).r;

    return tex2D(ContactHistoryPrevSampler, prevUV).r;
}

// Bilateral blur for contact (flattened 5x5 kernel)
float BlurContact_Bilateral(float2 uv)
{
    static const int K = 25;

    static const float2 OFFS[K] =
    {
        float2(-2,-2), float2(-1,-2), float2(0,-2), float2(1,-2), float2(2,-2),
        float2(-2,-1), float2(-1,-1), float2(0,-1), float2(1,-1), float2(2,-1),
        float2(-2, 0), float2(-1, 0), float2(0, 0), float2(1, 0), float2(2, 0),
        float2(-2, 1), float2(-1, 1), float2(0, 1), float2(1, 1), float2(2, 1),
        float2(-2, 2), float2(-1, 2), float2(0, 2), float2(1, 2), float2(2, 2)
    };

    float centerD = ReShade::GetLinearizedDepth(uv);

    float sum = 0.0;
    float wsum = 0.0;

    float2 px = ReShade::PixelSize;

    [loop]
    for (int i = 0; i < K; ++i)
    {
        float2 sUV = uv + OFFS[i] * px;

        float v = tex2D(ContactHistoryCurrSampler, sUV).r;

        float sd = ReShade::GetLinearizedDepth(sUV);

        float depthDiff = abs(centerD - sd);

        float w = exp(-depthDiff * 50.0);

        sum += v * w;
        wsum += w;
    }

    return sum / max(1e-6, wsum);
}

// ============================
// SSR (adaptive raymarch + temporal)
// ============================
float3 ReflectView(float3 viewDir, float3 normal)
{
    return normalize(viewDir - 2.0 * dot(viewDir, normal) * normal);
}

// SSR raymarch: coarse dynamic loop bounded by SSR_MAX_STEPS (no unroll)
bool SSR_Raymarch_Adaptive(float3 viewPos, float3 reflDir, out float2 hitUV, out float3 hitViewPos)
{
    float maxDist = SSRMaxDistance;

    int coarseSteps = SSRSteps / 4;
    coarseSteps = clamp(coarseSteps, 4, SSR_MAX_STEPS);

    float stepSize = maxDist / max(1.0, (float)coarseSteps);

    float3 pos = viewPos;

    float2 uv;
    float depth;
    float3 sp;

    [loop]
    for (int i = 0; i < SSR_MAX_STEPS; ++i)
    {
        if (i >= coarseSteps)
            break;

        pos += reflDir * stepSize;

        uv = ProjectToUV(pos);

        // out of screen
        if (uv.x < 0.0 || uv.x > 1.0 ||
            uv.y < 0.0 || uv.y > 1.0)
            continue;

        depth = ReShade::GetLinearizedDepth(uv);

        // sky / invalid depth
        if (depth >= 1.0)
            continue;

        sp = ReconstructViewPos(uv);

        float distScene = length(sp - viewPos);
        float distRay   = length(pos - viewPos);

        if (distScene + 0.01 < distRay)
        {
            hitUV = uv;
            hitViewPos = sp;
            return true;
        }
    }

    hitUV = 0.0;
    hitViewPos = 0.0;
    return false;
}

float4 ComputeSSR(float2 uv)
{
    float depth = ReShade::GetLinearizedDepth(uv);
    if (depth >= 1.0)
        return float4(0, 0, 0, 0);

    float3 viewPos = ReconstructViewPos(uv);
    float3 normal  = EstimateNormal(uv);

    float3 viewDir = normalize(-viewPos);

    float3 refl = ReflectView(viewDir, normal);

    // small stable jitter (don’t distort reflection direction too much)
    float2 jitter = BlueNoiseJitterUV(uv) * 0.002;
    refl = normalize(refl + float3(jitter.x, jitter.y, 0.0) * 0.001);

    float2 hitUV;
    float3 hitViewPos;

    bool hit = SSR_Raymarch_Adaptive(viewPos, refl, hitUV, hitViewPos);

    float3 reflectedColor;
    float weight;

    if (hit)
    {
        reflectedColor = tex2D(BackBuffer, hitUV).rgb;
        weight = 0.5;
    }
    else
    {
        reflectedColor = tex2D(BackBuffer, uv).rgb;
        weight = 0.0;
    }

    float4 curr = float4(reflectedColor, weight);

    float2 motion = EstimateMotionUV(uv);
    float4 hist = tex2D(SSRHistoryPrevSampler, uv);

    return MotionAwareBlendVec4(curr, hist, motion, TemporalBlend, TemporalClamp);
}
// ============================
// BLOOM: ping-pong + mip chain (safe)
// ============================
float4 PS_BloomDown0(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float3 c = tex2D(BackBuffer, uv).rgb;
    float l = Luma(c);
    float mask = saturate((l - BloomThreshold) * 2.0);
    return float4(c * mask, 1.0);
}

float4 PS_BloomPingPong(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float3 c = tex2D(BloomMip0ASampler, uv).rgb;
    return float4(c, 1.0);
}

float4 PS_BloomDown1(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float2 t = ReShade::PixelSize * 2.0;
    float3 sum = 0.0;
    sum += tex2D(BloomMip0BSampler, uv + float2( t.x, 0)).rgb;
    sum += tex2D(BloomMip0BSampler, uv + float2(-t.x, 0)).rgb;
    sum += tex2D(BloomMip0BSampler, uv + float2(0,  t.y)).rgb;
    sum += tex2D(BloomMip0BSampler, uv + float2(0, -t.y)).rgb;
    sum *= 0.25;
    return float4(sum, 1.0);
}

float4 PS_BloomBlurH(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float2 r = ReShade::PixelSize * BloomRadius * 2.0;
    float3 c0 = tex2D(BloomMip1Sampler, uv).rgb;
    float3 c1 = tex2D(BloomMip1Sampler, uv + float2(r.x, 0)).rgb;
    float3 c2 = tex2D(BloomMip1Sampler, uv - float2(r.x, 0)).rgb;
    float3 c3 = tex2D(BloomMip1Sampler, uv + float2(2.0*r.x, 0)).rgb;
    float3 c4 = tex2D(BloomMip1Sampler, uv - float2(2.0*r.x, 0)).rgb;
    float3 outc = (0.204164*c0 + 0.304005*(c1 + c2) + 0.093827*(c3 + c4));
    return float4(outc, 1.0);
}

float4 PS_BloomBlurV(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float2 r = ReShade::PixelSize * BloomRadius * 2.0;
    float3 c0 = tex2D(BloomMip1TempSampler, uv).rgb;
    float3 c1 = tex2D(BloomMip1TempSampler, uv + float2(0, r.y)).rgb;
    float3 c2 = tex2D(BloomMip1TempSampler, uv - float2(0, r.y)).rgb;
    float3 c3 = tex2D(BloomMip1TempSampler, uv + float2(0, 2.0*r.y)).rgb;
    float3 c4 = tex2D(BloomMip1TempSampler, uv - float2(0, 2.0*r.y)).rgb;
    float3 outc = (0.204164*c0 + 0.304005*(c1 + c2) + 0.093827*(c3 + c4));
    return float4(outc, 1.0);
}

float3 ApplyBloomComposite(float2 uv, float3 base)
{
    float3 b = tex2D(BloomMip2Sampler, uv).rgb;
    return base + b * BloomStrength;
}

// ============================
// EYE ADAPTATION (safe current->history write)
// ============================
float ComputeLogLuminance(float3 c)
{
    float l = max(0.0001, Luma(c));
    return log(l);
}

float4 PS_LumDown(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    static const int K = 25;
    static const float2 OFFS[K] =
    {
        float2(-2,-2), float2(-1,-2), float2(0,-2), float2(1,-2), float2(2,-2),
        float2(-2,-1), float2(-1,-1), float2(0,-1), float2(1,-1), float2(2,-1),
        float2(-2, 0), float2(-1, 0), float2(0, 0), float2(1, 0), float2(2, 0),
        float2(-2, 1), float2(-1, 1), float2(0, 1), float2(1, 1), float2(2, 1),
        float2(-2, 2), float2(-1, 2), float2(0, 2), float2(1, 2), float2(2, 2)
    };

    float2 step = float2(1.0 / 64.0, 1.0 / 64.0);
    float3 sum = float3(0.0, 0.0, 0.0);

    [loop]
    for (int i = 0; i < K; ++i)
    {
        float2 sampleUV = uv + OFFS[i] * step;
        sum += tex2D(BackBuffer, sampleUV).rgb;
    }

    sum /= float(K);
    float logL = ComputeLogLuminance(sum);
    return float4(logL, logL, logL, 1.0);
}

// Write current luminance into history (we avoid sampling LumHistoryTex while writing)
float4 PS_LumHistoryWrite(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float currLog = tex2D(LumCurrSampler, uv).r;
    return float4(currLog, currLog, currLog, 1.0);
}

// ============================
// DoF (depth-aware, temporal)
// ============================
float ComputeCoC(float depthViewZ)
{
    float z = -depthViewZ;
    float focalLength = 50.0;
    float fd = max(0.001, FocalDistance);
    float coc = abs((focalLength * (z - fd)) / (z * fd)) * (1000.0 / Aperture);
    float px = coc * 0.5;
    px = saturate(px / (MaxCoCRadius)) * MaxCoCRadius;
    return clamp(px, 0.0, MaxCoCRadius);
}

float DepthWeight(float centerDepth, float sampleDepth, float radius)
{
    float diff = abs(centerDepth - sampleDepth);
    float w = exp(-diff * 200.0 * (1.0 / max(0.1, radius * 0.02)));
    return w;
}

// DoF gathers (flattened weights, dynamic loops)
float4 DoF_GatherH(float2 uv, float cocPx)
{
    int taps = min(9, DOF_MAX_TAPS);
    float halfCount = taps * 0.5;

    float2 dir = float2(1.0, 0.0) * (cocPx / max(1.0, halfCount));

    float centerDepth = ReShade::GetLinearizedDepth(uv);

    float3 accum = float3(0, 0, 0);
    float wsum = 0.0;
    
    [loop]
    for (int i = 0; i < DOF_MAX_TAPS; ++i)
    {
        if (i >= taps)
            break;

        float idx = (float)i - halfCount;

        float2 sampleUV = uv + dir * idx;

        float sampleDepth = ReShade::GetLinearizedDepth(sampleUV);

        float depthDiff = abs(centerDepth - sampleDepth);

        float dw = exp(-depthDiff * 200.0);

        float3 sampleCol = tex2D(BackBuffer, sampleUV).rgb;

        float gw = DOF_WEIGHTS[i];

        float w = gw * dw;

        accum += sampleCol * w;
        wsum += w;
    }

    return float4(accum / max(1e-6, wsum), 1.0);
}

float4 DoF_GatherV(float2 uv, float cocPx)
{
    int taps = min(9, DOF_MAX_TAPS);
    float halfCount = taps * 0.5;

    float2 dir = float2(0.0, 1.0) * (cocPx / max(1.0, halfCount));

    float centerDepth = ReShade::GetLinearizedDepth(uv);

    float3 accum = float3(0, 0, 0);
    float wsum = 0.0;

    [loop]
    for (int i = 0; i < DOF_MAX_TAPS; ++i)
    {
        if (i >= taps)
            break;

        float idx = (float)i - halfCount;

        float2 sampleUV = uv + dir * idx;

        float sampleDepth = ReShade::GetLinearizedDepth(sampleUV);

        float depthDiff = abs(centerDepth - sampleDepth);

        float dw = exp(-depthDiff * 200.0);

        float3 sampleCol = tex2D(BackBuffer, sampleUV).rgb;

        float gw = DOF_WEIGHTS[i];

        float w = gw * dw;

        accum += sampleCol * w;
        wsum += w;
    }

    return float4(accum / max(1e-6, wsum), 1.0);
}

// Sample DoF history (reads Prev)
float4 SampleDoFHistory(float2 uv, float3 viewPos)
{
    float4 hist = tex2D(DoFHistoryPrevSampler, uv);

    if (!EnableDoF)
        return hist;

    float4 prevClip = mul(PrevViewProj, float4(viewPos, 1.0));

    if (abs(prevClip.w) < 1e-6)
        return hist;

    prevClip /= prevClip.w;

    float2 prevUV = prevClip.xy * 0.5 + 0.5;

    if (prevUV.x < 0.0 || prevUV.x > 1.0 ||
        prevUV.y < 0.0 || prevUV.y > 1.0)
        return hist;

    float4 reprojected = tex2D(DoFHistoryPrevSampler, prevUV);

    return reprojected;
}

float4 MotionAwareBlendDoF(float4 curr, float4 hist, float2 motionUV)
{
    float motionLen = length(motionUV) * float(BUFFER_WIDTH);
    float motionFactor = saturate(1.0 - smoothstep(0.5, 8.0, motionLen));
    float blend = lerp(0.25 * TemporalBlend, TemporalBlend, motionFactor);
    float4 clamped;
    clamped.rgb = clamp(hist.rgb, curr.rgb - TemporalClamp, curr.rgb + TemporalClamp);
    clamped.a = clamp(hist.a, curr.a - TemporalClamp, curr.a + TemporalClamp);
    return lerp(curr, clamped, blend);
}

float4 PS_DoFHistoryWrite(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float depth = ReShade::GetLinearizedDepth(uv);

    float3 color = tex2D(BackBuffer, uv).rgb;

    // sky / far plane → no DoF history
    if (depth >= 1.0)
        return float4(color, 0.0);

    float3 viewPos = ReconstructViewPos(uv);

    // IMPORTANT: use view-space Z consistently
    float coc = ComputeCoC(viewPos.z);

    float4 h = DoF_GatherH(uv, coc);
    float4 v = DoF_GatherV(uv, coc);

    float3 outCol = v.rgb;

    float outA = saturate(coc / MaxCoCRadius);

    float4 hist = SampleDoFHistory(uv, viewPos);

    float2 motion = EstimateMotionUV(uv);

    return MotionAwareBlendDoF(
        float4(outCol, outA),
        hist,
        motion
    );
}

float4 PS_DoFBlur(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float2 t = ReShade::PixelSize * 1.0;
    float4 s0 = tex2D(DoFHistoryCurrSampler, uv);
    float4 s1 = tex2D(DoFHistoryCurrSampler, uv + float2( t.x, 0));
    float4 s2 = tex2D(DoFHistoryCurrSampler, uv + float2(-t.x, 0));
    float4 s3 = tex2D(DoFHistoryCurrSampler, uv + float2(0,  t.y));
    float4 s4 = tex2D(DoFHistoryCurrSampler, uv + float2(0, -t.y));
    float4 avg = (s0 + s1 + s2 + s3 + s4) / 5.0;
    return avg;
}

float3 ApplyDepthOfField(float2 uv, float3 color)
{
    if (!EnableDoF)
        return color;

    float4 dof = tex2D(DoFBlurSampler, uv);

    if (dof.a <= 0.001)
        return color;

    float cocNorm = dof.a;
    float3 blurred = dof.rgb;

    // ReShade depth (consistent pipeline)    
    float3 viewPos = ReconstructViewPos(uv);

    float z = -viewPos.z;

    float nearFactor = saturate((FocalDistance - z) / FocalDistance);

    // NOTE: this was a no-op before (lerp(x,x,y) = x)
    float nearBoost = 1.0;

    float finalMix = saturate(cocNorm * nearBoost);

    float edge = smoothstep(0.02, 0.08, cocNorm);

    float mix = lerp(finalMix, edge * finalMix, 0.5);

    return lerp(color, blurred, mix);
}

// ============================
// SHARPEN & GRAIN (linear helpers)
// ============================
float3 ToLinear(float3 c) { return pow(c, 2.2); }
float3 ToSRGB(float3 c)   { return pow(c, 1.0/2.2); }

float3 SharpenPassLinear(float2 uv, float3 c)
{
    float2 t = ReShade::PixelSize;
    float3 n = tex2D(BackBuffer, uv + float2(0, -t.y)).rgb;
    float3 s = tex2D(BackBuffer, uv + float2(0,  t.y)).rgb;
    float3 e = tex2D(BackBuffer, uv + float2( t.x, 0)).rgb;
    float3 w = tex2D(BackBuffer, uv + float2(-t.x, 0)).rgb;
    float3 blur = (c + n + s + e + w) / 5.0;
    float3 detail = c - blur;
    float lum = Luma(c);
    float weight = saturate((lum - 0.2) * 1.5);
    return saturate(c + detail * Sharpness * weight);
}

float3 ApplyGrainLinear(float3 c, float2 uv)
{
    float2 scaled = uv * 1.0 * float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float n = Hash21(scaled + float2(FrameIndex, FrameIndex));
    float g = (n - 0.5) * 2.0 * GrainAmount;
    return saturate(c + g.xxx);
}

// ============================
// TONEMAP (simple ACES fallback)
// ============================
float3 TonemapACES_Simple(float3 x)
{
    const float a=2.51,b=0.03,c=2.43,d=0.59,e=0.14;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}
float3 ApplyTonemapLinear(float3 c)
{
    return TonemapACES_Simple(c);
}

// ============================
// MAIN PASS
// ============================
float4 PS_Photorealism(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float3 sceneSRGB = tex2D(BackBuffer, uv).rgb;
    float3 c = ToLinear(sceneSRGB);

    // AO
    if (EnableAO)
    {
        float ao = tex2D(AOBlurSampler, uv).r;
        if (ao == 0.0) ao = tex2D(AOHistoryCurrSampler, uv).r;
        c *= ao;
    }

    // Simple exposure
    c *= exp2(0.10);

    // Tonemap and color grading (apply tonemap in linear)
    c = ApplyTonemapLinear(c);
    c = ToSRGB(c);

    // Contact shadows
    float cs = tex2D(ContactBlurSampler, uv).r;
    if (cs == 0.0) cs = tex2D(ContactHistoryCurrSampler, uv).r;
    c *= cs;

    // SSR composite
    if (EnableSSR)
    {
        float4 ssr = tex2D(SSRBlurSampler, uv);
        float3 ssrColor = ssr.rgb;
        float ssrWeight = ssr.a;
        c = lerp(c, ssrColor, ssrWeight);
    }

    // Depth of Field
    if (EnableDoF)
    {
        c = ApplyDepthOfField(uv, c);
    }

    // Bloom composite
    if (EnableBloom)
        c = ApplyBloomComposite(uv, c);

    // Sharpen & Grain in linear space
    float3 cLinear = ToLinear(c);
    if (EnableSharpen)  cLinear = SharpenPassLinear(uv, cLinear);
    if (EnableGrain)    cLinear = ApplyGrainLinear(cLinear, uv);
    c = ToSRGB(cLinear);

    return float4(saturate(c), 1.0);
}

// ============================
// COPY PASS SHADERS (fallback swap Curr->Prev)
// ============================
float4 PS_CopyAOHistoryCurrToPrev(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float v = tex2D(AOHistoryCurrSampler, uv).r;
    return float4(v, v, v, 1.0);
}
float4 PS_CopyContactHistoryCurrToPrev(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float v = tex2D(ContactHistoryCurrSampler, uv).r;
    return float4(v, v, v, 1.0);
}
float4 PS_CopySSRHistoryCurrToPrev(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    return tex2D(SSRHistoryCurrSampler, uv);
}
float4 PS_CopyDoFHistoryCurrToPrev(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    return tex2D(DoFHistoryCurrSampler, uv);
}

// ============================
// MISSING PS ENTRY POINTS (AO/Contact/SSR)
// ============================
float4 PS_AOHistoryWrite(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float currAO = ComputeAO_Current(uv);
    
    float3 viewPos = ReconstructViewPos(uv);

    float histAO = SampleHistoryAO(uv, viewPos);

    float2 motion = EstimateMotionUV(uv);

    float blended = TemporalBlendAO_Motion(currAO, histAO, motion);

    return float4(blended, blended, blended, 1.0);
}

float4 PS_AOBlur(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float b = BlurAO_Bilateral(uv);
    return float4(b, b, b, 1.0);
}

float4 PS_ContactHistoryWrite(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float currC = ComputeContact_Current(uv);
    
    float3 viewPos = ReconstructViewPos(uv);

    float histC = SampleHistoryContact(uv, viewPos);

    float2 motion = EstimateMotionUV(uv);

    float blended = MotionAwareBlendScalar(currC, histC, motion, TemporalBlend, TemporalClamp);

    return float4(blended, blended, blended, 1.0);
}

float4 PS_ContactBlur(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float b = BlurContact_Bilateral(uv);
    return float4(b, b, b, 1.0);
}

float4 PS_SSRHistoryWrite(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float4 curr = ComputeSSR(uv);
    return curr;
}

float4 PS_SSRBlur(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float2 t = ReShade::PixelSize * 1.0;
    float4 s0 = tex2D(SSRHistoryCurrSampler, uv);
    float4 s1 = tex2D(SSRHistoryCurrSampler, uv + float2( t.x, 0));
    float4 s2 = tex2D(SSRHistoryCurrSampler, uv + float2(-t.x, 0));
    float4 s3 = tex2D(SSRHistoryCurrSampler, uv + float2(0,  t.y));
    float4 s4 = tex2D(SSRHistoryCurrSampler, uv + float2(0, -t.y));
    float4 avg = (s0 + s1 + s2 + s3 + s4) / 5.0;
    return avg;
}

// ============================
// TECHNIQUE (order matters)
// ============================
technique ReshadeTrueLight
{
    pass LumDown
    {
        RenderTarget = LumCurrTex;
        VertexShader = PostProcessVS;
        PixelShader  = PS_LumDown;
    }

    pass LumHistoryWrite
    {
        RenderTarget = LumHistoryTex;
        VertexShader = PostProcessVS;
        PixelShader  = PS_LumHistoryWrite;
    }

    pass AOHistoryWrite
    {
        RenderTarget = AOHistoryCurr;
        VertexShader = PostProcessVS;
        PixelShader  = PS_AOHistoryWrite;
    }

    pass AOBlur
    {
        RenderTarget = AOBlurTex;
        VertexShader = PostProcessVS;
        PixelShader  = PS_AOBlur;
    }

    pass ContactHistoryWrite
    {
        RenderTarget = ContactHistoryCurr;
        VertexShader = PostProcessVS;
        PixelShader  = PS_ContactHistoryWrite;
    }

    pass ContactBlur
    {
        RenderTarget = ContactBlurTex;
        VertexShader = PostProcessVS;
        PixelShader  = PS_ContactBlur;
    }

    pass SSRHistoryWrite
    {
        RenderTarget = SSRHistoryCurr;
        VertexShader = PostProcessVS;
        PixelShader  = PS_SSRHistoryWrite;
    }

    pass SSRBlur
    {
        RenderTarget = SSRBlurTex;
        VertexShader = PostProcessVS;
        PixelShader  = PS_SSRBlur;
    }

    pass BloomDown0
    {
        RenderTarget = BloomMip0A;
        VertexShader = PostProcessVS;
        PixelShader  = PS_BloomDown0;
    }

    pass BloomPingPong
    {
        RenderTarget = BloomMip0B;
        VertexShader = PostProcessVS;
        PixelShader  = PS_BloomPingPong;
    }

    pass BloomDown1
    {
        RenderTarget = BloomMip1;
        VertexShader = PostProcessVS;
        PixelShader  = PS_BloomDown1;
    }

    pass BloomBlurH
    {
        RenderTarget = BloomMip1Temp;
        VertexShader = PostProcessVS;
        PixelShader  = PS_BloomBlurH;
    }

    pass BloomBlurV
    {
        RenderTarget = BloomMip2;
        VertexShader = PostProcessVS;
        PixelShader  = PS_BloomBlurV;
    }

    pass DoFHistoryWrite
    {
        RenderTarget = DoFHistoryCurr;
        VertexShader = PostProcessVS;
        PixelShader  = PS_DoFHistoryWrite;
    }

    pass DoFBlur
    {
        RenderTarget = DoFBlurTex;
        VertexShader = PostProcessVS;
        PixelShader  = PS_DoFBlur;
    }

    pass CopyAOHistoryCurrToPrev
    {
        RenderTarget = AOHistoryPrev;
        VertexShader = PostProcessVS;
        PixelShader  = PS_CopyAOHistoryCurrToPrev;
    }

    pass CopyContactHistoryCurrToPrev
    {
        RenderTarget = ContactHistoryPrev;
        VertexShader = PostProcessVS;
        PixelShader  = PS_CopyContactHistoryCurrToPrev;
    }

    pass CopySSRHistoryCurrToPrev
    {
        RenderTarget = SSRHistoryPrev;
        VertexShader = PostProcessVS;
        PixelShader  = PS_CopySSRHistoryCurrToPrev;
    }

    pass CopyDoFHistoryCurrToPrev
    {
        RenderTarget = DoFHistoryPrev;
        VertexShader = PostProcessVS;
        PixelShader  = PS_CopyDoFHistoryCurrToPrev;
    }

    pass Main
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_Photorealism;
    }
}
