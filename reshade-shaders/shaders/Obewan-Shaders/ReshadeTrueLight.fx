//===========================================================================
// SKYRIM REALISTIC PIPELINE — ReshadeTrueLight
// Author: Obewan (https://github.com/obewan)
// Co-authors: Anthropic Claude, Microsoft Copilot, OpenAI ChatGPT, Google Gemini.
// Version: 0.1.0
// Requirement:
//     - BlueNoise.png in reshade-shaders/Textures (only if UseBlueNoise = true)
//     - a depth buffer correctly set (check it using the DisplayDepth shader)
//
// v0.1.0: single-frame pipeline (temporal reprojection removed). All effects
// use spatial bilateral filtering. See the header for the rationale.
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

float Luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

float3 ToLinear(float3 c) { return pow(abs(c), 2.2); }
float3 ToSRGB(float3 c)   { return pow(abs(c), 1.0 / 2.2); }

// Animated per-pixel jitter in [-0.5, 0.5]. Uses blue noise when available.
float2 JitterUV(float2 uv)
{
    float fi = float(FrameIndex);
    if (UseBlueNoise)
    {
        float2 bn = tex2D(BlueNoiseSampler, frac(uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT) / 64.0 + fi * 0.137)).rg;
        return bn - 0.5;
    }
    float n  = Hash21(uv * 1024.0 + fi);
    float n2 = Hash11(n + fi * 0.017);
    return float2(n, n2) - 0.5;
}

// Normalized linear depth in [0,1] (1 = far plane / sky).
float GetDepth(float2 uv)
{
    if (!HasDepth) return 1.0;
    return ReShade::GetLinearizedDepth(uv);
}

// Single FOV-based view-space reconstruction. Convention: forward = -Z.
// Returns position in world units (normalized depth * CameraFar).
float3 ReconstructViewPos(float2 uv)
{
    float z = GetDepth(uv) * CameraFar;
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float t = tan(radians(CameraFovY) * 0.5);
    float3 ray = float3(ndc.x * ReShade::AspectRatio * t, ndc.y * t, -1.0);
    return ray * z;
}

// Exact inverse of ReconstructViewPos: view-space position -> screen UV.
float2 ProjectToUV(float3 v)
{
    float t = tan(radians(CameraFovY) * 0.5);
    float invz = 1.0 / max(-v.z, 1e-4);
    float2 ndc = float2(v.x * invz / (ReShade::AspectRatio * t), v.y * invz / t);
    ndc.y = -ndc.y;
    return ndc * 0.5 + 0.5;
}

// View-space normal from depth, using the smaller of forward/backward
// differences on each axis to limit bleeding across depth discontinuities.
float3 EstimateNormal(float2 uv)
{
    float2 px = ReShade::PixelSize;
    float3 c = ReconstructViewPos(uv);

    float3 dxR = ReconstructViewPos(uv + float2(px.x, 0)) - c;
    float3 dxL = c - ReconstructViewPos(uv - float2(px.x, 0));
    float3 ddx = (abs(dxR.z) < abs(dxL.z)) ? dxR : dxL;

    float3 dyD = ReconstructViewPos(uv + float2(0, px.y)) - c;
    float3 dyU = c - ReconstructViewPos(uv - float2(0, px.y));
    float3 ddy = (abs(dyD.z) < abs(dyU.z)) ? dyD : dyU;

    float3 n = cross(ddx, ddy);
    float len = length(n);
    if (len < 1e-6 || any(isnan(n))) return float3(0, 0, 1);
    n /= len;

    // face toward the camera (camera at origin, so view dir to camera = -c)
    if (dot(n, normalize(-c)) < 0.0) n = -n;
    return n;
}

// Convert a world-space radius at viewPos into a screen-space UV radius.
float2 ViewRadiusToUV(float3 viewPos, float worldRadius)
{
    float z = max(-viewPos.z, 1e-4);
    float t = tan(radians(CameraFovY) * 0.5);
    float uvr = (worldRadius / (z * t)) * 0.5;
    return float2(uvr / ReShade::AspectRatio, uvr);
}

// ============================
// AO (half-res compute + bilateral upsample)
// ============================
float ComputeAO(float2 uv)
{
    float d = GetDepth(uv);
    if (d <= 0.0 || d >= 1.0) return 1.0;

    float3 p = ReconstructViewPos(uv);
    float3 n = EstimateNormal(uv);

    int samples = clamp(AOSamples, 4, AO_MAX_SAMPLES);
    float2 uvRadius = ViewRadiusToUV(p, AORadius);
    float2 jitter = JitterUV(uv) * ReShade::PixelSize * 2.0;

    float occlusion = 0.0;

    [loop]
    for (int i = 0; i < AO_MAX_SAMPLES; ++i)
    {
        if (i >= samples) break;

        float2 sampleUV = uv + AO_Dirs[i] * uvRadius + jitter;

        float sd = GetDepth(sampleUV);
        if (sd <= 0.0 || sd >= 1.0) continue;

        float3 sp = ReconstructViewPos(sampleUV);
        float3 v = sp - p;
        float dist = length(v);
        if (dist < 1e-4) continue;

        float nd = saturate(dot(n, v / dist));
        float rangeAtten = saturate(1.0 - dist / (AORadius * 2.0));

        occlusion += nd * rangeAtten;
    }

    occlusion /= float(samples);
    float ao = saturate(1.0 - occlusion * AOStrength);
    return pow(ao, 1.2);
}

// Depth-aware blur / upsample (reads the half-res AO target).
float BlurAO(float2 uv)
{
    float centerD = GetDepth(uv);
    float2 step = ReShade::PixelSize * 2.0; // half-res texel

    float sum = 0.0;
    float wsum = 0.0;

    [loop]
    for (int y = -2; y <= 2; ++y)
    {
        [loop]
        for (int x = -2; x <= 2; ++x)
        {
            float2 sUV = uv + float2(x, y) * step;
            float v = tex2D(AORawSampler, sUV).r;
            float sd = GetDepth(sUV);
            float w = exp(-abs(centerD - sd) * 50.0);
            sum += v * w;
            wsum += w;
        }
    }
    return sum / max(1e-6, wsum);
}

// ============================
// CONTACT SHADOWS (full-res raymarch + bilateral blur)
// ============================
float ComputeContact(float2 uv)
{
    float d = GetDepth(uv);
    if (d <= 0.0 || d >= 1.0) return 1.0;

    float3 p = ReconstructViewPos(uv);
    float3 n = EstimateNormal(uv);

    float3 L = normalize(float3(0.0, -0.7, -0.7)); // fixed key-light direction
    float ndotl = saturate(dot(n, -L));
    if (ndotl < 0.05) return 1.0;

    float stepLen = ContactMaxDist / 8.0;
    float2 jitter = JitterUV(uv) * ReShade::PixelSize;

    float3 samplePos = p;
    float occlusion = 0.0;

    [loop]
    for (int i = 1; i <= 8; ++i)
    {
        samplePos += -L * stepLen;

        float2 sampleUV = ProjectToUV(samplePos) + jitter;
        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) continue;

        float sd = GetDepth(sampleUV);
        if (sd <= 0.0 || sd >= 1.0) continue;

        float3 sp = ReconstructViewPos(sampleUV);
        float distScene = length(sp - p);
        float distRay   = length(samplePos - p);

        if (distScene + stepLen * 0.25 < distRay * 0.98)
        {
            float w = saturate(1.0 - distRay / ContactMaxDist);
            occlusion += w;
        }
    }

    occlusion = saturate(occlusion / 8.0);
    float shadow = lerp(1.0, 1.0 - occlusion, ContactStrength * ndotl);
    return saturate(shadow);
}

float BlurContact(float2 uv)
{
    float centerD = GetDepth(uv);
    float2 step = ReShade::PixelSize;

    float sum = 0.0;
    float wsum = 0.0;

    [loop]
    for (int y = -2; y <= 2; ++y)
    {
        [loop]
        for (int x = -2; x <= 2; ++x)
        {
            float2 sUV = uv + float2(x, y) * step;
            float v = tex2D(ContactRawSampler, sUV).r;
            float sd = GetDepth(sUV);
            float w = exp(-abs(centerD - sd) * 50.0);
            sum += v * w;
            wsum += w;
        }
    }
    return sum / max(1e-6, wsum);
}

// ============================
// SSR (half-res raymarch + bilateral blur)
// ============================
float3 ReflectView(float3 viewDir, float3 normal)
{
    return normalize(viewDir - 2.0 * dot(viewDir, normal) * normal);
}

// Coarse linear march, then binary refinement + thickness rejection.
bool SSR_Raymarch(float3 viewPos, float3 reflDir, out float2 hitUV)
{
    hitUV = 0.0;

    int steps = clamp(SSRSteps, 8, SSR_MAX_STEPS);
    float stepSize = SSRMaxDistance / float(steps);
    float3 prevPos = viewPos;
    float3 pos = viewPos;

    [loop]
    for (int i = 0; i < SSR_MAX_STEPS; ++i)
    {
        if (i >= steps) break;

        prevPos = pos;
        pos += reflDir * stepSize;

        float2 uv = ProjectToUV(pos);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) break;

        float sd = GetDepth(uv);
        if (sd >= 1.0) continue;

        float3 sp = ReconstructViewPos(uv);
        if (length(sp - viewPos) + stepSize * 0.5 < length(pos - viewPos))
        {
            // Binary search between the last in-front sample and this one.
            float3 a = prevPos, b = pos;
            [loop]
            for (int j = 0; j < 5; ++j)
            {
                float3 mid = (a + b) * 0.5;
                float2 muv = ProjectToUV(mid);
                float3 msp = ReconstructViewPos(muv);
                if (length(msp - viewPos) < length(mid - viewPos)) b = mid; // mid behind surface
                else a = mid;                                               // mid in front
            }

            float2 fuv = ProjectToUV(b);
            float3 fsp = ReconstructViewPos(fuv);

            // Reject if the surface lies far behind the ray (silhouette / gap bleed).
            float thickness = length(b - viewPos) - length(fsp - viewPos);
            if (abs(thickness) > SSRThickness) return false;

            hitUV = fuv;
            return true;
        }
    }
    return false;
}

float4 ComputeSSR(float2 uv)
{
    float d = GetDepth(uv);
    if (d >= 1.0) return float4(0, 0, 0, 0);

    float3 viewPos = ReconstructViewPos(uv);
    float3 normal  = EstimateNormal(uv);
    float3 viewDir = normalize(viewPos);   // camera -> surface (incident)
    float3 refl    = ReflectView(viewDir, normal);

    // Schlick Fresnel: surfaces facing the camera (NPCs, walls head-on) reflect
    // ~F0 (almost nothing); only grazing angles build up reflection. This is the
    // key term that removes the "everything is a mirror" artifact.
    float NdotV   = saturate(dot(normal, -viewDir));
    float fresnel = SSRBaseReflect + (1.0 - SSRBaseReflect) * pow(1.0 - NdotV, 5.0);

    float2 hitUV;
    if (!SSR_Raymarch(viewPos, refl, hitUV))
        return float4(0, 0, 0, 0);

    // Fade reflections near screen edges to hide marching artifacts.
    float2 edge = smoothstep(0.0, 0.1, hitUV) * smoothstep(0.0, 0.1, 1.0 - hitUV);
    float edgeFade = edge.x * edge.y;

    float3 reflectedColor = tex2D(BackBuffer, hitUV).rgb; // sRGB scene color
    float weight = saturate(SSRStrength * fresnel * edgeFade);
    return float4(reflectedColor, weight);
}

float4 BlurSSR(float2 uv)
{
    float2 t = ReShade::PixelSize * 2.0; // half-res texel
    float4 s0 = tex2D(SSRRawSampler, uv);
    float4 s1 = tex2D(SSRRawSampler, uv + float2( t.x, 0));
    float4 s2 = tex2D(SSRRawSampler, uv + float2(-t.x, 0));
    float4 s3 = tex2D(SSRRawSampler, uv + float2(0,  t.y));
    float4 s4 = tex2D(SSRRawSampler, uv + float2(0, -t.y));
    return (s0 + s1 + s2 + s3 + s4) / 5.0;
}

// ============================
// DoF (separable: horizontal gather -> vertical gather)
// ============================
float ComputeCoCNorm(float2 uv)
{
    float z = GetDepth(uv) * CameraFar;
    float blur = saturate((abs(z - FocalDistance) - FocalRange) / max(FocalRange, 1e-3));
    return blur; // 0 = sharp, 1 = max blur
}

float4 DoF_Gather(float2 uv, float2 dir)
{
    float cocNorm = ComputeCoCNorm(uv);
    float cocPx = cocNorm * MaxCoCRadius;

    int taps = DOF_MAX_TAPS;
    int halfT = taps / 2;
    float2 stepDir = dir * (cocPx / max(1.0, float(halfT)));

    float centerD = GetDepth(uv);
    float3 accum = 0.0;
    float wsum = 0.0;

    [loop]
    for (int i = 0; i < taps; ++i)
    {
        int idx = i - halfT;
        float2 sUV = uv + stepDir * float(idx);
        float sd = GetDepth(sUV);
        float dw = exp(-abs(centerD - sd) * 30.0);
        float w = DOF_WEIGHTS[i] * dw;
        accum += tex2D(BackBuffer, sUV).rgb * w;
        wsum += w;
    }

    return float4(accum / max(1e-6, wsum), cocNorm);
}

// ============================
// BLOOM (threshold in linear space, ping-pong + small mip chain)
// ============================
float4 PS_BloomDown0(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float3 c = ToLinear(tex2D(BackBuffer, uv).rgb);
    float l = Luma(c);
    float mask = saturate(l - BloomThreshold) / max(1e-3, 1.0 - BloomThreshold);
    return float4(c * mask, 1.0);
}

float4 PS_BloomPingPong(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    return float4(tex2D(BloomMip0ASampler, uv).rgb, 1.0);
}

float4 PS_BloomDown1(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float2 t = ReShade::PixelSize * 2.0;
    float3 sum = 0.0;
    sum += tex2D(BloomMip0BSampler, uv + float2( t.x, 0)).rgb;
    sum += tex2D(BloomMip0BSampler, uv + float2(-t.x, 0)).rgb;
    sum += tex2D(BloomMip0BSampler, uv + float2(0,  t.y)).rgb;
    sum += tex2D(BloomMip0BSampler, uv + float2(0, -t.y)).rgb;
    return float4(sum * 0.25, 1.0);
}

float4 PS_BloomBlurH(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float2 r = ReShade::PixelSize * BloomRadius * 2.0;
    float3 c0 = tex2D(BloomMip1Sampler, uv).rgb;
    float3 c1 = tex2D(BloomMip1Sampler, uv + float2(r.x, 0)).rgb;
    float3 c2 = tex2D(BloomMip1Sampler, uv - float2(r.x, 0)).rgb;
    float3 c3 = tex2D(BloomMip1Sampler, uv + float2(2.0 * r.x, 0)).rgb;
    float3 c4 = tex2D(BloomMip1Sampler, uv - float2(2.0 * r.x, 0)).rgb;
    return float4(0.204164 * c0 + 0.304005 * (c1 + c2) + 0.093827 * (c3 + c4), 1.0);
}

float4 PS_BloomBlurV(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float2 r = ReShade::PixelSize * BloomRadius * 2.0;
    float3 c0 = tex2D(BloomMip1TempSampler, uv).rgb;
    float3 c1 = tex2D(BloomMip1TempSampler, uv + float2(0, r.y)).rgb;
    float3 c2 = tex2D(BloomMip1TempSampler, uv - float2(0, r.y)).rgb;
    float3 c3 = tex2D(BloomMip1TempSampler, uv + float2(0, 2.0 * r.y)).rgb;
    float3 c4 = tex2D(BloomMip1TempSampler, uv - float2(0, 2.0 * r.y)).rgb;
    return float4(0.204164 * c0 + 0.304005 * (c1 + c2) + 0.093827 * (c3 + c4), 1.0);
}

// ============================
// EYE ADAPTATION (instant auto-exposure)
// ============================
float4 PS_LumDown(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    static const int K = 9;
    static const float2 OFFS[K] =
    {
        float2(-1,-1), float2(0,-1), float2(1,-1),
        float2(-1, 0), float2(0, 0), float2(1, 0),
        float2(-1, 1), float2(0, 1), float2(1, 1)
    };

    float2 step = 1.0 / 64.0;
    float sum = 0.0;
    [loop]
    for (int i = 0; i < K; ++i)
        sum += Luma(ToLinear(tex2D(BackBuffer, uv + OFFS[i] * step).rgb));

    return (sum / float(K)).xxxx;
}

// Coarse scene-average luminance from the 64x64 downsample.
float SceneAvgLuma()
{
    float sum = 0.0;
    [loop]
    for (int y = 0; y < 4; ++y)
    {
        [loop]
        for (int x = 0; x < 4; ++x)
        {
            float2 g = (float2(x, y) + 0.5) / 4.0;
            sum += tex2Dlod(LumCurrSampler, float4(g, 0, 0)).r;
        }
    }
    return max(sum / 16.0, 1e-4);
}

// Smoothly approach the current scene average (frame-rate independent).
float4 PS_Adapt(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float curr = SceneAvgLuma();
    float prev = tex2Dlod(AdaptPrevSampler, float4(0.5, 0.5, 0, 0)).r;
    if (prev <= 0.0) prev = curr; // first frame init

    float rate = saturate(AdaptSpeed * frametime * 0.001); // frametime in ms
    return lerp(prev, curr, rate).xxxx;
}

float4 PS_AdaptCopy(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    return tex2Dlod(AdaptSampler, float4(0.5, 0.5, 0, 0)).r.xxxx;
}

float ComputeExposure()
{
    if (!EnableAutoExposure)
        return exp2(ManualExposure);

    float avg = max(tex2Dlod(AdaptSampler, float4(0.5, 0.5, 0, 0)).r, 1e-4);
    return clamp(ExposureKey / avg, ExposureMin, ExposureMax);
}

// ============================
// SHARPEN & GRAIN (display space)
// ============================
float3 SharpenPass(float2 uv, float3 c)
{
    float2 t = ReShade::PixelSize;
    float3 n = tex2D(BackBuffer, uv + float2(0, -t.y)).rgb;
    float3 s = tex2D(BackBuffer, uv + float2(0,  t.y)).rgb;
    float3 e = tex2D(BackBuffer, uv + float2( t.x, 0)).rgb;
    float3 w = tex2D(BackBuffer, uv + float2(-t.x, 0)).rgb;
    float3 blur = (tex2D(BackBuffer, uv).rgb + n + s + e + w) / 5.0;
    float3 detail = tex2D(BackBuffer, uv).rgb - blur;
    float weight = saturate((Luma(c) - 0.2) * 1.5);
    return saturate(c + detail * Sharpness * weight);
}

float3 ApplyGrain(float3 c, float2 uv)
{
    float2 scaled = uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float n = Hash21(scaled + float(FrameIndex));
    float g = (n - 0.5) * 2.0 * GrainAmount;
    return saturate(c + g.xxx);
}

// Triangular-PDF dither (~1 LSB at 8-bit), animated. Breaks up banding on
// smooth gradients such as the night sky without a visible grain look.
float3 ApplyDither(float3 c, float2 uv)
{
    float2 seed = uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT) + float(FrameIndex);
    float r1 = Hash21(seed);
    float r2 = Hash21(seed + 17.31);
    float t = r1 + r2 - 1.0; // triangular in [-1, 1]
    return c + t * (1.0 / 255.0);
}

// ============================
// TONEMAP (ACES approximation)
// ============================
float3 TonemapACES(float3 x)
{
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// ============================
// EFFECT PASS ENTRY POINTS
// ============================
float4 PS_AOCompute(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float ao = EnableAO ? ComputeAO(uv) : 1.0;
    return ao.xxxx;
}
float4 PS_AOBlur(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    return BlurAO(uv).xxxx;
}

float4 PS_ContactCompute(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float c = EnableContact ? ComputeContact(uv) : 1.0;
    return c.xxxx;
}
float4 PS_ContactBlur(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    return BlurContact(uv).xxxx;
}

float4 PS_SSRCompute(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    return EnableSSR ? ComputeSSR(uv) : float4(0, 0, 0, 0);
}
float4 PS_SSRBlur(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    return BlurSSR(uv);
}

float4 PS_DoFGatherH(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    if (!EnableDoF) return float4(tex2D(BackBuffer, uv).rgb, 0.0);
    return DoF_Gather(uv, float2(1.0, 0.0));
}
float4 PS_DoFGatherV(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    if (!EnableDoF) return float4(tex2D(BackBuffer, uv).rgb, 0.0);
    // Vertical gather over the horizontally-pre-blurred buffer.
    float cocNorm = tex2D(DoFRawSampler, uv).a;
    float cocPx = cocNorm * MaxCoCRadius;
    int taps = DOF_MAX_TAPS;
    int halfT = taps / 2;
    float2 stepDir = float2(0.0, 1.0) * (cocPx / max(1.0, float(halfT)));

    float centerD = GetDepth(uv);
    float3 accum = 0.0;
    float wsum = 0.0;
    [loop]
    for (int i = 0; i < taps; ++i)
    {
        int idx = i - halfT;
        float2 sUV = uv + stepDir * float(idx);
        float sd = GetDepth(sUV);
        float dw = exp(-abs(centerD - sd) * 30.0);
        float w = DOF_WEIGHTS[i] * dw;
        accum += tex2D(DoFRawSampler, sUV).rgb * w;
        wsum += w;
    }
    return float4(accum / max(1e-6, wsum), cocNorm);
}

// ============================
// MAIN PASS
// ============================
float4 PS_Photorealism(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    // ---- Debug visualizers ----
    if (DebugMode != 0)
    {
        if (DebugMode == 1) { float d = GetDepth(uv); return float4(d.xxx, 1.0); }
        if (DebugMode == 2) { float3 n = EstimateNormal(uv); return float4(n * 0.5 + 0.5, 1.0); }
        if (DebugMode == 3) { float3 p = ReconstructViewPos(uv); return float4(frac(abs(p) * 0.1), 1.0); }
        if (DebugMode == 4) { float coc = ComputeCoCNorm(uv); return float4(coc.xxx, 1.0); }
        if (DebugMode == 5) { float ao = tex2D(AOBlurSampler, uv).r; return float4(ao.xxx, 1.0); }
        if (DebugMode == 6) { float cs = tex2D(ContactBlurSampler, uv).r; return float4(cs.xxx, 1.0); }
        if (DebugMode == 7) { float4 s = tex2D(SSRBlurSampler, uv); return float4(s.rgb * s.a, 1.0); }
        if (DebugMode == 8) { return float4(tex2D(BloomMip2Sampler, uv).rgb, 1.0); }
    }

    float3 c = ToLinear(tex2D(BackBuffer, uv).rgb);

    // AO (linear, pre-tonemap)
    if (EnableAO)
        c *= tex2D(AOBlurSampler, uv).r;

    // Contact shadows (linear)
    if (EnableContact)
        c *= tex2D(ContactBlurSampler, uv).r;

    // Exposure
    c *= ComputeExposure();

    // SSR (linear): reflected color stored in sRGB, convert before mixing
    if (EnableSSR)
    {
        float4 ssr = tex2D(SSRBlurSampler, uv);
        c = lerp(c, ToLinear(ssr.rgb), saturate(ssr.a));
    }

    // Bloom (linear, additive)
    if (EnableBloom)
        c += tex2D(BloomMip2Sampler, uv).rgb * BloomStrength;

    // Tonemap + encode
    c = TonemapACES(c);
    c = ToSRGB(c);

    // DoF (display space; lerp toward the blurred buffer by CoC)
    if (EnableDoF)
    {
        float4 dof = tex2D(DoFBlurSampler, uv);
        c = lerp(c, dof.rgb, saturate(dof.a));
    }

    // Sharpen & grain (display space)
    if (EnableSharpen) c = SharpenPass(uv, c);
    if (EnableGrain)   c = ApplyGrain(c, uv);

    // Dither last, just before 8-bit quantization, to suppress banding.
    c = ApplyDither(c, uv);

    return float4(saturate(c), 1.0);
}

// ============================
// TECHNIQUE (order matters)
// ============================
technique ReshadeTrueLight
{
    pass LumDown        { VertexShader = PostProcessVS; PixelShader = PS_LumDown;       RenderTarget = LumCurrTex; }
    pass Adapt          { VertexShader = PostProcessVS; PixelShader = PS_Adapt;         RenderTarget = AdaptTex; }
    pass AdaptCopy      { VertexShader = PostProcessVS; PixelShader = PS_AdaptCopy;     RenderTarget = AdaptPrevTex; }

    pass AOCompute      { VertexShader = PostProcessVS; PixelShader = PS_AOCompute;      RenderTarget = AORawTex; }
    pass AOBlur         { VertexShader = PostProcessVS; PixelShader = PS_AOBlur;         RenderTarget = AOBlurTex; }

    pass ContactCompute { VertexShader = PostProcessVS; PixelShader = PS_ContactCompute; RenderTarget = ContactRawTex; }
    pass ContactBlur    { VertexShader = PostProcessVS; PixelShader = PS_ContactBlur;    RenderTarget = ContactBlurTex; }

    pass SSRCompute     { VertexShader = PostProcessVS; PixelShader = PS_SSRCompute;     RenderTarget = SSRRawTex; }
    pass SSRBlur        { VertexShader = PostProcessVS; PixelShader = PS_SSRBlur;        RenderTarget = SSRBlurTex; }

    pass BloomDown0     { VertexShader = PostProcessVS; PixelShader = PS_BloomDown0;     RenderTarget = BloomMip0A; }
    pass BloomPingPong  { VertexShader = PostProcessVS; PixelShader = PS_BloomPingPong;  RenderTarget = BloomMip0B; }
    pass BloomDown1     { VertexShader = PostProcessVS; PixelShader = PS_BloomDown1;     RenderTarget = BloomMip1; }
    pass BloomBlurH     { VertexShader = PostProcessVS; PixelShader = PS_BloomBlurH;     RenderTarget = BloomMip1Temp; }
    pass BloomBlurV     { VertexShader = PostProcessVS; PixelShader = PS_BloomBlurV;     RenderTarget = BloomMip2; }

    pass DoFGatherH     { VertexShader = PostProcessVS; PixelShader = PS_DoFGatherH;     RenderTarget = DoFRawTex; }
    pass DoFGatherV     { VertexShader = PostProcessVS; PixelShader = PS_DoFGatherV;     RenderTarget = DoFBlurTex; }

    pass Main           { VertexShader = PostProcessVS; PixelShader = PS_Photorealism; }
}
