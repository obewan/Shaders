//===========================================================================
// SKYRIM REALISTIC PIPELINE — ReshadeTrueLight
// Author: Obewan (https://github.com/obewan)
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

// Noise value in [0,1]. Tiled blue noise (temporally animated, golden-ratio
// value rotation to preserve its spectrum) when enabled, else a white-noise hash.
// Blue noise is much less perceptible for grain and dithering.
//
// The temporal term animates the VALUE, never the coordinate: hashing
// (pixel + frame counter) loses float precision as the frame counter grows,
// which produced crawling vertical bands.
float BlueNoise(float2 uv)
{
    float2 p = uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float anim = frac(float(FrameIndex) * 0.61803399); // bounded golden-ratio step

    if (UseBlueNoise)
    {
        float v = tex2D(BlueNoiseSampler, frac(p * (1.0 / 64.0))).r;
        return frac(v + anim);
    }

    // Robust per-pixel hash (Hoskins): reduces coordinate magnitude first, so it
    // stays stable for large pixel coordinates.
    float3 p3 = frac(p.xyx * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    float h = frac((p3.x + p3.y) * p3.z);
    return frac(h + anim);
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
    // Optionally grow the radius with distance to keep a stable screen footprint.
    float effRadius = AORadius + max(-p.z, 0.0) * AODistantRadius;
    float2 uvRadius = ViewRadiusToUV(p, effRadius);
    // Static per-pixel rotation angle (no frame counter, so the blur can resolve it).
    float rnd = Hash21(uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT));
    float ca = cos(rnd * 6.2831853);
    float sa = sin(rnd * 6.2831853);

    float occlusion = 0.0;

    [loop]
    for (int i = 0; i < AO_MAX_SAMPLES; ++i)
    {
        if (i >= samples) break;

        // Rotate the whole spiral per pixel (not just the radius): rotating the
        // directions decorrelates neighbours so the bilateral blur resolves to
        // clean, smooth AO instead of structured noise.
        float  r  = (float(i) + 0.5) / float(samples); // 0..1 along radius
        float2 d0 = AO_Dirs[i];
        float2 dir = float2(d0.x * ca - d0.y * sa, d0.x * sa + d0.y * ca);
        float2 sampleUV = uv + dir * uvRadius * r;

        float sd = GetDepth(sampleUV);
        if (sd <= 0.0 || sd >= 1.0) continue;

        float3 sp = ReconstructViewPos(sampleUV);
        float3 v = sp - p;
        float dist = length(v);
        if (dist < 1e-4) continue;

        // Occluded when the neighbour rises above the tangent plane (beyond bias),
        // attenuated so distant samples count less.
        float nd = dot(n, v / dist) - AOBias / max(dist, 1e-3);
        float atten = saturate(1.0 - dist / effRadius);
        occlusion += saturate(nd) * atten;
    }

    occlusion /= float(samples);
    float ao = pow(saturate(1.0 - occlusion * AOStrength), AOPower);
    // Fade AO out at distance — far depth has poor precision and produces noise.
    return lerp(ao, 1.0, saturate((d - AOFadeStart) / max(1e-4, AOFadeEnd - AOFadeStart)));
}

// GTAO/HBAO-style: march along screen-space slices and track the highest
// occluder (horizon) on each side, instead of point-sampling a disk. More
// accurate falloff and contact darkening; heavier than SSAO.
float ComputeGTAO(float2 uv)
{
    float d = GetDepth(uv);
    if (d <= 0.0 || d >= 1.0) return 1.0;

    float3 p = ReconstructViewPos(uv);
    float3 n = EstimateNormal(uv);

    float effRadius = AORadius + max(-p.z, 0.0) * AODistantRadius;
    float2 radiusUV = ViewRadiusToUV(p, effRadius);
    float rnd = Hash21(uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT)); // static slice rotation

    const int SLICES = 4;
    const int STEPS  = 4;
    float occlusion = 0.0;

    [loop]
    for (int s = 0; s < SLICES; ++s)
    {
        float ang = (float(s) + rnd) * (3.14159265 / float(SLICES));
        float2 dir = float2(cos(ang), sin(ang));

        float hPlus = 0.0, hMinus = 0.0; // max elevation above tangent per side

        [loop]
        for (int k = 1; k <= STEPS; ++k)
        {
            float2 off = dir * radiusUV * ((float(k) - 0.5) / float(STEPS));

            float2 uvA = uv + off;
            float sdA = GetDepth(uvA);
            if (sdA > 0.0 && sdA < 1.0)
            {
                float3 v = ReconstructViewPos(uvA) - p;
                float l = length(v);
                if (l > 1e-4)
                    hPlus = max(hPlus, (dot(n, v) / l - AOBias) * saturate(1.0 - l / effRadius));
            }

            float2 uvB = uv - off;
            float sdB = GetDepth(uvB);
            if (sdB > 0.0 && sdB < 1.0)
            {
                float3 v = ReconstructViewPos(uvB) - p;
                float l = length(v);
                if (l > 1e-4)
                    hMinus = max(hMinus, (dot(n, v) / l - AOBias) * saturate(1.0 - l / effRadius));
            }
        }

        occlusion += (saturate(hPlus) + saturate(hMinus)) * 0.5;
    }

    occlusion /= float(SLICES);
    float ao = pow(saturate(1.0 - occlusion * AOStrength), AOPower);
    // Fade AO out at distance — far depth has poor precision and produces noise.
    return lerp(ao, 1.0, saturate((d - AOFadeStart) / max(1e-4, AOFadeEnd - AOFadeStart)));
}

// Depth-aware blur / upsample (reads the half-res AO target). Wide 7x7 Gaussian
// bilateral so the per-pixel rotated dither resolves to smooth AO.
float BlurAO(float2 uv)
{
    float centerD = GetDepth(uv);
    float2 step = ReShade::PixelSize * 2.0; // half-res texel

    float sum = 0.0;
    float wsum = 0.0;

    [loop]
    for (int y = -3; y <= 3; ++y)
    {
        [loop]
        for (int x = -3; x <= 3; ++x)
        {
            float2 o = float2(x, y);
            float2 sUV = uv + o * step;
            float v = tex2D(AORawSampler, sUV).r;
            float sd = GetDepth(sUV);
            float gw = exp(-dot(o, o) * 0.18);          // gaussian spatial weight
            float dw = exp(-abs(centerD - sd) * 50.0);  // bilateral depth weight
            sum += v * gw * dw;
            wsum += gw * dw;
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

    const int STEPS = 8;
    float stepLen   = ContactMaxDist / float(STEPS);
    float bias      = stepLen * 0.5;       // avoid self-occlusion
    float thickness = ContactMaxDist;      // reject occluders far behind (background)

    // Static (non-animated) per-pixel start offset: staggers samples without the
    // temporal flicker an animated jitter caused. The bilateral blur smooths the rest.
    float jitter = Hash21(uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT));
    float3 samplePos = p - L * (stepLen * jitter);

    float occlusion = 0.0;

    [loop]
    for (int i = 1; i <= STEPS; ++i)
    {
        samplePos -= L * stepLen; // march toward the light

        float2 sampleUV = ProjectToUV(samplePos);
        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) break;

        float sd = GetDepth(sampleUV);
        if (sd <= 0.0 || sd >= 1.0) continue;

        // Occluded when the scene surface sits in front of the ray, within a
        // bounded thickness band (rejects background bleed at silhouettes).
        float diff = length(samplePos) - length(ReconstructViewPos(sampleUV));
        if (diff > bias && diff < thickness)
        {
            float w = saturate(1.0 - length(samplePos - p) / ContactMaxDist);
            occlusion = max(occlusion, w); // closest occluder wins; stable vs summing
        }
    }

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

// Adaptive (sphere-trace-style) march: step proportional to the gap between the
// ray and the nearest surface, then binary refinement + thickness rejection.
bool SSR_Raymarch(float3 viewPos, float3 reflDir, float jitter, out float2 hitUV)
{
    hitUV = 0.0;

    int steps = clamp(SSRSteps, 8, SSR_MAX_STEPS);
    float baseStep = SSRMaxDistance / float(steps);
    float minStep = baseStep * 0.25;
    float maxStep = baseStep * 4.0;

    float t = lerp(minStep, baseStep, jitter); // jittered start to break up banding
    float3 prevPos = viewPos;

    [loop]
    for (int i = 0; i < SSR_MAX_STEPS; ++i)
    {
        if (i >= steps || t > SSRMaxDistance) break;

        float3 pos = viewPos + reflDir * t;

        float2 uv = ProjectToUV(pos);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) break;

        float sd = GetDepth(uv);
        if (sd >= 1.0) { prevPos = pos; t += maxStep; continue; } // sky: skip ahead

        float3 sp = ReconstructViewPos(uv);
        float gap = length(sp) - length(pos); // >0: ray in front of surface; <=0: crossed it

        if (gap <= 0.0)
        {
            // Binary search between the last in-front sample and this one.
            float3 a = prevPos, b = pos;
            [loop]
            for (int j = 0; j < 5; ++j)
            {
                float3 mid = (a + b) * 0.5;
                float3 msp = ReconstructViewPos(ProjectToUV(mid));
                if (length(msp) <= length(mid)) b = mid; // mid behind surface
                else a = mid;                            // mid in front
            }

            float2 fuv = ProjectToUV(b);
            float3 fsp = ReconstructViewPos(fuv);

            // Reject if the surface lies far behind the ray (silhouette / gap bleed).
            if (abs(length(b) - length(fsp)) > SSRThickness) return false;

            hitUV = fuv;
            return true;
        }

        // Advance proportionally to the gap (conservative sphere trace).
        prevPos = pos;
        t += clamp(gap * 0.7, minStep, maxStep);
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

    float jitter = JitterUV(uv).x + 0.5;   // [0,1]

    float2 hitUV;
    if (!SSR_Raymarch(viewPos, refl, jitter, hitUV))
        return float4(0, 0, 0, 0);

    // Schlick Fresnel with metallic F0. Dielectrics: F0 = SSRBaseReflect, so
    // head-on surfaces (NPCs/walls) reflect almost nothing and only grazing
    // angles build up. Metals: F0 tends toward the surface colour, giving a
    // stronger, albedo-tinted reflection.
    float  NdotV  = saturate(dot(normal, -viewDir));
    float3 albedo = tex2D(BackBuffer, uv).rgb;                 // surface's own colour (sRGB)
    float3 f0     = lerp(SSRBaseReflect.xxx, albedo, SSRMetallic);
    float3 F      = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);

    // Metals tint the reflection by their albedo.
    float3 reflectedColor = tex2D(BackBuffer, hitUV).rgb;      // sRGB scene colour
    reflectedColor = lerp(reflectedColor, reflectedColor * albedo, SSRMetallic);

    // Fade reflections near screen edges to hide marching artifacts.
    float2 edge = smoothstep(0.0, 0.1, hitUV) * smoothstep(0.0, 0.1, 1.0 - hitUV);
    float edgeFade = edge.x * edge.y;

    float weight = saturate(SSRStrength * Luma(F) * edgeFade);
    return float4(reflectedColor, weight);
}

// Variable-radius disk blur: rougher (lower glossiness) and more distant
// reflections blur more, faking glossy-vs-mirror surfaces. Two hex rings + center.
float4 BlurSSR(float2 uv)
{
    static const float2 RING[6] =
    {
        float2( 1.0, 0.0), float2( 0.5,  0.86603), float2(-0.5,  0.86603),
        float2(-1.0, 0.0), float2(-0.5, -0.86603), float2( 0.5, -0.86603)
    };

    float rough = 1.0 - SSRGlossiness;
    float depth = GetDepth(uv);
    // radius in half-res texels: roughness sets the base, distance widens it
    float radius = rough * rough * 8.0 * (0.5 + depth * 1.5);
    radius = min(radius, 12.0);

    float2 texel = ReShade::PixelSize * 2.0; // half-res texel

    float4 sum = tex2D(SSRRawSampler, uv);   // center, weight 1
    float wsum = 1.0;

    [loop]
    for (int i = 0; i < 6; ++i)
    {
        float2 inner = RING[i] * 0.6 * radius * texel;
        float2 outer = RING[i] * 1.0 * radius * texel;
        sum += tex2D(SSRRawSampler, uv + inner) * 0.7; wsum += 0.7;
        sum += tex2D(SSRRawSampler, uv + outer) * 0.4; wsum += 0.4;
    }

    return sum / wsum;
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
    // cocPx is in pixels -> convert the per-tap step to UV space.
    float2 stepDir = dir * (cocPx / max(1.0, float(halfT))) * ReShade::PixelSize;

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
        accum += tex2D(CompositeSampler, sUV).rgb * w;
        wsum += w;
    }

    return float4(accum / max(1e-6, wsum), cocNorm);
}

// ============================
// BLOOM (progressive pyramid: soft-knee prefilter -> 13-tap downsample chain
// -> 9-tap tent upsample chain, accumulating energy at every scale).
// ============================

// Unreal-style soft-knee threshold: smooth ramp around the threshold instead of
// a hard cutoff, which avoids flicker/popping on bright edges.
float3 BloomPrefilter(float3 c)
{
    float br = max(c.r, max(c.g, c.b));
    float knee = max(BloomThreshold * BloomSoftKnee, 1e-4);
    float soft = clamp(br - BloomThreshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    float contrib = max(soft, br - BloomThreshold) / max(br, 1e-4);
    return c * contrib;
}

// 13-tap downsample (overlapping 4-sample boxes) — smooth, firefly-resistant.
float3 Downsample13(sampler s, float2 uv, float2 t)
{
    float3 a = tex2D(s, uv + t * float2(-2, -2)).rgb;
    float3 b = tex2D(s, uv + t * float2( 0, -2)).rgb;
    float3 c = tex2D(s, uv + t * float2( 2, -2)).rgb;
    float3 d = tex2D(s, uv + t * float2(-2,  0)).rgb;
    float3 e = tex2D(s, uv + t * float2( 0,  0)).rgb;
    float3 f = tex2D(s, uv + t * float2( 2,  0)).rgb;
    float3 g = tex2D(s, uv + t * float2(-2,  2)).rgb;
    float3 h = tex2D(s, uv + t * float2( 0,  2)).rgb;
    float3 i = tex2D(s, uv + t * float2( 2,  2)).rgb;
    float3 j = tex2D(s, uv + t * float2(-1, -1)).rgb;
    float3 k = tex2D(s, uv + t * float2( 1, -1)).rgb;
    float3 l = tex2D(s, uv + t * float2(-1,  1)).rgb;
    float3 m = tex2D(s, uv + t * float2( 1,  1)).rgb;
    float3 r = (j + k + l + m) * 0.125;       // center box (weight 0.5)
    r += (a + b + d + e) * 0.03125;           // four corner boxes (weight 0.125 each)
    r += (b + c + e + f) * 0.03125;
    r += (d + e + g + h) * 0.03125;
    r += (e + f + h + i) * 0.03125;
    return r;
}

// 9-tap tent upsample filter.
float3 Upsample9(sampler s, float2 uv, float2 t)
{
    float3 a = tex2D(s, uv + t * float2(-1,  1)).rgb;
    float3 b = tex2D(s, uv + t * float2( 0,  1)).rgb;
    float3 c = tex2D(s, uv + t * float2( 1,  1)).rgb;
    float3 d = tex2D(s, uv + t * float2(-1,  0)).rgb;
    float3 e = tex2D(s, uv + t * float2( 0,  0)).rgb;
    float3 f = tex2D(s, uv + t * float2( 1,  0)).rgb;
    float3 g = tex2D(s, uv + t * float2(-1, -1)).rgb;
    float3 h = tex2D(s, uv + t * float2( 0, -1)).rgb;
    float3 i = tex2D(s, uv + t * float2( 1, -1)).rgb;
    return (e * 4.0 + (b + d + f + h) * 2.0 + (a + c + g + i)) * (1.0 / 16.0);
}

float4 PS_BloomPrefilter(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float3 c = ToLinear(Downsample13(BackBuffer, uv, ReShade::PixelSize));
    return float4(BloomPrefilter(c), 1.0);
}

float4 PS_BloomDown1(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return float4(Downsample13(BloomD0s, uv, ReShade::PixelSize *  2.0), 1.0); }
float4 PS_BloomDown2(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return float4(Downsample13(BloomD1s, uv, ReShade::PixelSize *  4.0), 1.0); }
float4 PS_BloomDown3(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return float4(Downsample13(BloomD2s, uv, ReShade::PixelSize *  8.0), 1.0); }
float4 PS_BloomDown4(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return float4(Downsample13(BloomD3s, uv, ReShade::PixelSize * 16.0), 1.0); }
float4 PS_BloomDown5(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return float4(Downsample13(BloomD4s, uv, ReShade::PixelSize * 32.0), 1.0); }

// Upsample chain: each level adds its own downsample mip to the upsampled smaller mip.
float4 PS_BloomUp4(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return float4(tex2D(BloomD4s, uv).rgb + Upsample9(BloomD5s, uv, ReShade::PixelSize * 64.0 * BloomRadius), 1.0); }
float4 PS_BloomUp3(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return float4(tex2D(BloomD3s, uv).rgb + Upsample9(BloomU4s, uv, ReShade::PixelSize * 32.0 * BloomRadius), 1.0); }
float4 PS_BloomUp2(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return float4(tex2D(BloomD2s, uv).rgb + Upsample9(BloomU3s, uv, ReShade::PixelSize * 16.0 * BloomRadius), 1.0); }
float4 PS_BloomUp1(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return float4(tex2D(BloomD1s, uv).rgb + Upsample9(BloomU2s, uv, ReShade::PixelSize *  8.0 * BloomRadius), 1.0); }
float4 PS_BloomUp0(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target { return float4(tex2D(BloomD0s, uv).rgb + Upsample9(BloomU1s, uv, ReShade::PixelSize *  4.0 * BloomRadius), 1.0); }

// ============================
// LIGHT POSITION + GOD RAYS
// ============================

// Luminance-weighted lightUV of the brightest areas -> the on-screen light
// position. Temporally smoothed so it doesn't jitter. rg = UV, b = confidence.
float4 PS_LightPos(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    const int N = 16;
    float2 sumUV = 0.0;
    float  sumW  = 0.0;

    [loop]
    for (int y = 0; y < N; ++y)
    {
        [loop]
        for (int x = 0; x < N; ++x)
        {
            float2 g = (float2(x, y) + 0.5) / float(N);
            float  l = tex2Dlod(LumCurrSampler, float4(g, 0, 0)).r;
            float  w = max(l - GodrayThreshold, 0.0);
            w *= w;
            sumUV += g * w;
            sumW  += w;
        }
    }

    float2 lightUV = (sumW > 1e-5) ? sumUV / sumW : float2(0.5, 0.5);
    float  conf = saturate(sumW / float(N * N) * 8.0);

    float4 prev = tex2Dlod(LightPosPrevSampler, float4(0.5, 0.5, 0, 0));
    float  rate = saturate(AdaptSpeed * frametime * 0.001);
    if (prev.b <= 0.0) prev = float4(lightUV, conf, 1.0); // first-frame init

    float2 smUV  = lerp(prev.rg, lightUV, rate);
    float  smCnf = lerp(prev.b,  conf,     rate);
    return float4(smUV, smCnf, 1.0);
}

float4 PS_LightPosCopy(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    return tex2Dlod(LightPosSampler, float4(0.5, 0.5, 0, 0));
}

// Bright source for the shafts. Only bright pixels (the sun / bright sky) emit;
// dark geometry is zero here, so the radial accumulation is naturally broken up
// by silhouettes — occlusion-by-darkness, no fragile sky-depth assumption.
float4 PS_GodrayBright(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    if (!EnableGodrays) return float4(0.0, 0.0, 0.0, 1.0);
    float3 c = ToLinear(tex2D(BackBuffer, uv).rgb);
    float  l = max(Luma(c) - GodrayThreshold, 0.0);
    l *= l;                                                  // emphasize the brightest source (the sun) over merely-bright snow
    float  sky = pow(saturate(GetDepth(uv)), GodraySkyBias); // optional: bias emission toward the distant sky (0 = any bright pixel)
    return float4(c * l * sky, 1.0);
}

// Radial scatter from the light position (GPU-Gems volumetric light shafts).
float4 PS_Godray(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    if (!EnableGodrays) return float4(0.0, 0.0, 0.0, 1.0);

    float2 lightPos = tex2Dlod(LightPosSampler, float4(0.5, 0.5, 0, 0)).rg;

    const int SAMPLES = 48;
    float2 delta = (uv - lightPos) * (GodrayDensity / float(SAMPLES));
    float2 coord = uv;
    float  illum = 1.0;
    float3 sum = 0.0;

    [loop]
    for (int i = 0; i < SAMPLES; ++i)
    {
        coord -= delta;
        sum += tex2D(GodrayBrightSampler, coord).rgb * illum;
        illum *= GodrayDecay;
    }

    // Normalize the decay-weighted sum (geometric series ~ 1/(1-decay)) so the
    // brightness stays stable across decay/sample settings.
    return float4(sum * (1.0 - GodrayDecay), 1.0);
}

// Sample the bright source with a per-channel offset along `dir` -> chromatic
// fringing on the flare (the coloured edges real lenses produce).
float3 LensFlareSample(float2 uv, float2 dir)
{
    float3 ca = float3(-LensFlareCA, 0.0, LensFlareCA) * ReShade::PixelSize.x * 4.0;
    return float3(
        tex2D(BloomD1s, uv + dir * ca.r).r,
        tex2D(BloomD1s, uv + dir * ca.g).g,
        tex2D(BloomD1s, uv + dir * ca.b).b);
}

// Screen-space lens flare (Chapman): ghosts mirrored through the screen centre
// plus a halo, sourced from the bloom prefilter (so it self-gates on brightness).
float4 PS_LensFlare(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    if (!EnableLensFlare) return float4(0.0, 0.0, 0.0, 1.0);

    float2 texcoord = 1.0 - uv;                                  // point-reflect through centre
    float2 ghostVec = (float2(0.5, 0.5) - texcoord) * LensFlareDispersal;
    float2 dir = ghostVec / (length(ghostVec) + 1e-4);

    int ghosts = clamp(LensFlareGhosts, 1, 8);
    float3 result = 0.0;

    [loop]
    for (int i = 0; i < 8; ++i)
    {
        if (i >= ghosts) break;
        float2 offset = frac(texcoord + ghostVec * float(i));
        float  w = length(float2(0.5, 0.5) - offset) / length(float2(0.5, 0.5));
        w = pow(saturate(1.0 - w), 10.0);                       // brightest near centre
        result += LensFlareSample(offset, dir) * w;
    }

    // Halo: a soft ring at a fixed offset toward the centre.
    float2 haloVec = dir * LensFlareHalo;
    float2 hoff = texcoord + haloVec;
    float  wh = length(float2(0.5, 0.5) - frac(hoff)) / length(float2(0.5, 0.5));
    wh = pow(saturate(1.0 - wh), 5.0);
    result += LensFlareSample(hoff, dir) * wh;

    return float4(result, 1.0);
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
// Local contrast ("clarity"): boost each pixel's luminance against its large-
// radius local average, weighted to midtones. Adds a crisp, near-HDR pop to
// textures that global contrast can't, without flattening colour. `focus`
// suppresses it in out-of-focus (DoF) regions.
float3 ApplyClarity(float2 uv, float3 c, float focus)
{
    static const float2 RING[6] =
    {
        float2( 1.0, 0.0), float2( 0.5,  0.86603), float2(-0.5,  0.86603),
        float2(-1.0, 0.0), float2(-0.5, -0.86603), float2( 0.5, -0.86603)
    };

    float2 rad = ReShade::PixelSize * ClarityRadius;
    float center = Luma(tex2D(CompositeSampler, uv).rgb);
    float sum = center;
    float wsum = 1.0;

    [unroll]
    for (int i = 0; i < 6; ++i)
    {
        sum += Luma(tex2D(CompositeSampler, uv + RING[i] * rad * 0.55).rgb) * 0.7;
        sum += Luma(tex2D(CompositeSampler, uv + RING[i] * rad).rgb) * 0.4;
        wsum += 1.1;
    }

    float detail = center - sum / wsum;
    float mid = 4.0 * saturate(center) * (1.0 - saturate(center)); // midtone weight
    return saturate(c + detail * ClarityAmount * mid * focus);
}

// Contrast Adaptive Sharpening (AMD FidelityFX CAS, simplified): the sharpening
// amount adapts to local contrast, so flat detail is sharpened but strong edges
// are spared the haloes/crunch of a plain unsharp mask. Returned as a detail add
// so the DoF blur is preserved; `focus` (0..1) skips out-of-focus regions.
float3 SharpenPass(float2 uv, float3 c, float focus)
{
    float2 t = ReShade::PixelSize;
    float3 a = tex2D(CompositeSampler, uv + t * float2(-1, -1)).rgb;
    float3 b = tex2D(CompositeSampler, uv + t * float2( 0, -1)).rgb;
    float3 cc= tex2D(CompositeSampler, uv + t * float2( 1, -1)).rgb;
    float3 d = tex2D(CompositeSampler, uv + t * float2(-1,  0)).rgb;
    float3 e = tex2D(CompositeSampler, uv).rgb;
    float3 f = tex2D(CompositeSampler, uv + t * float2( 1,  0)).rgb;
    float3 g = tex2D(CompositeSampler, uv + t * float2(-1,  1)).rgb;
    float3 h = tex2D(CompositeSampler, uv + t * float2( 0,  1)).rgb;
    float3 i = tex2D(CompositeSampler, uv + t * float2( 1,  1)).rgb;

    float3 mn = min(min(min(a, b), min(cc, d)), min(min(e, f), min(min(g, h), i)));
    float3 mx = max(max(max(a, b), max(cc, d)), max(max(e, f), max(max(g, h), i)));

    float3 amp = sqrt(saturate(min(mn, 1.0 - mx) / max(mx, 1e-4)));
    float3 w = -amp * lerp(0.0, 0.2, saturate(Sharpness)); // cross weights (negative = sharpen)
    float3 cas = (e + (b + d + f + h) * w) / (1.0 + 4.0 * w);

    return saturate(c + (cas - e) * focus); // add the sharpening detail, keep DoF
}

// Film grain: luminance-aware (peaks in midtones like real film, fades in
// shadows/highlights) using blue noise when enabled. GrainSize controls coarseness.
float3 ApplyGrain(float3 c, float2 uv)
{
    float n = BlueNoise(uv / max(GrainSize, 1.0)) - 0.5;
    float l = Luma(c);
    float response = 4.0 * l * (1.0 - l); // 0 at black/white, 1 at mid-grey
    return saturate(c + n * 2.0 * GrainAmount * response);
}

// Triangular-PDF dither (~1 LSB at 8-bit). Blue noise when enabled gives the
// least-perceptible result. Breaks up banding on smooth gradients (night sky).
float3 ApplyDither(float3 c, float2 uv)
{
    float r1 = BlueNoise(uv);
    float r2 = BlueNoise(uv + 0.5);
    float t = r1 + r2 - 1.0; // triangular in [-1, 1]
    return c + t * (1.0 / 255.0);
}

// ============================
// TONEMAP (selectable operator) — each returns linear; ToSRGB is applied after.
// ============================
float3 Tonemap_ACES(float3 x) // Narkowicz ACES approximation
{
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float3 Hable(float3 x)
{
    const float A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}
float3 Tonemap_Hable(float3 c) // Uncharted 2 filmic
{
    float3 curr = Hable(c * 2.0);
    float3 whiteScale = 1.0 / Hable(TonemapWhite.xxx);
    return saturate(curr * whiteScale);
}

float3 Tonemap_Reinhard(float3 c) // extended Reinhard with white point
{
    return saturate((c * (1.0 + c / (TonemapWhite * TonemapWhite))) / (1.0 + c));
}

// AgX (minimal implementation, Benjamin Wrensch / Troy Sobotka). The sigmoid
// output is display-encoded; the pow(2.2) here cancels with the later ToSRGB so
// AgX displays correctly through the shared encode path.
float3 agxContrast(float3 x)
{
    float3 x2 = x * x;
    float3 x4 = x2 * x2;
    return  15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4
          - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
float3 Tonemap_AgX(float3 val)
{
    const float3x3 agx_mat = float3x3(
        0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772, 0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    const float3x3 agx_mat_inv = float3x3(
         1.19687900512017,  -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417,   -0.0980434501171241,
        -0.0990297440797205,-0.0989611768448433,  1.15107367264116);
    const float min_ev = -12.47393, max_ev = 4.026069;

    val = mul(val, agx_mat);                          // input transform
    val = clamp(log2(max(val, 1e-10)), min_ev, max_ev);
    val = (val - min_ev) / (max_ev - min_ev);
    val = agxContrast(val);                           // sigmoid
    val = mul(val, agx_mat_inv);                      // output transform
    return pow(max(val, 0.0), 2.2);                   // -> linear (cancels ToSRGB)
}

float3 ApplyTonemap(float3 c)
{
    if      (TonemapOperator == 1) c = Tonemap_AgX(c);
    else if (TonemapOperator == 2) c = Tonemap_Hable(c);
    else if (TonemapOperator == 3) c = Tonemap_Reinhard(c);
    else                           c = Tonemap_ACES(c);
    return ToSRGB(c);
}

// Grading in display space (neutral at defaults).
float3 ApplyGrade(float3 c)
{
    // White balance: temperature (R-B axis) and tint (G axis), normalized by its
    // own luminance so neutral grey keeps its brightness (colour shift only).
    float t = WhiteTemp / 100.0;
    float n = WhiteTint / 100.0;
    float3 wb = float3(1.0 + 0.3 * t, 1.0 - 0.3 * n, 1.0 - 0.3 * t);
    wb /= max(Luma(wb), 1e-4);
    c *= wb;

    // Luminance contrast around mid-grey via an additive luma-delta shift (same
    // offset on every channel). Flattens/expands the tonal range without the
    // per-channel clipping of a multiplicative scale (which flashed saturated
    // pixels) and without collapsing colour to grey.
    float l  = Luma(c);
    float lc = (l - 0.5) * Contrast + 0.5;
    c += (lc - l);

    // Saturation
    c = lerp(Luma(c).xxx, c, Saturation);
    return saturate(c);
}

// ============================
// EFFECT PASS ENTRY POINTS
// ============================
float4 PS_AOCompute(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float ao = 1.0;
    if (EnableAO)
        ao = (AOMode == 0) ? ComputeAO(uv) : ComputeGTAO(uv);
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
    if (!EnableDoF) return float4(tex2D(CompositeSampler, uv).rgb, 0.0);
    return DoF_Gather(uv, float2(1.0, 0.0));
}
float4 PS_DoFGatherV(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    if (!EnableDoF) return float4(tex2D(DoFRawSampler, uv).rgb, 0.0);
    // Vertical gather over the horizontally-pre-blurred buffer.
    float cocNorm = tex2D(DoFRawSampler, uv).a;
    float cocPx = cocNorm * MaxCoCRadius;
    int taps = DOF_MAX_TAPS;
    int halfT = taps / 2;
    // cocPx is in pixels -> convert the per-tap step to UV space.
    float2 stepDir = float2(0.0, 1.0) * (cocPx / max(1.0, float(halfT))) * ReShade::PixelSize;

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
// COMPOSITE PASS — full processed scene (linear effects -> tonemap -> sRGB).
// Written to CompositeTex so DoF can blur the finished image, not the raw scene.
// ============================
float4 PS_Composite(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
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
        c += tex2D(BloomU0s, uv).rgb * BloomStrength;

    // God rays (linear, additive). The occluder-masked bright source self-gates
    // (no bright sky -> no shafts), so no extra confidence gate is needed.
    if (EnableGodrays)
        c += tex2D(GodraySampler, uv).rgb * GodrayIntensity;

    // Lens flare (linear, additive), faded toward the screen edges.
    if (EnableLensFlare)
    {
        float2 dc = uv - 0.5;
        float falloff = saturate(1.0 - dot(dc, dc) * 1.5);
        c += tex2D(LensFlareSampler, uv).rgb * LensFlareIntensity * falloff;
    }

    // Tonemap + encode (selectable operator), then grading
    c = ApplyTonemap(c);
    c = ApplyGrade(c);

    return float4(saturate(c), 1.0);
}

// ============================
// MAIN PASS — DoF over the composite, then sharpen / grain / dither / output.
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
        if (DebugMode == 8) { return float4(tex2D(BloomU0s, uv).rgb, 1.0); }
        if (DebugMode == 9) { return float4(tex2D(GodraySampler, uv).rgb * GodrayIntensity, 1.0); }
        if (DebugMode == 10) { return float4(tex2D(LensFlareSampler, uv).rgb * LensFlareIntensity, 1.0); }
    }

    float3 c = tex2D(CompositeSampler, uv).rgb;

    // Depth of Field: blend the sharp composite toward the DoF-blurred composite.
    float focus = 1.0;
    if (EnableDoF)
    {
        float4 dof = tex2D(DoFBlurSampler, uv);
        float coc = saturate(dof.a);
        c = lerp(c, dof.rgb, coc);
        focus = 1.0 - coc; // don't sharpen out-of-focus regions
    }

    // Local contrast, then sharpen & grain (display space)
    if (EnableClarity) c = ApplyClarity(uv, c, focus);
    if (EnableSharpen) c = SharpenPass(uv, c, focus);
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
    pass LightPos       { VertexShader = PostProcessVS; PixelShader = PS_LightPos;      RenderTarget = LightPosTex; }
    pass LightPosCopy   { VertexShader = PostProcessVS; PixelShader = PS_LightPosCopy;  RenderTarget = LightPosPrevTex; }

    pass AOCompute      { VertexShader = PostProcessVS; PixelShader = PS_AOCompute;      RenderTarget = AORawTex; }
    pass AOBlur         { VertexShader = PostProcessVS; PixelShader = PS_AOBlur;         RenderTarget = AOBlurTex; }

    pass ContactCompute { VertexShader = PostProcessVS; PixelShader = PS_ContactCompute; RenderTarget = ContactRawTex; }
    pass ContactBlur    { VertexShader = PostProcessVS; PixelShader = PS_ContactBlur;    RenderTarget = ContactBlurTex; }

    pass SSRCompute     { VertexShader = PostProcessVS; PixelShader = PS_SSRCompute;     RenderTarget = SSRRawTex; }
    pass SSRBlur        { VertexShader = PostProcessVS; PixelShader = PS_SSRBlur;        RenderTarget = SSRBlurTex; }

    pass BloomPrefilter { VertexShader = PostProcessVS; PixelShader = PS_BloomPrefilter; RenderTarget = BloomD0; }
    pass BloomDown1     { VertexShader = PostProcessVS; PixelShader = PS_BloomDown1;     RenderTarget = BloomD1; }
    pass BloomDown2     { VertexShader = PostProcessVS; PixelShader = PS_BloomDown2;     RenderTarget = BloomD2; }
    pass BloomDown3     { VertexShader = PostProcessVS; PixelShader = PS_BloomDown3;     RenderTarget = BloomD3; }
    pass BloomDown4     { VertexShader = PostProcessVS; PixelShader = PS_BloomDown4;     RenderTarget = BloomD4; }
    pass BloomDown5     { VertexShader = PostProcessVS; PixelShader = PS_BloomDown5;     RenderTarget = BloomD5; }
    pass BloomUp4       { VertexShader = PostProcessVS; PixelShader = PS_BloomUp4;       RenderTarget = BloomU4; }
    pass BloomUp3       { VertexShader = PostProcessVS; PixelShader = PS_BloomUp3;       RenderTarget = BloomU3; }
    pass BloomUp2       { VertexShader = PostProcessVS; PixelShader = PS_BloomUp2;       RenderTarget = BloomU2; }
    pass BloomUp1       { VertexShader = PostProcessVS; PixelShader = PS_BloomUp1;       RenderTarget = BloomU1; }
    pass BloomUp0       { VertexShader = PostProcessVS; PixelShader = PS_BloomUp0;       RenderTarget = BloomU0; }

    pass GodrayBright   { VertexShader = PostProcessVS; PixelShader = PS_GodrayBright;   RenderTarget = GodrayBrightTex; }
    pass Godray         { VertexShader = PostProcessVS; PixelShader = PS_Godray;         RenderTarget = GodrayTex; }

    pass LensFlare      { VertexShader = PostProcessVS; PixelShader = PS_LensFlare;      RenderTarget = LensFlareTex; }

    pass Composite      { VertexShader = PostProcessVS; PixelShader = PS_Composite;      RenderTarget = CompositeTex; }

    pass DoFGatherH     { VertexShader = PostProcessVS; PixelShader = PS_DoFGatherH;     RenderTarget = DoFRawTex; }
    pass DoFGatherV     { VertexShader = PostProcessVS; PixelShader = PS_DoFGatherV;     RenderTarget = DoFBlurTex; }

    pass Main           { VertexShader = PostProcessVS; PixelShader = PS_Photorealism; }
}
