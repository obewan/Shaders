//===========================================================================
// SKYRIM REALISTIC PIPELINE — ReshadeTrueLight
// Author: Obewan (https://github.com/obewan)
// Version: 1.2.0
// Requirement:
//     - bluenoise.png in reshade-shaders/textures (only if Use Blue Noise = true)
//     - a depth buffer correctly set (check it using the DisplayDepth shader)
//     - a separate AA shader (e.g. CShade DLAA or SMAA), ordered before this one
//
// Single-frame pipeline (no temporal reprojection): ReShade can't supply per-
// frame view/projection matrices, so all effects are single-frame with spatial
// bilateral filtering. The only cross-frame state is the 1x1 luminance/light
// adaptation ping-pong (no reprojection needed).
//
//---------------------------------------------------------------------------
// v1.2.0 - distance pass
//
// New
//   - Distant Shading. AO could never fix "the distance is too bright": its
//     radius is a world-space sphere projected to screen, so its footprint
//     shrinks with 1/z and past mid-distance it only darkens contact-scale
//     creases. Meanwhile Skyrim's own shadow map ends well before the horizon,
//     so far LOD gets sun + ambient with no occlusion and reads as a flat
//     bright cut-out. This shades it by the angle to the tracked light, over a
//     deliberately wide-baseline normal (per-pixel parallax at range is below
//     the depth buffer's precision, so a 1px normal there is just noise), and
//     ramps in with distance so the engine-shadowed near field is untouched.
//     Adds a flat Distant Ambient Dim that needs no light direction, for
//     overcast and night. Debug View > Distant Shading previews the term.
//
//   - Sharpen Mode. The CAS path only ever backs off, so faint texture stays
//     faint. The new Detail mode (after Marty McFly / Pascal Gilcher's qUINT
//     DELCS) divides the high-pass by local RMS contrast instead, normalising
//     amplitude so weak texture is boosted to the prominence of strong texture
//     while outlines are pushed down. Masks depth edges, log-compresses the
//     tail and composites as an overlay. Now the default.
//   - Distant Shading gained a Haze Fadeout, and its Highlight Protection now
//     tests display-referred brightness with the knee capped below 1.0 (at 1.0
//     the smoothstep was degenerate and quietly disabled protection).
//   - Sharpening is no longer gated on normalised CoC. Because CoC carries no
//     notion of how strong the blur actually is, any preset focusing past the
//     near field zeroed the sharpener there even when the blur was a pixel
//     wide - the sharpener was inheriting the DoF's focal distance as its own
//     working range. Now measured against the real blur radius, with a new
//     Defocus Suppression (px) control (0 decouples the two entirely).
//   - Detail sharpen's depth-edge mask no longer inherits the near-field radius
//     boost. It scaled with proximity and smothered the magnified near ground,
//     which is the one place the boost existed to help.
//   - Distant Shading gained Highlight Protection. Sunlit snowcaps were losing
//     their glare: the pass fakes a shadow test, but a peak the engine drew
//     that bright is one the sun is genuinely hitting, and a depth-derived
//     normal on distant LOD is too crude to overrule that.
//
// Fixed
//   - Sharpen read as anti-aliasing rather than as a texture enhancer. Its taps
//     were fixed at one pixel, making it a Nyquist-frequency Laplacian that can
//     only amplify pixel-scale edges - i.e. it was re-crisping the edges the
//     recommended pre-pass AA had just resolved. Added Sharpen Radius (px),
//     defaulting to 2, to move the response into the 2-6 px band where surface
//     texture actually lives. That radius is now also scaled by 1/distance
//     (Near Detail Boost): a fixed pixel radius bites on minified distant
//     geometry but slides over the magnified road at your feet, which is the
//     opposite of what a sharpen pass is for.
//   - Sharpen did nothing at any usable slider value. The CAS gain is
//     1/(1 + 4w) and only takes off as w approaches -0.25, but the slider was
//     mapped over lerp(0, 0.2, Sharpness) — so the whole useful band sat above
//     ~0.62 and the 0.25 default moved a midtone pixel by about one 8-bit
//     level. The kernel now runs at AMD's full peak and the slider scales the
//     detail it produces, which makes the response linear and 0 still off.
//
//---------------------------------------------------------------------------
// v1.1.0 - lighting pass
//
// New
//   - Indirect light. The AO loop now also gathers each neighbour's lit colour
//     using the SAME visibility weight it used for occlusion, so light returns
//     exactly where the occlusion took it away. Added back modulated by the
//     receiver's own colour (a stand-in for albedo), so a red wall bleeds red
//     and dark surfaces stay dark. AO targets are RGBA16F now: rgb = bounce,
//     a = occlusion. Not true GI - one screen-space bounce off the lit image.
//   - Light source tracking, as its own concept and UI category. Two passes:
//     find the brightest cell, then take the centroid of only the cells near
//     that peak. The old single-pass luminance centroid landed BETWEEN the sun
//     and a bright snowfield, i.e. on neither. Confidence now means "how bright
//     is the brightest thing on screen" and gates god rays, lens flare and fog
//     glow, so a windowless dungeon stops growing sun shafts.
//   - Contact shadows march toward that tracked light instead of a hardcoded
//     vector, so they swing with the sun through the day. Falls back to the old
//     fixed direction on its own when no confident light is on screen.
//
// Fixed
//   - God rays banded (48 fixed radii, no per-pixel offset) and streaked (the
//     CLAMP sampler repeated border pixels into hard smears when the light sat
//     near or off screen). Now dithered with interleaved gradient noise, and
//     off-buffer taps are dropped.
//   - Bloom ignored exposure: it was extracted from the raw scene and added
//     post-exposure, so it quietly weakened at night as the eye adapted upward.
//     Thresholded after exposure now, so the setting means one thing at any
//     hour. Prefilter also Karis-weighted, so one blown pixel can't flicker the
//     whole pyramid.
//   - Exposure metered off 16 bilinear taps, roughly a sixteenth of the frame,
//     and leaned on temporal smoothing to hide the wobble. 64 taps now, with
//     optional centre weighting and log-average metering.
//   - Fog sun glow used a raw UV distance, so it was an ellipse stretched
//     horizontally on any non-square frame.
//
// Retuning note: Bloom Threshold and God Ray Threshold are measured AFTER
// exposure as of this version, and log-average metering defaults on - existing
// presets may want a nudge to Exposure Key and those two thresholds.
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

// Interleaved gradient noise (Jimenez) in [0,1). Preferred over a white-noise
// hash for staggering the start of a ray march: its error is spread evenly over
// a tight neighbourhood, so a small blur or a bilinear upsample resolves it,
// where white noise leaves speckle behind. Static, like every other jitter here.
float IGN(float2 px)
{
    return frac(52.9829189 * frac(dot(px, float2(0.06711056, 0.00583715))));
}

// Scene exposure gain. Defined here rather than beside the adaptation passes
// because the bloom prefilter, the god ray source and the light tracker all run
// before them and must threshold in the SAME post-exposure space. Otherwise
// "bright enough to bloom" drifts away from "bright enough to be the sun" as the
// eye adapts, and bloom silently weakens at night while the gain climbs.
float ComputeExposure()
{
    if (!EnableAutoExposure)
        return exp2(ManualExposure);

    float avg = max(tex2Dlod(AdaptSampler, float4(0.5, 0.5, 0, 0)).r, 1e-4);
    return clamp(ExposureKey / avg, ExposureMin, ExposureMax);
}

float3 ToLinear(float3 c) { return pow(abs(c), 2.2); }
float3 ToSRGB(float3 c)   { return pow(abs(c), 1.0 / 2.2); }

// STATIC per-pixel jitter in [-0.5, 0.5]. Uses blue noise when available.
//
// Intentionally NOT animated: like AO and contact shadows, SSR has no temporal
// accumulation to average a per-frame jitter, so animating it would just make the
// reflections crawl/shimmer frame-to-frame. The spatial BlurSSR pass resolves the
// static dither instead. (Grain/dither still animate their VALUE — see BlueNoise.)
float2 JitterUV(float2 uv)
{
    if (UseBlueNoise)
    {
        float2 bn = tex2D(BlueNoiseSampler, frac(uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT) / 64.0)).rg;
        return bn - 0.5;
    }
    float2 p = uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float n  = Hash21(p);
    float n2 = Hash11(n);
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
// `px` is the sampling baseline. One pixel is right for contact-scale work, but
// it is useless at range: a single pixel of parallax on a far surface is smaller
// than the depth buffer's precision there, so the estimate degenerates into
// noise. A wider baseline recovers the LOW-frequency form instead — the shape of
// a hillside rather than the LOD triangle under it — which is what DistantShade
// needs. Hence the split.
float3 EstimateNormalAt(float2 uv, float2 px)
{
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

float3 EstimateNormal(float2 uv) { return EstimateNormalAt(uv, ReShade::PixelSize); }

// Convert a world-space radius at viewPos into a screen-space UV radius.
float2 ViewRadiusToUV(float3 viewPos, float worldRadius)
{
    float z = max(-viewPos.z, 1e-4);
    float t = tan(radians(CameraFovY) * 0.5);
    float uvr = (worldRadius / (z * t)) * 0.5;
    return float2(uvr / ReShade::AspectRatio, uvr);
}

// View-space direction from the scene toward the estimated key light (surface ->
// light). ReShade has no sun vector, but the light's screen position plus the FOV
// gives the direction it lies in, which is all a contact shadow needs. Blends
// back to a fixed up/behind key light when nothing bright is confidently on
// screen (interiors, overcast nights) so shadows don't swing at random.
float3 GetSunDirView()
{
    static const float3 FALLBACK = float3(0.0, 0.70710678, 0.70710678);

    float4 lp = tex2Dlod(LightPosSampler, float4(0.5, 0.5, 0, 0));

    // Same unprojection convention as ReconstructViewPos: forward = -Z, y flipped.
    float2 ndc = lp.rg * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float t = tan(radians(CameraFovY) * 0.5);
    float3 dir = normalize(float3(ndc.x * ReShade::AspectRatio * t, ndc.y * t, -1.0));

    float track = saturate(LightTrackAmount * lp.b);
    return normalize(lerp(FALLBACK, dir, track));
}

// How much to trust the tracked light this frame: a 0..1 multiplier for the
// effects that only make sense when a real light is actually on screen.
float LightGate()
{
    float conf = tex2Dlod(LightPosSampler, float4(0.5, 0.5, 0, 0)).b;
    return lerp(1.0, saturate(conf), LightConfidenceGate);
}

// ============================
// DISTANT SHADING
// ============================
// Why this exists, given there is already an AO pass: AO is a LOCAL term. Its
// radius is a world-space sphere projected to screen, so its footprint shrinks
// with 1/z — past mid-distance it covers a pixel or two and can only ever darken
// contact-scale creases. It cannot produce the thing that is actually missing at
// range, which is a shadowed mountain face. Skyrim itself stops helping too: its
// shadow map ends well before the horizon, so distant LOD is lit by sun plus
// ambient with no occlusion at all, and reads as a flat bright cut-out.
//
// So this stands in for the shadowing the engine gave up on: a wrapped Lambert
// term against the tracked light, ramped in with distance so anything the engine
// still shadows is left alone, plus a flat ambient dim for the aerial falloff.
// It is a cheat — there is no shadow test — but the cue that sells depth is
// large-scale form, and N.L over a low-frequency normal delivers exactly that.
float DistantShade(float2 uv, float3 lit)
{
    float d = GetDepth(uv);
    if (d <= 0.0 || d >= 1.0) return 1.0; // sky: leave it alone

    // Ramp in with distance. Below the start the engine's own shadows are still
    // running and doubling up on them would just crush the near field.
    float dist = d * CameraFar;
    float t = saturate((dist - DistantShadeStart) / max(1e-4, DistantShadeFull - DistantShadeStart));

    // ...then fade back OUT into the haze. Past a point, most of what reaches
    // the camera from a mountain is in-scattered atmosphere rather than light
    // off rock, and in-scattering is additive: multiplying it down does not
    // shade the mountain, it dims the air in front of it. That reads as grime on
    // the horizon, not as form — which is why a term that looks right at mid
    // range can make the far peaks merely dark. Fading out where the air takes
    // over keeps the shading on the surfaces that still have surface to shade.
    if (DistantHazeFade > 0.0)
        t *= 1.0 - saturate((dist - DistantHazeFade) / max(CameraFar - DistantHazeFade, 1e-4));

    if (t <= 0.0) return 1.0;

    // Wide baseline: the point is the shape of the landform, not its LOD facets.
    float3 n = EstimateNormalAt(uv, ReShade::PixelSize * DistantShadeScale);
    float3 L = GetSunDirView(); // surface -> light

    // Wrapped diffuse: the terminator softens and the unlit side settles at
    // ambient instead of black, which is what a hazy distance actually does.
    float w = saturate(DistantShadeWrap);
    float ndl = saturate((dot(n, L) + w) / (1.0 + w));

    // The directional half is only as trustworthy as the light estimate, so it
    // rides the same confidence gate as god rays and lens flare. The ambient dim
    // does not — it needs no direction, so it keeps working in an overcast or
    // sunless scene where the tracker has nothing to lock onto.
    float directional = lerp(1.0 - DistantShadeStrength * LightGate(), 1.0, ndl);
    float ambient     = 1.0 - DistantAmbientDim;
    float shade = lerp(1.0, directional * ambient, t);

    // Spare what is already bright. This term is a stand-in for a shadow test,
    // and a surface the engine drew that hot is one the sun is hitting — sunlit
    // snow on a peak, most obviously. Darkening it is not "restoring form", it
    // is deleting the highlight, and a depth-derived normal on a distant LOD is
    // far too crude to be trusted over the engine's own verdict on that pixel.
    // So brightness wins the argument: the shading works the duller rock and
    // leaves the snow its glare.
    // Tested display-referred, not on the linear value: `lit` is pre-exposure
    // linear, where a mid-grey rock reads about 0.2 and nothing but a specular
    // hit approaches 1.0, so a knee set by eye against the screen protected far
    // less than it looked like it would. The knee is also capped below 1.0 —
    // at 1.0 the smoothstep is degenerate and silently disables the whole term.
    float lum  = pow(saturate(Luma(lit)), 1.0 / 2.2);
    float knee = min(DistantHighlightKnee, 0.9);
    float protect = smoothstep(knee, 1.0, lum) * DistantHighlightProtect;
    return lerp(shade, 1.0, saturate(protect));
}

// ============================
// AO + INDIRECT BOUNCE (half-res compute + bilateral upsample)
// Returns rgb = bounced radiance, a = occlusion.
// ============================
float4 ComputeAO(float2 uv)
{
    float d = GetDepth(uv);
    if (d <= 0.0 || d >= 1.0) return float4(0, 0, 0, 1);

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

    float  occlusion = 0.0;
    float3 bounceSum = 0.0;   // visibility-weighted neighbour radiance
    float  bounceW   = 0.0;

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
        float vis = saturate(nd) * atten;
        occlusion += vis;

        // Indirect bounce: a neighbour that occludes us is also facing us, so it
        // reflects its own lit colour back. Weighting it by the SAME visibility
        // term means the light returns exactly where the occlusion took it away.
        if (EnableGI)
        {
            bounceSum += ToLinear(tex2Dlod(BackBuffer, float4(sampleUV, 0, 0)).rgb) * vis;
            bounceW   += vis;
        }
    }

    occlusion /= float(samples);
    float ao = pow(saturate(1.0 - occlusion * AOStrength), AOPower);

    // Average bounce colour scaled by how occluded we are: a pixel that sees
    // nothing gets no bleed, a deep crevice gets the most.
    float3 bounce = (bounceW > 1e-5) ? (bounceSum / bounceW) * occlusion : float3(0, 0, 0);

    // Fade AO (and the bounce with it) out at distance — far depth has poor
    // precision and produces noise.
    float fade = saturate((d - AOFadeStart) / max(1e-4, AOFadeEnd - AOFadeStart));
    return float4(bounce * (1.0 - fade), lerp(ao, 1.0, fade));
}

// GTAO/HBAO-style: march along screen-space slices and track the highest
// occluder (horizon) on each side, instead of point-sampling a disk. More
// accurate falloff and contact darkening; heavier than SSAO.
float4 ComputeGTAO(float2 uv)
{
    float d = GetDepth(uv);
    if (d <= 0.0 || d >= 1.0) return float4(0, 0, 0, 1);

    float3 p = ReconstructViewPos(uv);
    float3 n = EstimateNormal(uv);

    float effRadius = AORadius + max(-p.z, 0.0) * AODistantRadius;
    float2 radiusUV = ViewRadiusToUV(p, effRadius);
    float rnd = Hash21(uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT)); // static slice rotation

    const int SLICES = 4;
    const int STEPS  = 4;
    float  occlusion = 0.0;
    float3 bounceSum = 0.0;
    float  bounceW   = 0.0;

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
                {
                    float e = (dot(n, v) / l - AOBias) * saturate(1.0 - l / effRadius);
                    hPlus = max(hPlus, e);
                    if (EnableGI)
                    {
                        float w = saturate(e);
                        bounceSum += ToLinear(tex2Dlod(BackBuffer, float4(uvA, 0, 0)).rgb) * w;
                        bounceW   += w;
                    }
                }
            }

            float2 uvB = uv - off;
            float sdB = GetDepth(uvB);
            if (sdB > 0.0 && sdB < 1.0)
            {
                float3 v = ReconstructViewPos(uvB) - p;
                float l = length(v);
                if (l > 1e-4)
                {
                    float e = (dot(n, v) / l - AOBias) * saturate(1.0 - l / effRadius);
                    hMinus = max(hMinus, e);
                    if (EnableGI)
                    {
                        float w = saturate(e);
                        bounceSum += ToLinear(tex2Dlod(BackBuffer, float4(uvB, 0, 0)).rgb) * w;
                        bounceW   += w;
                    }
                }
            }
        }

        occlusion += (saturate(hPlus) + saturate(hMinus)) * 0.5;
    }

    occlusion /= float(SLICES);
    float ao = pow(saturate(1.0 - occlusion * AOStrength), AOPower);

    // Same normalised form as the SSAO path (average bounce colour * occlusion),
    // so switching AO mode doesn't change how strong the bleed reads.
    float3 bounce = (bounceW > 1e-5) ? (bounceSum / bounceW) * occlusion : float3(0, 0, 0);

    // Fade AO (and the bounce with it) out at distance — far depth has poor
    // precision and produces noise.
    float fade = saturate((d - AOFadeStart) / max(1e-4, AOFadeEnd - AOFadeStart));
    return float4(bounce * (1.0 - fade), lerp(ao, 1.0, fade));
}

// Depth-aware blur / upsample (reads the half-res AO target). Wide 7x7 Gaussian
// bilateral so the per-pixel rotated dither resolves to smooth AO.
float4 BlurAO(float2 uv)
{
    float centerD = GetDepth(uv);
    float2 step = ReShade::PixelSize * 2.0; // half-res texel

    float4 sum = 0.0;
    float wsum = 0.0;

    [loop]
    for (int y = -3; y <= 3; ++y)
    {
        [loop]
        for (int x = -3; x <= 3; ++x)
        {
            float2 o = float2(x, y);
            float2 sUV = uv + o * step;
            float4 v = tex2D(AORawSampler, sUV);
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

    // Direction the light travels (light -> surface), from the tracked on-screen
    // sun. Contact shadows now swing with the sun through the day instead of
    // always falling the same way.
    float3 L = -GetSunDirView();
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

// Karis average weight: dims a tap in proportion to its own brightness, so one
// blown-out pixel can't dominate (and flicker) the whole bloom pyramid.
float KarisWeight(float3 c) { return 1.0 / (1.0 + Luma(c)); }

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

// Same 13 taps as Downsample13, but each of the five overlapping boxes is
// weighted by its Karis average. Only worth the extra maths on the first
// downsample, where the full-res fireflies actually live.
float3 Downsample13Karis(sampler s, float2 uv, float2 t)
{
    float3 a = ToLinear(tex2D(s, uv + t * float2(-2, -2)).rgb);
    float3 b = ToLinear(tex2D(s, uv + t * float2( 0, -2)).rgb);
    float3 c = ToLinear(tex2D(s, uv + t * float2( 2, -2)).rgb);
    float3 d = ToLinear(tex2D(s, uv + t * float2(-2,  0)).rgb);
    float3 e = ToLinear(tex2D(s, uv + t * float2( 0,  0)).rgb);
    float3 f = ToLinear(tex2D(s, uv + t * float2( 2,  0)).rgb);
    float3 g = ToLinear(tex2D(s, uv + t * float2(-2,  2)).rgb);
    float3 h = ToLinear(tex2D(s, uv + t * float2( 0,  2)).rgb);
    float3 i = ToLinear(tex2D(s, uv + t * float2( 2,  2)).rgb);
    float3 j = ToLinear(tex2D(s, uv + t * float2(-1, -1)).rgb);
    float3 k = ToLinear(tex2D(s, uv + t * float2( 1, -1)).rgb);
    float3 l = ToLinear(tex2D(s, uv + t * float2(-1,  1)).rgb);
    float3 m = ToLinear(tex2D(s, uv + t * float2( 1,  1)).rgb);

    float3 b0 = (j + k + l + m) * 0.25;   // centre box
    float3 b1 = (a + b + d + e) * 0.25;
    float3 b2 = (b + c + e + f) * 0.25;
    float3 b3 = (d + e + g + h) * 0.25;
    float3 b4 = (e + f + h + i) * 0.25;

    float w0 = KarisWeight(b0) * 0.5;     // same box weights as Downsample13
    float w1 = KarisWeight(b1) * 0.125;
    float w2 = KarisWeight(b2) * 0.125;
    float w3 = KarisWeight(b3) * 0.125;
    float w4 = KarisWeight(b4) * 0.125;

    return (b0 * w0 + b1 * w1 + b2 * w2 + b3 * w3 + b4 * w4)
         / max(w0 + w1 + w2 + w3 + w4, 1e-4);
}

float4 PS_BloomPrefilter(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    // Threshold AFTER exposure, so Bloom Threshold means the same thing at noon
    // and at midnight. Previously bloom was extracted from the raw scene and
    // added post-exposure, so as the eye adapted upward at night the bloom stayed
    // put and quietly became weaker relative to the image.
    float3 c = Downsample13Karis(BackBuffer, uv, ReShade::PixelSize) * ComputeExposure();
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
    float exposure = ComputeExposure(); // meter in the same space as bloom/god rays

    // Pass 1 — find the single brightest cell. A plain luminance centroid over
    // everything above the threshold lands halfway between the sun and a bright
    // snowfield, i.e. on neither, which is why shafts used to radiate from empty
    // sky on snowy exteriors.
    float  peak   = 0.0;
    float2 peakUV = float2(0.5, 0.5);
    [loop]
    for (int y = 0; y < N; ++y)
    {
        [loop]
        for (int x = 0; x < N; ++x)
        {
            float2 g = (float2(x, y) + 0.5) / float(N);
            float  l = tex2Dlod(LumCurrSampler, float4(g, 0, 0)).r * exposure;
            if (l > peak) { peak = l; peakUV = g; }
        }
    }

    // Pass 2 — centroid of only the cells near that peak, which gives sub-cell
    // precision on the actual light while ignoring merely-bright background.
    float thr = max(GodrayThreshold, peak * LightPeakBias);
    float2 sumUV = 0.0;
    float  sumW  = 0.0;
    [loop]
    for (int y2 = 0; y2 < N; ++y2)
    {
        [loop]
        for (int x2 = 0; x2 < N; ++x2)
        {
            float2 g = (float2(x2, y2) + 0.5) / float(N);
            float  l = tex2Dlod(LumCurrSampler, float4(g, 0, 0)).r * exposure;
            float  w = max(l - thr, 0.0);
            w *= w;
            sumUV += g * w;
            sumW  += w;
        }
    }

    // Fall back to the peak cell, not the screen centre: at a high Peak Lock the
    // threshold can sit at (or above) the peak itself and zero every weight, and
    // snapping the light to the middle of the screen would visibly swing the
    // shafts and the contact shadows.
    float2 lightUV = (sumW > 1e-6) ? sumUV / sumW : peakUV;

    // Confidence is now "how bright is the brightest thing on screen", which is
    // what the consumers actually want to gate on — a dark dungeon scores 0.
    float conf = saturate((peak - GodrayThreshold) * 4.0);

    float4 prev = tex2Dlod(LightPosPrevSampler, float4(0.5, 0.5, 0, 0));
    float  rate = saturate(AdaptSpeed * frametime * 0.001);
    // Init off alpha, not confidence: confidence is legitimately 0 in the dark,
    // which would otherwise re-init (and so never smooth) every frame indoors.
    if (prev.a <= 0.0) prev = float4(lightUV, conf, 1.0);

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
    float3 c = ToLinear(tex2D(BackBuffer, uv).rgb) * ComputeExposure();
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

    // Static per-pixel start offset. Without it every pixel samples the same
    // radii and the shafts show concentric banding. Static (not animated) for the
    // usual reason: there is no temporal accumulation here to average an animated
    // jitter, so it would crawl instead. The half-res -> full-res upsample does
    // the resolving.
    float  jitter = IGN(uv * float2(BUFFER_WIDTH, BUFFER_HEIGHT));
    float2 coord = uv - delta * jitter;
    float  illum = 1.0;
    float3 sum = 0.0;

    [loop]
    for (int i = 0; i < SAMPLES; ++i)
    {
        coord -= delta;
        // The sampler CLAMPs, so a tap past the edge would repeat the border
        // pixel and smear it into a hard streak whenever the light sits near or
        // off screen. Drop those taps instead of letting them accumulate.
        float2 inside = step(float2(0.0, 0.0), coord) * step(coord, float2(1.0, 1.0));
        sum += tex2D(GodrayBrightSampler, coord).rgb * illum * (inside.x * inside.y);
        illum *= GodrayDecay;
    }

    // Normalize the decay-weighted sum (geometric series ~ 1/(1-decay)) so the
    // brightness stays stable across decay/sample settings.
    return float4(sum * (1.0 - GodrayDecay) * LightGate(), 1.0);
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

// Scene luminance from the 64x64 downsample, on an 8x8 grid (was 4x4, which
// metered off roughly a sixteenth of the frame and needed heavy temporal
// smoothing to hide the resulting wobble).
float SceneAvgLuma()
{
    const int N = 8;
    float sum = 0.0;
    float wsum = 0.0;

    [loop]
    for (int y = 0; y < N; ++y)
    {
        [loop]
        for (int x = 0; x < N; ++x)
        {
            float2 g = (float2(x, y) + 0.5) / float(N);
            float  l = tex2Dlod(LumCurrSampler, float4(g, 0, 0)).r;

            // Centre weighting: exposure should follow what the player is looking
            // at, not a bright corner of sky.
            float2 dc = g - 0.5;
            float  w = lerp(1.0, exp(-dot(dc, dc) * 5.0), MeteringCenterWeight);

            // Log (geometric) mean: a small very bright region — the sun, a torch
            // — barely moves it, where an arithmetic mean gets dragged up by it
            // and the exposure then crushes the rest of the scene dark.
            sum  += (MeteringLogAverage ? log(max(l, 1e-4)) : l) * w;
            wsum += w;
        }
    }

    float avg = sum / max(wsum, 1e-4);
    if (MeteringLogAverage) avg = exp(avg);
    return max(avg, 1e-4);
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
//
// The tap radius is what decides whether this reads as an edge treatment or as
// texture. At one pixel the kernel is a Nyquist-frequency Laplacian: the only
// thing it can amplify is pixel-scale edge structure, which - since this shader
// wants an AA pass ordered BEFORE it - largely means re-crisping the very edges
// SMAA/DLAA just resolved. That looks like AA being partly undone, not like
// stone and cloth gaining grain. Surface texture lives lower, around 2-6 px, so
// widening the taps moves the response into that band and stops the two passes
// fighting over the same frequencies.
//
// That radius still has to answer to perspective, though. Sharpening is a
// screen-space operation and so is scale-invariant, but the scene is not: the
// road under your feet is magnified, spreading one texel over many pixels, while
// the same material at range is minified into a couple. A fixed pixel radius
// therefore bites hard on distant geometry and slides straight over the near
// ground, which has no contrast left at that scale. Growing the kernel as 1/z
// tracks the projected texel size, so "sharpen" means the same thing to a
// material however close it is. It only ever grows - clamped at 1x - so the
// far-field look is unchanged.
// Tap spacing shared by both sharpen modes: the base radius, widened on near
// surfaces so the kernel tracks projected texel size. `scale` comes back out
// because the depth-edge mask has to be normalised against it.
float2 SharpenTexelStep(float2 uv, out float scale)
{
    float z = max(GetDepth(uv) * CameraFar, 1e-3);
    scale = clamp(SharpenNearDist / z, 1.0, max(SharpenNearBoost, 1.0));
    return ReShade::PixelSize * max(SharpenRadius, 1.0) * scale;
}

float3 SharpenPass(float2 uv, float3 c, float focus)
{
    float nearScale;
    float2 t = SharpenTexelStep(uv, nearScale);
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
    // Run the kernel at AMD's full strength (peak -0.2) and let the slider scale the
    // detail it produces. Folding the slider into `w` instead left the effect near-dead
    // below ~0.7: the 1/(1 + 4w) gain only takes off as w approaches -0.25, so at the
    // 0.25 default it moved a midtone pixel by about one 8-bit level.
    float3 w = -amp * 0.2; // cross weights (negative = sharpen)
    float3 cas = (e + (b + d + f + h) * w) / (1.0 + 4.0 * w);

    return saturate(c + (cas - e) * saturate(Sharpness) * focus); // scale detail, keep DoF
}

// Overlay blend. Multiplies in the shadows and screens in the highlights, so
// adding detail through it preserves the tonality of the pixel instead of
// pushing both ends toward clipping the way a plain add does. b = 0.5 is the
// identity, which is why the detail term below is centred there.
float3 BlendOverlay(float3 a, float3 b)
{
    return (a < 0.5) ? (2.0 * a * b) : (1.0 - 2.0 * (1.0 - a) * (1.0 - b));
}

float3 SharpTap(float2 uv) { return tex2D(CompositeSampler, uv).rgb; }

// Silhouette mask from the depth Laplacian, at a FIXED one-pixel baseline.
// Deliberately not tied to the colour tap radius: a depth discontinuity is a
// property of the scene, not of whichever detail band the sharpener was asked to
// enhance. Widening this baseline with the near-field radius boost made the mask
// scale with proximity and smother the magnified near ground — the exact place
// the boost existed to help. One pixel also keeps the threshold on the same
// calibration DELCS uses.
//
// A Laplacian is blind to linear ramps, so a flat receding floor barely registers
// and only real discontinuities and sharp curvature are masked.
float DepthEdgeMask(float2 uv)
{
    float2 t = ReShade::PixelSize;
    float A = GetDepth(uv + t * float2(-1, -1));
    float B = GetDepth(uv + t * float2( 0, -1));
    float C = GetDepth(uv + t * float2( 1, -1));
    float D = GetDepth(uv + t * float2(-1,  0));
    float E = GetDepth(uv);
    float F = GetDepth(uv + t * float2( 1,  0));
    float G = GetDepth(uv + t * float2(-1,  1));
    float H = GetDepth(uv + t * float2( 0,  1));
    float I = GetDepth(uv + t * float2( 1,  1));

    float edge = (A + C + G + I) + 2.0 * (B + D + F + H) - 12.0 * E;
    return saturate(1.0 - abs(edge) * DetailDepthMask);
}

// Detail-enhancing sharpen, after Marty McFly / Pascal Gilcher's qUINT DELCS.
//
// The difference from CAS is the polarity of the adaptation, and it is the whole
// reason this mode exists. CAS asks "is there an edge here?" and backs OFF, but
// it never boosts anything: faint texture stays faint, and what you notice is
// the crisper edges. This divides the high-pass by the local RMS contrast, so
// amplitude is normalised — barely-there weave and grain get amplified up to the
// same prominence as strong detail, while outlines, which carry most of the
// contrast in a frame, are divided down. An enhancer rather than a sharpener.
//
// Three details make it usable rather than merely loud:
//   - the depth Laplacian masks silhouettes, where sharpening only ever makes
//     haloes (a flat receding floor has a LINEAR depth ramp, and a Laplacian is
//     blind to those, so grazing ground is untouched — only real discontinuities
//     and sharp curvature register);
//   - a log knee compresses the tail, since dividing by a small RMS can produce
//     enormous values in near-flat regions;
//   - the result is composited as an overlay rather than added.
float3 DetailSharpen(float2 uv, float3 c, float focus)
{
    float nearScale;
    float2 t = SharpenTexelStep(uv, nearScale); // nearScale unused here: see DepthEdgeMask

    float3 A = SharpTap(uv + t * float2(-1, -1));
    float3 B = SharpTap(uv + t * float2( 0, -1));
    float3 C = SharpTap(uv + t * float2( 1, -1));
    float3 D = SharpTap(uv + t * float2(-1,  0));
    float3 E = SharpTap(uv);
    float3 F = SharpTap(uv + t * float2( 1,  0));
    float3 G = SharpTap(uv + t * float2(-1,  1));
    float3 H = SharpTap(uv + t * float2( 0,  1));
    float3 I = SharpTap(uv + t * float2( 1,  1));

    float3 corners = (A + C) + (G + I);
    float3 neigh   = (B + D) + (F + H);

    // Full 3x3 Laplacian (corners 1, cross 2, centre -12) — wider and more
    // isotropic than the cross-only kernel CAS uses.
    float3 detail = corners + 2.0 * neigh - 12.0 * E;

    // Local RMS contrast, the normaliser. DetailFloor keeps a genuinely flat
    // region (sky, fog) from dividing by ~0 and having its dither amplified.
    float3 mean = (corners + neigh + E) / 9.0;
    float3 rms = (mean - A) * (mean - A);
    rms += (mean - B) * (mean - B);
    rms += (mean - C) * (mean - C);
    rms += (mean - D) * (mean - D);
    rms += (mean - E) * (mean - E);
    rms += (mean - F) * (mean - F);
    rms += (mean - G) * (mean - G);
    rms += (mean - H) * (mean - H);
    rms += (mean - I) * (mean - I);
    detail *= rsqrt(rms + max(DetailFloor, 1e-6)) * 0.1;

    detail *= DepthEdgeMask(uv);

    // Luma-only avoids the coloured speckle a per-channel high-pass leaves on
    // red/blue detail.
    if (DetailLumaOnly) detail = Luma(detail).xxx;

    detail = -detail * saturate(Sharpness) * 0.1;
    detail = sign(detail) * log(abs(detail) * 10.0 + 1.0) * 0.3; // soft knee

    return saturate(BlendOverlay(c, 0.5 + detail * focus));
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

// Distance/aerial fog (linear). Geometry fades toward the fog colour with
// distance; the fog brightens toward the on-screen sun (forward scattering).
// The sky (far depth) is left alone so only the scene hazes, not the zenith.
float3 ApplyFog(float2 uv, float3 c)
{
    float d = GetDepth(uv);
    if (d >= 1.0) return c; // sky

    float dist = d * CameraFar;
    float f = saturate((1.0 - exp(-max(dist - FogStart, 0.0) * FogDensity)) * FogMax);

    float3 fogCol = ToLinear(FogColor);
    if (FogSunAmount > 0.0)
    {
        float2 lp = tex2Dlod(LightPosSampler, float4(0.5, 0.5, 0, 0)).rg;
        // Aspect-correct the distance, or the glow is an ellipse stretched
        // horizontally on any non-square frame.
        float2 dl = (uv - lp) * float2(ReShade::AspectRatio, 1.0);
        float toLight = saturate(1.0 - length(dl) * 1.2);
        fogCol = lerp(fogCol, ToLinear(FogSunColor), toLight * toLight * FogSunAmount * LightGate());
    }

    // Darken the fog in dark scenes (night) using the adapted scene luminance —
    // the same gate as Purkinje — so bright haze doesn't wash out far mountains.
    float adapt = tex2Dlod(AdaptSampler, float4(0.5, 0.5, 0, 0)).r;
    float night = saturate(1.0 - adapt / max(FogNightThreshold, 1e-3));
    fogCol *= lerp(1.0, FogNightDim, night);

    return lerp(c, fogCol, f);
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

// Purkinje effect: in darkness human vision shifts to rod (scotopic) response —
// desaturated and blue-shifted, with reds going dark. Gated by the adapted scene
// luminance so it engages at night / in dungeons, not in daylight.
float3 ApplyPurkinje(float3 c)
{
    float adapt = tex2Dlod(AdaptSampler, float4(0.5, 0.5, 0, 0)).r; // linear scene avg
    float night = saturate(1.0 - adapt / max(PurkinjeThreshold, 1e-3));
    if (night <= 0.0) return c;

    float l = Luma(c);
    float pix = saturate(1.0 - l * 1.5);               // bright spots (fire, torches) keep colour
    float s = saturate(night * pix * PurkinjeStrength);

    float rod = dot(c, float3(0.2, 0.5, 0.4));          // rod luminance (blue-green biased)
    float3 scotopic = rod * float3(0.82, 0.94, 1.25);   // desaturated, cool blue
    return lerp(c, scotopic, s);
}

// ============================
// EFFECT PASS ENTRY POINTS
// ============================
float4 PS_AOCompute(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    // The bounce gather rides along in the AO loop, so this pass also runs when
    // only indirect light is wanted. rgb = bounce, a = occlusion.
    if (!EnableAO && !EnableGI) return float4(0, 0, 0, 1);
    return (AOMode == 0) ? ComputeAO(uv) : ComputeGTAO(uv);
}
float4 PS_AOBlur(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    return BlurAO(uv);
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
    float3 albedo = c;   // stands in for surface albedo; ReShade can't read the real one

    // AO + indirect bounce (linear, pre-tonemap)
    if (EnableAO || EnableGI)
    {
        float4 ao = tex2D(AOBlurSampler, uv);

        if (EnableAO)
            c *= ao.a;

        // Bounced light goes back in modulated by the receiver's own colour, so a
        // red wall bleeds red and dark surfaces stay dark rather than everything
        // lifting toward grey. This is where AO stops reading as painted-on dirt:
        // the crevice is still darker, but it is now filled with light of the
        // right colour instead of neutral black.
        if (EnableGI)
        {
            float3 bounce = lerp(Luma(ao.rgb).xxx, ao.rgb, GISaturation);
            c += bounce * albedo * (GIStrength * 2.0);
        }
    }

    // Contact shadows (linear)
    if (EnableContact)
        c *= tex2D(ContactBlurSampler, uv).r;

    // Distant shading (linear): the large-scale form AO is too local to give.
    if (EnableDistantShade)
        c *= DistantShade(uv, c);

    // Exposure
    c *= ComputeExposure();

    // SSR (linear): reflected color stored in sRGB, convert before mixing
    if (EnableSSR)
    {
        float4 ssr = tex2D(SSRBlurSampler, uv);
        c = lerp(c, ToLinear(ssr.rgb), saturate(ssr.a));
    }

    // Distance fog (linear) before the light effects so haze blooms naturally
    if (EnableFog)
        c = ApplyFog(uv, c);

    // Bloom (linear, additive)
    if (EnableBloom)
        c += tex2D(BloomU0s, uv).rgb * BloomStrength;

    // God rays (linear, additive). The bright source self-gates on brightness and
    // the scatter pass already applied the light-confidence gate.
    if (EnableGodrays)
        c += tex2D(GodraySampler, uv).rgb * GodrayIntensity;

    // Lens flare (linear, additive), faded toward the screen edges and gated on
    // light confidence — a lens flare with no light in frame is the tell that the
    // tracker is guessing.
    if (EnableLensFlare)
    {
        float2 dc = uv - 0.5;
        float falloff = saturate(1.0 - dot(dc, dc) * 1.5);
        c += tex2D(LensFlareSampler, uv).rgb * LensFlareIntensity * falloff * LightGate();
    }

    // Tonemap + encode (selectable operator), then grading
    c = ApplyTonemap(c);
    c = ApplyGrade(c);
    if (EnablePurkinje) c = ApplyPurkinje(c);

    return float4(saturate(c), 1.0);
}

// Lateral (transverse) chromatic aberration: split the red and blue channels
// radially, the shift growing toward the frame edges like a real lens — the
// centre stays perfectly sharp. Reads the finished composite; an optical effect,
// so it runs before the sensor-side grain/dither.
float3 ApplyEdgeCA(float2 uv)
{
    float2 d = uv - 0.5;
    float  edge = saturate(dot(d, d) * 2.0);          // 0 at centre, ~1 at the corners
    float2 dir  = d * rsqrt(dot(d, d) + 1e-8);         // radial direction
    float2 off  = dir * edge * CAStrength * ReShade::PixelSize; // corner shift in px

    float r = tex2D(CompositeSampler, uv + off).r;     // red drifts outward
    float g = tex2D(CompositeSampler, uv).g;           // green stays put
    float b = tex2D(CompositeSampler, uv - off).b;     // blue drifts inward
    return float3(r, g, b);
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
        if (DebugMode == 5) { float ao = tex2D(AOBlurSampler, uv).a; return float4(ao.xxx, 1.0); }
        if (DebugMode == 6) { float cs = tex2D(ContactBlurSampler, uv).r; return float4(cs.xxx, 1.0); }
        if (DebugMode == 7) { float4 s = tex2D(SSRBlurSampler, uv); return float4(s.rgb * s.a, 1.0); }
        if (DebugMode == 8) { return float4(tex2D(BloomU0s, uv).rgb, 1.0); }
        if (DebugMode == 9) { return float4(tex2D(GodraySampler, uv).rgb * GodrayIntensity, 1.0); }
        if (DebugMode == 10) { return float4(tex2D(LensFlareSampler, uv).rgb * LensFlareIntensity, 1.0); }
        if (DebugMode == 11) { return float4(ToSRGB(tex2D(AOBlurSampler, uv).rgb * GIStrength * 2.0), 1.0); }
        if (DebugMode == 12)
        {
            // Crosshair on the tracked light, tinted by confidence (red = not
            // trusted, green = trusted). Handy for checking Peak Lock.
            float4 lp = tex2Dlod(LightPosSampler, float4(0.5, 0.5, 0, 0));
            float2 d = (uv - lp.rg) * float2(ReShade::AspectRatio, 1.0);
            float mark = saturate(1.0 - min(abs(d.x), abs(d.y)) * 400.0);
            float3 base = tex2D(CompositeSampler, uv).rgb * 0.35;
            return float4(base + mark * float3(1.0 - lp.b, lp.b, 0.0), 1.0);
        }
        if (DebugMode == 13) { float3 l = ToLinear(tex2D(BackBuffer, uv).rgb); return float4(DistantShade(uv, l).xxx, 1.0); }
    }

    // Chromatic aberration first (optical effect, before the sensor grain/dither).
    float3 c = EnableCA ? ApplyEdgeCA(uv) : tex2D(CompositeSampler, uv).rgb;

    // Depth of Field: blend the sharp composite toward the DoF-blurred composite.
    float focus = 1.0;
    if (EnableDoF)
    {
        float4 dof = tex2D(DoFBlurSampler, uv);
        float coc = saturate(dof.a);
        c = lerp(c, dof.rgb, coc);

        // Suppress sharpening by the ACTUAL blur in pixels, not by normalised
        // CoC. CoC is normalised to the focal range and knows nothing about how
        // strong the blur is, so a preset that focuses far but keeps a small max
        // radius reads coc = 1 close up while being barely defocused — and the
        // old gate switched sharpening off completely there. The sharpener was
        // silently inheriting the DoF's focal distance as its own working range.
        // Measured against the real radius, a gentle DoF now suppresses gently.
        // At the default 3 px radius this is identical to the old behaviour.
        float blurPx = coc * MaxCoCRadius;
        focus = (SharpenDefocusLimit <= 0.0)
              ? 1.0
              : saturate(1.0 - blurPx / SharpenDefocusLimit);
    }

    // Local contrast, then sharpen & grain (display space)
    if (EnableClarity) c = ApplyClarity(uv, c, focus);
    if (EnableSharpen) c = (SharpenMode == 0) ? SharpenPass(uv, c, focus)
                                              : DetailSharpen(uv, c, focus);
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
