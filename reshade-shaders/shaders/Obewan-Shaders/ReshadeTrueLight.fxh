//===========================================================================
// SKYRIM REALISTIC PIPELINE — ReshadeTrueLight Header
// Author: Obewan (https://github.com/obewan)
// Version: 0.1.0
// Requirements
//   - BlueNoise.png in reshade-shaders/Textures if UseBlueNoise = true
//   - A working depth buffer (verify with the stock DisplayDepth shader)
//
// NOTE (v0.1.0): temporal reprojection has been removed. Stock ReShade cannot
// feed per-frame view/projection matrices into a uniform, so the previous
// reprojection path could never work without a custom C++ addon. All effects
// are now single-frame and rely on spatial bilateral filtering for stability.
//===========================================================================

#include "ReShadeUI.fxh"
#include "ReShade.fxh"

// Compile-time bounds for bounded loops
static const int AO_MAX_SAMPLES = 12;
static const int DOF_MAX_TAPS   = 9;
static const int SSR_MAX_STEPS  = 32;

// Precomputed unit directions for AO sampling
static const float2 AO_Dirs[AO_MAX_SAMPLES] =
{
    float2( 1.000000,  0.000000),
    float2( 0.866025,  0.500000),
    float2( 0.500000,  0.866025),
    float2( 0.000000,  1.000000),
    float2(-0.500000,  0.866025),
    float2(-0.866025,  0.500000),
    float2(-1.000000,  0.000000),
    float2(-0.866025, -0.500000),
    float2(-0.500000, -0.866025),
    float2( 0.000000, -1.000000),
    float2( 0.500000, -0.866025),
    float2( 0.866025, -0.500000)
};

// Precomputed 9-tap Gaussian weights (center first)
static const float DOF_WEIGHTS[DOF_MAX_TAPS] =
{
    0.027027f, 0.054054f, 0.121621f, 0.242422f, 0.333333f,
    0.242422f, 0.121621f, 0.054054f, 0.027027f
};

// ============================
// FRAME INFO (engine-driven, hidden — no UI bar)
// ============================
uniform int FrameIndex < source = "framecount"; >;

// ============================
// DEPTH / CAMERA
// ReShade::GetLinearizedDepth returns normalized [0,1] (1 = far plane / sky).
// CameraFar scales that into the world units used for position reconstruction.
// CameraFovY replaces the (unavailable) projection matrix for unprojection.
// ============================
uniform bool  HasDepth  < ui_type = "checkbox"; ui_label = "Has Depth"; ui_tooltip = "Uncheck if the game exposes no usable depth buffer."; > = true;
uniform float CameraFar  < ui_type = "slider"; ui_min = 100.0; ui_max = 10000.0; ui_step = 10.0; ui_label = "Camera Far (depth scale)"; ui_tooltip = "World-unit scale applied to normalized depth."; > = 1000.0;
uniform float CameraFovY < ui_type = "slider"; ui_min = 30.0;  ui_max = 120.0;   ui_step = 1.0;  ui_label = "Vertical FOV (deg)"; ui_tooltip = "Match the game's vertical field of view."; > = 60.0;

// ============================
// TEXTURES / SAMPLERS
// ============================
texture2D BackBufferTex : COLOR;
sampler BackBuffer { Texture = BackBufferTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// Bloom pyramid (progressive down/up-sample, CoD/Jimenez style). RGBA16F, LINEAR.
#define BLOOM_SAMP(s, t) sampler s { Texture = t; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D BloomD0 { Width = BUFFER_WIDTH/2;  Height = BUFFER_HEIGHT/2;  Format = RGBA16F; }; BLOOM_SAMP(BloomD0s, BloomD0)
texture2D BloomD1 { Width = BUFFER_WIDTH/4;  Height = BUFFER_HEIGHT/4;  Format = RGBA16F; }; BLOOM_SAMP(BloomD1s, BloomD1)
texture2D BloomD2 { Width = BUFFER_WIDTH/8;  Height = BUFFER_HEIGHT/8;  Format = RGBA16F; }; BLOOM_SAMP(BloomD2s, BloomD2)
texture2D BloomD3 { Width = BUFFER_WIDTH/16; Height = BUFFER_HEIGHT/16; Format = RGBA16F; }; BLOOM_SAMP(BloomD3s, BloomD3)
texture2D BloomD4 { Width = BUFFER_WIDTH/32; Height = BUFFER_HEIGHT/32; Format = RGBA16F; }; BLOOM_SAMP(BloomD4s, BloomD4)
texture2D BloomD5 { Width = BUFFER_WIDTH/64; Height = BUFFER_HEIGHT/64; Format = RGBA16F; }; BLOOM_SAMP(BloomD5s, BloomD5)
texture2D BloomU0 { Width = BUFFER_WIDTH/2;  Height = BUFFER_HEIGHT/2;  Format = RGBA16F; }; BLOOM_SAMP(BloomU0s, BloomU0)
texture2D BloomU1 { Width = BUFFER_WIDTH/4;  Height = BUFFER_HEIGHT/4;  Format = RGBA16F; }; BLOOM_SAMP(BloomU1s, BloomU1)
texture2D BloomU2 { Width = BUFFER_WIDTH/8;  Height = BUFFER_HEIGHT/8;  Format = RGBA16F; }; BLOOM_SAMP(BloomU2s, BloomU2)
texture2D BloomU3 { Width = BUFFER_WIDTH/16; Height = BUFFER_HEIGHT/16; Format = RGBA16F; }; BLOOM_SAMP(BloomU3s, BloomU3)
texture2D BloomU4 { Width = BUFFER_WIDTH/32; Height = BUFFER_HEIGHT/32; Format = RGBA16F; }; BLOOM_SAMP(BloomU4s, BloomU4)

// Luminance downsample (used for instant auto-exposure)
texture2D LumCurrTex { Width = 64; Height = 64; Format = R16F; };
sampler LumCurrSampler { Texture = LumCurrTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// Adapted luminance (1x1, smoothed over time via a tiny ping-pong — the only
// cross-frame state in the shader; needs no reprojection matrices).
texture2D AdaptTex { Width = 1; Height = 1; Format = R16F; };
sampler AdaptSampler { Texture = AdaptTex; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D AdaptPrevTex { Width = 1; Height = 1; Format = R16F; };
sampler AdaptPrevSampler { Texture = AdaptPrevTex; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// AO (computed at half-res, upsampled in the blur)
texture2D AORawTex { Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = R8; };
sampler AORawSampler { Texture = AORawTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D AOBlurTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8; };
sampler AOBlurSampler { Texture = AOBlurTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// Contact shadows (full-res)
texture2D ContactRawTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8; };
sampler ContactRawSampler { Texture = ContactRawTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D ContactBlurTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8; };
sampler ContactBlurSampler { Texture = ContactBlurTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// SSR (computed at half-res, upsampled in the blur)
texture2D SSRRawTex { Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = RGBA8; };
sampler SSRRawSampler { Texture = SSRRawTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D SSRBlurTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler SSRBlurSampler { Texture = SSRBlurTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// DoF (separable: horizontal gather -> vertical gather). rgb = blurred color, a = normalized CoC.
texture2D DoFRawTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler DoFRawSampler { Texture = DoFRawTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D DoFBlurTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler DoFBlurSampler { Texture = DoFBlurTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// Fully processed scene (AO/contact/exposure/SSR/bloom/tonemap) BEFORE DoF, so
// DoF blurs the finished image instead of the raw backbuffer.
texture2D CompositeTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler CompositeSampler { Texture = CompositeTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// ============================
// BLUE-NOISE
// Place BlueNoise.png in reshade-shaders/Textures. Default off to avoid load failures.
// ============================
uniform bool UseBlueNoise < ui_type = "checkbox"; ui_label = "Use Blue Noise"; > = false;
texture2D BlueNoiseTex < source = "bluenoise.png"; pooled = true; > { Width = 64; Height = 64; Format = RGBA8; };
sampler BlueNoiseSampler { Texture = BlueNoiseTex; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = REPEAT; AddressV = REPEAT; };

// ============================
// EXPOSURE
// ============================
uniform float frametime < source = "frametime"; >; // milliseconds since last frame
uniform bool  EnableAutoExposure < ui_type = "checkbox"; ui_label = "Auto Exposure"; > = true;
uniform float ExposureKey  < ui_type = "slider"; ui_min = 0.05; ui_max = 0.50; ui_step = 0.01; ui_label = "Exposure Key (middle grey)"; > = 0.18;
uniform float AdaptSpeed   < ui_type = "slider"; ui_min = 0.1;  ui_max = 8.0;  ui_step = 0.1;  ui_label = "Adaptation Speed"; ui_tooltip = "Higher = faster eye adaptation. Lower = smoother."; > = 2.0;
uniform float ExposureMin  < ui_type = "slider"; ui_min = 0.05; ui_max = 2.0;  ui_step = 0.01; ui_label = "Min Exposure (gain floor)"; > = 0.25;
uniform float ExposureMax  < ui_type = "slider"; ui_min = 0.5;  ui_max = 8.0;  ui_step = 0.05; ui_label = "Max Exposure (gain ceiling)"; ui_tooltip = "Lower this to stop dark/night scenes from over-brightening."; > = 2.0;
uniform float ManualExposure < ui_type = "slider"; ui_min = -4.0; ui_max = 4.0; ui_step = 0.1; ui_label = "Manual Exposure (EV, when auto off)"; > = 0.0;

// ============================
// TONEMAP & GRADING
// ============================
uniform int   TonemapOperator < ui_type = "combo"; ui_items = "ACES\0AgX\0Hable (Uncharted 2)\0Reinhard\0"; ui_label = "Tonemap Operator"; ui_tooltip = "AgX handles bright saturated colours (fire, sunsets, magic) more gracefully than ACES."; > = 0;
uniform float TonemapWhite    < ui_type = "slider"; ui_min = 1.0; ui_max = 16.0; ui_step = 0.1; ui_label = "White Point (Hable/Reinhard)"; > = 4.0;
uniform float Contrast        < ui_type = "slider"; ui_min = 0.5; ui_max = 2.0;  ui_step = 0.01; ui_label = "Contrast"; > = 1.0;
uniform float Saturation      < ui_type = "slider"; ui_min = 0.0; ui_max = 2.0;  ui_step = 0.01; ui_label = "Saturation"; > = 1.0;

// ============================
// AO
// ============================
uniform bool  EnableAO       < ui_type = "checkbox"; ui_label = "Enable AO"; > = true;
uniform int   AOMode         < ui_type = "combo"; ui_items = "SSAO (fast, low-end)\0GTAO (quality)\0"; ui_label = "AO Mode"; ui_tooltip = "SSAO: cheap disk sampling. GTAO: horizon-marched, more accurate, a bit heavier."; > = 0;
uniform float AORadius       < ui_type = "slider"; ui_min = 0.05; ui_max = 20.0; ui_step = 0.05; ui_label = "AO Radius (world)"; > = 3.0;
uniform int   AOSamples      < ui_type = "slider"; ui_min = 4;    ui_max = 12;   ui_step = 1;    ui_label = "AO Samples"; > = 12;
uniform float AOStrength     < ui_type = "slider"; ui_min = 0.0;  ui_max = 4.0;  ui_step = 0.01; ui_label = "AO Strength"; > = 1.6;
uniform float AOPower        < ui_type = "slider"; ui_min = 0.5;  ui_max = 4.0;  ui_step = 0.05; ui_label = "AO Contrast (power)"; ui_tooltip = "Higher = deeper, more contrasty occlusion."; > = 2.0;
uniform float AOBias         < ui_type = "slider"; ui_min = 0.0;  ui_max = 0.5;  ui_step = 0.005; ui_label = "AO Bias"; ui_tooltip = "Raise to reduce self-occlusion / flat-surface noise."; > = 0.03;
uniform float AOFadeStart    < ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0;  ui_step = 0.01; ui_label = "AO Fade Start (depth)"; ui_tooltip = "Normalized depth where AO begins to fade out."; > = 0.6;
uniform float AOFadeEnd      < ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0;  ui_step = 0.01; ui_label = "AO Fade End (depth)"; ui_tooltip = "Normalized depth where AO is fully gone. Lower this if distant terrain (mountains) still shows AO noise."; > = 0.9;
uniform float AODistantRadius < ui_type = "slider"; ui_min = 0.0; ui_max = 0.1; ui_step = 0.002; ui_label = "AO Distant Radius"; ui_tooltip = "Grows the AO radius with distance so far objects keep a stable, less-noisy AO footprint (XeGTAO-style). 0 = off. An alternative to fading AO out."; > = 0.0;

// ============================
// CONTACT SHADOWS
// ============================
uniform bool  EnableContact   < ui_type = "checkbox"; ui_label = "Enable Contact Shadows"; > = true;
uniform float ContactMaxDist  < ui_type = "slider"; ui_min = 0.05; ui_max = 5.0; ui_step = 0.05; ui_label = "Contact Max Distance (world)"; > = 1.0;
uniform float ContactStrength < ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0; ui_step = 0.01; ui_label = "Contact Strength"; > = 0.65;

// ============================
// SSR
// ============================
uniform bool  EnableSSR       < ui_type = "checkbox"; ui_label = "Enable SSR"; > = true;
uniform int   SSRSteps        < ui_type = "slider"; ui_min = 8;    ui_max = 32;    ui_step = 1;   ui_label = "SSR Steps"; > = 20;
uniform float SSRMaxDistance  < ui_type = "slider"; ui_min = 1.0;  ui_max = 400.0; ui_step = 1.0; ui_label = "SSR Max Distance (world)"; > = 60.0;
uniform float SSRStrength     < ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0;   ui_step = 0.01; ui_label = "SSR Strength"; > = 0.5;
uniform float SSRBaseReflect  < ui_type = "slider"; ui_min = 0.0;  ui_max = 0.2;   ui_step = 0.005; ui_label = "SSR Base Reflectivity (F0)"; ui_tooltip = "Head-on reflectivity. Keep low (~0.02) so only grazing angles reflect — this stops the 'mirror on NPC' look."; > = 0.02;
uniform float SSRThickness    < ui_type = "slider"; ui_min = 0.5;  ui_max = 50.0;  ui_step = 0.5; ui_label = "SSR Thickness (world)"; ui_tooltip = "Reject hits where the scene lies far behind the ray (silhouette bleed). Raise if reflections vanish, lower if backgrounds smear into reflections."; > = 8.0;
uniform float SSRMetallic     < ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0;   ui_step = 0.01; ui_label = "SSR Metallic"; ui_tooltip = "0 = dielectric (clear reflection). 1 = metal (reflection tinted by the surface colour, stronger Fresnel)."; > = 0.0;
uniform float SSRGlossiness   < ui_type = "slider"; ui_min = 0.0;  ui_max = 1.0;   ui_step = 0.01; ui_label = "SSR Glossiness"; ui_tooltip = "1 = sharp mirror reflection. Lower = rougher / blurrier reflection. (Heuristic; ReShade cannot read TruePBR roughness.)"; > = 0.7;

// ============================
// DoF (focal range in world units; CoC mapped to pixel radius)
// ============================
uniform bool  EnableDoF      < ui_type = "checkbox"; ui_label = "Enable DoF"; > = true;
uniform float FocalDistance  < ui_type = "slider"; ui_min = 0.0;  ui_max = 500.0; ui_step = 1.0; ui_label = "Focal Distance (world)"; > = 30.0;
uniform float FocalRange     < ui_type = "slider"; ui_min = 1.0;  ui_max = 300.0; ui_step = 1.0; ui_label = "In-focus Range (world)"; > = 40.0;
uniform float MaxCoCRadius   < ui_type = "slider"; ui_min = 0.5;  ui_max = 16.0;  ui_step = 0.25; ui_label = "Max CoC Radius (px)"; ui_tooltip = "Maximum blur radius in pixels. Small values (1-3) give subtle defocus; raise for stronger bokeh."; > = 3.0;

// ============================
// BLOOM
// ============================
uniform bool  EnableBloom    < ui_type = "checkbox"; ui_label = "Enable Bloom"; > = true;
uniform float BloomThreshold < ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01; ui_label = "Bloom Threshold (linear)"; ui_tooltip = "Brightness where bloom starts."; > = 0.7;
uniform float BloomSoftKnee  < ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01; ui_label = "Bloom Soft Knee"; ui_tooltip = "Soft transition around the threshold (0 = hard cutoff, 1 = very soft). Reduces flicker/popping on bright edges."; > = 0.5;
uniform float BloomStrength  < ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.005; ui_label = "Bloom Strength"; > = 0.08;
uniform float BloomRadius    < ui_type = "slider"; ui_min = 0.5; ui_max = 2.0; ui_step = 0.05; ui_label = "Bloom Spread"; ui_tooltip = "Width of the glow (scales the upsample tent)."; > = 1.0;

// ============================
// SHARPEN & GRAIN
// ============================
uniform bool  EnableSharpen < ui_type = "checkbox"; ui_label = "Enable Sharpen"; > = true;
uniform float Sharpness     < ui_type = "slider"; ui_min = 0.0; ui_max = 1.0;  ui_step = 0.01;  ui_label = "Sharpness"; > = 0.25;
uniform bool  EnableGrain   < ui_type = "checkbox"; ui_label = "Enable Grain"; > = true;
uniform float GrainAmount   < ui_type = "slider"; ui_min = 0.0; ui_max = 0.07; ui_step = 0.001; ui_label = "Grain Amount"; > = 0.02;

// ============================
// DEBUG
// ============================
uniform int DebugMode < ui_type = "combo"; ui_items = "Off\0Depth\0Normals\0ViewPos\0CoC\0AO\0Contact\0SSR\0Bloom\0"; ui_label = "Debug View"; > = 0;
