//===========================================================================
// SKYRIM REALISTIC PIPELINE — ReshadeTrueLight Header (patched)
// Author: Obewan
// Co-authors: Microsoft Copilot, OpenAI ChatGPT, Google Gemini.
// Version: 0.0.1
// Requirements
//   - BlueNoise.png or bluenoise.png in reshade-shaders/textures if UseBlueNoise = true
//   - A depth buffer correctly setted (check it using the DisplayDepth shader)
//   - Injector should update PrevViewProj and optionally ProjectionMatrix/InvProjectionMatrix each frame
//===========================================================================

#include "ReShadeUI.fxh"
#include "ReShade.fxh"

// Conservative compile-time bounds for unrolled loops
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
// GLOBAL MATRICES / FRAME INFO
// - PrevViewProj must be updated each frame by the injector for reprojection to work
// - ProjectionMatrix and InvProjectionMatrix are optional but strongly recommended
// ============================
uniform float4x4 ProjectionMatrix < ui_type = "matrix"; ui_label = "ProjectionMatrix (injector)"; > = float4x4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
uniform float4x4 InvProjectionMatrix < ui_type = "matrix"; ui_label = "InvProjectionMatrix (injector)"; > = float4x4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
uniform float4x4 PrevViewProj < ui_type = "matrix"; ui_label = "PrevViewProj (injector)"; > = float4x4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
uniform int FrameIndex < ui_type = "slider"; ui_min = 0; ui_max = 100000; ui_step = 1; ui_label = "FrameIndex (injector)"; > = 0;

// ============================
// DEPTH / CAMERA CONVENTIONS
// - HasDepth false makes GetDepth return far plane (safe fallback)
// - ReversedZ must match the game/injector convention
// ============================
uniform bool HasDepth < ui_type = "checkbox"; ui_label = "HasDepth (set false if no depth)"; > = true;
uniform int ReversedZ    < ui_type = "combo"; ui_items = "Off\0On\0"; ui_label = "Reversed Z"; > = 0;
uniform float CameraNear < ui_type = "slider"; ui_min = 0.01; ui_max = 10.0; ui_step = 0.01; ui_label = "CameraNear"; > = 0.1;
uniform float CameraFar  < ui_type = "slider"; ui_min = 10.0; ui_max = 10000.0; ui_step = 1.0; ui_label = "CameraFar"; > = 1000.0;

// ============================
// TEXTURES / SAMPLERS
// - Use POINT for samplers sampled inside loops to avoid derivative generation
// - Use LINEAR for bloom and blur targets where filtering is desired
// ============================
texture2D BackBufferTex : COLOR;
sampler BackBuffer { Texture = BackBufferTex; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// Bloom ping-pong & mips
texture2D BloomMip0A { Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = RGBA8; };
sampler BloomMip0ASampler { Texture = BloomMip0A; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D BloomMip0B { Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = RGBA8; };
sampler BloomMip0BSampler { Texture = BloomMip0B; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D BloomMip1 { Width = BUFFER_WIDTH/4; Height = BUFFER_HEIGHT/4; Format = RGBA8; };
sampler BloomMip1Sampler { Texture = BloomMip1; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D BloomMip1Temp { Width = BUFFER_WIDTH/4; Height = BUFFER_HEIGHT/4; Format = RGBA8; };
sampler BloomMip1TempSampler { Texture = BloomMip1Temp; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D BloomMip2 { Width = BUFFER_WIDTH/8; Height = BUFFER_HEIGHT/8; Format = RGBA8; };
sampler BloomMip2Sampler { Texture = BloomMip2; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// Luminance current and history (64x64)
texture2D LumCurrTex { Width = 64; Height = 64; Format = R8; };
sampler LumCurrSampler { Texture = LumCurrTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D LumHistoryTex { Width = 64; Height = 64; Format = R8; };
sampler LumHistorySampler { Texture = LumHistoryTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// HISTORY PING-PONG TARGETS (POINT samplers for loop safety)
texture2D AOHistoryPrev { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8; };
sampler AOHistoryPrevSampler { Texture = AOHistoryPrev; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D AOHistoryCurr { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8; };
sampler AOHistoryCurrSampler { Texture = AOHistoryCurr; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D AOBlurTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8; };
sampler AOBlurSampler { Texture = AOBlurTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// Contact
texture2D ContactHistoryPrev { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8; };
sampler ContactHistoryPrevSampler { Texture = ContactHistoryPrev; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D ContactHistoryCurr { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8; };
sampler ContactHistoryCurrSampler { Texture = ContactHistoryCurr; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D ContactBlurTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8; };
sampler ContactBlurSampler { Texture = ContactBlurTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// SSR
texture2D SSRHistoryPrev { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler SSRHistoryPrevSampler { Texture = SSRHistoryPrev; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D SSRHistoryCurr { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler SSRHistoryCurrSampler { Texture = SSRHistoryCurr; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D SSRBlurTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler SSRBlurSampler { Texture = SSRBlurTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// DoF
texture2D DoFHistoryPrev { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler DoFHistoryPrevSampler { Texture = DoFHistoryPrev; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D DoFHistoryCurr { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler DoFHistoryCurrSampler { Texture = DoFHistoryCurr; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D DoFBlurTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler DoFBlurSampler { Texture = DoFBlurTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// ============================
// BLUE-NOISE
// - Place BlueNoise.png or bluenoise.png in reshade-shaders/Textures
// - Default UseBlueNoise = false to avoid runtime failures
// ============================
uniform bool UseBlueNoise < ui_type = "checkbox"; ui_label = "UseBlueNoise"; > = false;
texture2D BlueNoiseTex < source = "BlueNoise.png"; pooled = true; > { Width = 64; Height = 64; Format = RGBA8; };
sampler BlueNoiseSampler { Texture = BlueNoiseTex; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = REPEAT; AddressV = REPEAT; };

// ============================
// USER CONTROLS
// ============================
uniform bool EnableAO < ui_type = "checkbox"; ui_label = "Enable AO"; > = true;
uniform float AORadiusMeters < ui_type = "slider"; ui_min = 0.01; ui_max = 5.0; ui_step = 0.01; ui_label = "AO Radius (m)"; > = 1.2;
uniform int AOSamples < ui_type = "slider"; ui_min = 4; ui_max = 16; ui_step = 1; ui_label = "AO Samples"; > = 12;
uniform float AOStrength < ui_type = "slider"; ui_min = 0.0; ui_max = 2.0; ui_step = 0.01; ui_label = "AO Strength"; > = 0.8;

uniform bool UseReprojection < ui_type = "checkbox"; ui_label = "Use Reprojection (temporal)"; > = true;
uniform float TemporalBlend < ui_type = "slider"; ui_min = 0.0; ui_max = 0.99; ui_step = 0.01; ui_label = "Temporal Blend"; > = 0.85;
uniform float TemporalClamp < ui_type = "slider"; ui_min = 0.01; ui_max = 0.5; ui_step = 0.01; ui_label = "Temporal Clamp"; > = 0.12;

uniform bool EnableSSR < ui_type = "checkbox"; ui_label = "Enable SSR"; > = true;
uniform int SSRSteps < ui_type = "slider"; ui_min = 8; ui_max = 64; ui_step = 1; ui_label = "SSR Steps"; > = 20;
uniform float SSRMaxDistance < ui_type = "slider"; ui_min = 0.5; ui_max = 200.0; ui_step = 0.5; ui_label = "SSR Max Distance"; > = 50.0;

uniform bool EnableDoF < ui_type = "checkbox"; ui_label = "Enable DoF"; > = true;
uniform float FocalDistance < ui_type = "slider"; ui_min = 0.1; ui_max = 200.0; ui_step = 0.1; ui_label = "Focal Distance (m)"; > = 6.0;
uniform float Aperture < ui_type = "slider"; ui_min = 0.5; ui_max = 22.0; ui_step = 0.1; ui_label = "Aperture f-stop"; > = 2.8;
uniform float MaxCoCRadius < ui_type = "slider"; ui_min = 1.0; ui_max = 64.0; ui_step = 1.0; ui_label = "Max CoC Radius (px)"; > = 24.0;

uniform bool EnableBloom < ui_type = "checkbox"; ui_label = "Enable Bloom"; > = true;
uniform float BloomThreshold < ui_type = "slider"; ui_min = 0.0; ui_max = 2.0; ui_step = 0.01; ui_label = "Bloom Threshold"; > = 1.0;
uniform float BloomStrength < ui_type = "slider"; ui_min = 0.0; ui_max = 2.0; ui_step = 0.01; ui_label = "Bloom Strength"; > = 0.12;
uniform float BloomRadius < ui_type = "slider"; ui_min = 0.5; ui_max = 6.0; ui_step = 0.1; ui_label = "Bloom Radius"; > = 2.0;

uniform bool EnableSharpen < ui_type = "checkbox"; ui_label = "Enable Sharpen"; > = true;
uniform float Sharpness < ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01; ui_label = "Sharpness"; > = 0.25;
uniform bool EnableGrain < ui_type = "checkbox"; ui_label = "Enable Grain"; > = true;
uniform float GrainAmount < ui_type = "slider"; ui_min = 0.0; ui_max = 0.07; ui_step = 0.001; ui_label = "Grain Amount"; > = 0.02;

// Debug mode to visualize intermediate buffers
uniform int DebugMode < ui_type = "combo"; ui_items = "Off\0Depth\0Normals\0Motion\0CoC\0SSR Hits\0History\0"; ui_label = "Debug Mode"; > = 0;
