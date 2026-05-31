//===========================================================================
// SKYRIM REALISTIC PIPELINE — ReshadeTrueLight Header
// Author: Obewan (https://github.com/obewan)
// Co-authors: Anthropic Claude, Microsoft Copilot, OpenAI ChatGPT, Google Gemini.
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

// Bloom ping-pong & mips
texture2D BloomMip0A { Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = RGBA16F; };
sampler BloomMip0ASampler { Texture = BloomMip0A; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D BloomMip0B { Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = RGBA16F; };
sampler BloomMip0BSampler { Texture = BloomMip0B; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D BloomMip1 { Width = BUFFER_WIDTH/4; Height = BUFFER_HEIGHT/4; Format = RGBA16F; };
sampler BloomMip1Sampler { Texture = BloomMip1; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D BloomMip1Temp { Width = BUFFER_WIDTH/4; Height = BUFFER_HEIGHT/4; Format = RGBA16F; };
sampler BloomMip1TempSampler { Texture = BloomMip1Temp; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };
texture2D BloomMip2 { Width = BUFFER_WIDTH/8; Height = BUFFER_HEIGHT/8; Format = RGBA16F; };
sampler BloomMip2Sampler { Texture = BloomMip2; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

// Luminance downsample (used for instant auto-exposure)
texture2D LumCurrTex { Width = 64; Height = 64; Format = R16F; };
sampler LumCurrSampler { Texture = LumCurrTex; MinFilter = LINEAR; MagFilter = LINEAR; MipFilter = POINT; AddressU = CLAMP; AddressV = CLAMP; };

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

// ============================
// BLUE-NOISE
// Place BlueNoise.png in reshade-shaders/Textures. Default off to avoid load failures.
// ============================
uniform bool UseBlueNoise < ui_type = "checkbox"; ui_label = "Use Blue Noise"; > = false;
texture2D BlueNoiseTex < source = "BlueNoise.png"; pooled = true; > { Width = 64; Height = 64; Format = RGBA8; };
sampler BlueNoiseSampler { Texture = BlueNoiseTex; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; AddressU = REPEAT; AddressV = REPEAT; };

// ============================
// EXPOSURE
// ============================
uniform bool  EnableAutoExposure < ui_type = "checkbox"; ui_label = "Auto Exposure"; > = true;
uniform float ExposureKey  < ui_type = "slider"; ui_min = 0.05; ui_max = 0.50; ui_step = 0.01; ui_label = "Exposure Key (middle grey)"; > = 0.18;
uniform float ManualExposure < ui_type = "slider"; ui_min = -4.0; ui_max = 4.0; ui_step = 0.1; ui_label = "Manual Exposure (EV, when auto off)"; > = 0.0;

// ============================
// AO
// ============================
uniform bool  EnableAO       < ui_type = "checkbox"; ui_label = "Enable AO"; > = true;
uniform float AORadius       < ui_type = "slider"; ui_min = 0.05; ui_max = 20.0; ui_step = 0.05; ui_label = "AO Radius (world)"; > = 2.0;
uniform int   AOSamples      < ui_type = "slider"; ui_min = 4;    ui_max = 12;   ui_step = 1;    ui_label = "AO Samples"; > = 12;
uniform float AOStrength     < ui_type = "slider"; ui_min = 0.0;  ui_max = 2.0;  ui_step = 0.01; ui_label = "AO Strength"; > = 0.8;

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

// ============================
// DoF (focal range in world units; CoC mapped to pixel radius)
// ============================
uniform bool  EnableDoF      < ui_type = "checkbox"; ui_label = "Enable DoF"; > = true;
uniform float FocalDistance  < ui_type = "slider"; ui_min = 0.0;  ui_max = 500.0; ui_step = 1.0; ui_label = "Focal Distance (world)"; > = 30.0;
uniform float FocalRange     < ui_type = "slider"; ui_min = 1.0;  ui_max = 300.0; ui_step = 1.0; ui_label = "In-focus Range (world)"; > = 40.0;
uniform float MaxCoCRadius   < ui_type = "slider"; ui_min = 1.0;  ui_max = 32.0;  ui_step = 1.0; ui_label = "Max CoC Radius (px)"; > = 12.0;

// ============================
// BLOOM
// ============================
uniform bool  EnableBloom    < ui_type = "checkbox"; ui_label = "Enable Bloom"; > = true;
uniform float BloomThreshold < ui_type = "slider"; ui_min = 0.0; ui_max = 1.0; ui_step = 0.01; ui_label = "Bloom Threshold (linear)"; > = 0.6;
uniform float BloomStrength  < ui_type = "slider"; ui_min = 0.0; ui_max = 2.0; ui_step = 0.01; ui_label = "Bloom Strength"; > = 0.12;
uniform float BloomRadius    < ui_type = "slider"; ui_min = 0.5; ui_max = 6.0; ui_step = 0.1;  ui_label = "Bloom Radius"; > = 2.0;

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
