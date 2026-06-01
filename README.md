# Obewan Shaders

Post-processing shaders for ReShade, aimed at a realistic image — primarily for *The Elder Scrolls V: Skyrim Special Edition*.

## ReshadeTrueLight

An all-in-one realistic pipeline in a single technique:

- Ambient occlusion (SSAO and GTAO modes, with distance fade / distant-radius controls)
- Contact shadows
- Screen-space reflections (Fresnel-masked, adaptive march, metallic + glossy/rough reflections)
- Depth of field (composite-based, separable bokeh)
- Bloom
- Auto-exposure with smooth eye adaptation
- ACES tonemapping, sharpen, film grain and dithering

### Requirements

- A working depth buffer (verify with the stock `DisplayDepth` shader).
- `bluenoise.png` in `reshade-shaders/textures/` — only if **Use Blue Noise** is enabled.

### Installation

Copy the contents of `reshade-shaders/` into your ReShade `reshade-shaders/` folder, then enable **ReshadeTrueLight** in the ReShade overlay.

## License

See repository.

## Acknowledgements

Developed with the help of AI assistants (Anthropic Claude, and earlier scaffolding from Microsoft Copilot, OpenAI ChatGPT and Google Gemini) for code review, debugging and implementation. The design decisions, testing and final tuning are the author's own.
