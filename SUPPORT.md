# Support Policy

Starman Tier 1 development targets:

- Windows x64 with D3D12
- Linux x64 with Vulkan
- macOS x64 with Metal
- macOS arm64 with Metal

M0 treats adapter absence as a failed smoke test. Hosted CI may need software adapters or self-hosted runners for reliable GPU coverage.

Public APIs are unstable before the first public baseline. Persisted file formats are versioned and must migrate forward with backups.
