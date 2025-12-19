# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.3.0] - 2025-12-19

### Added

- Upgrade stdgpu for CUDA 13 support [@stotko](https://github.com/stotko) ([\#33](https://github.com/vc-bonn/torchhull/pull/33))
- Support OpenCV convention for transforms [@stotko](https://github.com/stotko) ([\#19](https://github.com/vc-bonn/torchhull/pull/19))

### Changed

- Change partial masks condition to treat non-visible parts as unknown [@stotko](https://github.com/stotko) ([\#18](https://github.com/vc-bonn/torchhull/pull/18))

### Removed

- Drop Python 3.9 support [@stotko](https://github.com/stotko) ([\#27](https://github.com/vc-bonn/torchhull/pull/27))

### Fixed

- Fix unused warning in Gaussian blur [@stotko](https://github.com/stotko) ([\#32](https://github.com/vc-bonn/torchhull/pull/32))
- Do not use define inside macro evaluation [@stotko](https://github.com/stotko) ([\#26](https://github.com/vc-bonn/torchhull/pull/26))
- Fix pixel rounding in revised partial mask condition [@stotko](https://github.com/stotko) ([\#20](https://github.com/vc-bonn/torchhull/pull/20))
- Fix kernel error if sparse visual hull field is empty [@stotko](https://github.com/stotko) ([\#17](https://github.com/vc-bonn/torchhull/pull/17))
- Add missing docs for Gaussian blur [@stotko](https://github.com/stotko) ([\#16](https://github.com/vc-bonn/torchhull/pull/16))


## [0.2.0] - 2025-07-10

### Added

- Add standard and optimized Gaussian blur for masks [@stotko](https://github.com/stotko) ([\#14](https://github.com/vc-bonn/torchhull/pull/14))
- Support non-binary masks from matting approaches [@stotko](https://github.com/stotko) ([\#13](https://github.com/vc-bonn/torchhull/pull/13))
- Add Python 3.13 support [@stotko](https://github.com/stotko) ([\#6](https://github.com/vc-bonn/torchhull/pull/6))

### Changed

- Define visual hull isosurface multiplicatively [@stotko](https://github.com/stotko) ([\#12](https://github.com/vc-bonn/torchhull/pull/12))
- Move numpy to dev dependencies [@stotko](https://github.com/stotko) ([\#5](https://github.com/vc-bonn/torchhull/pull/5))


## [0.1.0] - 2024-11-06

- Initial version

[0.3.0]: https://github.com/vc-bonn/torchhull/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/vc-bonn/torchhull/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/vc-bonn/torchhull/releases/tag/v0.1.0
