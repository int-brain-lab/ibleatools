# Building ibleatools Documentation

This directory contains the Sphinx documentation for ibleatools. This README explains how to build and serve the documentation locally for development purposes.

## Prerequisites

Before building the documentation, ensure you have the following installed:

1. **Python 3.10+** - The project requires Python 3.10 or higher
2. **ibleatools package** - Install the package in development mode:
   ```bash
   pip install -e .
   ```
3. **Sphinx and documentation dependencies** - Install the required packages:
   ```bash
   pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
   ```

## Building the Documentation

### Quick Start

To build the documentation, navigate to the `docs/` directory and run:

```bash
cd docs/
make html
```

This will generate the HTML documentation in the `build/html/` directory.

### Available Build Commands

The documentation supports several build targets:

- **`make html`** - Build HTML documentation (default)
- **`make clean`** - Clean the build directory
- **`make help`** - Show all available build options
- **`make livehtml`** - Build HTML with live reload (requires `sphinx-autobuild`)

### Windows Users

If you're on Windows, use the provided batch file instead:

```cmd
cd docs/
make.bat html
```

## Viewing the Documentation

After building, you can view the documentation by:

1. **Opening the HTML files directly:**
   - Navigate to `docs/build/html/`
   - Open `index.html` in your web browser

2. **Using a local web server:**
   ```bash
   cd docs/build/html/
   python -m http.server 8000
   ```
   Then open http://localhost:8000 in your browser

## Development Workflow

### Making Changes

1. Edit the source files in `docs/source/`
2. Rebuild the documentation: `make html`
3. Refresh your browser to see changes

### Live Reload (Optional)

For automatic rebuilding when files change, install `sphinx-autobuild` and use:

```bash
pip install sphinx-autobuild
make livehtml
```

This will start a local server that automatically rebuilds and refreshes the documentation when you make changes.

## Documentation Structure

The documentation is organized as follows:

```
docs/
├── source/           # Sphinx source files
│   ├── index.rst     # Main documentation page
│   ├── ephysatlas.rst # Package overview
│   ├── features.rst  # Feature computation module
│   ├── plots.rst     # Visualization module
│   ├── reveal.rst    # High-level interface
│   ├── utils.rst     # Utility functions
│   ├── how-to/       # Tutorials and guides
│   └── reference/    # API reference
├── build/            # Generated documentation (git-ignored)
├── Makefile          # Build commands
└── make.bat          # Windows build commands
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure ibleatools is installed in development mode (`pip install -e .`)
2. **Missing dependencies**: Install all required packages listed in the Prerequisites section
3. **Build errors**: Check that all source files are valid RST format
4. **Permission errors**: Ensure you have write permissions to the `build/` directory

### Cleaning Build Artifacts

If you encounter build issues, clean the build directory:

```bash
make clean
make html
```

## Contributing to Documentation

When contributing to the documentation:

1. Follow the existing RST format and structure
2. Use Google-style docstrings in Python code (as specified in CONTRIBUTING.md)
3. Test your changes by building the documentation locally
4. Ensure all links and references work correctly

## Online Documentation

The latest documentation is available online at:(This is TODO) [https://int-brain-lab.github.io/ibleatools](https://int-brain-lab.github.io/ibleatools)

For questions or issues with the documentation, please open an issue on the [ibleatools GitHub repository](https://github.com/int-brain-lab/ibleatools).
