#!/usr/bin/env python3
"""
Edge Impulse Model (EIM) Diagnostic and Runner Script
This script loads and diagnoses an Edge Impulse model, providing inference capabilities.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Try importing Edge Impulse SDK
try:
    import ei_python_sdk
except ImportError:
    print("Warning: ei_python_sdk not found. Install with: pip install edge-impulse-linux-runner")
    ei_python_sdk = None


class EIMDiagnostics:
    """Diagnose and run Edge Impulse Models"""
    
    def __init__(self, model_path):
        """
        Initialize the EIM diagnostics
        
        Args:
            model_path: Path to the .eim model file
        """
        self.model_path = Path(model_path)
        self.model = None
        self.model_info = {}
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"✓ Model file found: {self.model_path}")
        print(f"  File size: {self.model_path.stat().st_size / (1024*1024):.2f} MB")
    
    def diagnose(self):
        """Run diagnostic checks on the model"""
        print("\n" + "="*60)
        print("EDGE IMPULSE MODEL DIAGNOSTICS")
        print("="*60)
        
        # Check file properties
        self._check_file_properties()
        
        # Try to load model metadata
        self._check_model_metadata()
        
        # System information
        self._check_system_info()
        
        print("\n" + "="*60)
        print("DIAGNOSTICS COMPLETE")
        print("="*60)
    
    def _check_file_properties(self):
        """Check EIM file properties"""
        print("\n[1] FILE PROPERTIES")
        print("-" * 60)
        print(f"Path: {self.model_path}")
        print(f"Filename: {self.model_path.name}")
        print(f"Size: {self.model_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"Architecture: {self._extract_arch_from_filename()}")
    
    def _extract_arch_from_filename(self):
        """Extract architecture info from filename"""
        filename = self.model_path.stem
        if 'aarch64' in filename:
            return "ARM64 (aarch64)"
        elif 'armv7' in filename:
            return "ARM32 (armv7)"
        elif 'x86_64' in filename:
            return "x86_64"
        else:
            return "Unknown"
    
    def _check_model_metadata(self):
        """Check model metadata"""
        print("\n[2] MODEL METADATA")
        print("-" * 60)
        print("Note: Detailed metadata extraction requires model loading")
        print("This requires Edge Impulse SDK and model quantization info")
        
        try:
            # Try to extract version from filename
            if 'v' in self.model_path.stem:
                version = self.model_path.stem.split('v')[-1]
                print(f"Model Version: v{version}")
        except Exception as e:
            print(f"Could not extract version: {e}")
    
    def _check_system_info(self):
        """Check system information"""
        print("\n[3] SYSTEM INFORMATION")
        print("-" * 60)
        
        import platform
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Python Version: {platform.python_version()}")
        print(f"Architecture: {platform.machine()}")
        
        # Check required packages
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required packages are installed"""
        print("\n[4] DEPENDENCIES")
        print("-" * 60)
        
        required_packages = {
            'numpy': 'Numerical computing',
            'edge_impulse_linux_runner': 'Edge Impulse SDK (optional)',
            'tensorflow': 'TensorFlow (optional)',
            'onnx': 'ONNX Runtime (optional)',
        }
        
        for package, description in required_packages.items():
            try:
                __import__(package)
                print(f"✓ {package:<35} - {description}")
            except ImportError:
                optional = "(optional)" in description
                symbol = "⚠" if optional else "✗"
                print(f"{symbol} {package:<35} - {description}")
    
    def load_model(self):
        """Attempt to load the model"""
        print("\n[5] MODEL LOADING")
        print("-" * 60)
        
        try:
            # This is a placeholder - actual loading depends on the SDK
            print("Attempting to load model...")
            print("Note: Actual model loading requires Edge Impulse Linux Runner")
            print("Install with: pip install edge-impulse-linux-runner")
            return True
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False
    
    def run_inference(self, input_data):
        """
        Run inference on the model
        
        Args:
            input_data: Input features as numpy array
            
        Returns:
            Predictions from the model
        """
        print("\n[6] RUNNING INFERENCE")
        print("-" * 60)
        
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data, dtype=np.float32)
        
        print(f"Input shape: {input_data.shape}")
        print(f"Input dtype: {input_data.dtype}")
        
        try:
            if self.model is None:
                print("Model not loaded. Load model first with load_model()")
                return None
            
            # predictions = self.model.predict(input_data)
            # print(f"Output shape: {predictions.shape}")
            # return predictions
            
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            return None


def print_usage_example():
    """Print example usage of the diagnostics"""
    print("\n" + "="*60)
    print("USAGE EXAMPLE")
    print("="*60)
    
    example_code = '''
# Basic Usage:
from run_eim_model import EIMDiagnostics

# Initialize diagnostics
model_path = "eim/nutribin-test-linux-aarch64-v3.eim"
diagnostics = EIMDiagnostics(model_path)

# Run diagnostics
diagnostics.diagnose()

# Load model (if SDK available)
diagnostics.load_model()

# Run inference with sample data
sample_input = np.random.rand(1, 224, 224, 3)  # Example: image data
results = diagnostics.run_inference(sample_input)
    '''
    
    print(example_code)


def main():
    """Main entry point"""
    
    # Define model path
    model_path = Path(__file__).parent / "eim" / "nutribin-test-linux-aarch64-v3.eim"
    
    # Create diagnostics instance
    try:
        diagnostics = EIMDiagnostics(str(model_path))
        
        # Run diagnostics
        diagnostics.diagnose()
        
        # Try to load model
        diagnostics.load_model()
        
        # Print usage example
        print_usage_example()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
