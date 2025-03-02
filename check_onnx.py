#!/usr/bin/env python3
"""
Script to check the ONNX model structure.
"""

import os
import onnxruntime as ort
import numpy as np

def main():
    """Check the ONNX model structure."""
    print("Checking ONNX model structure...")
    
    # Path to the ONNX model
    model_path = os.path.join('onnx', 'model.onnx')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load the ONNX model
    print(f"Loading model from {model_path}...")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    # Print the input names
    print("\nModel Inputs:")
    for i, input_info in enumerate(session.get_inputs()):
        print(f"  {i+1}. Name: {input_info.name}")
        print(f"     Shape: {input_info.shape}")
        print(f"     Type: {input_info.type}")
    
    # Print the output names
    print("\nModel Outputs:")
    for i, output_info in enumerate(session.get_outputs()):
        print(f"  {i+1}. Name: {output_info.name}")
        print(f"     Shape: {output_info.shape}")
        print(f"     Type: {output_info.type}")
    
    # Try to run the model with dummy inputs
    print("\nTrying to run the model with dummy inputs...")
    
    # Create dummy inputs based on the input shapes
    dummy_inputs = {}
    for input_info in session.get_inputs():
        # Create a dummy input with the right shape and type
        shape = input_info.shape
        # Replace dynamic dimensions with small values
        shape = [2 if dim is None or dim <= 0 else dim for dim in shape]
        
        # Create the dummy input
        if input_info.type == 'tensor(int64)':
            dummy_inputs[input_info.name] = np.zeros(shape, dtype=np.int64)
        else:
            dummy_inputs[input_info.name] = np.zeros(shape, dtype=np.float32)
    
    # Print the dummy inputs
    print("\nDummy Inputs:")
    for name, value in dummy_inputs.items():
        print(f"  {name}: shape={value.shape}, dtype={value.dtype}")
    
    # Try to run the model
    try:
        outputs = session.run(None, dummy_inputs)
        print("\nModel ran successfully!")
        print(f"Output shapes: {[output.shape for output in outputs]}")
    except Exception as e:
        print(f"\nError running the model: {str(e)}")
    
    print("\nCheck completed.")

if __name__ == "__main__":
    main() 