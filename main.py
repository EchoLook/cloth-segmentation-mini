#!/usr/bin/env python3
"""
Main script for cloth segmentation model usage.
This script demonstrates how to use the process_image function to segment clothing in an image.
"""

import os
import argparse
from process import process_image

def main():
    """Example of using the cloth segmentation API"""
    
    parser = argparse.ArgumentParser(description='Cloth Segmentation Tool')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--save_overlay', action='store_true', help='Save overlay image')
    parser.add_argument('--save_centroids', action='store_true', help='Save centroids image')
    parser.add_argument('--threshold', type=float, default=0.0, help='Confidence threshold for segmentation (0.0-1.0)')
    args = parser.parse_args()
    
    # Make sure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Process the image
    print(f"Processing image: {args.image}")
    print(f"Using CUDA: {'Yes' if args.cuda else 'No'}")
    print(f"Output directory: {args.output}")
    print(f"Save overlay: {'Yes' if args.save_overlay else 'No'}")
    print(f"Save centroids: {'Yes' if args.save_centroids else 'No'}")
    print(f"Confidence threshold: {args.threshold}")
    print("Processing...")
    
    result = process_image(
        image_path=args.image,
        output_dir=args.output,
        use_cuda=args.cuda,
        save_overlay=args.save_overlay,
        save_centroids=args.save_centroids,
        confidence_threshold=args.threshold
    )
    
    # Print results
    print("\n=== Results ===")
    
    # Print optional paths if they were created
    if args.save_overlay and 'overlay_path' in result:
        print(f"Overlay image: {result['overlay_path']}")
    
    if args.save_centroids and 'centroids_path' in result:
        print(f"Centroid image: {result['centroids_path']}")
    
    print(f"JSON data: {result['json_path']}")
    
    print("\nClass images:")
    class_names = ["", "Upper body (red)", "Lower body (green)", "Full body (blue)"]
    for i, path in enumerate(result['class_paths']):
        cls = result['class_indices'][i]
        print(f"  Class {cls} ({class_names[cls]}): {path}")
    
    print("\nSegmentation info:")
    info = result['segmentation_info']
    print(f"Number of classes detected: {info['num_classes']}")
    for cls, data in info['classes'].items():
        class_name = ["", "Upper body (red)", "Lower body (green)", "Full body (blue)"][int(cls)]
        print(f"  Class {cls} ({class_name}):")
        print(f"    Center: ({data['center'][0]:.1f}, {data['center'][1]:.1f})")

if __name__ == "__main__":
    main() 