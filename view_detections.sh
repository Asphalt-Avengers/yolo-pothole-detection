#!/bin/bash

# Check if exactly two arguments are provided: input and output directories
# if [ "$#" -ne 2 ]; then
#   echo "Usage: $0 <input_directory> <output_directory>"
#   exit 1
# fi

# Assign arguments to variables
input_dir="/home/admin/dev/aws-utils/detections"
output_dir="/home/admin/dev/aws-utils/detection_images_only"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Initialize a counter for renaming files
counter=1

# Recursively find all files named "frame.jpg" (case-insensitive) in the input directory
find "$input_dir" -type f -iname "frame.jpg" | while read -r file; do
  # Create a new unique filename using the counter
  new_filename="frame_$(printf "%03d" "$counter").jpg"
  
  # Copy the file to the output directory with the new filename
  cp "$file" "$output_dir/$new_filename"
  
  echo "Copied: $file -> $output_dir/$new_filename"
  
  # Increment the counter for the next file
  counter=$((counter + 1))
done
