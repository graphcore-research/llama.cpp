#!/bin/bash

# Compiler
CXX=g++

# Compiler flags
CXXFLAGS="-O3 -march=native -ffast-math -Wall -Wextra -Werror"

# Source file
SRC="sparq.cpp"

# Output executable
OUT="sparq"

# Compile
$CXX $CXXFLAGS $SRC -o $OUT

# Check if compilation was successful
if [ $? -eq 0 ]; then
  # Run the executable
  ./$OUT
else
  echo "Compilation failed."
fi
