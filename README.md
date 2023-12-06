To compile on Saga:

ml CUDA/12.0.0

nvcc multiple_streams.cu -o m_streams -lcusparse

To compile on LUMI-G:

ml rocm

hipcc multiple_streams.hip -o m_streams -I hipsparse -lhipsparse
